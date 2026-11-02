#!/usr/bin/env python3

##########################################
##                                      ##
##            ~ Light DQM ~             ##
##                                      ##
##########################################
##
##  Written 09.07.2025

##  Updated 15.06.2026
##
##    - James Mead <jmead@nikhef.nl>
##    - Sindhujha Kumaran <s.kumaran@uci.edu>
##    - Angela White <ajwhite@uchicago.edu> *active editor
##
###########################################
'''
    # example usage:

python light_dqm.py
                       --input_path /global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_light/warm_commission/
                       --file_syntax mpd_run_dbg_rctl_
                       --channel_status_file light_dqm/channel_status.csv
                       --output_dir dqm_plots/
                       --tmp_dir tmp/
                       --units ADC16
                       --ptps16bit 150
                       --start_run 0
                       --nfiles 10
                       --ncomp 5
                       --powspec_nevts 200
                       --max_evts 500
                       --write_json_blobs False
                       --merge_grafana_plots False
                       --plot_all_clipped False
                       --plot_all_negatives False
'''
############################################

import os
import sys
import time
import glob
import argparse
import traceback
from ascii import header, footer
import textwrap

import json
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from scipy.fft import rfft, rfftfreq
from scipy.stats import beta

from PyPDF2 import PdfMerger
from PyPDF2 import PdfReader, PdfWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Process and plot DUNE light file data.")
    parser.add_argument('--input_path', type=str, default='.', help='Path to input file')
    parser.add_argument('--file_syntax', type=str, default='.', help='File name syntax')
    parser.add_argument('--channel_status_file', type=str, default='actions/light_dqm/channel_status.csv', help='Channel status file')
    parser.add_argument('--output_dir', type=str, default='dqm_plots/', help='Directory to save final output plots')
    parser.add_argument('--tmp_dir', type=str, default='tmp/', help='Directory to save temporary output plots')
    parser.add_argument('--units', type=str, default='ADC16', choices=['ADC16', 'ADC14', 'V'], help='Units for waveform')
    parser.add_argument('--ptps16bit', type=int, default=500, help='Peak-to-peak threshold for 16-bit ADC')
    parser.add_argument('--start_run', type=int, default=0, help='Starting file index for processing')
    parser.add_argument('--nfiles', type=int, default=1, help='Number of files to process')
    parser.add_argument('--ncomp', type=int, default=-1, help='Number of previous files to compare')
    parser.add_argument('--powspec_nevts', type=int, default=500, help='Number of events to process per file for noise spectra')
    parser.add_argument('--max_evts', type=int, default=500, help='Maximum number of events to process for the whole file')
    parser.add_argument('--write_json_blobs', type=bool, default=False, help='Write noise spectra blobs to json files')
    parser.add_argument('--merge_grafana_plots', type=bool, default=False, help='Merge baselines and flatlines grafana plots')
    parser.add_argument('--plot_all_clipped', type=bool, default=False, help='Plot all clipped waveform plots')
    parser.add_argument('--plot_all_negatives', type=bool, default=False, help='Plot all negative baseline plots')

    return parser.parse_args()

# ----------------------------- #
#         Global Constants      #
# ----------------------------- #

args = parse_args()
ptps16bit = args.ptps16bit
default_units = args.units

SAMPLE_RATE = 0.016  # us per sample
MIN_GB_PER_FILE = 1  # GB value used to determine how many events in each file are skipped.
adc14_max = 8191
adc14_16 = 2**2      # Conversion factor between 14 and 16 bit ADC

ADC_V_range = 2.0    # Voltage range of ADC
ADC_V_offset = -1.0  # Voltage offset

# Precomputed peak-to-peak values for different units
ptps_16 = np.array([ptps16bit]*8)
ptps_14 = ptps_16 / adc14_16
ptps_V = ptps_16 * (ADC_V_range) / (adc14_max * adc14_16)

EXT = f".{os.getenv('ND_NEARLINE_TAG')}.FLOW.hdf5"

def get_ptps(units):
    """
    Get peak-to-peak values for the specified units.

    Args:
        units (str): 'ADC16', 'ADC14', or 'V'

    Returns:
        np.ndarray: Peak-to-peak values for each ADC
    """
    if units == 'ADC16':
        return ptps_16
    elif units == 'ADC14':
        return ptps_14
    elif units == 'V':
        return ptps_V
    else:
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")

# Frequency regions of interest (MHz) and window width (MHz)
vlist = [0.5e6, 1.8e6, 4.6e6, 7.1e6, 8.5e6, 10e6, 11.5e6, 19e6, 20e6, 25e6, 30e6]
window = 0.4e6

# Channel mapping: select channels 4-15 in each group of 16
channels = []
for group_start in range(0, 64, 16):
    channels.extend(range(group_start + 4, min(group_start + 16, 64)))

# Load channel status (0 = good, nonzero = bad)
channel_status_csv = args.channel_status_file
cs = None  # default if load fails

try:
    cs_df = pd.read_csv(channel_status_csv, header=None)
    cs = cs_df.to_numpy()
    print(f"Channel status loaded successfully from: {channel_status_csv}")
except FileNotFoundError:
    print(f"Channel status file not found, skipping: {channel_status_csv}")
except pd.errors.EmptyDataError:
    print(f"Channel status file is empty: {channel_status_csv}")
except Exception as e:
    print(f"Error loading channel status from {channel_status_csv}: {e}")


# ----------------------------- #
#         Core Functions        #
# ----------------------------- #

# def numpy friendly clopper-pearson function
def clopper_pearson(passed, total, interval=0.68):

    # raise error if passed > total
    if np.any(passed > total):
        raise ValueError("Passed events cannot be greater than total events.")

    alpha = 1 - interval
    # Ensure arrays are numpy arrays
    passed = np.asarray(passed)
    total = np.asarray(total)
    # Initialize lower and upper arrays
    lower = np.zeros_like(passed, dtype=float)
    upper = np.ones_like(passed, dtype=float)
    # Mask for valid entries (total > 0)
    valid = total > 0
    # For valid entries, compute lower and upper bounds
    lower[valid] = beta.ppf(alpha/2, passed[valid], total[valid] - passed[valid] + 1)
    upper[valid] = beta.ppf(1 - alpha/2, passed[valid] + 1, total[valid] - passed[valid])
    # Replace NaNs with 0 and 1
    lower = np.nan_to_num(lower, nan=0.0)
    upper = np.nan_to_num(upper, nan=1.0)
    # Fraction calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.true_divide(passed, total)
        frac = np.nan_to_num(frac, nan=0.0)
    # If total is zero, set frac to 0
    frac = np.where(total == 0, 0.0, frac)
    # If frac is 0 and total is not zero, set lower and upper to 0 and 1 respectively
    mask = (frac == 0) & (total != 0)
    lower = np.where(mask, 0.0, lower)
    upper = np.where(mask, 1.0, upper)

    # error if err low < 0 or err high > 100
    if np.any(frac < lower) or np.any(frac > upper):
        raise ValueError("Calculated fraction is outside the confidence interval.")
    if np.any(lower < 0) or np.any(upper > 1):
        raise ValueError("Confidence interval bounds are out of range [0, 1].")

    # Calculate errors
    err_low = 100 * (frac - lower)
    err_up = 100 * (upper - frac)
    pct = 100 * frac

    # Ensure negative error is zero when central value is zero (i.e., no entries)
    mask_no_entries = (total == 0) | (pct == 0)
    err_low[mask_no_entries] = 0.0

    return pct, err_low, err_up


# Convert ADC16 counts to voltage
def adc16_to_voltage(adc_counts, mask=None):
    if mask is not None:
        adc_counts = np.where(mask[..., np.newaxis], adc_counts, 0)
    return adc_counts * (ADC_V_range + ADC_V_offset) / (adc14_max * adc14_16)


# Get waveform information for a given event
def get_waveform_info(waveform, units='ADC16', mask=None, ths=None):
    if mask is not None:
        # mask is expected to be a list/array of event indices
        waveform = waveform[mask]
    # check if the waveform is clipped
    #clipped = np.any(waveform >= (adc14_max-1)*adc14_16, axis=-1)
    # define units
    if units == 'ADC14':
        waveform = waveform / adc14_16
    elif units == 'V':
        waveform = adc16_to_voltage(waveform)
    elif units != 'ADC16':
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")
    # take the stdandard deviation of the first 50 samples
    noise = np.std(waveform[:, :, :, :50], axis=3)
    # take the mean of the first 50 sample
    baseline = np.mean(waveform[:, :, :, :50], axis=3)
    # subtract the baseline from the waveform
    max_value = np.max(waveform - baseline[:, :, :, np.newaxis], axis=3)
    # check for negative values below ptps
    #negs = np.any(waveform - baseline[:, :, :, np.newaxis] < -ths[np.newaxis, :, np.newaxis, np.newaxis], axis=-1)
    # wvfms of specified events
    wvfms = waveform[:, :, :, :]

    # return the baseline, max value, and clipped status
    return wvfms, noise, baseline, max_value#, clipped, negs


### GRAFANA PLOTS ###

# function for checking if channels max range flatlines at any point
def check_flatline(max_values, threshold=0.1):
    # check if the max range is flatlined at any point
    flatlined = np.all(max_values < threshold, axis=0)
    return flatlined

# function to plot an 8x64 grid with red cross for flatlined, green circle for not flatlined
def plot_flatline_mask(flatline_mask, channel_status=None, times=None,
                       output_name='flatline_mask.png', grafana=True):
    n_adcs = flatline_mask.shape[0]
    n_channels = flatline_mask.shape[1]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(-0.5, n_adcs - 0.5)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_adcs))
    ax.set_xlabel('Channel')
    ax.set_ylabel('ADC')
    ax.set_xticklabels(np.arange(n_channels))
    ax.set_yticklabels(np.arange(n_adcs))

    for i in range(n_adcs):
        for j in range(n_channels):
            alpha = 1.0
            # Skip inactive channels
            if channel_status is not None and channel_status[i, j] == -1:
                continue
            if grafana:
                if channel_status is not None and channel_status[i, j] > 0: #in [1,3]:
                        ax.plot(j, i, marker='.', color='black', markersize=10)
                elif flatline_mask[i, j]:
                        ax.text(
                            j, i-0.1, "☹️",  # or "😢"
                            color="red",
                            fontsize=14,
                            ha="center",
                            va="center"
                        )
                else:
                    ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')


            else:
                if channel_status is not None and channel_status[i, j] > 0: #in [1,3]:
                    ax.plot(j, i, marker='.', color='black', markersize=10)
                elif flatline_mask[i, j]:
                    ax.plot(j, i, marker='x', color='red', markersize=12, markeredgewidth=2)
                else:
                    ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')


    plt.title("Alive and dead channels", y=1.18, fontsize=14)
    plt.tight_layout()

    # add faded line at x=31.5
    ax.axvline(x=31.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    # add faded line at y = 1.5, 3.5, 5.5, 7.5
    for y in range(1, n_adcs):
        if y % 2 == 0:
            ax.axhline(y=y - 0.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)


    legend_handles = [
        Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=10,
               label='Ignore known problem channels'),
        Line2D([0], [0], marker='o', color='green', markerfacecolor='none',
               linestyle='None', markersize=10, label='Still living')
    ]

    if grafana:
        legend_handles.append(
            Line2D([0], [0], linestyle='None', marker=r"$☹$", color='red',
                   markersize=12, markeredgewidth=0.2, label='Dead (contact LRS expert)')
        )
    else:
        legend_handles.append(
            Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10,
                   label='Dead (contact LRS expert)')
        )

    # Place legend above the plot, in one lineer', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    # add start and end times to the top left
    if times is not None:
        # Place the time text above the plot, aligned with the legend
        ax.text(0.01, 1.15, f'Start: {times[0]}\nEnd:  {times[1]}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # Make the figure slightly larger to fit the legend
    fig.set_size_inches(16, 4)
    # Increase the top margin to fit the legend
    plt.subplots_adjust(top=0.82)
    output = f"{output_name}"
    plt.savefig(output)


def check_baseline(prev_baseline, current_baseline, units='ADC16', threshold=200, k_sigma=3.0, adc14_16=4.0):
    """
    Compare previous and current baseline arrays.
    Each baseline is [central, lower, upper], each shape (8, 64).
    Returns a boolean mask (8, 64) where True indicates a significant change,
    and a status array (8, 64) indicating the reason:
      1 = current uncertainty too large
      2 = previous uncertainty too large
      3 = significant directional change
    Directional test:
      - If curr > prev, use sqrt(curr_lower^2 + prev_upper^2)
      - If curr < prev, use sqrt(curr_upper^2 + prev_lower^2)
    """
    # Adjust threshold for ADC14 units
    thr = threshold / adc14_16 if units == 'ADC14' else threshold

    curr_c, curr_l, curr_u = current_baseline
    mask = np.zeros_like(curr_c, dtype=bool)
    status = np.zeros_like(curr_c, dtype=int)

    # check against assumed range
    min_allowed = np.zeros_like(curr_c)
    max_allowed = np.zeros_like(curr_c)
    min_allowed[...] = -30000
    max_allowed[...] = -26000
    # for index 6 (ADC 6), set min_allowed to 15000, max_allowed to 20000
    #min_allowed[6, :] = -26000
    #max_allowed[6, :] = -22000
    mask_range = (curr_c < min_allowed) | (curr_c > max_allowed)
    mask |= mask_range
    status[mask_range] = -1

    # Current uncertainty too large
    curr_margin = np.maximum(curr_l, curr_u)
    mask_curr = curr_margin > thr
    mask |= mask_curr
    status[mask_curr] = 1

    if prev_baseline is None:
        return mask, status

    # checking against previous baseline
    prev_c, prev_l, prev_u = prev_baseline

    # Directional difference
    diff = curr_c - prev_c
    abs_diff = np.abs(diff)

    # Directional combined uncertainty
    sigma_sq = np.where(
        diff > 0,
        curr_l**2 + prev_u**2,
        curr_u**2 + prev_l**2
    )
    sigma = np.sqrt(np.maximum(sigma_sq, np.finfo(float).eps))
    mask_dir = abs_diff > (k_sigma * sigma)

    # Only apply mask_dir where mask is not already True
    mask |= mask_dir & (~mask)
    status[mask_dir] = 2

    # status == 3 if both masks are True
    status[mask & mask_dir] = 3

    return mask, status


def plot_baseline_mask(baseline_mask, channel_status=None, times=None,
                       output_name='baseline_mask.png', grafana=True):
    """
    Plot an 8x64 grid with red cross for baseline outliers, green circle for normal.
    Optionally mark known bad channels with a black dot.
    """
    n_adcs = baseline_mask.shape[0]
    n_channels = baseline_mask.shape[1]
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(-0.5, n_adcs - 0.5)
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_adcs))
    ax.set_xlabel('Channel')
    ax.set_ylabel('ADC')
    ax.set_xticklabels(np.arange(n_channels))
    ax.set_yticklabels(np.arange(n_adcs))

    for i in range(n_adcs):
        for j in range(n_channels):
            # Skip inactive channels
            if channel_status is not None and channel_status[i, j] == -1:
                continue

            if grafana:
                if channel_status is not None and channel_status[i, j] in [2,3]:
                        ax.plot(j, i, marker='.', color='black', markersize=10)
                elif baseline_mask[i, j]:
                    ax.text(
                        j, i-0.1, "☹️",  # or "😢"
                        color="red",
                        fontsize=14,
                        ha="center",
                        va="center"
                    )

                else:
                    ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')

            else:
                if baseline_mask[i, j]:
                    ax.plot(j, i, marker='x', color='red', markersize=12, markeredgewidth=2)
                else:
                    ax.plot(j, i, marker='o', color='green', markersize=10, fillstyle='none')
                if channel_status is not None and channel_status[i, j] in [2,3]:
                    ax.plot(j, i, marker='.', color='black', markersize=10)

    plt.title("Channels baseline status", y=1.18, fontsize=14)
    plt.tight_layout()
    # add faded line at x=31.5
    ax.axvline(x=31.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    # add faded line at y = 1.5, 3.5, 5.5, 7.5
    for y in range(1, n_adcs):
        if y % 2 == 0:
            ax.axhline(y=y - 0.5, color='grey', linestyle='-', linewidth=1, alpha=0.5)
    legend_handles = [
        Line2D([0], [0], marker='.', color='black', linestyle='None', markersize=10,
               label='Ignore known problem channels'),
        Line2D([0], [0], marker='o', color='green', markerfacecolor='none',
               linestyle='None', markersize=10, label='Stable')
    ]

    if grafana:
        legend_handles.append(
            Line2D([0], [0], linestyle='None', marker=r"$☹$", color='red',
                   markersize=12, markeredgewidth=0.2, label='Deviation (contact LRS expert)')
        )
    else:
        legend_handles.append(
            Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=10,
                   label='Deviation (contact LRS expert)')
        )


    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    # add start and end times to the top left
    if times is not None:
        # Place the time text above the plot, aligned with the legend
        ax.text(0.01, 1.15, f'Start: {times[0]}\nEnd:  {times[1]}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # Make the figure slightly larger to fit the legend
    fig.set_size_inches(16, 4)
    plt.subplots_adjust(top=0.82)
    output = f"{output_name}"
    plt.savefig(output)


### DQM PLOTS ###

def tag_large_events(WVFM_ARRAY, ADC_LIST=None, TICKS=600):

    large_event_wvfm = np.zeros((8,64,TICKS), dtype=np.int64)
    SIPM_CHANNELS = ([4,5,6,7,8,9] + \
                     [10,11,12,13,14,15] + \
                     [20,21,22,23,24,25] + \
                     [26,27,28,29,30,31] + \
                     [36,37,38,39,40,41] + \
                     [42,43,44,45,46,47] + \
                     [52,53,54,55,56,57] + \
                     [58,59,60,61,62,63])

    if ADC_LIST is None:
        ADC_LIST = [0,1,2,3,4,5,6,7]
    for adc in ADC_LIST:
        CHANNEL_LIST = [6,10,24,26,36,42,52,58]
        for ch in CHANNEL_LIST:
            mask = np.argmax(np.max(WVFM_ARRAY[:, adc, ch, :TICKS], axis=-1))
            #max_value = np.max(WVFM_ARRAY[mask, adc, ch, :], axis=-1)

            for e_ch in range(6):
                large_event_wvfm[adc, ch+e_ch, :] += np.array(WVFM_ARRAY[mask, adc, ch+e_ch, :TICKS]/4, dtype=np.int16)
            del mask
            #del max_value

    return large_event_wvfm

# Get the sum waveform for a given event and units
def get_sum_waveform(waveform, units='ADC16', mask=None, clip=True):
    # waveform shape: (n_events, n_adcs, n_channels, n_samples)
    if mask is not None:
        waveform = waveform[mask]
    if units == 'ADC14':
        waveform = waveform / adc14_16
    elif units == 'V':
        waveform = adc16_to_voltage(waveform)
    elif units != 'ADC16':
        raise ValueError("Units must be 'ADC14', 'ADC16', or 'V'.")
    # sum over samples axis
    sum_waveform = np.sum(waveform, axis=0)
    # clip the waveform if requested to mimic non-baselined oscilloscope limit
    if clip:
        sum_waveform = np.clip(sum_waveform, 0, 8192*4 - 1)
    return sum_waveform


# Plot the sum waveform per EPCB for a single event
def plot_sum_waveform(waveform, units='ADC16', output_name='sum_waveform.pdf'):
    # waveform shape: (n_events, n_adcs, n_channels, n_samples)
    n_epcbs = int(len(channels) // 6)
    n_adcs = waveform.shape[0]
    n_samples = waveform.shape[-1]
    nrows = n_adcs * (n_epcbs // 2)
    ncols = 2
    _, axes = plt.subplots(nrows, ncols, figsize=(20, 2 * nrows), sharex=True)

    # Ensure axes is always 2D
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i * (n_epcbs // 2) + (j // 2)
            tt_idx = j % 2
            channels_list = channels[j * 6:(j + 1) * 6]
            epcb_waveform = waveform[i, channels_list, :]
            #else:
            #    epcb_waveform = waveform[signal_mask[0], i, channels_list, :]
            # loop over channels
            for ch in range(len(channels_list)):
                axes[idx, tt_idx].plot(epcb_waveform[ch, :], label=f'Channel {channels_list[ch]}')
            # plot total
            sum_wvfm = get_sum_waveform(epcb_waveform, units=units)
            # set title
            #axes[idx, tt_idx].set_title(f'Event {i_evt} - Non-beam trigger timing')
            axes[idx, tt_idx].set_title(f'Sanity Check: IN-PROGRESS')
            axes[idx, tt_idx].plot(sum_wvfm, color='black', alpha=0.5, label='Sum ADC')
            axes[idx, tt_idx].set_title(f'ADC {i} - EPCB {j} summed waveforms')
            axes[idx, tt_idx].set_ylabel(f'{units} counts')
            axes[idx, tt_idx].legend(loc='upper right', fontsize='small')
            axes[idx, tt_idx].grid(True)
            axes[idx, tt_idx].set_xlim(0, n_samples)
            axes[idx, tt_idx].set_ylim(-5000, 38000)
    axes[-1, 0].set_xlabel('Samples/ time (ticks)')
    axes[-1, 1].set_xlabel('Samples/ time (ticks)')
    plt.tight_layout()
    #plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()


# Plot noise for each ADC channel
def plot_noises(prev_noises, noises, i_evt, mask_inactive=True,
                format_bad_channels=True, output_name='noises.pdf'):
    n_adcs = noises.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]

    medians = np.median(noises[i_evt,:,:], axis=0)
    lowers = np.percentile(noises[i_evt,:,:], 16, axis=0) - medians
    uppers = np.percentile(noises[i_evt,:,:], 84, axis=0) - medians

    for i in range(n_adcs):
        ax = axes[i]
        channels_idx = np.arange(noises.shape[2])
        # mask inactive channels
        if mask_inactive:
            mask = np.isin(channels_idx, channels)
        else:
            mask = np.ones_like(channels_idx, dtype=bool)
        # Default values
        median = np.full(noises.shape[2], np.nan)
        lower  = np.full(noises.shape[2], np.nan)
        upper  = np.full(noises.shape[2], np.nan)

        if isinstance(i_evt, np.ndarray) and len(i_evt) > 1:
            # Get 68% central quantile range (16th and 84th percentiles)
            median = medians[i, :]
            lower = -lowers[i, :]
            upper = uppers[i, :]
            # Set alpha to 0.5 for bad channels (cs != 0)
            if format_bad_channels and cs.shape == (n_adcs, len(channels_idx)):
                alphas = np.ones_like(median)
                bad_mask = cs[i, :] != 0
                alphas[bad_mask] = 0.25
                # Plot previous files with error bars
                if prev_noises is not None:
                    prev_centre, prev_lower, prev_upper = prev_noises
                    prev_centre = prev_centre[i, mask]
                    prev_lower = prev_lower[i, mask]
                    prev_upper = prev_upper[i, mask]
                    for ch, c, l, u in zip(
                        channels_idx[mask], prev_centre, prev_lower, prev_upper
                    ):
                        ax.step(
                            [ch - 0.5, ch + 0.5], [c, c],
                            color='red', alpha=alphas[ch]/2, where='post', linewidth=1
                        )
                        ax.fill_between(
                            [ch - 0.5, ch + 0.5], [c - l, c - l], [c + u, c + u],
                            color='red', alpha=alphas[ch]/4, step='post'
                        )
                # Plot current noise with error bars
                for ch in channels_idx[mask]:
                    ax.errorbar(
                        ch, median[ch], yerr=[[lower[ch]], [upper[ch]]],
                        fmt='.', color='b', markersize=5,
                        linewidth=0.5, capsize=5, capthick=1,
                        alpha=alphas[ch]
                    )
        # formatting
        ax.set_title(f'ADC {i} Noise')
        ax.set_ylabel('Noise (ADC counts)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel('channel #')
    # adjust layout
    plt.tight_layout()

    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return medians, lowers, uppers

#Check for HV instabilities
def get_fprompt_estimate(WVFM_ARRAY, ARRIVAL_TICK=77, THRESHOLD=600):

    NUM_EVENTS = np.shape(WVFM_ARRAY)[0]
    N_ADCS = np.shape(WVFM_ARRAY)[1]

    save_array = np.zeros((NUM_EVENTS, N_ADCS))   

    fast_max = WVFM_ARRAY[:,:,:,ARRIVAL_TICK:(ARRIVAL_TICK+14)].max(axis=-1)
    slow_max = WVFM_ARRAY[:,:,:,(ARRIVAL_TICK+14):(ARRIVAL_TICK+201)].max(axis=-1)
    max_mask = (fast_max > slow_max) & (fast_max > THRESHOLD//4)
    del slow_max
    del fast_max
    # Integrate the fast component
    fast_int = WVFM_ARRAY[:,:,:,ARRIVAL_TICK:(ARRIVAL_TICK+14)].sum(axis=-1)
    # Integrate the total signal
    total_int = WVFM_ARRAY[:,:,:,ARRIVAL_TICK:(ARRIVAL_TICK+201)].sum(axis=-1)
    #del WVFM_ARRAY
    # build the fprompt array
    fprompt = fast_int / total_int
    del fast_int
    del total_int

    fprompt_mask = ((fprompt > 0) & (fprompt < 0.25)) & (max_mask==1)
    save_array[:,:] = (fprompt_mask).sum(axis=-1)

    del fprompt
    del fprompt_mask
    del max_mask

    return save_array

# Plot the count of HV instability occurrences
def plot_fprompt_rates(HVINST_ARRAY, FPROMPT_THD=20, FILE_LENGTH=300, MULT=4, N_ADCS=8, output_name='hv_instabilities_01.pdf'):
    fig, ax = plt.subplots(figsize=(12,4))
    x_array = np.arange(0, FILE_LENGTH, np.float32(FILE_LENGTH/(np.shape(HVINST_ARRAY)[0])))
    if len(x_array) > np.shape(HVINST_ARRAY)[0]:
        x_array = x_array[1:]
    elif len(x_array) < np.shape(HVINST_ARRAY)[0]:
        HVINST_ARRAY = HVINST_ARRAY[:len(x_array),:,:]
        
    color_array = ['#733957', '#A3672C', '#C3A34B',
                '#D6D893', '#B4DEC6', '#74BBCD',
                '#4F88B9', '#5C538B']

    style_array = [':', '-', ':',
                '-', ':', '-',
                ':', '-']

    for i in range(N_ADCS):
        ax.plot(x_array, np.cumsum(HVINST_ARRAY[:,i]), color=color_array[i], linestyle=style_array[i], label=f'ADC {i}', linewidth=2)
    #print('End Game Sums:', (HVINST_ARRAY.sum(axis=0)*MULT)//FILE_LENGTH, ' Hz')
    ax.axhline(y=FPROMPT_THD, color='crimson', linestyle='--', label='Limit', linewidth=2)
    ax.axhspan(ymin=FPROMPT_THD, ymax=1500, color='crimson', alpha=0.2)
    ax.legend(loc='upper right', ncols=3, fontsize=12)
    ax.set_ylim(0.5, 1500)
    ax.set_yscale('log')
    ax.set_xlim(x_array[0]-2,x_array[-1]+2)
    ax.set_xlabel('Seconds Into File [s]', fontsize=14)
    ax.set_ylabel('Cumulative Irregularities', fontsize=14)
    textstr = f'If Any ADC Exceeds Limit, Screenshot & Send to @2x2_lrs_expert on Slack'
    #textstr = f'Particle Type: protons\n Vertex in 2x2'
    #props = dict(boxstyle='round', facecolor='white', edgecolor='')
    ax.text(
        0.5, 0.55, textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='center',
        color='maroon'
    )
    #plt.text(x=5,  y=8, s='If Any ADC Exceeds Limit, Screenshot & Send to @2x2_lrs_expert on Slack', rotation=0, verticalalignment='center', fontsize=12, color='crimson')
    plt.title(f'Channels with Fprompt [0, 0.25] Per File', fontsize=16, loc='left', y=0.88, x=0.02)
    plt.tight_layout()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

    return

# Plot the time of HV instability occurrences
def plot_fprompt_occurrences(HVINST_ARRAY, FILE_LENGTH=300, N_ADCS=8, output_name='hv_instabilities_02.pdf'):
    fig, ax = plt.subplots(figsize=(12,3))
    x_array = np.arange(0, FILE_LENGTH, np.float32(FILE_LENGTH/(np.shape(HVINST_ARRAY)[0])))
    if len(x_array) > np.shape(HVINST_ARRAY)[0]:
        x_array = x_array[1:]
    elif len(x_array) < np.shape(HVINST_ARRAY)[0]:
        HVINST_ARRAY = HVINST_ARRAY[:len(x_array),:,:]
        
    color_array = ['#733957', '#A3672C', '#C3A34B',
                '#D6D893', '#B4DEC6', '#74BBCD',
                '#4F88B9', '#5C538B']

    style_array = [':', '-', ':',
                '-', ':', '-',
                ':', '-']

    for i in range(N_ADCS):
        ax.plot(x_array, HVINST_ARRAY[:,i], color=color_array[i], linestyle=style_array[i], label=f'ADC {i}', linewidth=2)
    #ax.axhline(y=6, color='crimson', linestyle='--', label='Limit', linewidth=2)
    #ax.axhspan(ymin=6, ymax=49, color='crimson', alpha=0.2)
    ax.legend(loc='upper right', ncols=3, fontsize=12)
    ax.set_ylim(-1, 12)
    ax.set_xlim(x_array[0]-2,x_array[-1]+2)
    ax.set_xlabel('Seconds Into File [s]', fontsize=14)
    ax.set_ylabel('Irregular Channels', fontsize=14)
    plt.title(f'Channels with Fprompt [0, 0.25] Per File', fontsize=16, loc='left', y=0.80, x=0.02)
    plt.tight_layout()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

    return

# Plot baselines for each ADC channel
def plot_baselines(prev_baselines, baselines, i_evt, mask_inactive=True,
                   format_bad_channels=True, output_name='baselines.pdf'):
    n_adcs = baselines.shape[1]
    axes = plt.subplots(n_adcs, 1, figsize=(10, 2*n_adcs), sharex=True)[1]
    # Initialize arrays to store median, lower, and upper bounds
    medians = np.median(baselines[i_evt,:,:], axis=0)
    lowers = np.percentile(baselines[i_evt,:,:], 16, axis=0) - medians
    uppers = np.percentile(baselines[i_evt,:,:], 84, axis=0) - medians
    # Loop over each ADC
    for i in range(n_adcs):
        ax = axes[i]
        channels_idx = np.arange(baselines.shape[2])
        # mask inactive channels
        if mask_inactive:
            mask = np.isin(channels_idx, channels)
        else:
            mask = np.ones_like(channels_idx, dtype=bool)

        if isinstance(i_evt, np.ndarray) and len(i_evt) > 1:
            # Get 68% central quantile range (16th and 84th percentiles)
            median = medians[i, mask]
            lower = -lowers[i, mask]
            upper = uppers[i, mask]

            # Set alpha to 0.5 for bad channels (cs != 0)
            if format_bad_channels and cs.shape == (n_adcs, len(channels_idx)):
                alphas = np.ones_like(median)
                bad_mask = cs[i, :] != 0
                alphas[bad_mask] = 0.25

                # Plot previous files with error bars
                if prev_baselines is not None:
                    prev_centre, prev_lower, prev_upper = prev_baselines
                    prev_centre = prev_centre[i, mask]
                    prev_lower = prev_lower[i, mask]
                    prev_upper = prev_upper[i, mask]
                    for ch, c, l, u in zip(
                        channels_idx[mask], prev_centre, prev_lower, prev_upper
                    ):
                        ax.step(
                            [ch - 0.5, ch + 0.5], [c, c],
                            color='red', alpha=alphas[ch]/2, where='post', linewidth=1
                        )
                        ax.fill_between(
                            [ch - 0.5, ch + 0.5], [c - l, c - l], [c + u, c + u],
                            color='red', alpha=alphas[ch]/4, step='post'
                        )
                # Plot current baseline with error bars
                for ch in channels_idx[mask]:
                    ax.errorbar(
                        ch, median[ch], yerr=[[lower[ch]], [upper[ch]]],
                        fmt='.', color='b', markersize=5,
                        linewidth=0.5, capsize=5, capthick=1,
                        alpha=alphas[ch]
                    )
        # formatting
        ax.set_title(f'ADC {i} Baselines')
        ax.set_ylabel('Baseline (ADC counts)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # adjust layout
    axes[-1].set_xlabel('channel #')
    plt.tight_layout()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()
    return medians, lowers, uppers


def get_max_value_mask(max_values, ptps, cs=None):
    # if ptps is a single value, convert it to a list of the same length as the number of ADCs
    if isinstance(ptps, (int, float)):
        ptps = [ptps] * 8
    elif len(ptps) != 8:
        raise ValueError("ptps must be a single value or a list of length 8")
    # Vectorized mask: True where max_values < ptps[adc] and channel_status == 0
    max_mask = max_values < (np.array(ptps)[np.newaxis, :, np.newaxis])
    if cs is not None:
        ch_mask = (cs == 0)[np.newaxis, :, :]
        mask = max_mask & ch_mask
    else:
        mask = max_mask
    return mask

def get_noise_spectra(waveform, mask=None):
    # mask waveform, set to Nan
    if mask is not None:
        waveform = np.where(mask[..., np.newaxis], waveform, np.nan)
    n_samples = waveform.shape[3]
    # calculate the FFT of the waveform
    fft_N = rfft(waveform, axis=-1) / int(n_samples/2+1)
    # calculate the power spectrum
    power_spectra = 2*np.abs(fft_N)**2
    upper_quantile, power_spectrum, lower_quantile = np.nanquantile(power_spectra, [0.84, 0.5, 0.16], axis=0)
    # calculate the frequency bins
    freq_bins = rfftfreq(n_samples, d=SAMPLE_RATE * 1e-6)
    return freq_bins, power_spectra, power_spectrum, upper_quantile, lower_quantile

# plot the average noise spectrum
def plot_noise_spectra_epcb(
    freq_bins, noise_spectra, upper_quantile, lower_quantile,
    skip_bad_channels=True, nevts=None, output_name='noise_spectra_epcb.pdf'):

    n_epcbs = int(len(channels)/6)
    n_adcs = noise_spectra.shape[0]
    axes = plt.subplots(n_adcs, 1, figsize=(20, 2*n_adcs), sharex=True)[1]

    # Mask the first frequency bin (set to nan)
    freq_mask = np.ones_like(freq_bins, dtype=bool)
    freq_mask[0] = False

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i
            # Mask for epcb channels
            epcb_mask = np.zeros(noise_spectra.shape[1], dtype=bool)
            channels_list = channels[j*6:(j+1)*6]
            epcb_mask[channels_list] = True
            # Mask for good channels
            if skip_bad_channels and cs.shape == (n_adcs, len(channels_list)):
                ch_mask = (cs[i, :] == 0)
            else:
                ch_mask = np.ones(noise_spectra.shape[1], dtype=bool)
            ch_mask = np.logical_and(ch_mask, epcb_mask)

            # Average over channels
            avg = np.nanmean(noise_spectra[i, ch_mask], axis=0)
            axes[idx].step(
                freq_bins[freq_mask]/1e6,
                avg[freq_mask],
                where='mid',
                label=f'EPCB {j}'
            )

            axes[idx].set_title(f'ADC {i}, Average Noise Spectrum per EPCB: equidistant {nevts} events')
            axes[idx].set_ylabel('V^2 / bin')
            axes[idx].set_yscale('log')
            axes[idx].set_ylim(5e-10, 1e-5)
            axes[idx].set_xlim(0, 31.25)
            axes[idx].legend(loc='upper right', fontsize='x-small')

            # add grid lines
            axes[idx].grid(True, which='major', linestyle='-', linewidth=0.5)

    axes[-1].set_xlabel('Frequency (MHz)')
    plt.tight_layout()
    #plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

# plot the channel by channel noise spectrum
def plot_noise_spectra_channels(
    freq_bins, noise_spectra, skip_bad_channels=True,
    nevts=None, output_name='noise_spectra_channels.pdf'):

    n_epcbs = int(len(channels)/6)
    n_adcs = noise_spectra.shape[0]
    axes = plt.subplots(n_adcs*n_epcbs, 1, figsize=(20, 2*n_adcs*n_epcbs), sharex=True)[1]

    # Mask the first frequency bin (set to nan)
    freq_mask = np.ones_like(freq_bins, dtype=bool)
    freq_mask[0] = False

    # defining regions of interest
    vlist = [0.5e6, 1.8e6, 4.6e6, 7.1e6, 8.5e6, 10e6, 11.5e6, 19e6, 20e6, 25e6, 30e6]
    window = 0.4e6

    for i in range(n_adcs):
        for j in range(n_epcbs):
            idx = i * n_epcbs + j
             # mask for epcb channels
            epcb_mask = np.zeros(noise_spectra.shape[1], dtype=bool)
            channels_list = channels[j*6:(j+1)*6]
            epcb_mask[channels_list] = True
            # Mask for good channels
            if skip_bad_channels and cs.shape == (n_adcs, len(channels_list)):
                ch_mask = (cs[i, :] == 0)
            else:
                ch_mask = np.ones(noise_spectra.shape[1], dtype=bool)
            ch_mask = np.logical_and(ch_mask, epcb_mask)

            # Plot each epcb
            for k in ch_mask.nonzero()[0]:

                # plot step for centre line and fill between for error bands
                axes[idx].step(
                    freq_bins[freq_mask]/1e6,
                    noise_spectra[i, k][freq_mask],
                    where='mid', label=f'Channel {k}'
                )

            # Add horizontal line at the minimum value
            min_val = np.nanmin(noise_spectra[i, :, freq_mask])
            axes[idx].axhline(y=min_val, color='k', linestyle='--')
            axes[idx].set_title(f'ADC {i} EPCB {j} average noise spectrum: equidistant {nevts} events')
            axes[idx].legend(loc='upper right', fontsize='small')

            axes[idx].set_ylabel('V^2 / bin')
            axes[idx].set_yscale('log')
            axes[idx].grid(True, which='major', linestyle='-', linewidth=0.5)
            axes[idx].set_ylim(5e-10, 1e-5)
            axes[idx].set_xlim(0, 31.25)

    axes[-1].set_xlabel('Frequency (MHz)')
    plt.tight_layout()
    #plt.show()
    # save as pdf
    output_pdf = f"{args.tmp_dir}/{output_name}"
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()


# save data as JSON
def save_as_json(file_index, data_c, data_l, data_u, output_dir, filename):
    """
    Save data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    If the file exists, append to it; otherwise, create a new file.
    """
    data = {
        'file_index': int(file_index),
        'data_c': data_c.tolist(),
        'data_l': data_l.tolist(),
        'data_u': data_u.tolist()
    }
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(output_dir + '/' + filename) else 'w'
    with open(output_dir + '/' + filename, mode) as f:
        json.dump(data, f)
        f.write('\n')
    #print(f"Data for file_index {file_index} ({filename}) saved to {output_dir}")

def read_from_json(file_indices, output_dir, filename):
    """
    Read data from JSON file, loop over file indices, add in quadrature.
    If no matching entries are found, return None.
    """

    # Return None if file_indices is None or empty
    if file_indices is None or len(file_indices) == 0:
        return None

    # Check if file exists
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        print(f"JSON file not found: {filepath}, not making comparisons")
        return None

    # Read the JSON file and collect data for the specified file indices
    data_c, data_l, data_u = [], [], []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('file_index') in file_indices:
                        data_c.append(entry['data_c'])
                        data_l.append(entry['data_l'])
                        data_u.append(entry['data_u'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filepath}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # if data_c is empty return none
    if len(data_c) == 0:
        return None
    data_c = np.array(data_c)
    data_l = np.array(data_l)
    data_u = np.array(data_u)
    # sum in quadrature
    data_c = np.mean(data_c, axis=0)
    data_l = np.sqrt(np.sum(data_l**2, axis=0))
    data_u = np.sqrt(np.sum(data_u**2, axis=0))
    return data_c, data_l, data_u

def save_spectra_as_json(file_index, spectra_roi, output_dir, filename):
    """
    Save frequency spectra data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    """
    data = {
        'file_index': int(file_index),
        'spectra_roi': spectra_roi.tolist()
    }
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(output_dir + '/' + filename) else 'w'
    with open(output_dir + '/' + filename, mode) as f:
        json.dump(data, f)
        f.write('\n')
    #print(f"Data for file_index {file_index} ({filename}) saved to {output_dir}")

def save_eff_as_json(file_index, passed, totals, output_dir, filename):
    """
    Save efficiency data as JSON file.
    Data is 2D: i_adc, i_ch, where i_adc is the ADC index and i_ch is the channel index.
    """
    data = {
        'file_index': int(file_index),
        'pass': passed.tolist(),
        'totals': totals.tolist()
    }
    # Use append mode if file exists, else write mode
    mode = 'a' if os.path.exists(output_dir + '/' + filename) else 'w'
    with open(output_dir + '/' + filename, mode) as f:
        json.dump(data, f)
        f.write('\n')
        f.close()
    #print(f"Data for file_index {file_index} ({filename}) saved to {output_dir}")


def read_eff_from_json(file_indices, output_dir, filename):
    """
    Read efficiency data from JSON file, loop over file indices,
    combine the passes and totals
    """
    # Return None if file_indices is None or empty
    if file_indices is None or len(file_indices) == 0:
        return None

    # Check if file exists
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        print(f"Efficiency JSON file not found: {filepath}, not making comparisons")
        return None

    # Read the JSON file and collect data for the specified file indices
    passed, totals = [], []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('file_index') in file_indices:
                        passed.append(entry['pass'])
                        totals.append(entry['totals'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line in {filepath}: {line.strip()}")
            f.close()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # if passed is empty return none
    if len(passed) == 0 or len(totals) == 0:
        return None
    # sum
    passed = np.sum(np.array(passed), axis=0)
    totals = np.sum(np.array(totals), axis=0)

    return passed, totals

# function to produce placeholder pdf when try/except fails
def placeholder_pdf(output_dir, filename, message):
    """
    Create a placeholder PDF with an error message.
    """
    output_pdf = os.path.join(output_dir, filename)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, fontsize=12, ha='center', va='center')
    ax.axis('off')
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
        plt.close()
    print(f"Placeholder PDF created: {output_pdf}")


# ----------------------------- #
#             Main              #
# ----------------------------- #

def main():
    """
    Main entry point for DUNE light file processing and plotting.
    """
    # start timer
    start_time = time.time()

    # print header
    header()

    args = parse_args()
    if args.units not in ['ADC16', 'ADC14', 'V']:
        print(f"Invalid units: {args.units}. Must be 'ADC16', 'ADC14', or 'V'.")
        sys.exit(1)

    if args.powspec_nevts > args.max_evts:
        print("Number of events for the power spectrum can't be greater than the total events")
        # set powerspec_nevts to max_evts
        args.powspec_nevts = args.max_evts
        print(f"Setting powerspec_nevts to {args.max_evts} (beware of memory issues with large nevts)")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory is ready: {args.output_dir}")
    except Exception as e:
        print(f"Error creating output directory {args.output_dir}: {e}")
        raise

    try:
        os.makedirs(args.tmp_dir, exist_ok=True)
        print(f"Temporary directory is ready: {args.tmp_dir}")
    except Exception as e:
        print(f"Error creating temporary directory {args.tmp_dir}: {e}")
        raise

    # search for any files in --input_path starting with args.name_conv and ending with '.FLOW.hdf5'
    all_files = [f for f in os.listdir(args.input_path) if f.startswith(args.file_syntax) and f.endswith(EXT)]
    all_files.sort()            # HERE BE DRAGONS
    n_available_files = len(all_files)
    print(f"Found {n_available_files} files in {args.input_path} starting with {args.file_syntax} and ending with {EXT}")
    if n_available_files == 0:
        print("No files found. Exiting.")
        sys.exit(1)
    #if args.start_run >= n_available_files:
    #    print(f"Start file index {args.start_run} is out of range. There are only {n_available_files} files available. Exiting.")
    #    sys.exit(1)
    #if args.start_run + args.nfiles > n_available_files:
    if args.nfiles > n_available_files:
        args.nfiles = n_available_files - args.start_run
        print(f"Adjusting number of files to process to {args.nfiles} to avoid going out of range.")

    file_time = start_time
    proc_files = 0

    for i_file in range(args.start_run, args.start_run + args.nfiles, 1):

        # Construct the filename based on the input path and file index
        # filename = f'{args.input_path}{all_files[i_file]}'
        # HACK: Use start_run literally, instead of as an index into a (wrongly-)sorted list
        # (this is OK for the nearline, which just runs on one file at a time)
        #filename = f'{args.input_path}{args.file_syntax}{i_file}.FLOW.hdf5'
        filename = f'{args.input_path}{args.file_syntax}{i_file:05d}{EXT}'
        # just the file name, no path and cutting off .FLOW.hdf5
        # short_filename = all_files[i_file][:-10]
        short_filename = os.path.basename(filename)[:-10]

        # Skip if file does not exist
        if not os.path.exists(filename):
            print(f"File not found, skipping: {filename}")
            continue
        proc_files += 1
        print(f"Processing file: {filename} with units: {args.units}")

        ptps = get_ptps(args.units)

        try:
            file = h5py.File(filename, 'r')
            print(f"File opened successfully: {filename}")

            size_bytes = os.path.getsize(filename)
            size_gb = size_bytes / (1024 ** 3)
            print(f"File size: {size_gb:.2f} GB")

            MULT = int(size_gb // MIN_GB_PER_FILE)  # Determine MULT based on file size
            if MULT < 1:
                MULT = 1
            print(f"Using MULT = {MULT} for data reduction.")

        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue

        # Get start and end timestamps in ms
        start_timestamp = file["light/events/data"][0]['utime_ms'][0]
        end_timestamp = file["light/events/data"][-1]['utime_ms'][0]

        # Convert to UTC datetime string
        start_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_timestamp / 1000))
        end_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_timestamp / 1000))

        # Convert to US Central Time (UTC-6)
        start_central = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_timestamp / 1000 - 6 * 3600))
        end_central = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_timestamp / 1000 - 6 * 3600))
        print(f"File start time (US Central, UTC-6): {start_central}")
        print(f"File end time (US Central, UTC-6): {end_central}")

        # Count beam and off-beam event rates, the file length and the minimum dead time
        beam_mask = (file['light/events/data']['trig_type']==1)
        timestamps = file['light/events/data']['tai_ns'][:,0]
        file_length = (np.max(timestamps) - np.min(timestamps)) / 1e9
        time_differences = timestamps[1:] - timestamps[:-1]
        print(file_length)
        print(f"File length: {file_length:.2f} seconds")
        print(f"Total Number of Events: {len(beam_mask):.2f} events")
        print(f"Total Number of Beam Events: {np.sum(beam_mask):.2f} events")
        print(f"Total Number of Off-Beam Events: {np.sum(beam_mask==0):.2f} events")
        print(f"Beam Trigger Rate: {(np.sum(beam_mask)/file_length):.2f} Hz")
        beam_trigger_rate = (np.sum(beam_mask)/file_length)
        print(f"Self-Trigger Rate: {(np.sum(beam_mask==0)/file_length):.2f} Hz")
        self_trigger_rate = (np.sum(beam_mask==0)/file_length)
        #smallest_time_differences = time_differences[time_differences/1e3 < 5e5]
        smallest_time_diff = np.min(time_differences/1e3)
        print(f"Minimum Dead Time: {(np.min(time_differences/1e3)):.2f} μs")
        del beam_mask, timestamps, time_differences

        # files for comparison
        if args.ncomp == -1 or args.ncomp > i_file:
            ncomps = np.arange(0, i_file, 1)
        elif args.ncomp == 0:
            ncomps = np.array([])
        else:
            ncomps = np.arange(i_file-args.ncomp, i_file, 1)

        # get n events to process
        nevents_total = file["light/wvfm/data"]['samples'][::MULT,:,:,:].shape[0]
        #evts_to_process = args.max_evts
        #if args.max_evts > nevents_total:
        #    print(f"Total number of events to process {args.max_evts} is less than the number of events in the file {nevents_total}, changing to total events")
        evts_to_process = nevents_total

        # select evenly spaced indices up to max_evts
        #sel_idx = np.linspace(0, nevents_total - 1, evts_to_process, dtype=int)
        #sel_idx = np.unique(sel_idx)  # ensure unique indices
        sel_idx = np.linspace(0, nevents_total - 1, evts_to_process, dtype=int)

        # define datasets for beam trigger type
        #beam_mask = sel_idx[file["light/events/data"]['trig_type'][::MULT] == 1]
        #beam_wvfms, beam_noises, beam_baselines, beam_max_values = get_waveform_info(
        #    file["light/wvfm/data"]['samples'][::MULT, :, :], args.units, mask=beam_mask, ths=ptps)
        #nbeam_evts = beam_wvfms.shape[0]


        # define datasets for self-trigger type
        #strig_mask = sel_idx[file["light/events/data"]['trig_type'][::MULT] == 0]
        #strig_wvfms, strig_noises, strig_baselines, strig_max_values = get_waveform_info(
        #    file["light/wvfm/data"]['samples'][::MULT,:,:,:], args.units, mask=strig_mask, ths=ptps)
        #nstrig_evts = strig_wvfms.shape[0]


        # define datasets for all events
        wvfms, noises, baselines, max_values = get_waveform_info(
            file["light/wvfm/data"]['samples'][::MULT,:,:,:], args.units, mask=sel_idx, ths=ptps)
        nevts = wvfms.shape[0]

        ###############################################################
        # WORK IN PROGRESS TO TRACK RATIO OF BEAM TO SELF-TRIG EVENTS #
        ###############################################################

        #print(f"Waveform info extracted for file: {filename}")
        #print("    Total number of events:", wvfms.shape[0])
        #print("    Number of beam events:", beam_wvfms.shape[0],
        #      (f"{100*beam_wvfms.shape[0]/wvfms.shape[0]:.1f}%" if wvfms.shape[0]>0 else "N/A"))
        #print("    Number of self-trigger events:", strig_wvfms.shape[0],
        #      (f"{100*strig_wvfms.shape[0]/wvfms.shape[0]:.1f}%" if wvfms.shape[0]>0 else "N/A"))


        ### DQM PLOTS ###

        # identify event with largest total integral
        #integrals = np.sum(wvfms, axis=(1,2,3))
        #max_integral_evt = np.argmax(integrals)

        # number of samples over noise threshold
        #samples_over_th = wvfms > ptps_16[np.newaxis, :, np.newaxis, np.newaxis]
        #nsamples_over_th = np.sum(samples_over_th, axis=(1,2,3))
        #max_nsamples_evt = np.argmax(nsamples_over_th)

        # get event number containing the wvfm with the largest diff between consecutive samples
        #diffs = np.abs(np.diff(wvfms, axis=-1))
        #max_diffs = np.max(diffs, axis=(1,2,3))
        #max_diff_evt = np.argmax(max_diffs)

        ####################################################################
        # WIP: find a way to make this a tag for sparking / discharge events
        #evts = np.array([max_integral_evt])
        #evts = np.array([max_nsamples_evt])
        #evts = np.array([max_diff_evt])
        ####################################################################

        # sum waveform baselined
        large_event_array = tag_large_events(WVFM_ARRAY=(wvfms - baselines[:, :, :, np.newaxis]) , ADC_LIST=None, TICKS=np.shape(wvfms)[-1])
        try:
            #for evt in evts:
            plot_sum_waveform(large_event_array, args.units, output_name='plot1_sumwvfm.pdf')
            print(f"Sum waveform plotted for file: {filename}")
        except Exception as e:

            placeholder_pdf(args.tmp_dir, 'plot1_sumwvfm.pdf',
                            "Failed to plot summed waveforms: plot1_sumwvfm.pdf")

            traceback.print_exc()

        # HV instability check:
        low_fprompt_array = get_fprompt_estimate(WVFM_ARRAY=(wvfms - baselines[:, :, :, np.newaxis])/4, ARRIVAL_TICK=77, THRESHOLD=600)
        max_fprompt_array = np.round((low_fprompt_array.sum(axis=0)*MULT)/file_length, 1)
        try:
            #for evt in evts:
            plot_fprompt_rates(HVINST_ARRAY=low_fprompt_array, FPROMPT_THD=20, FILE_LENGTH=np.round(file_length,2), MULT=MULT, output_name='plot6_hv_instabilities.pdf')
            print(f"HV Instability Rate plotted for file: {filename}")
        except Exception as e:

            placeholder_pdf(args.tmp_dir, 'plot6_hv_instabilities.pdf',
                            "Failed to plot HV instability rates: plot6_hv_instabilities.pdf")
            traceback.print_exc()

        try:
            #for evt in evts:
            plot_fprompt_occurrences(HVINST_ARRAY=low_fprompt_array, FILE_LENGTH=np.round(file_length,2), output_name='plot7_hv_instabilities.pdf')
            print(f"HV Instability Time plotted for file: {filename}")
        except Exception as e:

            placeholder_pdf(args.tmp_dir, 'plot7_hv_instabilities.pdf',
                            "Failed to plot HV instability occurences: plot7_hv_instabilities.pdf")
            traceback.print_exc()

        # noise power spectra and rois
        if args.powspec_nevts > 0:
            try:
                # get mask
                max_mask = get_max_value_mask(
                    max_values[:args.powspec_nevts], ptps
                )
                wvfms_v = adc16_to_voltage(wvfms[:args.powspec_nevts])
                process_powsp_evts = len(wvfms_v)

                freq_bins, noise_spectra, noise_spectrum, upper, lower = get_noise_spectra(
                    wvfms_v) #, max_mask)
                rois_mhz = np.array([0.5, 1.8, 4.6, 7.1, 8.5, 10, 11.5, 19, 20, 25, 30])
                # convert to index from frequency bins
                rois_bins = np.array([np.argmin(np.abs(freq_bins*1e-6 - roi)) for roi in rois_mhz])

                # loop over rois, and save spectrum bin to json for tracking rois over time
                if args.write_json_blobs:
                    for i_roi in range(len(rois_bins)):
                        roi_bin = rois_bins[i_roi]
                        spur = noise_spectrum[:, :, roi_bin]
                        # convert to string and replace '.' with '_' for naming convention
                        roi_mhz = str(rois_mhz[i_roi]).replace('.', '_')
                        # save to json
                        save_spectra_as_json(
                            i_file, spur, args.output_dir,
                            f'noise_spectra_roi_{roi_mhz}mhz.json'
                        )
                # free memory
                del wvfms_v

                # plot full spectrum
                plot_noise_spectra_epcb(
                    freq_bins, noise_spectrum, None, None,
                    skip_bad_channels=True, nevts=process_powsp_evts,
                    output_name='plot5_powspec.pdf',
                )
                print(f"Noise spectra calculated for file: {filename}")

            except Exception as e:
                placeholder_pdf(args.tmp_dir, 'plot5_powspec.pdf',
                                "Failed to plot noise spectra: plot5_powspec.pdf")
                traceback.print_exc()


        # reading and writing noise widths
        try:
            # read noises
            prev_noises = read_from_json(
                ncomps, args.output_dir, 'noises.json'
            ) if i_file - args.start_run > 0 else None
            # Plot noises
            noise_c, noise_l, noise_u = plot_noises(prev_noises,
                noises, i_evt=np.arange(0, baselines.shape[0], 1),
                mask_inactive=False, output_name='plot4_noises.pdf'
            )
            # write noises
            if args.write_json_blobs:
                save_as_json(
                    i_file, noise_c, noise_l, noise_u,
                    args.output_dir, 'noises.json'
                )
            print(f"Noises plotted for file: {filename}")

        except Exception as e:
            placeholder_pdf(args.tmp_dir, 'plot4_noises.pdf',
                            "Failed to plot noises: plot4_noises.pdf")
            traceback.print_exc()


        # reading and writing baselines
        try:
            # read baselines
            prev_baselines = read_from_json(
                ncomps, args.output_dir, 'baselines.json'
            ) if i_file - args.start_run > 0 else None
            # Plot baselines
            bline_c, bline_l, bline_u = plot_baselines(prev_baselines,
                baselines, i_evt=np.arange(0, baselines.shape[0], 1),
                mask_inactive=False, output_name='plot2_baselines.pdf'
            )
            # write baselines
            if args.write_json_blobs: save_as_json(i_file, bline_c, bline_l, bline_u,
                         args.output_dir, 'baselines.json')
            print(f"Baselines plotted for file: {filename}")

        except Exception as e:
            placeholder_pdf(args.tmp_dir, 'plot2_baselines.pdf',
                            "Failed to plot baselines: plot2_baselines.pdf")
            traceback.print_exc()



        ### GRAFANA PLOTS ###

        # flatlined waveforms
        try:
            # check for flatlining channels
            flatlined = check_flatline(
                max_values, threshold=3e2
            )
            # plotting flatlined channels for grafana
            #plot_flatline_mask(
            #    flatlined, cs, output_name=f'{args.output_dir}/{short_filename}_light_dqm_flatline.png',
            #    times = (start_central, end_central)
            #)
            # plotting flatlined channels
            plot_flatline_mask(
                flatlined, cs, output_name=f'{args.tmp_dir}/plot0_flatline.pdf',
                times = (start_central, end_central), grafana=False
            )
        except Exception as e:
            placeholder_pdf(args.output_dir, f'{short_filename}_light_dqm_flatline.png',
                            "Failed to plot grafana flatline: ..light_dqm_flatline.png")
            placeholder_pdf(args.tmp_dir, 'plot0_flatline.pdf', "Failed to plot flatlined channels: plot0_flatline.pdf")
            traceback.print_exc()

        # baseline fluctuations
        try:
            # checking for baseline fluctuations
            baselined, status = check_baseline(
                prev_baselines, (bline_c, bline_l, bline_u),
                units=args.units, threshold=900
            )
            #############################################################
            # WIP: plot using status to detail whether deviation within #
            #      this file or compared to previous files              #
            #############################################################

            # plotting baseline fluctuations for grafana
            #plot_baseline_mask(
            #    baselined, cs, output_name=f'{args.output_dir}/{short_filename}_light_dqm_baseline.png',
            #    times = (start_central, end_central)
            #)
            # plotting baseline fluctuations
            plot_baseline_mask(
                baselined, cs, output_name=f'{args.tmp_dir}/plot0_baseline.pdf',
                times = (start_central, end_central), grafana=False
            )

        except Exception as e:
            placeholder_pdf(args.output_dir, f'{short_filename}_light_dqm_baseline.png',
                            "Failed to plot grafana baseline fluctuations: ..light_dqm_baseline.png")
            placeholder_pdf(args.tmp_dir, 'plot0_baseline.pdf',
                            "Failed to plot baseline fluctuations: plot0_baseline.pdf")
            traceback.print_exc()

        print(f"Plots generated for file: {args.output_dir}, now merging PDFs...")
        # search for files in the output directory and merge pdfs
        merger = PdfMerger()
        print('a')
        merger_grafana = PdfMerger()
        print('b')
        # Create a PDF with the arguments, file index, and unix timestamp
        input_path_lines = []
        max_line_length = 40
        input_path = args.input_path
        print('c')
        print(f"Input path: {input_path}, length: {len(input_path)}")
        inputpath_wrapped = textwrap.wrap(
            input_path,
            width=max_line_length,
            break_long_words=True,
            break_on_hyphens=False
        )
        #while len(input_path) > max_line_length:
        #    split_idx = input_path.rfind('/', 1, max_line_length)
        #    if split_idx <= 0:
        #        split_idx = max_line_length
        #    input_path_lines.append(input_path[:split_idx])
        #    input_path = input_path[split_idx:].lstrip('/')
        #    print('f')
        #    input_path = input_path[split_idx:]
        #input_path_lines.append(input_path)
        #print('g')
        #input_path_str = "--input_path \\\n" + "\n".join("    " + l for l in input_path_lines)
        print('h')
        # split filename at '/' closest to the centre
        filename_lines = []
        max_line_length = 60
        filename_str = filename
        print('i')
        filename_wrapped = textwrap.wrap(
            filename,
            width=max_line_length,
            break_long_words=True,
            break_on_hyphens=False
        )
        #while len(filename_str) > max_line_length:
        #    split_idx = filename_str.rfind('/', 0, max_line_length)
        #    print('j')
        #    if split_idx == -1:
        #        split_idx = max_line_length
        #        print('k')
        #    filename_lines.append(filename_str[:split_idx])
        #    filename_str = filename_str[split_idx:]
        #    print('l')
        #filename_lines.append(filename_str)
        #filename_formatted = "\n".join("    " + l for l in filename_lines)
        print('m')

        # Build args_list dynamically, avoiding None entries
        args_list = [f"File index: {i_file:05d}"]

        # Add formatted filename lines
        #args_list.extend([line for line in filename_lines if line])
        args_list.extend(filename_wrapped)

        args_list.append("")
        args_list.append(f"Data start timestamp (CT):   {start_central}")
        args_list.append(f"Data end timestamp   (CT):   {end_central}")
        args_list.append(f"DQM runtime          (CT):   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - 21600))}")
        args_list.append("")
        args_list.append(f"Beam Trigger Rate:           {(beam_trigger_rate):.2f} Hz")
        args_list.append(f"Self Trigger Rate:           {(self_trigger_rate):.2f} Hz")
        args_list.append(f"HV Instability Rate:         {max_fprompt_array} Hz")   
        args_list.append(f"Minimum DAQ Dead/Reset Time: {(smallest_time_diff):.2f} μs")  
        args_list.append("")
        print('n')

        # Add all relevant arguments
        print('Started Writing Arguments')
        for arg_name in [
            "nfiles", "start_run", "ncomp", "output_dir",
            "units", "ptps16bit", "max_evts", "powspec_nevts"
        ]:
            arg_value = getattr(args, arg_name)
            args_list.append(f"--{arg_name} {arg_value}")

        # remove None
        args_list = [line for line in args_list if line is not None]
        args_pdf = os.path.join(args.output_dir, f"args_list_{i_file:05d}.pdf")

        with PdfPages(args_pdf) as pdf:
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.axis('off')
            text = "\n".join(args_list)
            ax.text(0.01, 0.99, text, va='top', ha='left', fontsize=12, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)

        # Insert the args PDF at the top of the merged file
        merger.append(args_pdf)
        merger_grafana.append(args_pdf)
        os.remove(args_pdf)


        # Append all DQM plot PDFs to the merger
        plot_files = sorted(glob.glob(os.path.join(args.tmp_dir, "*plot*.pdf")))
        for plot_path in plot_files:
            if os.path.exists(plot_path):
                merger.append(plot_path)
                os.remove(plot_path)

        merged_pdf_path = os.path.join(args.output_dir, f"{short_filename}_light_dqm_main.pdf")
        merger.write(merged_pdf_path)
        merger.close()

        #TO-DO: change pngs to pdfs if you want this option
        # if args.merge_grafana_plots:
            # # Append all Grafana plot PDFs to the merger
            # grafana_plotnames = sorted(
            #     glob.glob(os.path.join(args.output_dir_dir, "*light_dqm_baseline.pdf"))
            #     + glob.glob(os.path.join(args.output_dir_dir, "*light_dqm_flatline.pdf"))
            # )
            # for grafana_plotname in grafana_plotnames:
            #     grafana_plot_path = f"{args.output_dir}{grafana_plotname}"
            #     if os.path.exists(grafana_plot_path):
            #         merger_grafana.append(grafana_plot_path)
            #         os.remove(grafana_plot_path)
            # merged_grafana_pdf_path = f"{args.output_dir}merged_grafana_plots_{args.start_run}.pdf"
            # merger_grafana.write(merged_grafana_pdf_path)
            # merger_grafana.close()
            # print(f"Merged Grafana plots saved to: {merged_grafana_pdf_path}")

        # timing for file processing
        print(f"Processing completed for file: {filename}")
        print(f"Time taken for file {i_file:05d}: {time.time() - file_time:.2f} seconds")
        file_time = time.time()

    if not proc_files:
        raise ValueError("None of the files were found")

    footer()
    print(f"Time taken all files: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
