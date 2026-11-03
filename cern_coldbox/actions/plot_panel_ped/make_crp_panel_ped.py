#!/usr/bin/env python3
# Plot pedestal mean and RMS (std) on the CRP anode plane from the JSON
# written by make_mean_std_json.py, using a multi-tile layout yaml.
# hot_r colour map, fixed colour scales
# (mean 0-600, RMS 0-5 ADC), chip grid, tiles labelled io_group-n.
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 15})


def load_geometry(geometry_path):
    with open(geometry_path) as fi:
        geo = yaml.load(fi, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))

    # (io_group, io_channel) -> tile
    if 'tile_io_channels' in geo:
        tile_io = {tile: [int(v) for v in vals or []] for tile, vals in geo['tile_io_channels'].items()}
    else:
        tile_io = {tile: sorted(set(chips.values())) for tile, chips in geo['tile_chip_to_io'].items()}
    ioc_to_tile = {divmod(v, 1000): tile for tile, vals in tile_io.items() for v in vals}
    # tile label io_group-n, n = 1 for io_channels 1-4, 2 for 5-8, ...
    labels = {}
    for tile, vals in tile_io.items():
        iog_ioc = [divmod(v, 1000) for v in vals]
        labels[tile] = '{}-{}'.format(iog_ioc[0][0], (min(c for _, c in iog_ioc) - 1) // 4 + 1) if vals else str(tile)

    pos = np.array(list(geo['chip_channel_to_position'].values()))
    geo['nx'], geo['ny'] = pos.max(axis=0) + 1
    # chip footprint in pixels, for the chip grid lines
    first_chip = min(int(k) // 1000 for k in geo['chip_channel_to_position'])
    cpos = np.array([v for k, v in geo['chip_channel_to_position'].items() if int(k) // 1000 == first_chip])
    geo['chip_nx'], geo['chip_ny'] = np.ptp(cpos, axis=0) + 1
    # tile centres, shifted by the TPC centre if the yaml has one
    tpc_centers = geo.get('tpc_centers') or {}
    geo['tile_xy'] = {}
    for tile, p in geo['tile_positions'].items():
        off = tpc_centers.get(geo['tile_indeces'][tile][0], [0., 0., 0.])
        geo['tile_xy'][tile] = (p[2] + off[0], p[1] + off[1])
    geo['ioc_to_tile'] = ioc_to_tile
    geo['tile_labels'] = labels
    return geo

def convert_to_mv(d, asic_version, VDDA, vref_dac, vcm_dac):
    adc_lsb_mv = None #(mV/ADC bin)
    if asic_version == '2d': adc_lsb_mv = VDDA * (vref_dac - vcm_dac)/256 /256
    elif asic_version == '3': adc_lsb_mv = VDDA * (vref_dac / 256) * (2 / 1023)
    if not adc_lsb_mv: raise RuntimeError('Unrecognized ASIC version, examples "2d" or "3"')
    return d*adc_lsb_mv

def convert_to_ke(d, asic_version, VDDA, vref_dac, vcm_dac):
    if asic_version == '2d': gain = 2.46 #(mV/ke-)
    elif asic_version == '3': gain = 3.4
    return convert_to_mv(d, asic_version, VDDA, vref_dac, vcm_dac)/gain #(ke-/ADC bin)


V3_IO_GROUPS = (3, 4)  # LArPix-v3, the others are v2d
VCM_DAC_V2D = 68       # v3 does not use vcm: 0 is a placeholder


def convert_json_to_ke(d, geo, vdda_spec):
    # Same structure as the make_mean_std_json.py JSON, with mean and std (columns 0, 1)
    # in ke- instead of ADC. vdda_spec: {'iog-n': [vref, vdda]} (both mV, n = tile number as
    # in the tile labels); vref = VDDA * vref_dac / 256, so vref_dac = 256 * vref / VDDA.
    # Chips whose tile is not in the geometry or not in vdda_spec are dropped (listed).
    with open(vdda_spec) as f:
        spec = json.load(f)
    out, missing = {}, set()
    for chip_key, channels in d.items():
        iog, ioc, _ = map(int, chip_key.split('-'))
        tile = geo['ioc_to_tile'].get((iog, ioc))
        label = geo['tile_labels'].get(tile)
        if label not in spec:
            missing.add(label if tile is not None else 'io_group {} io_channel {}'.format(iog, ioc))
            continue
        vref, vdda = spec[label]
        vref_dac = 256 * vref / vdda
        if iog in V3_IO_GROUPS:
            asic, vcm_dac = '3', 0
        else:
            asic, vcm_dac = '2d', VCM_DAC_V2D
        out[chip_key] = [
            [convert_to_ke(v[0], asic, vdda, vref_dac, vcm_dac),
             convert_to_ke(v[1], asic, vdda, vref_dac, vcm_dac)] + list(v[2:])
            if v[0] >= 0 else list(v)  # -1 = no data, kept as is
            for v in channels]
    if missing:
        print('No vref/VDDA (not converted to ke-): {}'.format(sorted(missing)))
    return out


def percentile_range(values, lo=1, hi=99):
    # colour range from the 1st-99th percentile of the valid (>= 0) values
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]
    if len(values) == 0:
        return 0., 1.
    vmin, vmax = np.percentile(values, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 0.1, 1e-3)
    return vmin, vmax


def hist_per_iog(d, column, xlabel, title, output_file, nbins):
    # 1D histogram of one JSON column, all channels + one line per io_group
    values = {}  # io_group -> values
    for chip_key, channels in d.items():
        iog = int(chip_key.split('-')[0])
        values.setdefault(iog, []).extend(v[column] for v in channels if len(v) > column and v[column] >= 0)
    all_values = np.concatenate([np.array(r) for r in values.values()]) if values else np.array([])

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(all_values):
        bins = np.linspace(0, max(all_values.max(), 1e-3) * 1.01, nbins + 1)
        ax.hist(all_values, bins=bins, color='lightgrey', label='all ({} channels)'.format(len(all_values)))
        for iog in sorted(values):
            ax.hist(values[iog], bins=bins, histtype='step', label='io_group {}'.format(iog))
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data available\n(no data or no valid packets)',
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')
    ax.set_title(title)
    print('Saving to: ', output_file)
    fig.savefig(output_file)
    plt.close(fig)


def tile_images(geo, d, metric):
    # one (ny, nx) image per tile, pixel index = chip_channel_to_position
    nx, ny = geo['nx'], geo['ny']
    images = {tile: np.full((ny, nx), np.nan) for tile in geo['tile_positions']}
    missing_ioc, missing_chip = set(), set()
    for chip_key, channels in d.items():
        iog, ioc, chip = map(int, chip_key.split('-'))
        tile = geo['ioc_to_tile'].get((iog, ioc))
        if tile is None:
            missing_ioc.add((iog, ioc))
            continue
        for channel, values in enumerate(channels):
            if values[0] == -1:
                continue
            pos = geo['chip_channel_to_position'].get(chip * 1000 + channel)
            if pos is None:
                missing_chip.add(chip_key)
                continue
            ix, iy = pos
            if metric == -1:
                images[tile][iy, ix] = len(values[2])
            elif values[metric] >= 0:  # -1 = not available (e.g. rate without unix-time packets): grey
                images[tile][iy, ix] = values[metric]

    if metric == 0 and (missing_ioc or missing_chip):
        print('Not in geometry (skipped): {} (io_group, io_channel), {} chips: {}'.format(
            len(missing_ioc), len(missing_chip), sorted(missing_chip)[:20]))

    for tile, img in images.items():
        orientation = geo['tile_orientations'][tile]
        if orientation[2] == -1: img = img[:, ::-1]
        if orientation[1] == -1: img = img[::-1, :]
        images[tile] = img
    return images


def anode_plot(ax, geo, d, metric, vmax, label, min_bar=0):
    pitch = geo['pixel_pitch']
    nx, ny = geo['nx'], geo['ny']
    x_size, y_size = nx * pitch, ny * pitch
    cmap = matplotlib.colormaps['hot_r'].copy()
    cmap.set_bad('lightgrey')

    for tile, img in tile_images(geo, d, metric).items():
        xc, yc = geo['tile_xy'][tile]
        x0, y0 = xc - x_size / 2, yc - y_size / 2
        im = ax.imshow(img, extent=[x0, x0 + x_size, y0, y0 + y_size], origin='lower',
                       interpolation='nearest', cmap=cmap, vmin=min_bar, vmax=vmax)
        for i in range(geo['chip_nx'], nx, geo['chip_nx']):
            ax.plot([x0 + i * pitch] * 2, [y0, y0 + y_size], 'k:', lw=0.3)
        for j in range(geo['chip_ny'], ny, geo['chip_ny']):
            ax.plot([x0, x0 + x_size], [y0 + j * pitch] * 2, 'k:', lw=0.3)
        ax.add_patch(Rectangle((x0, y0), x_size, y_size, fill=False, ec='k', lw=1.2))
        ax.text(xc, yc, geo['tile_labels'][tile], ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=1))

    xs = [x for x, _ in geo['tile_xy'].values()]
    ys = [y for _, y in geo['tile_xy'].values()]
    xmin, xmax = min(xs) - x_size / 2, max(xs) + x_size / 2
    ymin, ymax = min(ys) - y_size / 2, max(ys) + y_size / 2
    margin = 0.02 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position [mm]')
    ax.set_ylabel('Y Position [mm]')
    ax.text(0.95, 1.005, 'LArPix', ha='center', va='bottom', transform=ax.transAxes)
    plt.colorbar(im, ax=ax, label=label, shrink=0.6)


def main(input_file, output_file, geometry, vdda_spec, mean_max, rms_max, rate_max, occupancy_max, rate_min, occupancy_min, nbins=100):
    with open(input_file) as f:
        d = json.load(f)
    geo = load_geometry(geometry)

    fig, axes = plt.subplots(1, 2, figsize=(52, 25), constrained_layout=True)
    fig.suptitle(Path(input_file).stem, fontsize=22)
    anode_plot(axes[0], geo, d, 0, mean_max, 'ADC mean [ADC]')
    anode_plot(axes[1], geo, d, 1, rms_max, 'ADC RMS [ADC]')

    print('Saving to: ', output_file)
    fig.savefig(output_file, dpi=100)

    # same but in ke- instead of ADC (converting ADC in mV and then to ke-),
    # ASIC version is v3 for iog3 and 4 and v2d for the others 
    # per tile vref and vdda are specified in the json file in 
    # cern_coldbox/vdda_spec/vref_vdda_cablelength.json where the key is 
    # the tile name indicated as 'iog-tilenumber', like for example '1-1' 
    # and the value is an array with 2 entries: vref and vdda. Additionally,
    # for v2d we assume vcm_dac = 68, while v3 doesnt need one and we can use 0 as placeholer.
    # all numbers taken from https://docs.google.com/spreadsheets/d/13QrENecXsHJahfXHwA1SGwCsAAyuePiqCspJ3QlDJXQ/edit?usp=sharing
    # (the json vref is in mV, = VDDA * vref_dac / 256: convert_json_to_ke turns it back into vref_dac)
    d_ke = convert_json_to_ke(d, geo, vdda_spec)
    mean_ke_min, mean_ke_max = percentile_range([v[0] for c in d_ke.values() for v in c])
    rms_ke_min, rms_ke_max = percentile_range([v[1] for c in d_ke.values() for v in c])
    print('ke- colour range: mean {:g}-{:g}, RMS {:g}-{:g}'.format(mean_ke_min, mean_ke_max, rms_ke_min, rms_ke_max))
    fig, axes = plt.subplots(1, 2, figsize=(52, 25), constrained_layout=True)
    fig.suptitle(Path(input_file).stem, fontsize=22)
    anode_plot(axes[0], geo, d_ke, 0, mean_ke_max, 'Pedestal mean [ke-]', min_bar=mean_ke_min)
    anode_plot(axes[1], geo, d_ke, 1, rms_ke_max, 'Pedestal RMS [ke-]', min_bar=rms_ke_min)

    print('Saving to: ', output_file[:-4]+'.Conversion2ke.png')
    fig.savefig(output_file[:-4]+'.Conversion2ke.png', dpi=100)
    plt.close(fig)

    ADCs = {}  # io_group -> ADCs
    for chip_key, channels in d.items():
        iog = int(chip_key.split('-')[0])
        ADCs.setdefault(iog, []).extend(v[0] for v in channels if len(v) > 0 and v[0] >= 0)
    all_ADCs = np.concatenate([np.array(r) for r in ADCs.values()]) if ADCs else np.array([])

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(all_ADCs):
        bins = np.linspace(0, max(all_ADCs.max(), 1e-3) * 1.01, nbins + 1)
        ax.hist(all_ADCs, bins=bins, color='lightgrey', label='all ({} channels)'.format(len(all_ADCs)))
        for iog in sorted(ADCs):
            ax.hist(ADCs[iog], bins=bins, histtype='step', label='io_group {}'.format(iog))
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No ADC available\n(no data or no valid packets)',
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True)
    ax.set_xlabel('ADC Means [ADC]')
    ax.set_ylabel('Counts')
    ax.set_title(Path(input_file).stem)

    print('Saving to: ', output_file[:-4]+'.1DhistMean.png')
    fig.savefig(output_file[:-4]+'.1DhistMean.png')

    # same but in ke- instead of ADC (converting ADC in mV and then to ke-)
    hist_per_iog(d_ke, 0, 'Pedestal mean [ke-]', Path(input_file).stem,
                 output_file[:-4]+'.Conversion2ke.1DhistMean.png', nbins)

    del fig
    ADCs = {}  # io_group -> ADCs
    for chip_key, channels in d.items():
        iog = int(chip_key.split('-')[0])
        ADCs.setdefault(iog, []).extend(v[1] for v in channels if len(v) > 1 and v[1] >= 0)
    all_ADCs = np.concatenate([np.array(r) for r in ADCs.values()]) if ADCs else np.array([])

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(all_ADCs):
        bins = np.linspace(0, max(all_ADCs.max(), 1e-3) * 1.01, nbins + 1)
        ax.hist(all_ADCs, bins=bins, color='lightgrey', label='all ({} channels)'.format(len(all_ADCs)))
        for iog in sorted(ADCs):
            ax.hist(ADCs[iog], bins=bins, histtype='step', label='io_group {}'.format(iog))
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No ADC available\n(no data or no valid packets)',
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True)
    ax.set_xlabel('ADC RMS [ADC]')
    ax.set_ylabel('Counts')
    ax.set_title(Path(input_file).stem)

    print('Saving to: ', output_file[:-4]+'.1DhistRMS.png')
    fig.savefig(output_file[:-4]+'.1DhistRMS.png')

    # same but in ke- instead of ADC (converting ADC in mV and then to ke-)
    hist_per_iog(d_ke, 1, 'Pedestal RMS [ke-]', Path(input_file).stem,
                 output_file[:-4]+'.Conversion2ke.1DhistRMS.png', nbins)

    del fig
    fig, axes = plt.subplots(1, 2, figsize=(52, 25), constrained_layout=True)
    fig.suptitle(Path(input_file).stem, fontsize=22)
    # automatic colour range (-1): 1st-99th percentile of the plotted column only
    # (JSON columns: 0 mean, 1 std, 2 rate, 3 occupancy), ignoring the -1 "no data" entries
    arr = np.array([c for k in d for c in d[k]], dtype=float).reshape(-1, 4)

    def auto_range(col, vmin, vmax):
        vals = arr[arr[:, col] >= 0, col]
        if len(vals) == 0:
            return (0. if vmin == -1 else vmin), (1. if vmax == -1 else vmax)
        lo = np.percentile(vals, 1) if vmin == -1 else vmin
        hi = np.percentile(vals, 99) if vmax == -1 else vmax
        if hi <= lo:  # all values (nearly) equal: open the range so pixels stay visible
            hi = lo + max(abs(lo) * 0.1, 1e-3)
        return lo, hi

    rate_min, rate_max = auto_range(2, rate_min, rate_max)
    occupancy_min, occupancy_max = auto_range(3, occupancy_min, occupancy_max)
    print('Rate colour range: {:g}-{:g} Hz, occupancy: {:g}-{:g} packets'.format(
        rate_min, rate_max, occupancy_min, occupancy_max))
    anode_plot(axes[0], geo, d, 2, rate_max, 'Rate [Hz]',min_bar=rate_min)
    anode_plot(axes[1], geo, d, 3, occupancy_max, 'Occupancy [number of packets]', min_bar=occupancy_min)

    print('Saving to: ', output_file[:-4]+'.RateOccupancy.png')
    fig.savefig(output_file[:-4]+'.RateOccupancy.png', dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str,
                        help='''JSON-formatted dict of chip_key:channel''')
    parser.add_argument('--output_file', required=True, type=str, help='''Output plot file''')
    parser.add_argument('--geometry', required=True, type=str, help='''Multi-tile layout yaml''')
    parser.add_argument('--vdda_spec', type=str,
                        default=str(Path(__file__).resolve().parent.parent / 'vdda_spec' / 'vref_vdda_cablelength.json'),
                        help='''JSON {"iog-tile": [vref, vdda]} in mV, for the ke- conversion (default: %(default)s)''')
    parser.add_argument('--mean_max', default=600., type=float,
                        help='''Color scale max for mean (default: %(default)s)''')
    parser.add_argument('--rms_max', default=5., type=float,
                        help='''Color scale max for RMS (default: %(default)s)''')
    parser.add_argument('--rate_max', default=-1, type=float,
                        help='''Color scale max for mean (default: %(default)s)''')
    parser.add_argument('--occupancy_max', default=-1, type=float,
                        help='''Color scale max for RMS (default: %(default)s)''')
    parser.add_argument('--rate_min', default=-1, type=float,
                        help='''Color scale max for mean (default: %(default)s)''')
    parser.add_argument('--occupancy_min', default=-1, type=float,
                        help='''Color scale max for RMS (default: %(default)s)''')
    args = parser.parse_args()
    main(**vars(args))
