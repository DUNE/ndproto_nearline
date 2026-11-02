#!/usr/bin/env python3
# 1D histogram of the per-channel data rate from the JSON written by
# make_mean_std_json.py, as plot_1d(metric='rate') in
# https://github.com/larpix/crs_daq/blob/fsd-run-delay/analysis/plot_metric.py
# but one plot for the whole CRP, with one line per io_group.
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(input_file, output_file, nbins):
    with open(input_file) as f:
        d = json.load(f)

    rates = {}  # io_group -> rates
    for chip_key, channels in d.items():
        iog = int(chip_key.split('-')[0])
        rates.setdefault(iog, []).extend(v[2] for v in channels if len(v) > 2 and v[2] >= 0)
    all_rates = np.concatenate([np.array(r) for r in rates.values()]) if rates else np.array([])

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(all_rates):
        bins = np.linspace(0, max(all_rates.max(), 1e-3) * 1.01, nbins + 1)
        ax.hist(all_rates, bins=bins, color='lightgrey', label='all ({} channels)'.format(len(all_rates)))
        for iog in sorted(rates):
            ax.hist(rates[iog], bins=bins, histtype='step', label='io_group {}'.format(iog))
        ax.set_yscale('log')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No rate available\n(no data or no unix-time packets)',
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True)
    ax.set_xlabel('Trigger Rate [Hz]')
    ax.set_ylabel('Channel Count')
    ax.set_title(Path(input_file).stem)

    print('Saving to: ', output_file)
    fig.savefig(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str,
                        help='''JSON written by make_mean_std_json.py''')
    parser.add_argument('--output_file', required=True, type=str, help='''Output plot file''')
    parser.add_argument('--nbins', default=100, type=int, help='''Number of bins (default: %(default)s)''')
    args = parser.parse_args()
    main(**vars(args))
