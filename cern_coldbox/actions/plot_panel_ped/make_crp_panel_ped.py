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


def main(input_file, output_file, geometry, mean_max, rms_max, rate_max, occupancy_max, rate_min, occupancy_min, nbins=100):
    with open(input_file) as f:
        d = json.load(f)
    geo = load_geometry(geometry)

    fig, axes = plt.subplots(1, 2, figsize=(52, 25), constrained_layout=True)
    fig.suptitle(Path(input_file).stem, fontsize=22)
    anode_plot(axes[0], geo, d, 0, mean_max, 'ADC mean [ADC]')
    anode_plot(axes[1], geo, d, 1, rms_max, 'ADC RMS [ADC]')

    print('Saving to: ', output_file)
    fig.savefig(output_file, dpi=100)

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
