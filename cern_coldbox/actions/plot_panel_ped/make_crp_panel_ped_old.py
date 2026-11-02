#!/usr/bin/env python3
# Plot pedestal mean and RMS on the CRP anode plane from the JSON
# written by make_mean_std_json.py, using a multi-tile layout yaml.
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_geometry(geometry_path):
    with open(geometry_path) as fi:
        geo = yaml.load(fi, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))

    # (io_group, io_channel) -> tile
    ioc_to_tile = {}
    for tile, chips in geo['tile_chip_to_io'].items():
        for iog_ioc in chips.values():
            ioc_to_tile[(iog_ioc // 1000, iog_ioc % 1000)] = tile

    pos = np.array(list(geo['chip_channel_to_position'].values()))
    geo['nx'], geo['ny'] = pos.max(axis=0) + 1
    geo['ioc_to_tile'] = ioc_to_tile
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


def anode_plot(ax, geo, d, metric, vmax, title):
    pitch = geo['pixel_pitch']
    x_size, y_size = geo['nx'] * pitch, geo['ny'] * pitch
    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad('lightgrey')

    for tile, img in tile_images(geo, d, metric).items():
        y0, x0 = geo['tile_positions'][tile][1:3]
        extent = [x0 - x_size/2, x0 + x_size/2, y0 - y_size/2, y0 + y_size/2]
        im = ax.imshow(img, extent=extent, origin='lower', interpolation='none',
                       cmap=cmap, vmin=0, vmax=vmax)
        ax.text(x0, y0, str(tile), ha='center', va='center', fontsize=8, alpha=0.5)

    xs = [p[2] for p in geo['tile_positions'].values()]
    ys = [p[1] for p in geo['tile_positions'].values()]
    ax.set_xlim(min(xs) - x_size, max(xs) + x_size)
    ax.set_ylim(min(ys) - y_size, max(ys) + y_size)
    ax.set_aspect('equal')
    ax.set_xlabel('[mm]')
    ax.set_ylabel('[mm]')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='ADC Counts', fraction=0.046, pad=0.02)


def main(input_file, output_file, geometry, mean_max, rms_max):
    with open(input_file) as f:
        d = json.load(f)
    geo = load_geometry(geometry)

    # default color scale: 99th percentile of the data
    values = np.array([v for channels in d.values() for v in channels if v[0] != -1])
    if mean_max is None: mean_max = np.percentile(values[:, 0], 99)
    if rms_max is None: rms_max = np.percentile(values[:, 2], 99)

    fig, axes = plt.subplots(1, 2, figsize=(24, 12), constrained_layout=True)
    fig.suptitle(Path(input_file).stem, fontsize=18)
    anode_plot(axes[0], geo, d, 0, mean_max, 'Mean')
    anode_plot(axes[1], geo, d, 2, rms_max, 'RMS')

    print('Saving to: ', output_file)
    fig.savefig(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str,
                        help='''JSON-formatted dict of chip_key:channel''')
    parser.add_argument('--output_file', required=True, type=str, help='''Output plot file''')
    parser.add_argument('--geometry', required=True, type=str, help='''Multi-tile layout yaml''')
    parser.add_argument('--mean_max', default=None, type=float,
                        help='''Color scale max for mean (default: 99th percentile)''')
    parser.add_argument('--rms_max', default=None, type=float,
                        help='''Color scale max for RMS (default: 99th percentile)''')
    args = parser.parse_args()
    main(**vars(args))
