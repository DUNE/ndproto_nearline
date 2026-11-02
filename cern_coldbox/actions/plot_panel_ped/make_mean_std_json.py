#!/usr/bin/env python3
# Per-channel pedestal mean, std and rate from a LArPix packet file.
# Output JSON: {"iog-ioc-chip": [[mean, std, rate, occupancy] x 64 channels]}, [-1, -1, -1, -1] = no data.
# Same selection as plot_roger.py: data packets (packet_type 1) with valid parity,
# dropping ADC == 0 packets on the LArPix-v3 io_groups
# rate [Hz] = selected data packets / livetime, livetime from the unix-time
# packets (type 4); it is -1 if the file has no unix-time packets.
import argparse
import json

import h5py
import numpy as np

FIELDS = ['io_group', 'io_channel', 'chip_id', 'channel_id', 'packet_type', 'valid_parity', 'dataword', 'timestamp']
BLOCK = 10_000_000  # packets read at a time


def main(input_file, output_file, v3_io_groups):
    v3_io_groups = [int(x) for x in v3_io_groups.split(',') if x.strip()]
    keys, n, s, s2 = [], [], [], []
    unixtime = []  # (min, max) per block
    n_zero_v3 = 0
    with h5py.File(input_file, 'r') as f:
        dset = f['packets']
        fields = [x for x in FIELDS if x in dset.dtype.names]
        missing = set(FIELDS) - set(fields) - {'timestamp'}
        if missing:
            raise SystemExit('{}: packets have no field {}'.format(input_file, ', '.join(sorted(missing))))
        for start in range(0, len(dset), BLOCK):
            p = dset.fields(fields)[start:start + BLOCK]
            if 'timestamp' in fields:
                t = p['timestamp'][p['packet_type'] == 4]
                if len(t):
                    unixtime.append((t.min(), t.max()))
            # coldbox data packets are packet_type 1 (the 2x2 nearline used 0)
            p = p[(p['valid_parity'] == 1) & (p['packet_type'] == 1)]
            zero_v3 = (p['dataword'] == 0) & np.isin(p['io_group'], v3_io_groups)
            n_zero_v3 += int(zero_v3.sum())
            p = p[~zero_v3]
            # one integer per (io_group, io_channel, chip_id, channel_id)
            key = (((p['io_group'].astype(np.int64) << 8 | p['io_channel']) << 8
                    | p['chip_id']) << 8 | p['channel_id'])
            k, inv = np.unique(key, return_inverse=True)
            adc = p['dataword'].astype(float)
            keys.append(k)
            n.append(np.bincount(inv))
            s.append(np.bincount(inv, weights=adc))
            s2.append(np.bincount(inv, weights=adc**2))

    # merge the blocks
    k, inv = np.unique(np.concatenate(keys), return_inverse=True)
    n = np.bincount(inv, weights=np.concatenate(n))
    mean = np.bincount(inv, weights=np.concatenate(s)) / n
    std = np.sqrt(np.maximum(np.bincount(inv, weights=np.concatenate(s2)) / n - mean**2, 0))

    livetime = 0
    if unixtime:
        unixtime = np.array(unixtime, dtype=float)
        livetime = unixtime[:, 1].max() - unixtime[:, 0].min()
    rate = n / livetime if livetime > 0 else np.full(len(n), -1.)

    output = {}
    for key, m, sd, r, nn in zip(k, mean, std, rate, n):
        iog, ioc, cid, chan = key >> 24, key >> 16 & 255, key >> 8 & 255, key & 255
        chip_key = '{}-{}-{}'.format(iog, ioc, cid)
        if chip_key not in output:
            output[chip_key] = [(-1, -1, -1, -1)] * 64
        output[chip_key][chan] = (m, sd, r, nn)

    print('Dropped {} ADC == 0 packets on v3 io_groups {}'.format(n_zero_v3, v3_io_groups))
    print('Channels with data: {}, livetime: {:g} s'.format(len(k), livetime))
    if livetime <= 0:
        print('No unix-time packets (type 4 with a timestamp field): rates set to -1')
    with open(output_file, 'w') as fout:
        json.dump(output, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='''HDF5 packet file''')
    parser.add_argument('--output_file', required=True, type=str, help='''JSON output file''')
    parser.add_argument('--v3_io_groups', default='3,4', type=str,
                        help='''Comma-separated LArPix-v3 io_groups, where ADC == 0 packets
                        are dropped (default: %(default)s)''')
    args = parser.parse_args()
    main(**vars(args))
