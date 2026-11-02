#!/usr/bin/env python3

import argparse
import os
import json
from pathlib import Path
from zlib import adler32

RETENTION_CLASS = 'test'        # change to 'physics' when ready
NAMESPACE = 'usertests'         # change to 'neardet-2x2-lar-light' when ready


def get_checksum(datapath: Path, chunksize=1_000_000_000):
    cksum = 1
    with open(datapath, 'rb') as f:
        while data := f.read(chunksize):
            cksum = adler32(data, cksum)
    cksum = cksum & 0xffffffff
    return f'{cksum:08x}'


def dump(json_infile: Path, data_outfile: Path):
    with open(json_infile) as f:
        json_in = json.load(f)
    md_in = json_in['metadata']

    result = {
        'name': data_outfile.name,
        'namespace': NAMESPACE,
        'checksums': {'adler32': get_checksum(data_outfile)},
        'size': data_outfile.stat().st_size,
        'parents': [{'did': f'vd-coldbox-pixels:{Path(json_infile).stem}'}],
    }

    result['metadata'] = {
        'core.application.family': 'larpix-control',
        'core.application.name': 'bin2packet',
        'core.application.version': os.getenv('ND_NEARLINE_TAG', 'develop'),
        'core.data_tier': 'decoded-raw',
        'core.data_stream': md_in['core.data_stream'],
        'core.start_time': md_in['core.start_time'],
        'core.end_time': md_in['core.end_time'],

        'core.first_event_number': md_in['core.first_event_number'],
        'core.last_event_number': md_in['core.last_event_number'],

        'core.file_content_status': 'good',
        'core.file_format': 'hdf5',
        'core.file_type': 'detector',
        'core.run_type': 'vd-coldbox-pixels',
        'core.runs': md_in['core.runs'],
        'core.runs_subruns': md_in['core.runs_subruns'],
        'retention.class': RETENTION_CLASS,
        'retention.status': 'active',
    }

    json_outfile = data_outfile.with_suffix(data_outfile.suffix + '.json')
    with open(json_outfile, 'w') as f:
        json.dump(result, f, indent=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json_infile', type=Path,
                    help='Metadata for input (path to JSON file)')
    ap.add_argument('data_outfile', type=Path,
                    help='File produced by the job')
    args = ap.parse_args()

    dump(**vars(args))


if __name__ == '__main__':
    main()
