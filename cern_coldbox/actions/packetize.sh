#!/usr/bin/env bash
# Make sure PACKET_FILE is a packet-format version of INPUT_FILE:
#  - raw (binary) LArPix file -> converted with larpix-control, as in
#    ndproto_nearline (branch justin) justin/packetize_charge_data.jobscript;
#  - already a packet file -> PACKET_FILE is a symlink to it (no conversion).
# Usage: packetize.sh INPUT_FILE PACKET_FILE
# Needs larpix-control branch direct-convert-mixed-v2-v3 (Python >= 3.10) in the environment.
# ASIC version per io_group can be overridden with the ASIC_VERSION env variable.

set -euo pipefail

if [[ "$#" != "2" ]]; then
    echo "Usage: $0 INPUT_FILE PACKET_FILE"
    exit 1
fi

inpath=$1
outpath=$2
# coldbox CRP: io_groups 3,4 are LArPix-v3, the others LArPix-v2
asic_version=${ASIC_VERSION:-1:2,2:2,3:3,4:3,5:2,6:2}

if [[ ! -f "$inpath" ]]; then
    echo "Input file not found: $inpath" >&2
    exit 1
fi

# raw files have msgs + msg_headers, packet files have packets
kind=$(python3 - "$inpath" <<'EOF'
import sys, h5py
try:
    with h5py.File(sys.argv[1], 'r') as f:
        if 'packets' in f: print('packet')
        elif 'msgs' in f and 'msg_headers' in f: print('raw')
        else: print('unknown')
except OSError:
    print('unreadable')
EOF
)

mkdir -p "$(dirname "$outpath")"
# the converter appends to an existing file
rm -f "$outpath"

case $kind in
    packet)
        echo "Already a packet file, not converting: $inpath"
        ln -s "$(cd "$(dirname "$inpath")" && pwd)/$(basename "$inpath")" "$outpath"
        ;;
    raw)
        if ! command -v convert_rawhdf5_to_hdf5.py > /dev/null; then
            echo "convert_rawhdf5_to_hdf5.py not found in PATH: activate the larpix-control environment first" >&2
            exit 1
        fi
        echo "Raw file, converting to packet format: $inpath"
        convert_rawhdf5_to_hdf5.py \
            --input_filename "$inpath" --output_filename "$outpath" \
            --asic-version "$asic_version" \
            --format-version 3.0 --direct
        ;;
    unreadable)
        echo "Cannot open as HDF5 (empty, truncated or still being written?): $inpath" >&2
        exit 1
        ;;
    *)
        echo "Neither a raw LArPix file (msgs) nor a packet file (packets): $inpath" >&2
        exit 1
        ;;
esac
