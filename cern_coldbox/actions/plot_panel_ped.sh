#!/usr/bin/env bash
# Pedestal mean/RMS map of the CRP from one LArPix packet file.
# Usage: plot_panel_ped.sh PACKET_FILE OUTPUT_PNG
# Also writes, next to OUTPUT_PNG (X.png):
#   X.rate.png  1D histogram of the per-channel data rate
#   X.json      per-channel [mean, std, rate]
# The geometry can be overridden with the GEOMETRY_FILE env variable.
# Files larger than MAX_FILE_SIZE bytes (env variable, default 10 GB; 0 = no limit) are not
# skipped: only the first part of the packets is used (MAX_FILE_SIZE / file size of them).

set -euo pipefail

if [[ "$#" != "2" ]]; then
    echo "Usage: $0 PACKET_FILE OUTPUT_PNG"
    exit 1
fi

inpath=$1
plotpath=$2
jsonpath=${plotpath%.*}.json
ratepath=${plotpath%.*}.rate.png

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
geometry=${GEOMETRY_FILE:-$here/geometry/multi_tile_layout_crp.yaml}
max_file_size=${MAX_FILE_SIZE:-10000000000}

if [[ ! -f "$inpath" ]]; then
    echo "Input file not found: $inpath" >&2
    exit 1
fi


mkdir -p "$(dirname "$plotpath")"

python3 "$here/plot_panel_ped/make_mean_std_json.py" --input_file "$inpath" --output_file "$jsonpath" --max_file_size "$max_file_size"
python3 "$here/plot_panel_ped/make_crp_panel_ped.py" --input_file "$jsonpath" --output_file "$plotpath" --geometry "$geometry"
python3 "$here/plot_panel_ped/make_rate_hist.py" --input_file "$jsonpath" --output_file "$ratepath"
