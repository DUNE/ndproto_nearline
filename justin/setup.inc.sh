#!/usr/bin/env bash

did_pfn_rse=$("$JUSTIN_PATH/justin-get-file")
result=$?
if [[ $result != 0 ]]; then
    echo "Got nothing from justin-get-file; bailing"
    exit
fi

set -euo pipefail

echo "$did_pfn_rse"
did=$(echo "$did_pfn_rse" | cut -f1 -d' ')
pfn=$(echo "$did_pfn_rse" | cut -f2 -d' ')

default_env_dir=/cvmfs/fifeuser4.opensciencegrid.org/sw/dune/a814e4fd48c5195883cbdbf1f2f666d4f266edd7
env_dir=${ND_NEARLINE_ENV_DIR:-$default_env_dir}
export PATH="$env_dir/env/bin:$PATH"
export LD_LIBRARY_PATH="$env_dir/env/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$env_dir/env/lib/libXrdPosixPreload.so:$LD_PRELOAD"
export METACAT_SERVER_URL=https://metacat.fnal.gov:9443/dune_meta_prod/app
export METACAT_AUTH_SERVER_URL=https://metacat.fnal.gov:8143/auth/dune

export ND_NEARLINE_TAG="${ND_NEARLINE_TAG:-test}"

scripts_dir=$ND_NEARLINE_SCRIPTS_DIR

input_fname=$(echo "$did" | cut -f2 -d:)
