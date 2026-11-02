#!/usr/bin/env bash

set -euo pipefail

src=$1; shift
dest=${1:-/data/DAQ}; shift

copy_one() {
    f=$1; shift

    f_meta="$f.json"

    if [[ ! -f "$f_meta" ]]; then
        echo "Missing metadata for $f. Skipping."
        return
    fi

    out_name=$(basename "$f" .h5).hdf5

    f_out="$dest/$out_name"

    cp "$f" "$f_out"

    f_out_meta="$f_out.json"

    cp -i "$f_meta" "$f_out_meta.tmp"

    cksum=$(xrdadler32 "$f_out" | awk '{print $1}')
    size=$(stat -c %s "$f_out")

    # my apologies
    sed -i "s/\.h5\",/.hdf5\",/" "$f_out_meta.tmp"
    sed -i "s/\"adler32\": \"[^\"]*\"/\"adler32\": \"$cksum\"/" "$f_out_meta.tmp"
    sed -i "s/\"size\": [^,]*/\"size\": $size/" "$f_out_meta.tmp"
    mv -i "$f_out_meta.tmp" "$f_out_meta"
}

while true; do
    find "$src" -type f -name '*.h5' -mmin +5 -print0 | while IFS= read -r -d '' file; do
        f_out=$dest/$(basename "$file" .h5).hdf5
        if [[ -e "$f_out" || -e "$f_out.copied" ]]; then
            continue
        fi
        echo $file
        copy_one "$file"
    done
    echo zzz
    sleep 60
done

