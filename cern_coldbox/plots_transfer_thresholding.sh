#!/usr/bin/env bash

while true; do
    rsync -avhP --chmod=u=rwX,g=rX,o=rX -e "ssh -J mkramer@lxplus" /nfs/rscratch/pCRP/plots_thresholding/ labpix.lbl.gov:/var/www/html/np04daq
    sleep 10
done