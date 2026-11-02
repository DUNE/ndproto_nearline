#!/usr/bin/env bash

while true; do
    rsync -avhP --chmod=u=rwX,g=rX,o=rX -e "ssh -J mkramer@lxplus" /data3/pCRP_nearline_generated_packet_files/panel_ped labpix.lbl.gov:/var/www/html/np04daq
    sleep 30
done

