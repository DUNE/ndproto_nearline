# 2x2 Raw-Packet Electron Lifetime

This package contains the packet-level electron-lifetime code used for the
preliminary comparison with the Brooke/Russell hit-based measurement.

## Files

- `packet_lifetime.py`
  Main electron-lifetime analysis code.

- `brooke_russell_comparison.png`
  Comparison plot with the Brooke/Russell result.

- `README.md`
  This file.

## Environment

At NERSC the code was run with:

    module load python
    conda activate e-lifetime

The main Python dependencies are:

    numpy
    scipy
    pandas
    h5py
    matplotlib
    pylandau

## Inputs

The program takes:

1. A raw 2x2 packet HDF5 file.
2. A corresponding FLOW HDF5 file used to recover the static
   per-channel pedestal.
3. A panel-pedestal JSON argument is also required by the current
   command-line interface.

The analysis itself does not use reconstructed tracks, reconstructed t0,
or reconstructed Y-Z positions.

## Example

    python packet_lifetime.py \
        /path/to/packet-file.h5 \
        --ped /path/to/panel_ped.json \
        --ped-source flow \
        --flow-file /path/to/matching.FLOW.hdf5 \
        --charge-mode adc-minus-ped \
        --successive-min-ticks 28 \
        --successive-max-ticks 28 \
        --mpv-bootstrap 20 \
        --outdir output_directory

The raw packet HDF5 file is the first positional argument.

To see all available options:

    python packet_lifetime.py --help

## Analysis procedure

The code approximately performs the following steps:

1. Read raw charge packets and external-trigger packets.
2. Determine packet time relative to EXT.
3. Recover a static pedestal for each physical charge channel.
4. Subtract the pedestal from individual charge triggers.
5. Group successive triggers belonging to the same physical channel.
6. Sum the pedestal-subtracted charge within each group.
7. Use the last successive hit as the time assigned to the summed object.
8. Fill charge versus delta-t from EXT.
9. Divide the drift-time range into slices.
10. Fit the charge distribution in each slice with a Landau-Gaussian model.
11. Extract the MPV from each slice.
12. Fit the MPVs versus drift time with an exponential attenuation model.
13. Extract the electron lifetime tau.

The lifetime follows approximately

    Q(t) = Q0 * exp(-t / tau)

where tau is the electron lifetime.

## Important caveat about successive hits

The exact historical Brooke/Russell definition of "successive" triggers has
not yet been recovered.

For the comparison shown here, the code used

    --successive-min-ticks 28
    --successive-max-ticks 28

because a very strong 28-tick same-channel separation feature was observed
in the packet data.

This 28-tick requirement is therefore a provisional implementation and
should NOT be interpreted as a confirmed Brooke/Russell requirement.

Brooke confirmed the more general procedure:

- pedestal-subtract individual triggers;
- sum successive triggers on the same channel;
- use the last successive hit as the reference time;
- fit a Landau x Gaussian distribution in each delta-T slice.

## Pedestal

The FLOW-derived static pedestal used in this implementation points to the
June 5, 2024 cold-pedestal calibration.

A packet-level pedestal JSON from the same June 5 calibration period was
also found in the nearline code.

## Notes for pipeline integration

The main script is intentionally provided as a standalone reference
implementation.

When integrating it into another pipeline, the pieces most likely to need
adaptation are:

- input file discovery;
- mapping between packet and FLOW files;
- pedestal source;
- external-trigger selection;
- definition of successive same-channel triggers;
- output directory handling.

The physics fitting portion can then be reused largely unchanged.
