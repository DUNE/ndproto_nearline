#!/usr/bin/env python3

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------

DEFAULT_DATA_FILE = Path(__file__).with_name("purity_monitor_history.xlsx")

MEASUREMENTS_SHEET = "Measurements"
EVENTS_SHEET = "Events"
GAP_SHEET = "Gap"
SETTINGS_SHEET = "Settings"


def load_data(data_file):
    """Load measurements, plot events, gap information and settings from Excel."""
    data_file = Path(data_file)

    measurements = pd.read_excel(data_file, sheet_name=MEASUREMENTS_SHEET)
    events = pd.read_excel(data_file, sheet_name=EVENTS_SHEET)
    gap = pd.read_excel(data_file, sheet_name=GAP_SHEET)
    settings_df = pd.read_excel(data_file, sheet_name=SETTINGS_SHEET)

    measurements["Date"] = pd.to_datetime(measurements["Date"])
    events["Date/time"] = pd.to_datetime(events["Date/time"])
    gap["Start"] = pd.to_datetime(gap["Start"])
    gap["End"] = pd.to_datetime(gap["End"])

    settings = dict(zip(settings_df["Parameter"], settings_df["Value"]))

    return measurements, events, gap, settings


def setting_timestamp(settings, name):
    """Read a timestamp setting stored in the workbook."""
    value = settings[name]
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return datetime.strptime(str(value), "%Y/%d/%m %H:%M")


# ---------------------------------------------------------------------------
# Purity-monitor / gas-analyser calculation
# ---------------------------------------------------------------------------

def E_field_strenght_Vpercm(cathode, anode_grid, drift_length):
    return (anode_grid - cathode) / drift_length


# O2: field-dependent rational-polynomial fit
# From "Parameterization of Electron Attachment Rate Constants for Impurities
# in LArTPC Detectors", JINST 17 T11007 (2022), Table 2
# https://iopscience.iop.org/article/10.1088/1748-0221/17/11/T11007/pdf
# 1 ppb = 10^-9 in mole fraction basis
_O2_FIT = dict(
    p=11,
    a1=76.2749,
    a2=4.24596,
    a3=0.0,
    a4=0.0,  # no a4 in table 2?
    b1=1.88083,
    b2=2.62643,
    b3=0.0632332,
    b4=-0.000211009,
)


def k_A_O2_per_s(E_field_Vpercm):
    E = E_field_Vpercm / 1000  # paper gives values in kV/cm
    f = _O2_FIT
    numerator = (
        (f["a1"] / f["b1"])
        + (f["a1"] * E)
        + (f["a2"] * E**2)
        + (f["a3"] * E**3)
        + (f["a4"] * E**4)
    )
    denominator = (
        1
        + (f["b1"] * E)
        + (f["b2"] * E**2)
        + (f["b3"] * E**3)
        + (f["b4"] * E**4)
    )
    return 10**f["p"] * (numerator / denominator)


# H2O: only one data point exists (Carls et al., ~32 V/cm), no established
# field dependence -> treated as constant. Table 4 value:
# 0.093 (ms ppb)^-1 = 9.3e10 s^-1 on a mole-fraction basis.
K_A_H2O_PER_S = 9.3e10


def _to_per_us_per_ppb(k_A_per_s):
    # s^-1 (mole fraction) -> us^-1 ppb^-1
    return k_A_per_s * 1e-15


def lifetime_gas_analyser_us(O2, H2O, field_strength_Vpercm):
    k_O2 = _to_per_us_per_ppb(k_A_O2_per_s(field_strength_Vpercm))
    k_H2O = _to_per_us_per_ppb(K_A_H2O_PER_S)
    one_over_tau_total = (k_O2 * O2) + (k_H2O * H2O)
    return 1 / one_over_tau_total


def lifetime_single_species_us(conc_ppb, k_per_us_ppb):
    conc_ppb = np.asarray(conc_ppb, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(
            conc_ppb > 0,
            1.0 / (k_per_us_ppb * conc_ppb),
            np.inf,
        )


def impurity_concentration_from_PrM_lifetime(
    lifetime_us, field_strength_Vpercm, H2O_ppb=0.0
):
    """O2-equivalent concentration [ppb] implied by a PrM lifetime."""
    k_O2 = _to_per_us_per_ppb(k_A_O2_per_s(field_strength_Vpercm))
    k_H2O = _to_per_us_per_ppb(K_A_H2O_PER_S)
    remaining_rate = (1.0 / lifetime_us) - k_H2O * H2O_ppb
    if remaining_rate <= 0:
        return 0.0
    return remaining_rate / k_O2


def O2_equivalent_from_gas(O2_ppb, H2O_ppb, field_strength_Vpercm):
    """Combine gas-analyser O2 and H2O into an O2-equivalent concentration."""
    k_O2 = k_A_O2_per_s(field_strength_Vpercm)
    k_H2O = K_A_H2O_PER_S
    return O2_ppb + H2O_ppb * (k_H2O / k_O2)


def O2_equivalent_from_PrM(lifetime_us, field_strength_Vpercm):
    """O2-equivalent concentration [ppb] implied by a PrM lifetime."""
    return impurity_concentration_from_PrM_lifetime(
        lifetime_us, field_strength_Vpercm, H2O_ppb=0.0
    )


# ---------------------------------------------------------------------------
# Gas-sampling corrections
# ---------------------------------------------------------------------------

def gas_sampling_source(date, ullage_start, liquid_again):
    if date < ullage_start:
        return "Liquid"
    elif date < liquid_again:
        return "Ullage"
    else:
        return "Liquid"


def apply_liquid_O2_offset(
    O2_ppb, dates, offset_ppb, ullage_start, liquid_again
):
    """Subtract the liquid-mode O2 zero offset; leave ullage readings untouched."""
    O2_ppb = np.asarray(O2_ppb, dtype=float)
    corrected = O2_ppb.copy()

    sources = np.array([
        gas_sampling_source(d, ullage_start, liquid_again) for d in dates
    ])
    liquid_mask = sources == "Liquid"
    corrected[liquid_mask] = np.clip(
        corrected[liquid_mask] - offset_ppb, 0.0, None
    )
    return corrected


# ---------------------------------------------------------------------------
# Plot annotations
# ---------------------------------------------------------------------------

def add_events_and_gap(ax, events, gap, label_events=True):
    """Draw the vertical event lines and the no-signal shaded interval."""
    # The gap sheet is deliberately separate so the interval can be edited
    # without changing the plotting code.
    for _, row in gap.iterrows():
        start = row["Start"]
        end = row["End"]
        label = str(row["Label"]) if not pd.isna(row["Label"]) else ""
        ax.axvspan(
            start,
            end,
            color="0.75",
            alpha=0.35,
            zorder=0,
            label=label if label else None,
        )

    for _, row in events.iterrows():
        event_time = row["Date/time"]
        label = str(row["Label"])
        linestyle = str(row["Line style"])

        ax.axvline(
            event_time,
            color="black",
            linestyle=linestyle,
            linewidth=1.5,
            zorder=1,
        )

        if label_events:
            ax.text(
                event_time,
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
            )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_lifetime(
    lifetime_monitor,
    lifetime_monitor_err,
    lifetime_gas,
    O2_gas,
    dates,
    field_strength_Vpercm,
    events,
    gap,
    projection_days=15,
    lifetime_O2_only=None,
    lifetime_H2O_only=None,
    output_file="lifetime_history.png",
    trustworthy_monitor_start_index=7,
):
    x = np.asarray(dates)
    gas_sources = np.array([
        gas_sampling_source(d, ULLAGE_START, LIQUID_AGAIN) for d in x
    ])
    y_monitor = np.asarray(lifetime_monitor, dtype=float)
    y_monitor_err = np.asarray(lifetime_monitor_err, dtype=float)
    y_gas = np.asarray(lifetime_gas, dtype=float) * GAS_LIFETIME_SCALE

    # Keep the original projection calculation.
    pump_is_turned_back_on_index = 8
    x_days = np.array([
        (d - x[0]).total_seconds() / 86400 for d in x
    ])
    fit_monitor = np.polyfit(
        x_days[pump_is_turned_back_on_index:],
        y_monitor[pump_is_turned_back_on_index:],
        1,
    )
    fit_gas = np.polyfit(
        x_days[pump_is_turned_back_on_index:],
        y_gas[pump_is_turned_back_on_index:],
        1,
    )
    future_x = np.array([
        x[-1] + timedelta(days=i) for i in range(1, projection_days + 1)
    ])
    future_days = np.array([
        (d - x[0]).total_seconds() / 86400 for d in future_x
    ])
    future_monitor = np.polyval(fit_monitor, future_days)
    future_gas = np.polyval(fit_gas, future_days)

    fig, ax = plt.subplots(figsize=(10, 6))

    add_events_and_gap(ax, events, gap, label_events=True)

    # Measurements before the anode signal are excluded from the main PrM plot.
    ax.errorbar(
        x[trustworthy_monitor_start_index:],
        y_monitor[trustworthy_monitor_start_index:],
        yerr=y_monitor_err[trustworthy_monitor_start_index:],
        fmt="o-",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        capsize=4,
        elinewidth=1.2,
        ecolor="tab:blue",
        alpha=0.9,
        label="Purity monitor",
    )

    ax.plot(x[trustworthy_monitor_start_index:], lifetime_O2_only[trustworthy_monitor_start_index:], "s--", color="tab:green", linewidth=2, label="O2-like impurities lifetime")

    ax2 = ax.twinx()
    ax2.plot(
        x,
        O2_gas,
        "^-.",
        color="tab:orange",
        linewidth=2,
        markersize=7,
        label="O2 concentration",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Lifetime [µs] (Purity Monitor)", fontsize=12)
    ax2.set_ylabel("Impurities [ppb]", fontsize=12)

    ax.set_ylim(top=1.1*np.max(y_monitor))
    ax.grid(True, alpha=0.25)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="center right")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close(fig)


def plot_concentration(
    lifetime_monitor,
    lifetime_monitor_err,
    O2_gas,
    H2O_gas,
    dates,
    field_strength_Vpercm,
    events,
    gap,
    output_file="concentration_history.png",
    trustworthy_monitor_start_index=7,
):
    x = np.asarray(dates)
    gas_sources = np.array([
        gas_sampling_source(d, ULLAGE_START, LIQUID_AGAIN) for d in x
    ])

    y_monitor = np.asarray(lifetime_monitor, dtype=float)
    y_monitor_err = np.asarray(lifetime_monitor_err, dtype=float)

    # O2-equivalent from the purity monitor.
    conc_monitor = np.array([
        O2_equivalent_from_PrM(t, field_strength_Vpercm)
        for t in y_monitor
    ])
    conc_monitor_err = conc_monitor * (y_monitor_err / y_monitor)

    # O2-equivalent from the gas analysers.
    conc_gas = np.array([
        O2_equivalent_from_gas(o2, h2o, field_strength_Vpercm)
        for o2, h2o in zip(O2_gas, H2O_gas)
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    add_events_and_gap(ax, events, gap, label_events=True)

    ax.errorbar(
        x[trustworthy_monitor_start_index:],
        conc_monitor[trustworthy_monitor_start_index:],
        yerr=conc_monitor_err[trustworthy_monitor_start_index:],
        fmt="o-",
        color="tab:blue",
        linewidth=2,
        markersize=7,
        capsize=4,
        elinewidth=1.2,
        ecolor="tab:blue",
        alpha=0.9,
        label="Purity monitor (O2-equiv.)",
    )

    liquid = gas_sources == "Liquid"
    ullage = gas_sources == "Ullage"

    ax.plot(
        x[liquid],
        conc_gas[liquid],
        "o-",
        color="tab:orange",
        linewidth=2,
        markersize=7,
        label="Gas analysers (O2-equiv.) - Liquid",
    )
    ax.plot(
        x[ullage],
        conc_gas[ullage],
        "s--",
        color="tab:red",
        linewidth=2,
        markersize=7,
        label="Gas analysers (O2-equiv.) - Ullage",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("O2-equivalent impurity concentration [ppb]", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot purity-monitor lifetime and gas-analyser history."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Excel workbook containing the measurements and plot annotations.",
    )
    args = parser.parse_args()

    measurements, events, gap, settings = load_data(args.data_file)

    global GAS_LIFETIME_SCALE, ULLAGE_START, LIQUID_AGAIN
    GAS_LIFETIME_SCALE = float(settings["Gas lifetime scale"])

    CATHODE_V = float(settings["Cathode voltage"])
    ANODE_GRID_V = float(settings["Anode grid voltage"])
    DRIFT_LENGTH_CM = float(settings["Drift length"])

    O2_ZERO = float(settings["Gas analyser O2 zero offset"])
    trustworthy_index = int(settings["Trustworthy monitor start index"])
    projection_days = int(settings["Projection days"])

    ULLAGE_START = setting_timestamp(settings, "Ullage start")
    LIQUID_AGAIN = setting_timestamp(settings, "Liquid again")
    SCALE_CHANGE_DATE = setting_timestamp(settings, "O2 scale changed")

    dates = measurements["Date"].dt.to_pydatetime()
    lifetime_monitor = measurements["Purity monitor lifetime [us]"].to_numpy(float)
    lifetime_monitor_err = measurements["Purity monitor RMS [us]"].to_numpy(float)
    O2_gas = measurements["Gas analyser O2 [ppb]"].to_numpy(float)
    H2O_gas = measurements["Gas analyser H2O [ppb]"].to_numpy(float)

    field = E_field_strenght_Vpercm(
        CATHODE_V,
        ANODE_GRID_V,
        DRIFT_LENGTH_CM,
    )

    O2_corrected = apply_liquid_O2_offset(
        O2_gas,
        dates,
        O2_ZERO,
        ULLAGE_START,
        LIQUID_AGAIN,
    )

    k_O2 = _to_per_us_per_ppb(k_A_O2_per_s(field))
    k_H2O = _to_per_us_per_ppb(K_A_H2O_PER_S)

    dates_array = np.asarray(dates)
    O2_for_lifetime = O2_gas.copy()
    correction_mask = ((dates_array >= LIQUID_AGAIN) & (dates_array < SCALE_CHANGE_DATE))
    O2_for_lifetime[correction_mask] = apply_liquid_O2_offset(O2_gas[correction_mask], dates[correction_mask], O2_ZERO, ULLAGE_START, LIQUID_AGAIN)#(O2_gas[correction_mask] - O2_ZERO)
    O2_for_lifetime = np.clip(O2_for_lifetime, 0.0, None)


    lifetime_O2_only = lifetime_single_species_us(O2_for_lifetime, k_O2)
    lifetime_H2O_only = lifetime_single_species_us(H2O_gas, k_H2O)

    lifetime_gas = lifetime_gas_analyser_us(
        O2_corrected,
        H2O_gas,
        field,
    )

    print(f"Electric field: {field:.3f} V/cm")
    print("Gas analyser lifetimes [us]:")
    print(lifetime_gas)

    plot_lifetime(
        lifetime_monitor,
        lifetime_monitor_err,
        lifetime_gas,
        O2_corrected,
        dates,
        field,
        events,
        gap,
        projection_days=projection_days,
        lifetime_O2_only=lifetime_O2_only,
        lifetime_H2O_only=lifetime_H2O_only,
        trustworthy_monitor_start_index=trustworthy_index,
    )

    plot_concentration(
        lifetime_monitor,
        lifetime_monitor_err,
        O2_corrected,
        H2O_gas,
        dates,
        field,
        events,
        gap,
        trustworthy_monitor_start_index=trustworthy_index,
    )


if __name__ == "__main__":
    main()
