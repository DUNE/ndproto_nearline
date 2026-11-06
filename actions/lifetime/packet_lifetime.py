#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd

import os
import sys
import matplotlib

# Use a headless backend only for terminal execution on machines with no display.
# When imported from a Jupyter kernel, leave the inline backend alone.
if ("ipykernel" not in sys.modules) and (not os.environ.get("DISPLAY")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit, least_squares

try:
    import pylandau
except ImportError:
    raise SystemExit("Install pylandau first: pip install pylandau")

TICK_US = 0.1
ROLLOVER = 10_000_000

def args():
    p = argparse.ArgumentParser()
    p.add_argument("packet_file")
    p.add_argument("--ped", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--chunk", type=int, default=2_000_000)
    p.add_argument("--time-bin-us", type=float, default=10.0)
    p.add_argument("--drift-max-us", type=float, default=200.0)
    p.add_argument("--fit-tmin-us", type=float, default=20.0)
    p.add_argument("--fit-tmax-us", type=float, default=180.0)
    p.add_argument("--min-io-bin-entries", type=int, default=50)
    p.add_argument("--max-langau-chi2", type=float, default=3.0)
    p.add_argument("--mpv-bootstrap", type=int, default=30)
    p.add_argument("--ext-io", type=int, default=None)
    p.add_argument("--ext-trigger-type", type=int, default=2)
    p.add_argument(
        "--charge-mode",
        choices=["abs", "adc-minus-ped", "ped-minus-adc"],
        default="abs",
        help=(
            "Per-trigger pedestal subtraction convention. "
            "'abs' reproduces old v2 behavior; "
            "'adc-minus-ped' uses ADC-P; "
            "'ped-minus-adc' uses P-ADC."
        ),
    )
    p.add_argument(
        "--successive-min-ticks",
        type=int,
        default=None,
        help=(
            "Minimum allowed consecutive same-channel hit spacing "
            "in 0.1-us timestamp ticks. If used together with "
            "--successive-max-ticks, tick-window grouping replaces "
            "--successive-gap-us."
        ),
    )
    p.add_argument(
        "--successive-max-ticks",
        type=int,
        default=None,
        help=(
            "Maximum allowed consecutive same-channel hit spacing "
            "in 0.1-us timestamp ticks."
        ),
    )

    p.add_argument(
        "--successive-gap-us",
        type=float,
        default=10.0,
        help=(
            "Maximum separation in microseconds between successive hits "
            "on the same channel and EXT. Hits separated by more than "
            "this start a new summed-channel object. Default: 10 us. "
            "This remains a configurable implementation choice until "
            "the exact original grouping interval is confirmed."
        ),
    )
    p.add_argument("--keep-single", action="store_true")

    p.add_argument(
        "--ped-source",
        choices=["panel", "flow"],
        default="panel",
        help=(
            "Pedestal source. 'panel' uses the nearline panel_ped JSON; "
            "'flow' recovers the static per-channel pedestal implicit "
            "in FLOW Q_raw."
        ),
    )

    p.add_argument(
        "--flow-file",
        default=None,
        help=(
            "Matching FLOW HDF5 file used only when --ped-source flow. "
            "No FLOW event building or corrected charge is used."
        ),
    )
    return p.parse_args()

def find_ped(packet):
    m = re.search(r"packet-(\d+)", packet.name)
    cands = [packet.with_name(packet.name+".panel_ped.json"),
             packet.with_name(packet.stem+".panel_ped.json")]
    if m:
        run = m.group(1)
        cands += [packet.with_name(f"packet{run}.panel_ped.json"),
                  packet.with_name(f"packet-{run}.panel_ped.json")]
        cands += list(packet.parent.glob(f"*{run}*panel_ped*.json"))
    for c in cands:
        if c.exists():
            return c.resolve()
    return None

def fill_last(x):
    idx = np.where(x != 0, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.zeros_like(x)
    good = idx >= 0
    out[good] = x[idx[good]]
    return out

def unroll_mclk(p):
    raw = p["timestamp"].astype(np.int64) % (2**31)
    out = np.zeros(len(p), dtype=np.int64)
    for io in np.unique(p["io_group"]):
        gi = np.where(p["io_group"] == io)[0]
        q, qts = p[gi], raw[gi]
        sync = (q["packet_type"] == 6) & (q["trigger_type"] == 83)
        inc = np.zeros(len(q), dtype=np.int64)
        inc[sync] = (np.round(qts[sync] / ROLLOVER) * ROLLOVER).astype(np.int64)
        offsets = np.cumsum(inc) - inc
        oops = (q["packet_type"] == 0) & (q["receipt_timestamp"].astype(np.int64) < qts)
        offsets[oops] -= fill_last(inc)[oops]
        out[gi] = (qts % ROLLOVER) + offsets
    return out

def window_count(data, trig, lo_us, hi_us):
    lo, hi = int(lo_us/TICK_US), int(hi_us/TICK_US)
    n = 0
    for t0 in trig:
        n += np.searchsorted(data, t0+hi, side="right") - np.searchsorted(data, t0+lo, side="left")
    return n

def choose_ext(tim, mclk, data_sorted, trigger_type):
    rows = []
    ios = np.unique(tim["io_group"][(tim["packet_type"]==7)&(tim["trigger_type"]==trigger_type)])
    for io in ios:
        m = (tim["packet_type"]==7)&(tim["trigger_type"]==trigger_type)&(tim["io_group"]==io)
        tt = np.sort(mclk[m])
        if len(tt) < 10: continue
        pre = window_count(data_sorted, tt, -200, 0)/len(tt)
        post = window_count(data_sorted, tt, 0, 200)/len(tt)
        rows.append(dict(io_group=int(io), n_triggers=len(tt), pre_per_trigger=pre,
                         post_per_trigger=post, excess_per_trigger=post-pre,
                         post_over_pre=(post/pre if pre>0 else np.inf)))
    d = pd.DataFrame(rows).sort_values(["excess_per_trigger","post_over_pre"], ascending=False)
    if len(d)==0: raise RuntimeError("No usable EXT stream")
    return int(d.iloc[0].io_group), d

def dt_hist(data, trig, lo_us=-100, hi_us=300, bin_us=2):
    lo, hi, bw = int(lo_us/TICK_US), int(hi_us/TICK_US), int(bin_us/TICK_US)
    edges = np.arange(lo, hi+bw, bw)
    c = np.zeros(len(edges)-1, dtype=np.int64)
    for t0 in trig:
        i0 = np.searchsorted(data, t0+lo, side="left")
        i1 = np.searchsorted(data, t0+hi, side="right")
        c += np.histogram(data[i0:i1]-t0, bins=edges)[0]
    return 0.5*(edges[:-1]+edges[1:])*TICK_US, c

def fit_langau(q, bin_width=1.0, qmin=0.0, qmax=50.0):
    q = np.asarray(q, float)
    q = q[np.isfinite(q)&(q>=qmin)&(q<qmax)]
    edges = np.arange(qmin, qmax+bin_width, bin_width)
    hist, edges = np.histogram(q, bins=edges)
    ctr = 0.5*(edges[:-1]+edges[1:])
    sm = gaussian_filter1d(hist.astype(float), 1.0)
    ip = np.argmax(sm); half = sm[ip]/2
    left=ip
    while left>0 and sm[left]>=half: left-=1
    right=ip
    while right<len(sm)-1 and sm[right]>=half: right+=1
    mask = np.zeros(len(hist), bool); mask[left:right+1]=True
    xfit, yfit = ctr[mask], hist[mask].astype(float)
    if len(xfit)<5: raise RuntimeError("Too few FWHM bins")
    def model(x, mu, eta, ratio, area):
        return area*pylandau.langau_pdf(x, mu=mu, eta=eta, sigma=ratio*eta)*bin_width
    pars, cov = curve_fit(model, xfit, yfit, p0=[ctr[ip],2,0.5,len(q)],
                          bounds=([qmin,.1,.02,0],[qmax,15,10,np.inf]),
                          max_nfev=10000)
    xx = np.linspace(qmin,qmax,5000); yy = model(xx,*pars)
    mpv = xx[np.argmax(yy)]
    err = np.sqrt(yfit+1.0); chi2=np.sum(((yfit-model(xfit,*pars))/err)**2)
    ndf=len(xfit)-4
    return dict(mpv=mpv, red_chi2=(chi2/ndf if ndf>0 else np.nan),
                pars=pars, cov=cov, x=ctr, hist=hist, mask=mask, xx=xx, yy=yy)

def bootstrap_mpv(q, nboot, seed):
    central = fit_langau(q)["mpv"]
    if nboot <= 0: return central, np.nan, 0
    q=np.asarray(q,float); rng=np.random.default_rng(seed); vals=[]
    for _ in range(nboot):
        try:
            r=fit_langau(q[rng.integers(0,len(q),len(q))])
            if np.isfinite(r["mpv"]): vals.append(r["mpv"])
        except Exception: pass
    vals=np.asarray(vals)
    return central, (np.std(vals,ddof=1) if len(vals)>=10 else np.nan), len(vals)

def common_fit(d, tmin, tmax):
    d=d[(d.time_us>=tmin)&(d.time_us<=tmax)&np.isfinite(d.mpv)&np.isfinite(d.mpv_err)&(d.mpv_err>0)].copy()
    ios=sorted(d.io_group.unique()); lookup={io:i for i,io in enumerate(ios)}
    t=d.time_us.to_numpy()/1000.; q=d.mpv.to_numpy(); e=d.mpv_err.to_numpy()
    ii=np.array([lookup[x] for x in d.io_group])
    def resid(p):
        return (q-p[1:][ii]*np.exp(-p[0]*t))/e
    A0=[d.loc[d.io_group==io,"mpv"].median() for io in ios]
    r=least_squares(resid,[.5]+A0,bounds=([-5]+[0]*len(ios),[5]+[50]*len(ios)),max_nfev=10000)
    alpha=float(r.x[0]); chi2=float(np.sum(resid(r.x)**2)); ndf=len(d)-(1+len(ios))
    return alpha,{int(io):float(r.x[i+1]) for i,io in enumerate(ios)},chi2,ndf,chi2/ndf,d


def common_fit_unweighted(d, tmin, tmax):
    """
    Central common-IO attenuation fit without MPV uncertainties.

    Fits:
        Q_i(t) = A_i * exp(-alpha * t)

    with one independent normalization A_i per IO group and one common alpha.

    This is the pre-bootstrap central lifetime fit. Because no MPV uncertainties
    are supplied, this is NOT a chi-square fit and no chi2/ndf is reported.
    """
    d = d[
        (d["time_us"] >= tmin)
        & (d["time_us"] <= tmax)
        & np.isfinite(d["mpv"])
        & (d["mpv"] > 0)
    ].copy()

    if len(d) == 0:
        raise RuntimeError("No MPV points are available for the unweighted common fit.")

    ios = sorted(d["io_group"].unique())
    io_to_index = {io: i for i, io in enumerate(ios)}

    t_ms = d["time_us"].to_numpy() / 1000.0
    q_obs = d["mpv"].to_numpy()
    io_index = np.array([io_to_index[x] for x in d["io_group"]])

    def residuals(p):
        alpha = p[0]
        amplitudes = p[1:]
        q_pred = amplitudes[io_index] * np.exp(-alpha * t_ms)
        return q_obs - q_pred

    A0 = [
        d.loc[d["io_group"] == io, "mpv"].median()
        for io in ios
    ]

    result = least_squares(
        residuals,
        [0.4] + A0,
        bounds=(
            [-5.0] + [0.0] * len(ios),
            [5.0] + [50.0] * len(ios),
        ),
        max_nfev=10_000,
    )

    alpha = float(result.x[0])
    amplitudes = {
        int(io): float(result.x[i + 1])
        for i, io in enumerate(ios)
    }

    residual_sse = float(np.sum(residuals(result.x) ** 2))
    residual_rms = float(np.sqrt(np.mean(residuals(result.x) ** 2)))

    return alpha, amplitudes, residual_sse, residual_rms, d


def main():
    a=args()
    packet=Path(a.packet_file).expanduser().resolve()
    if not packet.exists(): raise SystemExit(f"Missing {packet}")
    ped=Path(a.ped).expanduser().resolve() if a.ped else find_ped(packet)
    if ped is None or not ped.exists():
        raise SystemExit("Pedestal JSON not found automatically; pass --ped PATH")
    out=Path(a.outdir).expanduser().resolve() if a.outdir else packet.parent/f"{packet.stem}_lifetime"
    out.mkdir(parents=True,exist_ok=True)
    print("packet:",packet,"\nped:",ped,"\nout:",out)

    with h5py.File(packet,"r") as f:
        packets=f["packets"]
        print("HDF5 keys:",list(f.keys())); print("rows:",len(packets)); print("dtype:",packets.dtype)

        # counts
        pt=Counter(); combos=Counter()
        for s in range(0,len(packets),a.chunk):
            z=packets.fields(["packet_type","io_group","trigger_type"])[s:min(s+a.chunk,len(packets))]
            v,c=np.unique(z["packet_type"],return_counts=True)
            for x,n in zip(v,c): pt[int(x)]+=int(n)
            u,c=np.unique(np.stack([z["packet_type"],z["io_group"],z["trigger_type"]],axis=1),axis=0,return_counts=True)
            for row,n in zip(u,c): combos[tuple(map(int,row))]+=int(n)
        pd.DataFrame([{"packet_type":k,"count":v} for k,v in sorted(pt.items())]).to_csv(out/"packet_type_counts.csv",index=False)
        pd.DataFrame([{"packet_type":p,"io_group":io,"trigger_type":tr,"count":n} for (p,io,tr),n in sorted(combos.items())]).to_csv(out/"packet_io_trigger_counts.csv",index=False)
        print("packet types:",dict(sorted(pt.items())))

        # raw ADC
        z=packets.fields(["packet_type","dataword"])[:2_000_000]
        adc=z["dataword"][z["packet_type"]==0]
        plt.figure(figsize=(8,5)); plt.hist(adc,bins=np.arange(257)-.5,histtype="step")
        plt.yscale("log"); plt.xlabel("Raw ADC"); plt.ylabel("Packets"); plt.tight_layout()
        plt.savefig(out/"raw_adc.png",dpi=150); plt.close()

        # timing
        fields=["packet_type","io_group","trigger_type","timestamp","receipt_timestamp"]
        print("loading timing fields...")
        tim=packets.fields(fields)[:]
        mclk=unroll_mclk(tim)
        data_mask=tim["packet_type"]==0
        data_sorted=np.sort(mclk[data_mask])

        auto_io,score=choose_ext(tim,mclk,data_sorted,a.ext_trigger_type)
        ext_io=a.ext_io if a.ext_io is not None else auto_io
        score.to_csv(out/"ext_scores.csv",index=False)
        print(score.to_string(index=False)); print("using EXT IO",ext_io)

        plt.figure(figsize=(10,6))
        for io in sorted(score.io_group.unique()):
            m=(tim["packet_type"]==7)&(tim["io_group"]==io)&(tim["trigger_type"]==a.ext_trigger_type)
            tt=np.sort(mclk[m]); x,y=dt_hist(data_sorted,tt)
            plt.step(x,y/max(len(tt),1),where="mid",label=f"IO{io}")
        plt.axvline(0,ls="--"); plt.xlabel(r"$t_{packet}-t_{EXT}$ [$\mu$s]"); plt.ylabel("packets / trigger / 2 us")
        plt.legend(); plt.tight_layout(); plt.savefig(out/"ext_activity.png",dpi=150); plt.close()

        em=(tim["packet_type"]==7)&(tim["io_group"]==ext_io)&(tim["trigger_type"]==a.ext_trigger_type)
        ext_t=np.sort(mclk[em])
        data_idx=np.where(data_mask)[0]; data_mclk=mclk[data_idx]
        ext_index=np.searchsorted(ext_t,data_mclk,side="right")-1
        good=ext_index>=0
        dt=np.full(len(data_mclk),-1,dtype=np.int64)
        dt[good]=data_mclk[good]-ext_t[ext_index[good]]
        dt_us=dt*TICK_US
        drift=good&(dt_us>=0)&(dt_us<=a.drift_max_us)
        print("EXT:",len(ext_t),"charge:",len(data_mclk),"selected:",int(drift.sum()))

        # selected fields
        sf=["packet_type","io_group","io_channel","chip_id","channel_id","dataword"]
        parts=[]; cursor=0
        for s in range(0,len(packets),a.chunk):
            z=packets.fields(sf)[s:min(s+a.chunk,len(packets))]
            md=z["packet_type"]==0; nd=int(md.sum()); use=drift[cursor:cursor+nd]; q=z[md]
            if use.any(): parts.append(q[use][["io_group","io_channel","chip_id","channel_id","dataword"]])
            cursor+=nd
        selected=np.concatenate(parts)
        sel_dt=dt_us[drift]; sel_ext=ext_index[drift]

    np.savez_compressed(out/"selected_packets.npz",selected=selected,selected_dt_us=sel_dt,selected_ext_index=sel_ext)
    df=pd.DataFrame(dict(ext=sel_ext,dt_us=sel_dt,io_group=selected["io_group"],io_channel=selected["io_channel"],
                         chip_id=selected["chip_id"],channel_id=selected["channel_id"],adc=selected["dataword"]))

    # ========================================================
    # Pedestal source
    # ========================================================

    if a.ped_source == "panel":

        print("pedestal source: panel_ped JSON")
        print("pedestal file:", ped)

        with open(ped) as fh:
            pj = json.load(fh)

        lookup = {}

        for k, vals in pj.items():
            io, ioch, chip = map(int, k.split("-"))

            for ch, pair in enumerate(vals):
                if float(pair[0]) >= 0:
                    lookup[(io, ioch, chip, ch)] = float(pair[0])

        keys = zip(
            df.io_group.astype(int),
            df.io_channel.astype(int),
            df.chip_id.astype(int),
            df.channel_id.astype(int),
        )

        df["pedestal"] = [
            lookup.get(k, np.nan)
            for k in keys
        ]


    elif a.ped_source == "flow":

        if a.flow_file is None:
            raise RuntimeError(
                "--flow-file is required when --ped-source flow"
            )

        flow = Path(a.flow_file).expanduser().resolve()

        if not flow.exists():
            raise FileNotFoundError(
                f"FLOW file does not exist: {flow}"
            )

        print("pedestal source: FLOW static Q_raw pedestal")
        print("FLOW file:", flow)

        # Reuse the already-tested pedestal reconstruction from
        # packet_lifetime_final.py.
        #
        # This does NOT use FLOW event building, corrected charge,
        # reconstructed t0, or Y-Z information.
        from packet_lifetime_final import (
            flow_ped_map,
            key as flow_key,
            KE_PER_ADC,
        )

        lookup = flow_ped_map(
            flow,
            KE_PER_ADC,
            chunk=1_000_000,
        )

        print("FLOW pedestal channels:", len(lookup))

        kk = flow_key(
            df.io_group.to_numpy(dtype=np.int64),
            df.io_channel.to_numpy(dtype=np.int64),
            df.chip_id.to_numpy(dtype=np.int64),
            df.channel_id.to_numpy(dtype=np.int64),
        )

        df["pedestal"] = np.fromiter(
            (
                lookup.get(int(k), np.nan)
                for k in kk
            ),
            dtype=float,
            count=len(kk),
        )


    coverage = float(df.pedestal.notna().mean())

    print("pedestal coverage:", coverage)

    if coverage < 0.95:
        print(
            "WARNING: pedestal coverage is below 95%; "
            "check FLOW/packet channel mapping."
        )
    df=df.dropna(subset=["pedestal"]).copy()
    # --------------------------------------------------------
    # Brooke-like packet charge construction
    #
    # Keep the v2 charge convention for the moment:
    #     q_trigger = |ADC - pedestal|
    #
    # The absolute-value convention is still provisional; the
    # successive-trigger construction below is the part clarified
    # directly by Brooke.
    # --------------------------------------------------------
    delta_adc = (
        df.adc.astype(float) - df.pedestal
    )

    if a.charge_mode == "abs":
        df["q_abs"] = np.abs(delta_adc)

    elif a.charge_mode == "adc-minus-ped":
        df["q_abs"] = delta_adc

    elif a.charge_mode == "ped-minus-adc":
        df["q_abs"] = -delta_adc

    print("charge mode:", a.charge_mode)

    # --------------------------------------------------------
    # Group SUCCESSIVE hits on the SAME physical channel
    # belonging to the SAME EXT.
    #
    # Brooke clarification:
    #   - pedestal subtract individual triggers
    #   - sum successive triggers on a channel
    #   - use the LAST successive hit as the reference time
    #
    # --successive-gap-us determines when a new group starts.
    # --------------------------------------------------------
    channel_keys = [
        "ext",
        "io_group",
        "io_channel",
        "chip_id",
        "channel_id",
    ]

    df = df.sort_values(
        channel_keys + ["dt_us"]
    ).reset_index(drop=True)

    gap_us = df.groupby(
        channel_keys,
        sort=False
    )["dt_us"].diff()

    # Convert observed time differences back into the native
    # 0.1-us timestamp tick units.
    gap_ticks = np.rint(
        gap_us / TICK_US
    )

    if (
        a.successive_min_ticks is not None
        and a.successive_max_ticks is not None
    ):
        # Retrigger-window mode:
        # hits belong to the same successive sequence only when
        # their separation lies inside the selected tick window.
        is_successive = (
            gap_ticks.notna()
            & (gap_ticks >= a.successive_min_ticks)
            & (gap_ticks <= a.successive_max_ticks)
        )

        new_group = ~is_successive

        print(
            "successive-hit mode: tick window "
            f"{a.successive_min_ticks}-"
            f"{a.successive_max_ticks} ticks "
            f"({a.successive_min_ticks*TICK_US:.1f}-"
            f"{a.successive_max_ticks*TICK_US:.1f} us)"
        )

    else:
        # Original v3 maximum-gap mode.
        new_group = (
            gap_us.isna()
            | (gap_us > a.successive_gap_us)
        )

        print(
            "successive-hit mode: max gap "
            f"{a.successive_gap_us} us"
        )

    # Number successive groups independently inside each
    # EXT/channel combination.
    df["_new_group"] = new_group.astype(np.int64)

    df["successive_group"] = (
        df.groupby(
            channel_keys,
            sort=False
        )["_new_group"]
        .cumsum()
    )

    group_keys = channel_keys + ["successive_group"]

    sc = (
        df.groupby(
            group_keys,
            sort=False
        )
        .agg(
            q_sum=("q_abs", "sum"),
            n_packets=("q_abs", "size"),

            # Brooke: LAST successive hit is reference time
            dt_last=("dt_us", "max"),

            # keep mean only as a diagnostic
            dt_mean=("dt_us", "mean"),
        )
        .reset_index()
    )

    # The summed-channel object is located at the LAST hit time.
    sc["time_us"] = sc["dt_last"]

    # Only now divide the physical drift window into slices
    # for the Langau fits.
    sc["time_bin"] = np.floor(
        sc["time_us"] / a.time_bin_us
    ).astype(int)

    # Preserve existing v2 charge-fit window.
    # No additional fixed near-threshold ADC cut is applied.
    fit = sc[
        (sc.q_sum > 0)
        & (sc.q_sum < 50)
        & (sc.time_us >= 0)
        & (sc.time_us < a.drift_max_us)
    ].copy()

    # Brooke-like low-threshold selection:
    # disregard isolated single-trigger objects by default.
    if not a.keep_single:
        fit = fit[
            fit.n_packets > 1
        ].copy()

    df.drop(
        columns=["_new_group"],
        inplace=True,
        errors="ignore",
    )
    sc.to_pickle(out/"slice_charge.pkl"); fit.to_pickle(out/"fit_sample.pkl")

    plt.figure(figsize=(10,6)); plt.hist2d(fit.time_us,fit.q_sum,
        bins=[np.arange(0,a.drift_max_us+a.time_bin_us,a.time_bin_us),np.arange(0,50.5,.5)],norm=LogNorm())
    plt.xlabel(r"$\Delta t_{\mathrm{last\ hit}-EXT}$ [$\mu$s]"); plt.ylabel(r"$\sum |ADC-P|$ [ADC]")
    plt.colorbar(label="objects"); plt.tight_layout(); plt.savefig(out/"charge_map.png",dpi=150); plt.close()

    # IO/time Langau
    rows=[]
    for io in sorted(fit.io_group.unique()):
        dio=fit[fit.io_group==io]
        for tb in range(int(a.drift_max_us/a.time_bin_us)):
            q=dio.loc[dio.time_bin==tb,"q_sum"].to_numpy()
            if len(q)<a.min_io_bin_entries: continue
            try:
                r=fit_langau(q)
                if not np.isfinite(r["red_chi2"]) or r["red_chi2"]>=a.max_langau_chi2: continue
                mpv,err,nok=bootstrap_mpv(q,a.mpv_bootstrap,100000+1000*int(io)+tb)
                rows.append(dict(io_group=int(io),time_bin=tb,time_us=(tb+.5)*a.time_bin_us,N=len(q),
                                 mpv=mpv,mpv_err=err,langau_chi2_ndf=r["red_chi2"],n_boot_success=nok))
            except Exception as e:
                print("fit failed",io,tb,e)
    md = pd.DataFrame(
        rows,
        columns=[
            "io_group",
            "time_bin",
            "time_us",
            "N",
            "mpv",
            "mpv_err",
            "langau_chi2_ndf",
            "n_boot_success",
        ],
    )
    md.to_csv(out/"io_time_langau_mpvs.csv", index=False)

    if md.empty:
        print(
            "No valid Langau MPV fits for this configuration. "
            "The successive-hit grouping may be too restrictive "
            "or all candidate fits failed the Langau quality cut."
        )
        return

    plt.figure(figsize=(10,6))
    for io in sorted(md.io_group.unique()):
        d=md[md.io_group==io]
        plt.errorbar(d.time_us,d.mpv,yerr=d.mpv_err,fmt="o-",capsize=2,label=f"IO {io}")
    plt.xlabel(r"$\Delta t_{EXT}$ [$\mu$s]"); plt.ylabel("Langau MPV [ADC]"); plt.legend(ncol=2); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out/"io_mpv.png",dpi=150); plt.close()

    # ------------------------------------------------------------
    # Lifetime fits
    # ------------------------------------------------------------
    # Always make the central PRE-BOOTSTRAP common-IO fit.
    unweighted_alpha = np.nan
    unweighted_tau = np.nan
    unweighted_sse = np.nan
    unweighted_rms = np.nan
    unweighted_A = {}
    unweighted_fd = pd.DataFrame()

    try:
        (
            unweighted_alpha,
            unweighted_A,
            unweighted_sse,
            unweighted_rms,
            unweighted_fd,
        ) = common_fit_unweighted(
            md,
            a.fit_tmin_us,
            a.fit_tmax_us,
        )

        unweighted_tau = (
            1.0 / unweighted_alpha
            if unweighted_alpha > 0
            else np.nan
        )

        print("\nUNWEIGHTED CENTRAL LIFETIME FIT")
        print("alpha =", unweighted_alpha, "1/ms")
        print("tau   =", unweighted_tau, "ms")
        print("residual SSE =", unweighted_sse)
        print("residual RMS =", unweighted_rms, "ADC")
        print("(No chi2/ndf here because MPV uncertainties were not used.)")

        tt = np.linspace(a.fit_tmin_us, a.fit_tmax_us, 300)

        plt.figure(figsize=(10, 7))
        for io in sorted(unweighted_fd.io_group.unique()):
            d = unweighted_fd[unweighted_fd.io_group == io]
            plt.plot(
                d.time_us,
                d.mpv,
                "o",
                label=f"IO {io}",
            )
            plt.plot(
                tt,
                unweighted_A[int(io)]
                * np.exp(-unweighted_alpha * tt / 1000.0),
            )

        plt.xlabel(r"$\Delta t_{EXT}$ [$\mu$s]")
        plt.ylabel("Langau MPV [ADC]")
        plt.title(
            "Unweighted central common-IO fit\n"
            + rf"$\alpha={unweighted_alpha:.3f}\,\mathrm{{ms}}^{{-1}}$, "
            + (
                rf"$\tau={unweighted_tau:.3f}\,\mathrm{{ms}}$"
                if np.isfinite(unweighted_tau)
                else r"$\tau$ undefined for non-positive $\alpha$"
            )
        )
        plt.legend(ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "unweighted_common_fit.png", dpi=150)
        plt.close()

    except Exception as exc:
        print("\nUNWEIGHTED CENTRAL LIFETIME FIT FAILED:", exc)

    # If MPV bootstrap errors exist, additionally perform the weighted fit.
    weighted_alpha = np.nan
    weighted_tau = np.nan
    chi2 = np.nan
    red = np.nan
    ndf = 0
    weighted_A = {}
    weighted_fd = pd.DataFrame()

    if a.mpv_bootstrap > 0 and md.mpv_err.notna().sum():
        try:
            (
                weighted_alpha,
                weighted_A,
                chi2,
                ndf,
                red,
                weighted_fd,
            ) = common_fit(
                md,
                a.fit_tmin_us,
                a.fit_tmax_us,
            )

            weighted_tau = (
                1.0 / weighted_alpha
                if weighted_alpha > 0
                else np.nan
            )

            print("\nWEIGHTED LIFETIME FIT")
            print("alpha =", weighted_alpha, "1/ms")
            print("tau   =", weighted_tau, "ms")
            print("chi2/ndf =", chi2, "/", ndf, "=", red)

            tt = np.linspace(a.fit_tmin_us, a.fit_tmax_us, 300)

            plt.figure(figsize=(10, 7))
            for io in sorted(weighted_fd.io_group.unique()):
                d = weighted_fd[weighted_fd.io_group == io]
                plt.errorbar(
                    d.time_us,
                    d.mpv,
                    yerr=d.mpv_err,
                    fmt="o",
                    capsize=2,
                    label=f"IO {io}",
                )
                plt.plot(
                    tt,
                    weighted_A[int(io)]
                    * np.exp(-weighted_alpha * tt / 1000.0),
                )

            plt.xlabel(r"$\Delta t_{EXT}$ [$\mu$s]")
            plt.ylabel("Langau MPV [ADC]")
            plt.title(
                "Weighted common-IO fit\n"
                + rf"$\alpha={weighted_alpha:.3f}\,\mathrm{{ms}}^{{-1}}$, "
                + (
                    rf"$\tau={weighted_tau:.3f}\,\mathrm{{ms}}$, "
                    if np.isfinite(weighted_tau)
                    else ""
                )
                + rf"$\chi^2/\mathrm{{ndf}}={red:.2f}$"
            )
            plt.legend(ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out / "weighted_common_fit.png", dpi=150)
            plt.close()

        except Exception as exc:
            print("\nWEIGHTED LIFETIME FIT FAILED:", exc)
    else:
        print(
            "\nWeighted lifetime fit skipped: "
            "no bootstrap MPV uncertainties were requested."
        )

    summary = dict(
        packet_file=str(packet),
        pedestal_file=str(ped),
        packet_type_counts=dict(pt),
        ext_io_group=int(ext_io),
        ext_trigger_type=int(a.ext_trigger_type),
        n_ext=int(len(ext_t)),
        n_selected_packets=int(len(df)),
        pedestal_coverage=coverage,
        time_bin_us=a.time_bin_us,
        charge_convention="sum of abs(ADC-pedestal) over successive channel hits [PROVISIONAL sign]",
        successive_hit_grouping=(
            "same EXT + same physical channel; split when consecutive "
            f"hit separation exceeds {a.successive_gap_us} us"
        ),
        summed_object_time_reference="last successive hit",
        successive_gap_us=float(a.successive_gap_us),
        single_packet_rejection=(not a.keep_single),
        n_fit_objects=int(len(fit)),
        mpv_bootstrap=int(a.mpv_bootstrap),

        # Always available if the central fit succeeds:
        unweighted_alpha_per_ms=(
            float(unweighted_alpha)
            if np.isfinite(unweighted_alpha)
            else None
        ),
        unweighted_tau_ms=(
            float(unweighted_tau)
            if np.isfinite(unweighted_tau)
            else None
        ),
        unweighted_residual_sse=(
            float(unweighted_sse)
            if np.isfinite(unweighted_sse)
            else None
        ),
        unweighted_residual_rms_adc=(
            float(unweighted_rms)
            if np.isfinite(unweighted_rms)
            else None
        ),

        # Populated only when bootstrap MPV uncertainties are present:
        weighted_alpha_per_ms=(
            float(weighted_alpha)
            if np.isfinite(weighted_alpha)
            else None
        ),
        weighted_tau_ms=(
            float(weighted_tau)
            if np.isfinite(weighted_tau)
            else None
        ),
        weighted_chi2=(
            float(chi2)
            if np.isfinite(chi2)
            else None
        ),
        weighted_ndf=int(ndf),
        weighted_chi2_ndf=(
            float(red)
            if np.isfinite(red)
            else None
        ),
    )

    with open(out/"summary.json","w") as fh: json.dump(summary,fh,indent=2)
    print("saved:",out)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
