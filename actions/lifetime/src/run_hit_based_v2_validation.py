#!/usr/bin/env python3
"""DUNE 2x2 hit-based lifetime V2 *diagnostic*, not a replacement measurement.

Uses the same previously cached IO6-EXT packet selection, FLOW-v9 pedestal,
EXT/channel grouping, compact-span cut, charge fitter, and 40-us drift slices
as run_paired_lifetime_timeline_all.py and run_io6_n2_split_io.py.

Changes just one question: treat exactly n=2 and exactly n=3 packet groups
as two separate charge populations; give each (charge IO, n) its own MPV
normalization, then test whether one common attenuation slope is supported.

Never edits packet_lifetime_final.py, the older scripts, or their outputs.

WARNING: No n=1 acceptance, no threshold correction. All significance tests
are diagnostic approximations, conditional on bootstrap MPV errors.
"""
from __future__ import annotations

import argparse
import json
import re
import zlib
import io
import lzma
import tarfile
from pathlib import Path
import sys

# Historical analysis dependencies are preserved in old/.
# This changes import locations, not the physics analysis.
_OLD_DIR = Path(__file__).resolve().parents[1] / "old"
if _OLD_DIR.is_dir():
    sys.path.insert(0, str(_OLD_DIR))

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_distribution

TIME_RANGES = [(20, 60), (60, 100), (100, 140), (140, 180)]
CHANNEL_KEYS = ['io_group', 'io_channel', 'chip_id', 'channel_id']


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--outdir', type=Path, default=Path('hit_based_v2_validation'))
    p.add_argument('--cached-root', type=Path, default=Path('paired_lifetime_full_io5_io6/selected_cache'))
    p.add_argument('--flow-root', type=Path, default=Path('/global/cfs/cdirs/dune/www/data/2x2/reflows/v9/flow/beam'))
    p.add_argument('--frozen-root', type=Path, action='append', default=None)
    p.add_argument('--only', type=str, help='Only filenames containing this substring')
    p.add_argument('--max-files', type=int, help='Process at most this many acquisitions (for quick tests)')
    p.add_argument('--bootstrap', type=int, default=30, help='New Langau MPV bootstrap iterations')
    p.add_argument('--bootstrap-unit', choices=['group', 'ext'], default='group',
                   help='Resample groups independently, or EXT clusters within each IO/n/time bin')
    p.add_argument('--min-bin-entries', type=int, default=100)
    p.add_argument('--min-io-bins', type=int, default=4)
    p.add_argument('--reuse-n2-csv', type=Path,
                   default=Path('io6_n2_split_full/io6_n2_per_io_mpvs.csv'),
                   help='Reuse previous n=2 MPVs/errors if this CSV exists; use --refit-n2 to force refit')
    p.add_argument('--refit-n2', action='store_true', help='Independently recompute n=2 MPVs instead of reusing older split-IO CSV')
    p.add_argument('--daq-audit', action='store_true', help='Audit configured/enabled channels using raw-file DAQ JSON; NO filtering')
    p.add_argument('--per-file-plots', action='store_true')
    p.add_argument('--list-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args()


def select_multiplicity(groups, n):
    """Same FULL EXT/channel group definition; no gap-splitting, no n>=2 mixing."""
    if n not in (2, 3):
        raise ValueError('Only exact n=2 or exact n=3 is supported')
    return groups.loc[(groups['n'] == n) & (groups['span_us'] <= 10.0) &
                      (groups['t_last'] >= 20.0) & (groups['t_last'] < 180.0)].copy()


def sample_bootstrap(q, exts, rng, unit):
    if unit == 'group':
        return q[rng.integers(0, len(q), size=len(q))]
    # Cluster bootstrap INSIDE A SINGLE time/IO/multiplicity bin, not across
    # the entire file. This does not include cross-bin/IO covariance.
    _, inverse = np.unique(exts, return_inverse=True)
    n_ext = int(inverse.max()) + 1
    weights = np.bincount(rng.integers(0, n_ext, size=n_ext), minlength=n_ext)
    return np.repeat(q, weights[inverse])


def fit_spectra(selected, n, filename, core, n_boot, min_bin, unit):
    rows = []
    for charge_io in range(1, 9):
        sub = selected.loc[selected.io_group == charge_io]
        for left, right in TIME_RANGES:
            d = sub.loc[(sub.t_last >= left) & (sub.t_last < right)]
            q = d.q_sum_ADC.to_numpy(float)
            exts = d.ext_index.to_numpy(np.int64)
            keep = np.isfinite(q)
            q, exts = q[keep], exts[keep]
            row = dict(io_group=charge_io, n=n, t_start=left, t_end=right,
                       t_center=(left + right) / 2., N=len(q),
                       MPV_ADC=np.nan, MPV_error=np.nan, chi2_ndf=np.nan,
                       n_boot_success=0, fit_fraction=np.nan, failure='',
                       source='new_fit', bootstrap_unit=unit)
            if len(q) < min_bin:
                row['failure'] = f'Only {len(q)} groups, min {min_bin}'
                rows.append(row)
                continue
            row['fit_fraction'] = float(np.mean((q >= 0) & (q < 50)))
            try:
                fit = core.fit_langau(q, bin_width=1., qmin=0., qmax=50.)
                row['MPV_ADC'] = float(fit['mpv'])
                row['chi2_ndf'] = float(fit['red_chi2'])
                if not np.isfinite(row['MPV_ADC']) or row['MPV_ADC'] <= 0:
                    raise ValueError('Nonphysical central MPV')
            except Exception as exc:
                row['failure'] = f'Langau: {type(exc).__name__}: {exc}'
                rows.append(row)
                continue
            seed = zlib.crc32(f'{filename}|IO{charge_io}|n{n}|{left}-{right}|{unit}|v2'.encode())
            rng = np.random.default_rng(seed)
            boot = []
            for _ in range(n_boot):
                trial = sample_bootstrap(q, exts, rng, unit)
                if len(trial) < min_bin:
                    continue
                try:
                    v = float(core.fit_langau(trial, bin_width=1., qmin=0., qmax=50.)['mpv'])
                    if np.isfinite(v) and v > 0:
                        boot.append(v)
                except Exception:
                    pass
            row['n_boot_success'] = len(boot)
            if len(boot) >= max(2, (n_boot + 1)//2):
                row['MPV_error'] = float(np.std(boot, ddof=1))
            if not np.isfinite(row['MPV_error']) or row['MPV_error'] <= 0:
                row['failure'] = f'Insufficient valid bootstrap error ({len(boot)}/{n_boot})'
            rows.append(row)
    return pd.DataFrame(rows)


def read_old_n2(csv_path, filename):
    d = pd.read_csv(csv_path)
    needed = {'packet_file', 'io_group', 't_center', 'MPV_ADC', 'MPV_error', 'chi2_ndf', 'N'}
    if not needed.issubset(d.columns):
        raise ValueError(f'Existing n2 CSV missing fields: {sorted(needed - set(d.columns))}')
    d = d.loc[d.packet_file == filename].copy()
    if d.empty:
        return None
    if d.duplicated(['io_group', 't_center']).any():
        raise ValueError(f'Duplicate n2 MPV rows in {csv_path} for {filename}')
    if d.io_group.nunique() > 8 or len(d) > 32:
        raise ValueError('More than eight charge IOs or four time bins in reused n2 CSV')
    allowed = {40., 80., 120., 160.}
    if not set(d.t_center.astype(float)).issubset(allowed):
        raise ValueError('Unexpected time-bin center in cached n2 MPVs')
    d['n'] = 2
    d['source'] = 'existing_split_io_n2_csv'
    d['bootstrap_unit'] = 'group (existing)'
    return d


def valid_fit_rows(mpvs):
    d = mpvs.copy()
    for c in ['MPV_ADC', 'MPV_error', 'chi2_ndf']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d[np.isfinite(d.MPV_ADC) & (d.MPV_ADC > 0) &
             np.isfinite(d.MPV_error) & (d.MPV_error > 0)].copy()


def fit_model(mpvs, n_types, split, min_io_bins):
    """Use the PREVIOUS tested shared-alpha fit directly.

    For combined model create unique synthetic group IDs 1..8 (n2),
    11..18 (n3); actual IO group and multiplicity remain unchanged in CSV.
    """
    d = valid_fit_rows(mpvs[mpvs.n.isin(n_types)])
    if split is None:
        raise RuntimeError('Original split-IO fitting module unavailable')
    if len(n_types) == 2:
        d['io_group'] = d['io_group'].astype(int) + np.where(d.n.astype(int) == 3, 10, 0)
    result, amplitudes, accepted = split.fit_common_alpha(d, min_io_bins=min_io_bins)
    result['n_classes'] = ','.join(map(str, sorted(n_types)))
    return result, amplitudes, accepted


def do_fits(mpvs, split, min_io_bins):
    results = {}
    for tag, classes in [('n2', [2]), ('n3', [3]), ('shared', [2, 3])]:
        try:
            r, a, accepted = fit_model(mpvs, classes, split, min_io_bins)
            results[tag] = (r, a, accepted)
        except Exception as exc:
            results[tag] = (None, None, f'{type(exc).__name__}: {exc}')
    # This Δchi² is only interpretable when the exact same MPVs entered both
    # the 1-slope and 2-slope models (as checked below).
    n2, n3, shared = (results[tag][0] for tag in ['n2', 'n3', 'shared'])
    comparison = dict(delta_chi2=np.nan, approx_p_value=np.nan,
                      n_mpvs_nested=np.nan, nested_compatible=False)
    if all(r is not None for r in (n2, n3, shared)):
        same_n = int(n2['n_mpvs']) + int(n3['n_mpvs']) == int(shared['n_mpvs'])
        same_amplitudes = int(n2['n_io_groups']) + int(n3['n_io_groups']) == int(shared['n_io_groups'])
        comparison['nested_compatible'] = bool(same_n and same_amplitudes)
        if same_n and same_amplitudes:
            delta = shared['chi2'] - (n2['chi2'] + n3['chi2'])
            # Tiny negative numerical discrepancies are tolerated, not physics.
            if delta >= -1.e-6:
                comparison['delta_chi2'] = max(0., float(delta))
                comparison['approx_p_value'] = float(chi2_distribution.sf(max(0., delta), df=1))
                comparison['n_mpvs_nested'] = int(shared['n_mpvs'])
    return results, comparison


def list_runs(core, opts):
    roots = opts.frozen_root or [Path('frozen_brooke10_bs20'), Path('frozen_hit_timeseries_bs20')]
    runs = core.collect_runs(roots, opts.flow_root)
    if opts.only:
        runs = [r for r in runs if opts.only in Path(r['packet_file']).name]
    if opts.max_files is not None:
        runs = runs[:opts.max_files]
    return runs


def load_daq_channel_settings(packet_path):
    """Original acquisition ASIC settings, optional audit only (NOT a cut)."""
    import h5py
    if not Path(packet_path).is_file():
        raise FileNotFoundError(f'Raw packet file for DAQ audit not found: {packet_path}')
    with h5py.File(packet_path, 'r') as f:
        raw = f['daq_configs'][:].tobytes()
    regex = re.compile(r'(?:^|/)asic_configs/config_(\d+)-(\d+)-(\d+)\.json$')
    records = []
    with lzma.LZMAFile(io.BytesIO(raw)) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                match = regex.search(member.name)
                if not match:
                    continue
                io_group, io_channel, chip_id = map(int, match.groups())
                payload = json.load(tar.extractfile(member))
                for channel_id in range(64):
                    records.append((io_group, io_channel, chip_id, channel_id,
                                    payload['channel_mask'][channel_id],
                                    payload['csa_enable'][channel_id]))
    if not records:
        raise RuntimeError('No ASIC config JSON files found in compressed DAQ payload')
    return pd.DataFrame(records, columns=CHANNEL_KEYS + ['channel_mask', 'csa_enable'])


def daq_audit(selected_frames, filename, packet_path):
    config = load_daq_channel_settings(packet_path)
    if config.duplicated(CHANNEL_KEYS).any():
        raise RuntimeError('DAQ configuration contains duplicate electronics channels')
    rows = []
    for n, selected in selected_frames.items():
        d = selected[CHANNEL_KEYS].copy()
        d = d.merge(config, on=CHANNEL_KEYS, how='left', validate='many_to_one', indicator=True)
        d['configured'] = d['_merge'] == 'both'
        d['enabled'] = d['configured'] & (d['channel_mask'] == 0) & (d['csa_enable'] == 1)
        for io_group, sub in d.groupby('io_group'):
            rows.append(dict(packet_file=filename, io_group=int(io_group), n=int(n),
                             total_groups=len(sub), matched_daq=int(sub.configured.sum()),
                             active_groups=int(sub.enabled.sum()),
                             disabled_or_masked=int((sub.configured & ~sub.enabled).sum()),
                             unconfigured=int((~sub.configured).sum())))
    return pd.DataFrame(rows)


def plot_timeline(timeline, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = timeline.copy()
    if d.empty:
        return
    d['time_CDT'] = pd.to_datetime(d.time_CDT)
    fig, ax = plt.subplots(figsize=(12, 6))
    specs = [
        ('n2', 'n=2, separate IO amplitudes', 'o-'),
        ('n3', 'n=3, separate IO amplitudes', 's-'),
        ('shared', 'n=2+n=3, separate IO/n amplitudes', '^-'),
    ]
    for k, label, fmt in specs:
        y = pd.to_numeric(d.get(k+'_tau_ms'), errors='coerce')
        e = pd.to_numeric(d.get(k+'_tau_error_ms'), errors='coerce')
        good = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if good.any():
            ax.errorbar(d.loc[good, 'time_CDT'], y[good], yerr=e[good], fmt=fmt,
                        capsize=2.5, label=label)
    ax.set_title('DUNE 2x2 hit-based V2: EXACT n=2 vs EXACT n=3 | IO6 EXT\n'
                 'Independent (charge IO, multiplicity) amplitudes; same compact <=10 us selection')
    ax.set_xlabel('Acquisition start (CDT)')
    ax.set_ylabel('Effective attenuation timescale [ms]')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_per_file(mpvs, fits, output, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {2: 'C0', 3: 'C1'}
    for n in [2, 3]:
        sub = valid_fit_rows(mpvs[mpvs.n == n])
        if sub.empty:
            continue
        # Normalize within each IO using its own early-time MPV. The resulting
        # points are a visual aid, not a covariance-aware normalized fit.
        for charge_io, part in sub.groupby('io_group'):
            early = part.loc[part.t_center == 40, 'MPV_ADC']
            if early.empty or early.iloc[0] <= 0:
                continue
            a = float(early.iloc[0])
            ax.plot(part.t_center, part.MPV_ADC/a, marker='o' if n==2 else 's',
                    ls='-', alpha=.28, color=colors[n])
        # Plot the representative best slope derived from *weighted amplitudes*.
        result = fits['n2' if n == 2 else 'n3'][0]
        if result is not None:
            xx = np.linspace(40, 160, 200)
            ax.plot(xx, np.exp(-result['alpha_per_ms']*(xx-40)/1000),
                    color=colors[n], lw=2.5,
                    label=f'n={n} fit alpha={result["alpha_per_ms"]:.3f} /ms')
    shared = fits['shared'][0]
    if shared is not None:
        xx = np.linspace(40, 160, 200)
        ax.plot(xx, np.exp(-shared['alpha_per_ms']*(xx-40)/1000), '--',
                color='black', lw=2., label=f'shared alpha={shared["alpha_per_ms"]:.3f} /ms')
    ax.set_title(f'V2 normalized MPVs, {filename}\nLight lines: each IO (illustrative normalization at 40 us)')
    ax.set_xlabel('Last packet time relative to IO6 EXT [us]')
    ax.set_ylabel('MPV(t) / MPV(40us), by charge IO and n')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=175)
    plt.close(fig)


def self_test(split):
    groups = pd.DataFrame(dict(n=[1,2,2,3,3,4], span_us=[0,2.8,15,2.7,10.1,2.],
                               t_last=[40,40,80,60,80,60]))
    assert list(select_multiplicity(groups, 2).index) == [1]
    assert list(select_multiplicity(groups, 3).index) == [3]
    rows = []
    for n, alpha in [(2, 0.85), (3, 0.85)]:
        for charge_io, amp in [(1, 22.+5*n), (2, 26.+5*n)]:
            for t in [40., 80., 120., 160.]:
                q = amp * np.exp(-alpha*(t-40.)/1000.)
                rows.append(dict(io_group=charge_io, n=n, t_center=t, MPV_ADC=q,
                                 MPV_error=.3, chi2_ndf=1.))
    d = pd.DataFrame(rows)
    fits, nested = do_fits(d, split, min_io_bins=4)
    for tag in ['n2','n3','shared']:
        assert fits[tag][0] is not None, fits[tag][2]
        assert abs(fits[tag][0]['alpha_per_ms']-.85) < 1.e-5
    assert nested['nested_compatible'] and nested['delta_chi2'] < 1.e-7
    d.loc[d.n == 3, 'MPV_ADC'] *= np.exp(-.65*(d.loc[d.n == 3, 't_center']-40.)/1000.) / np.exp(-.85*(d.loc[d.n == 3, 't_center']-40.)/1000.)
    fits, nested = do_fits(d, split, min_io_bins=4)
    assert abs(fits['n3'][0]['alpha_per_ms']-.65) < 1.e-5
    assert nested['delta_chi2'] > 0.0
    print('SELF-TEST PASSED: exact-n selection, shared vs separate slopes, nested chi2 test')


def main():
    opts = args_parser()
    if opts.bootstrap < 2:
        raise SystemExit('--bootstrap must be >= 2')
    if opts.min_io_bins not in [3,4]:
        raise SystemExit('--min-io-bins must be 3 or 4')
    import run_io6_n2_split_io as split
    if opts.self_test:
        self_test(split)
        return
    import run_paired_lifetime_timeline_all as core
    runs = list_runs(core, opts)
    if not runs:
        raise SystemExit('No runs discovered; check frozen roots and --only')
    print(f'Found {len(runs)} runs; EXT reference fixed to IO6')
    for run in runs:
        filename = Path(run['packet_file']).name
        cache = opts.cached_root / (Path(filename).stem + '.IO6.npz')
        print(f'  {filename}: cache={cache.is_file()}, FLOW={bool(run["flow_path"] and run["flow_path"].is_file())}')
    if opts.list_only:
        return
    opts.outdir.mkdir(parents=True, exist_ok=True)
    timeline = []
    mpv_parts = []
    audit_parts = []
    reuse_old = (not opts.refit_n2) and opts.reuse_n2_csv.is_file() and opts.bootstrap_unit == 'group'
    print('Reusing existing n=2 split-IO MPVs:', reuse_old)
    if reuse_old and opts.bootstrap != 30:
        print('WARNING: reused n=2 CSV was typically generated with 30 resamples; your n=3 bootstrap count differs.')
    for index, run in enumerate(runs, 1):
        filename = Path(run['packet_file']).name
        print(f'\n[{index}/{len(runs)}] {filename}', flush=True)
        out = dict(packet_file=filename, time_CDT=run['timestamp'], ext_io_group=6,
                   bootstrap_unit=opts.bootstrap_unit, n_bootstrap_new=opts.bootstrap,
                   status='FAILED', failure='')
        cache = opts.cached_root / (Path(filename).stem + '.IO6.npz')
        try:
            if not cache.is_file():
                raise FileNotFoundError(f'Missing IO6 cache: {cache} (do not regenerate silently)')
            if run['flow_path'] is None or not run['flow_path'].is_file():
                raise FileNotFoundError(f'Missing FLOW v9 file: {run["flow_path"]}')
            groups, n_all, n_good, coverage = core.load_and_group(cache, run['flow_path'])
            out.update(n_packet_cache=n_all, n_packet_ped_matched=n_good, pedestal_coverage=coverage)
            selected = {n: select_multiplicity(groups, n) for n in [2,3]}
            out['n2_groups'] = len(selected[2]); out['n3_groups'] = len(selected[3])
            print(f'  Selected groups: n2={len(selected[2])}, n3={len(selected[3])}; pedestal={coverage:.2%}', flush=True)
            n2 = read_old_n2(opts.reuse_n2_csv, filename) if reuse_old else None
            if n2 is None:
                n2 = fit_spectra(selected[2], 2, filename, core, opts.bootstrap,
                                 opts.min_bin_entries, opts.bootstrap_unit)
            else:
                print('  Using existing n=2 MPV/error table; no n=2 spectral refits', flush=True)
            n3 = fit_spectra(selected[3], 3, filename, core, opts.bootstrap,
                             opts.min_bin_entries, opts.bootstrap_unit)
            mpvs = pd.concat([n2, n3], ignore_index=True)
            mpvs['packet_file'] = filename
            mpvs['time_CDT'] = run['timestamp']
            mpv_parts.append(mpvs)
            print('  Fitted spectral bins n2/n3:',
                  [int(valid_fit_rows(mpvs[mpvs.n == n]).shape[0]) for n in [2,3]], flush=True)
            fits, comparison = do_fits(mpvs, split, opts.min_io_bins)
            for tag in ['n2','n3','shared']:
                r, amps, acc_or_err = fits[tag]
                if r is None:
                    out[f'{tag}_failure'] = acc_or_err
                    print(f'  {tag}: FIT FAILED: {acc_or_err}', flush=True)
                    continue
                for field in ['tau_ms','tau_error_ms','alpha_per_ms','alpha_error_per_ms',
                              'chi2','ndf','chi2_ndf','n_io_groups','n_mpvs','n_poor_spectral_fits']:
                    out[tag+'_'+field] = r[field]
                out[f'{tag}_used_groups'] = r['io_groups']
                print(f'  {tag:6s}: alpha={r["alpha_per_ms"]:.4f} +/- {r["alpha_error_per_ms"]:.4f} /ms, '
                      f'tau={r["tau_ms"]:.3f} +/- {r["tau_error_ms"]:.3f} ms, '
                      f'chi2/ndf={r["chi2_ndf"]:.2f}, MPVs={r["n_mpvs"]}', flush=True)
            out.update(comparison)
            if np.isfinite(comparison['delta_chi2']):
                print(f'  Common vs separate alpha: delta_chi2={comparison["delta_chi2"]:.3f}; '
                      f'approx p={comparison["approx_p_value"]:.4g}', flush=True)
            if opts.daq_audit:
                try:
                    audit = daq_audit(selected, filename, run['packet_file'])
                    audit_parts.append(audit)
                    print('  DAQ audit groups:', len(audit), flush=True)
                except Exception as exc:
                    print(f'  DAQ audit unavailable: {exc}', flush=True)
                    out['daq_audit_error'] = f'{type(exc).__name__}: {exc}'
            if opts.per_file_plots:
                plot_per_file(mpvs, fits, opts.outdir / (Path(filename).stem+'_normalized_mpvs.png'), filename)
            out['status'] = 'OK' if fits['shared'][0] is not None else 'PARTIAL'
        except Exception as exc:
            out['failure'] = f'{type(exc).__name__}: {exc}'
            print(f'  FAILED: {out["failure"]}', flush=True)
        timeline.append(out)
        pd.DataFrame(timeline).to_csv(opts.outdir/'v2_lifetime_timeline.csv', index=False)
        if mpv_parts:
            pd.concat(mpv_parts, ignore_index=True).to_csv(opts.outdir/'v2_per_io_multiplicity_mpvs.csv', index=False)
        if audit_parts:
            pd.concat(audit_parts, ignore_index=True).to_csv(opts.outdir/'v2_daq_channel_audit.csv', index=False)
    time_df = pd.DataFrame(timeline).sort_values('time_CDT')
    if any(col in time_df for col in ['n2_tau_ms', 'n3_tau_ms', 'shared_tau_ms']):
        plot_timeline(time_df, opts.outdir/'v2_n2_n3_shared_lifetimes.png')
    print(f'\nDone. OK={sum(time_df.status == "OK")}/{len(time_df)}; outputs in {opts.outdir}', flush=True)
    print('CAUTION: Shared slope is an exploratory compatibility test, not a validated electron-lifetime result.')


if __name__ == '__main__':
    main()
