"""Fetch ZTF g/r lightcurves for ALeRCE-classified Cepheids (CEP) and RR Lyrae (RRL),
compute multi-band Lomb-Scargle periods, and cache the result for the notebook.

Run once:  python3 fetch_alerce.py
Output:    data/alerce_variables.json

The notebook (`timeseries_examples.ipynb`) reads from the cache; nothing in this
file is needed at notebook runtime.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy import signal

API = 'https://api.alerce.online/ztf/v1'
DATA = Path(__file__).parent / 'data'
OUTFILE = DATA / 'alerce_variables.json'

# class_name (LC classifier short code), min probability, target #objects,
# period search range (days)
CLASSES = {
    'RRL': dict(min_prob=0.95, n_target=60, period_range=(0.20, 1.20)),
    'CEP': dict(min_prob=0.80, n_target=60, period_range=(1.0,  60.0)),
}


def query_oids(class_name: str, min_prob: float, n_candidates: int):
    """Page through the LC-classifier object index for one class."""
    items = []
    page = 1
    while len(items) < n_candidates:
        r = requests.get(f'{API}/objects/', params={
            'classifier': 'lc_classifier',
            'class': class_name,
            'probability': min_prob,
            'page_size': min(200, n_candidates - len(items)),
            'page': page,
            'order_by': 'probability',
            'order_mode': 'DESC',
        }, timeout=30)
        r.raise_for_status()
        chunk = r.json().get('items') or []
        if not chunk:
            break
        items.extend(chunk)
        page += 1
    return items[:n_candidates]


def fetch_lightcurve(oid: str):
    r = requests.get(f'{API}/objects/{oid}/lightcurve', timeout=60)
    r.raise_for_status()
    return r.json()['detections']


def split_bands(detections):
    """Return {1: (t,y,s), 2: (t,y,s)} for g (fid=1) and r (fid=2).

    Uses the template-corrected magpsf_corr / sigmapsf_corr — what you
    want for *persistent* variables (RR Lyrae, Cepheids). The raw magpsf
    is from the difference image and is always positive, which mangles
    phase folds of steady-state variables. Detections with corrected=False
    are dropped (no template reference available).
    """
    rows = {1: [], 2: []}
    for d in detections:
        fid = d.get('fid')
        if fid not in (1, 2):
            continue
        if not d.get('corrected', False):
            continue
        m = d.get('magpsf_corr')
        s = d.get('sigmapsf_corr')
        if m is None or s is None or s <= 0:
            continue
        rows[fid].append((d['mjd'], m, s))
    out = {}
    for fid, lst in rows.items():
        if not lst:
            continue
        arr = np.array(lst)
        out[fid] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def multiband_period(bands, period_range):
    """Combined LS: sum per-band normalized powers. Returns (best_f, freq, power)."""
    f_min = 1.0 / period_range[1]
    f_max = 1.0 / period_range[0]
    t_all = np.concatenate([bands[b][0] for b in bands])
    T = t_all.max() - t_all.min()
    df = (1.0 / T) / 5.0
    freq = np.arange(f_min, f_max, df)
    omega = 2 * np.pi * freq

    total = np.zeros_like(freq)
    for b, (t, y, s) in bands.items():
        w = 1.0 / s ** 2
        p = signal.lombscargle(t, y, omega, weights=w,
                               floating_mean=True, normalize='normalize')
        # convert to z-form so per-band powers are exp-distributed under H0
        z = p * (len(t) - 1) / 2.0
        total += z
    best_f = float(freq[int(np.argmax(total))])
    return best_f, freq, total


def process_one(item, period_range, min_band_obs=15):
    """Use whatever bands have ≥ min_band_obs detections.

    Many ZTF Cepheids in ALeRCE only have g-band data; we still
    compute a (single-band) period for those instead of skipping.
    Returns dict with `bands_used` listing the fids actually used.
    """
    oid = item['oid']
    try:
        det = fetch_lightcurve(oid)
    except Exception as exc:
        return None, f'fetch failed: {exc}'
    raw = split_bands(det)
    bands = {fid: trio for fid, trio in raw.items()
             if len(trio[0]) >= min_band_obs}
    if not bands:
        present = {fid: len(trio[0]) for fid, trio in raw.items()}
        return None, f'no band has ≥{min_band_obs} obs (counts: {present})'

    best_f, _, _ = multiband_period(bands, period_range)

    rec = {
        'oid': oid,
        'class': item['class'],
        'prob': item['probability'],
        'ra':   item['meanra'],
        'dec':  item['meandec'],
        'ndet_g': len(raw[1][0]) if 1 in raw else 0,
        'ndet_r': len(raw[2][0]) if 2 in raw else 0,
        'bands_used': sorted(bands.keys()),  # which fids contributed to the period
        'period': 1.0 / best_f,
    }
    # Save lightcurves for every band that had any data (notebook can re-plot)
    for fid, letter in [(1, 'g'), (2, 'r')]:
        if fid in raw:
            t, y, s = raw[fid]
            rec[f't_{letter}'] = [round(float(x), 6) for x in t]
            rec[f'y_{letter}'] = [round(float(x), 4) for x in y]
            rec[f's_{letter}'] = [round(float(x), 4) for x in s]
        else:
            rec[f't_{letter}'] = rec[f'y_{letter}'] = rec[f's_{letter}'] = []
    return rec, None


def main():
    DATA.mkdir(exist_ok=True)
    out = []
    t0_all = time.time()
    for cls, cfg in CLASSES.items():
        print(f'\n=== {cls}  (target {cfg["n_target"]}, prob ≥ {cfg["min_prob"]}) ===')
        cands = query_oids(cls, cfg['min_prob'], cfg['n_target'] * 3)
        print(f'  {len(cands)} candidates returned')
        n_done = 0
        for it in cands:
            if n_done >= cfg['n_target']:
                break
            t0 = time.time()
            res, err = process_one(it, cfg['period_range'])
            if res is None:
                print(f'  {it["oid"]:14s}  SKIP  {err}')
                continue
            band_tag = '+'.join({1: 'g', 2: 'r'}[b] for b in res['bands_used'])
            print(f'  {res["oid"]:14s}  P = {res["period"]:7.4f} d   '
                  f'(g={res["ndet_g"]}, r={res["ndet_r"]}, used={band_tag}, '
                  f'{time.time()-t0:.1f}s)')
            out.append(res)
            n_done += 1
    with open(OUTFILE, 'w') as f:
        json.dump(out, f)
    size_kb = OUTFILE.stat().st_size / 1024
    print(f'\nWrote {len(out)} objects to {OUTFILE}  ({size_kb:.0f} KB, '
          f'{time.time()-t0_all:.0f}s total)')


if __name__ == '__main__':
    main()
