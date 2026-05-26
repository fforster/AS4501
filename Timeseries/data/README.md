# Data files used by `timeseries_examples.ipynb`

## GX 5-1 — Ginga X-ray timeseries

| File | Format | Samples | Sampling | Use case |
|------|--------|---------|----------|----------|
| `GX.dat` | 16 integers per row, 4096 rows | 65536 (=2¹⁶) | 1/128 s | Part I demo dataset — autocorrelation, AR/MA/ARMA, AIC/BIC, periodogram |

Photon counts from the galactic X-ray binary **GX 5-1** observed by the Ginga satellite, binned at 1/128 s (~512 s total). This is the same series the PDF uses on slides 20, 25, 27, 29, 40–42, 44. Mean ≈ 69 cts/bin, std ≈ 10. The variance exceeds the pure-Poisson expectation $\\sqrt{\\bar N} \\approx 8.3$, and a broad QPO appears near $f \\approx 0.19$ in units of the sampling frequency.

## Supernova lightcurve for the Gaussian-process section

| File | Source | Bands | N obs | Span | Use case |
|------|--------|-------|-------|------|----------|
| `sn_lightcurve.json` | ALeRCE / ZTF, `ZTF20abvtozi` (SN Ia) | g, r | 83 + 91 | 106 d | GP interpolation per band and joint multi-band GP |

Difference-image PSF magnitudes (`magpsf` / `sigmapsf`) — appropriate for transients, since the template flux is zero by construction. Fetched once with the inline downloader at the top of the GP notebook section.

## OGLE-IV RR Lyrae lightcurves

Two real RR Lyrae lightcurves from the **OGLE-IV Collection of Variable Stars** (Bulge fields, I-band photometry). Used by `timeseries_examples.ipynb` for the period-finding sections.

| File | Type | Period (d) | N obs | Mean I | Use case in notebook |
|------|------|------------|-------|--------|----------------------|
| `OGLE-BLG-RRLYR-01000.dat` | RRc  | 0.26013412 | 2627  | 17.773 | Lomb–Scargle basics (near-sinusoidal lightcurve) |
| `OGLE-BLG-RRLYR-05000.dat` | RRab | 0.49725278 | 13874 | 17.210 | Multi-term LS / truncated Fourier (sawtooth shape) |

## File format

Three whitespace-separated columns, no header:
```
HJD-2450000   I_mag   I_mag_error
```

Example:
```
5265.81251 17.599 0.012
5266.79609 17.767 0.016
```

## Source

OGLE-IV Collection of Variable Stars, Bulge RR Lyrae sample.
- Project: https://ogle.astrouw.edu.pl/
- Catalog page: https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/rrlyr/
- Direct photometry URL pattern: `https://www.astrouw.edu.pl/ogle/ogle4/OCVS/blg/rrlyr/phot/I/OGLE-BLG-RRLYR-<ID>.dat`

Reference: Soszyński et al., *Acta Astronomica*, OGLE-IV RR Lyrae papers.

Please cite the OGLE team if you use these data outside of class.
