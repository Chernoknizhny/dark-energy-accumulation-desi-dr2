# Dark energy accumulation model vs DESI DR2

Code and manuscript for the paper:
**"Accumulation Dark Energy Model: Testing the DESI Anomaly, Perturbation Stability, and Limits of Applicability"**


**"Модель накопления тёмной энергии: проверка на аномалию DESI, устойчивость возмущений и границы применимости"**

Author: Evgeny V. Chernoknizhny (independent researcher)

## Overview

Phenomenological accumulation dark energy model:

    d ln ρ_x / dN = κ_eff (1 − x) − λ(N),   λ(N) = λ₀ e^(−αN),   κ_eff = κ[1 + β a^(−3)]

Equation of state:

    w = −1 + [λ(N) − κ_eff (1 − x)] / 3

Structural ceiling: w ≤ −1 + λ₀/3 ≈ −0.97 (thawing zone unreachable at any finite x).

## Key results

| Dataset | ΛCDM χ² | CPL Δχ² | cand Δχ² |
|---|---|---|---|
| DESI DR2 BAO only | 10.283 | −4.67 | 0.000 |
| DESI + CMB (DESI prior, TH_CAL=1.0026) | 12.680 | −4.09 | +0.001 |
| DESI + CMB (CHW prior) | 12.424 | −3.50 | 0.000 |

- The pipeline reproduces the official DESI DR2 ΛCDM fit (χ² = 10.283, Ω_m = 0.2972).
- The accumulation model degenerates to ΛCDM (κ = λ₀ = 0) in all fits: the matter channel and decay are not required by the data.
- Perturbative bound κ ≤ 0.101 (no-ghost condition w > −1); the χ²(κ) profile is monotonic, Δχ² = 0.53 < 1, so the bound is perturbative, not observational.

## Repository contents

- `pipeline_desi_dr2.py` — main pipeline: BAO fits, CMB prior, profile, figures.
- `desi_bao_all_gaussian_bao_ALL_GCcomb_mean.txt`, `..._cov.txt` — official DESI DR2 BAO data (13 quantities, 7 redshifts).
- `Модель накопления тёмной энергии.tex` — LaTeX source of the manuscript.
- `Accumulation Dark Energy Model.pdf` - compiled English manuscript.
- `dark_energy_accumulation_desi_dr2.tex` — English LaTeX source.
- `profile_kappa.png`, `wz_curves.png` — figures.

## How to run

    pip install numpy scipy matplotlib
    python pipeline_desi_dr2.py
    

Requires the two DESI BAO files in the same directory.

## CMB prior

Compressed DESI DR2 prior (θ*, ω_b, ω_bc) from Appendix A of DESI DR2 Results II (CamSpec):

    μ = (0.0104110, 0.02223, 0.14208)
    C = 1e-9 × [[0.006621, 0.12444, −1.1929],
                [0.12444, 21.344, −94.001],
                [−1.1929, −94.001, 1488.4]]

Calibration: TH_CAL = 0.0104110 / θ*_th(h=0.6766, ω_b=0.02223, ω_bc=0.14208) = 1.0026.

## References

1. DESI Collaboration, DESI DR2 Results II, arXiv:2503.14738 (2025).
2. DESI Collaboration, DESI DR2 Results I, arXiv:2503.14739 (2025).
3. Planck Collaboration, A&A 641, A6 (2020), arXiv:1807.06209.
4. Chen, Huang, Wang, JCAP 02, 028 (2019), arXiv:1808.05725.
