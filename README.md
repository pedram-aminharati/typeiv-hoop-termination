# Friction-bounded design of the cylinder–dome transition of Type IV pressure vessels

Code and result data for

> P. Aminharati, M. Shirinbayan, K. Benfriha, J. Fitoussi, *Friction-bounded design of the
> cylinder–dome transition of Type IV composite pressure vessels: hoop termination, stacking
> order and hybrid reinforcement layers*, International Journal of Hydrogen Energy, submitted, 2026.

The study maps the hoop termination (end position, drop-off length, stacking position) and
hybrid reinforcement layers of eleven 35/70 MPa carbon/epoxy hydrogen vessels and two open
benchmark vessels with a CalculiX composite-shell model (542 analyses) and a layer-by-layer
axisymmetric model of the drop-off (106 analyses), under a friction (slippage) bound on the
windable hoop end.

## Requirements
* [CalculiX](http://www.calculix.de/) `ccx` 2.21 on the `PATH`
* Python ≥ 3.10 with the packages in `requirements.txt`

Run all scripts from the repository root.

## Contents
| File | Purpose |
|---|---|
| `core.py`, `vessel_gen.py`, `vessel_tr.py` | geometry, winding laws, CalculiX shell input generator (transition design variables, hybrid layers), Lekhnitskii solution |
| `sizing.py`, `run_sizing_*.py`, `sizing.json` | dome-consistent wall sizing (helical factor `k_hel`) |
| `p1_study.py` | vessel set, design grid, friction limit `e_crit(mu)`, shell study driver |
| `axi.py`, `interlam.py`, `il_study.py`, `il_fields.py` | layer-wise axisymmetric model of the drop-off and Brewer–Lagace delamination index |
| `p1_analysis.py`, `p1_rule.py`, `p1_metrics.py`, `p1_verify.py` | post-processing, design rule (hoop-strain mismatch chi), verification |
| `bench_agne.py`, `tr_eval.py` | open benchmark vessels (Agne et al., Compos. Struct. 370 (2025) 119424) |
| `fig_paper1.py`, `fig_p1.py`, `fig_il_contour.py`, `schematics/` | figures and rendered schematics |
| `make_tables_p1.py`, `make_refs_*.py` | manuscript tables and bibliography |
| `rel/thermal_p1.py` | cure cool-down check of the transition efficiency |

## Data
* `results_p1full/` – one JSON per shell analysis (fibre-stress ratio profiles, critical location, p_f, mass, verification checks)
* `results_il_metrics/` – delamination index and fibre efficiency per layer-wise analysis
* `rel_results/thermal_p1.json` – efficiency with and without cure residual stress
* The full layer-wise stress fields (about 150 MB) are available from the authors on request.

## Licence
MIT (see `LICENSE`). Please cite the article above when using the code or data.
