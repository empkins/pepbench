# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.3.0 - <Month DD, 2026>
### Added
- `pepbench.algorithms.icg`: Added `BPointExtractionAbelStuehler2026`, a machine-learning-based B-point extraction
  algorithm (from `biopsykit`), together with `get_b_point_abelstuehler2026_model` to download and load its
  pretrained model (Abel et al., 2026, Frontiers in Digital Health).
- `pepbench.algorithms.icg`: Also exposed `BPointExtractionMiljkovic2022` and `BPointExtractionPale2021` in the API
  reference and user guide.
- `pepbench.io`: Added `load_best_performing_algos_b_point`, `load_best_performing_algos_q_wave`, and
  `compute_abs_error`.

### Changed
- Requires `biopsykit>=0.14.0`.
- The experiment notebooks, scripts, and results previously in `experiments/` moved to the separate
  [pepbench-experiments](https://github.com/empkins/pepbench-experiments) repository.
- Documentation: added descriptions of all B-point algorithms to the user guide, including a recommendation of
  `BPointExtractionAbelStuehler2026`.

### Removed
- Removed unused runtime dependencies (`notebook`, `ipykernel`, `lv`).
  Install Jupyter separately if you need it to run the example notebooks.

## v0.2.0 - July 30, 2025
Release after incorporating changes due to the revision process of the PEPbench paper.
