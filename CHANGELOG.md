# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-17

### Added

- `requirements.txt` with numpy, scipy, and matplotlib dependencies
- Test suite (`test_hh.py`) with 16 tests covering gating kinetics,
  original model simulation, linearized model integration, and transfer
  function computation
- `if __name__ == '__main__'` guards so modules can be imported without
  triggering simulation

### Fixed

- Replace deprecated `scipy` namespace functions (`sp.exp`, `sp.arange`)
  with `numpy` equivalents for compatibility with modern scipy
- Remove deprecated `seaborn-deep` matplotlib style
- Fix numerical instability in `alpha_m` (at V = -40 mV) and `alpha_n`
  (at V = -55 mV) using `np.expm1` with L'Hopital limit fallback
- Fix typo "membrance" to "membrane" in `linearized_hh.py`
- Replace unused `scipy.constants.pi` with `numpy.pi` in
  `transfer_function_hh.py`

### Changed

- Matplotlib backend selection now auto-detects headless environments
  via the `DISPLAY` environment variable instead of hardcoding `Agg`
- Expanded `.gitignore` to cover Python bytecode, pytest cache, virtual
  environments, and build artifacts

### Removed

- Unused imports: `scipy.stats`, `scipy.linalg`, `scipy.constants`

## [0.1.0] - 2019-12-04

Initial release of the Hodgkin-Huxley model implementations.

### Added

- Original nonlinear Hodgkin-Huxley model (`original_hh.py`)
- Linearized Hodgkin-Huxley model (`linearized_hh.py`)
- Transfer function analysis (`transfer_function_hh.py`)
