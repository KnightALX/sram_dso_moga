# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SRAM DSO-MOGA** — SRAM Design Space Optimization using Multi-Objective Genetic Algorithm (NSGA-II/NSGA-III) for peripheral circuit PPA (Power, Performance, Area) tuning.

## Architecture

```
sram_dso_moga/
├── src/
│   ├── config.py      # Config class with validation, property accessors
│   ├── evaluator.py   # PPAEvaluator with analytical models
│   ├── nsga.py        # NSGA class, operators (SBX, polynomial mutation, tournament)
│   └── dashboard.py   # Dash-based interactive dashboard
├── config/
│   └── sram_config_v2.yaml  # Enhanced YAML schema
├── tests/
│   ├── test_nsga.py        # Unit tests for NSGA operators
│   ├── test_config_eval.py # Config validation & evaluator tests
│   └── validate.py         # Integration validation
├── run_moga.py     # CLI entry point
├── sram_dso_moga.py # Original baseline (legacy)
└── README.md        # Documentation
```

## Core Flow

```
YAML Config → Config (validate) → get_tunables()
                                        ↓
                    NSGA (evolve) → evaluate (PPAEvaluator)
                                        ↓
                        fast_non_dominated_sort() + crowding_distance
                                        ↓
                    Pareto Front → save_results() / Dashboard
```

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `Config` | config.py | Load/validate YAML, property accessors |
| `PPAEvaluator` | evaluator.py | Analytical PPA models (area/power/delay) |
| `NSGA` | nsga.py | Main optimizer with evolve() method |
| `Individual` | nsga.py | Gene array + objectives + rank + crowding |
| `AlgorithmConfig` | nsga.py | Hyperparameters dataclass |
| `Dashboard` | dashboard.py | Dash app for interactive visualization |

## Commands

```bash
# Run optimization
python run_moga.py --config config/sram_config_v2.yaml

# Run with custom params
python run_moga.py --config config/sram_config_v2.yaml --pop-size 100 --n-gen 100

# Interactive dashboard
python run_moga.py --config config/sram_config_v2.yaml --dashboard --dashboard-port 8050

# Run tests
python -m pytest tests/ -v

# Self-validation
python tests/validate.py
```

## Configuration Schema

Key fields in `config/sram_config_v2.yaml`:
- `algorithm.pop_size` / `algorithm.n_gen` / `algorithm.seed`
- `algorithm.crossover.prob` / `algorithm.crossover.eta`
- `algorithm.mutation.prob` / `algorithm.mutation.eta`
- `objectives[]` — list of {name, label, direction, weight, enabled}
- `groups[]` — device groups with bundle_flag + devices list
- `active_bundles[]` — which groups to optimize
- `ppa_model` — coefficients for area/power/delay formulas

## Gene Encoding

Genes are integer indices into option arrays:
- `_vt` → threshold voltage variants (vt0, vt1, vt2)
- `_gl` → gate length in um (e.g., 0.018–0.030)
- `_nfin` → number of fins (e.g., 1–8)

Decoding: `decode_genes(genes, tunables)` maps indices to actual values.

## Algorithm Details

**NSGA-II** (default):
1. Initialize random population
2. Evaluate objectives
3. Fast non-dominated sorting O(MN²)
4. Calculate crowding distance per front
5. Tournament selection based on (rank, -crowding_distance)
6. SBX crossover (90%) + polynomial mutation (15%)
7. Combine parent+offspring, select next generation
8. Repeat until n_gen reached

**NSGA-III**: Same flow but uses reference point association instead of crowding distance for diversity preservation.

## Known Issues (from baseline)

1. **Selection pressure was zero** in original — `random.sample()` instead of tournament. Fixed: now uses tournament selection.
2. **Crowding distance could produce NaN** when fmax==fmin. Fixed: skip division when range is zero.
3. **No YAML validation** — crashes with unhelpful errors. Fixed: `Config._validate()` checks all required fields.
4. **No checkpointing** — long runs lost on crash. Consider adding checkpoint support if needed.
5. **No parallelism** — sequential evaluation. Could add `concurrent.futures` for parallel PPA eval.