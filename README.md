# SRAM DSO-MOGA

**SRAM Design Space Optimization using Multi-Objective Genetic Algorithm**

NSGA-II/NSGA-III based optimizer for SRAM peripheral circuit PPA (Power, Performance, Area) tuning.

## Project Structure

```
sram_dso_moga/
├── src/
│   ├── config.py      # Configuration management & validation
│   ├── evaluator.py   # PPA evaluation models
│   ├── nsga.py        # NSGA-II/III implementation
│   └── dashboard.py   # Interactive Plotly Dash dashboard
├── config/
│   └── sram_config_v2.yaml  # Enhanced configuration schema
├── tests/
│   ├── test_nsga.py        # NSGA algorithm tests
│   ├── test_config_eval.py # Config & evaluator tests
│   └── validate.py         # Self-validation script
├── run_moga.py     # Main entry point
├── CLAUDE.md       # Claude Code guidance
└── README.md       # This file
```

## Quick Start

```bash
# Run optimization with default config
python run_moga.py --config config/sram_config_v2.yaml

# Custom parameters
python run_moga.py --config config/sram_config_v2.yaml --pop-size 100 --n-gen 100

# Run with dashboard
python run_moga.py --config config/sram_config_v2.yaml --dashboard

# Verbose logging
python run_moga.py --config config/sram_config_v2.yaml --verbose --log
```

## Configuration

Edit `config/sram_config_v2.yaml` to define:
- **algorithm**: pop_size, n_gen, seed, crossover/mutation parameters
- **objectives**: Area, Power, Delay with weights and directions
- **groups**: Device parameters (vt_options, gl_options, nfin_options)
- **ppa_model**: Coefficients for analytical PPA models

Example groups definition:
```yaml
groups:
  - bundle_flag: readpath
    devices:
      - name: precharge_pmos
        vt_options: ["vt0", "vt1", "vt2"]
        gl_options: [0.020, 0.025, 0.030]
        nfin_options: [2, 3, 4, 5, 6]
```

## Output

Results saved to `./results/`:
- `results.json` - Full optimization results with history
- `pareto_solutions.csv` - Pareto optimal solutions
- `dashboard.html` - Static visualization (or use `--dashboard` for interactive)

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Self-validation
python tests/validate.py
```

## Algorithm

- **NSGA-II**: Fast non-dominated sorting with crowding distance
- **NSGA-III**: Reference point-based niching for many-objective optimization
- **Selection**: Tournament selection based on rank then crowding distance
- **Crossover**: Simulated Binary Crossover (SBX) adapted for integer genes
- **Mutation**: Polynomial mutation for integer-coded parameters

## PPA Model

Simplified analytical models:
- **Area** = Σ(nfin) × 0.15 + μ(gl) × 120 + 0.8
- **Power** = Σ(nfin) × 12.5 × VT_penalty + 5.0
- **Delay** = 180 / (Σ(nfin) × 0.9 + ε) + μ(gl) × 800

Where VT_penalty depends on threshold voltage variant (vt0/vt1/vt2).