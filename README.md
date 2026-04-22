# SRAM DSO-MOGA

**SRAM Design Space Optimization using Multi-Objective Genetic Algorithm**

A two-phase multi-objective optimization framework for SRAM peripheral circuit PPA (Power, Performance, Area) tuning, built on NSGA-II/NSGA-III.

---

## Overview

SRAM DSO-MOGA explores the design space of SRAM peripheral circuits (sense amplifiers, precharge, write drivers) by optimizing device parameters — threshold voltage (`_vt`), gate length (`_gl`), and number of fins (`_nfin`) — across multiple conflicting objectives.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Stage 1: Coarse Search                  │
│  generate_coarse() ──► NSGA ──► Pareto Front (N solutions)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (Pareto seeds)
┌─────────────────────────────────────────────────────────────────┐
│                        Stage 2: Fine Search                    │
│  generate_fine(Pareto) ──► NSGA ──► Refined Pareto Front        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Run both stages (default behavior)
python run_moga.py --config config/sram_config_v2.yaml

# Run Stage 1 only
python run_moga.py --config config/sram_config_v2.yaml --stage 1

# Run Stage 2 only (requires Stage 1 checkpoint)
python run_moga.py --config config/sram_config_v2.yaml --stage 2

# Override algorithm parameters
python run_moga.py --config config/sram_config_v2.yaml --pop-size 100 --n-gen 80 --seed 42

# Save log to file
python run_moga.py --config config/sram_config_v2.yaml --log --verbose

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
sram_dso_moga/
├── src/
│   ├── config.py            # YAML config loading & validation
│   ├── evaluator.py         # Analytical PPA evaluation models
│   ├── nsga.py              # NSGA-II/III core algorithms
│   ├── spf_handler.py       # SPF netlist interface (user-implemented)
│   ├── fitness_collector.py  # Fitness collection & combination generation
│   ├── dashboard.py          # Interactive Plotly Dash dashboard
│   └── exporter.py          # JSON/CSV/summary export
├── config/
│   └── sram_config_v2.yaml   # Configuration schema
├── tests/
│   ├── test_nsga.py         # NSGA operator unit tests
│   ├── test_config_eval.py   # Config & evaluator tests
│   └── validate.py          # Integration validation
├── run_moga.py              # Main entry point (two-phase workflow)
├── sram_dso_moga.py         # Legacy baseline (single-stage)
├── CLAUDE.md                # Claude Code project guidance
└── README.md                # This file
```

---

## Two-Phase Workflow

### Stage 1 — Coarse Search

Explores the full design space using Latin Hypercube Sampling (LHS) when the combinatorial space is large. A population-based NSGA-II/III optimization identifies the initial Pareto front.

```
Step 1: Combination Generation
    └─ generate_coarse(n_samples=200) → coarse combinations

Step 2: Batch Simulation (user-implemented interface)
    └─ submit_batch_simulation() → submission_info

Step 3: PPA Result Collection
    └─ poll_and_collect_results() → {combo_idx: PPA}

Step 4: NSGA Optimization
    └─ Pareto front → checkpoint: stage1_pareto_solutions
```

### Stage 2 — Fine Search

Refines the search around Stage 1 Pareto solutions. Generates perturbations around each Pareto-optimal point using parameter variation, then runs a fresh NSGA optimization.

```
Input: Stage 1 Pareto solutions (checkpoint loaded)

Step 1: Combination Generation
    └─ generate_fine(pareto_solutions, n_samples_per=10, variation=0.2)
        → fine combinations around Pareto seeds

Step 2-4: Same as Stage 1 → Final Pareto front
```

**Note**: Stage 2's combination generation does not require Stage 1 simulation results — it uses the same analytical PPA model. Users implementing external simulation may extend `CombinedPPAEvaluator` to leverage pre-computed PPA values from Stage 1.

---

## Configuration

Edit `config/sram_config_v2.yaml` to define your optimization:

### Algorithm Settings

```yaml
algorithm:
  name: NSGA-II        # NSGA-II or NSGA-III
  pop_size: 80
  n_gen: 60
  seed: 42
  crossover:
    prob: 0.9
    eta: 15
  mutation:
    prob: 0.15
    eta: 20
```

### Sampling Settings

```yaml
sampling:
  coarse_samples: 200       # Stage 1 sample count
  fine_samples_per: 10     # Stage 2 samples per Pareto solution
  parameter_variation: 0.2  # +/-20% perturbation for fine search
  max_combo: 10000         # Switch to LHS sampling above this
```

### Device Groups

```yaml
active_bundles:
  - readpath
  - writepath

groups:
  - bundle_flag: readpath
    devices:
      - name: precharge_pmos
        vt_options: ["vt0", "vt1", "vt2"]
        gl_options: [0.020, 0.025, 0.030]
        nfin_options: [2, 3, 4, 5, 6]
```

### PPA Model Coefficients

```yaml
ppa_model:
  area:
    nfin_coef: 0.15
    gl_coef: 120.0
    base: 0.8
  power:
    nfin_coef: 12.5
    vt_penalty:
      vt0: 1.0
      vt1: 0.8
      vt2: 1.2
    base: 5.0
  delay:
    nfin_coef: 0.9
    gl_coef: 800.0
```

---

## Analytical PPA Model

When no external simulation is configured, the built-in model estimates PPA from device parameters:

```
Area   = Σ(nfin) × 0.15 + μ(gl) × 120 + 0.8
Power  = Σ(nfin) × 12.5 × VT_penalty + 5.0
Delay  = 180 / (Σ(nfin) × 0.9 + ε) + μ(gl) × 800
```

---

## External Simulation Interface

For real circuit simulation (HSPICE, Spectre), implement the interface in `src/spf_handler.py`:

```yaml
# In config YAML
spf_path: /path/to/your/sense_amp.spf

simulation:
  enabled: true
  tool: hspice  # or 'spectre'
  model_card: /path/to/models.sp
  temperature: 85
```

Your implementation should:
1. Load the SPF netlist as a template
2. Substitute device parameters (W, L, NF) based on combination
3. Run circuit simulation
4. Parse PPA from simulation output

The `submit_batch_simulation()` and `poll_and_collect_results()` functions in `run_moga.py` are the main integration points for your HPC batch system.

---

## Output

Results are saved to `./results/` (or `--output-dir`):

```
results/
├── checkpoints/
│   ├── stage1_pareto_solutions.json
│   ├── stage1_pop_gen{0..N}.json     # Per-generation population snapshots
│   ├── stage2_pareto_solutions.json
│   └── stage2_pop_gen{0..N}.json
├── stage1/
│   ├── results.json
│   ├── pareto_solutions.csv
│   └── summary.txt
├── stage2/
│   ├── results.json
│   ├── pareto_solutions.csv
│   └── summary.txt
└── moga.log                          # If --log is used
```

---

## Checkpoint & Recovery

Each generation's population is automatically saved during NSGA evolution (via `checkpoint_fn`). To recover from a crash:

1. Find the latest checkpoint: `results/checkpoints/stage{N}_pop_gen{G}.json`
2. Use `StageManager.load_population_checkpoint(stage, gen)` to restore
3. Modify `start_step` in `run_stage()` to skip combination generation and simulation

Example recovery scenario:
```python
from run_moga import StageManager
mgr = StageManager(Path('./results'))
latest_gen = mgr.get_latest_population_gen(1)  # e.g., 47
if latest_gen:
    gen, population = mgr.load_population_checkpoint(1, latest_gen)
    print(f"Resuming from generation {gen} with {len(population)} individuals")
```

---

## Algorithm Details

| Component | Implementation |
|-----------|---------------|
| Non-dominated sorting | Deb et al. O(MN²) fast sort |
| Diversity preservation | NSGA-II: crowding distance; NSGA-III: reference point association |
| Selection | Binary tournament (rank → crowding distance) |
| Crossover | SBX (Simulated Binary), adapted for integer genes |
| Mutation | Polynomial mutation, adapted for integer genes |
| Gene encoding | Integer indices into option arrays |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --tb=short

# Self-validation
python tests/validate.py
```

Current test status: **43 tests passing**

---

## References

- Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
- Deb & Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach" (NSGA-III)