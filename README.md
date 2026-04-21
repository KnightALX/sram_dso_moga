# SRAM DSO-MOGA

**SRAM Design Space Optimization using Multi-Objective Genetic Algorithm**

NSGA-II/NSGA-III based optimizer for SRAM peripheral circuit PPA (Power, Performance, Area) tuning.

## Project Structure

```
sram_dso_moga/
├── src/
│   ├── config.py          # Configuration management & validation
│   ├── evaluator.py       # PPA evaluation models
│   ├── nsga.py            # NSGA-II/III implementation
│   ├── spf_handler.py     # SPF netlist manipulation interface
│   ├── fitness_collector.py # PPA results collection interface
│   └── dashboard.py       # Interactive Plotly Dash dashboard
├── config/
│   └── sram_config_v2.yaml  # Enhanced configuration schema
├── tests/
│   ├── test_nsga.py        # NSGA algorithm tests
│   ├── test_config_eval.py # Config & evaluator tests
│   └── validate.py         # Self-validation script
├── run_moga.py             # Main entry point
├── CLAUDE.md               # Claude Code guidance
└── README.md               # This file
```

## Quick Start

```bash
# Run optimization with default config (uses built-in PPA model)
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

## Execution Flow

The optimization follows a 4-step pipeline:

### Step 1: Parse YAML → Calculate Combinations
Configuration is parsed and all possible parameter combinations are calculated.
LHS (Latin Hypercube Sampling) is used when combination count exceeds `max_combo`.

### Step 2: SPF Netlist Hack + Simulation (USER-IMPLEMENTED)
For real circuit simulation, replace the placeholder in `src/spf_handler.py`:

```python
def run_simulation(combo_params: dict, spf_template: str) -> dict:
    """
    User implements this function to:
    1. Replace parameters in SPF netlist based on combo_params
    2. Run circuit simulation
    3. Return PPA results: {'area': float, 'power': float, 'delay': float}
    """
    # TODO: User implements SPF manipulation and simulation
    pass
```

The interface provides:
- `SPFSession` class for netlist manipulation
- `apply_combination()` to replace parameters in SPF template
- `run_hspice()` / `run_spectre()` placeholder methods
- Configuration via `simulation:` section in YAML

### Step 3: Collect PPA Results
Results from Step 2 are collected via `FitnessCollector`:
- Supports batch collection
- Handles missing/invalid simulations gracefully
- Falls back to analytical model if no external results provided

### Step 4: NSGA Optimization
Pareto-optimal solutions are found using NSGA-II/III:
- Tournament selection based on rank and crowding distance
- SBX crossover for real-valued parameters
- Polynomial mutation

## SPF Simulation Interface

If you have an SPF netlist and want to use real simulation instead of the
built-in PPA model, implement the interface in `src/spf_handler.py`:

```yaml
# In your config YAML, add:
spf_path: /path/to/your/sense_amp.spf

simulation:
  enabled: true
  tool: hspice  # or 'spectre', 'spectre_spectre'
  model_card: /path/to/models.sp
  temperature: 85  # Celsius
```

Your implementation should:
1. Read `spf_path` as template
2. Replace device parameters (W, L, NF) based on combination
3. Run simulation
4. Parse output for power/delay

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

## PPA Model (Built-in)

Simplified analytical models (used when no external simulation is configured):
- **Area** = Σ(nfin) × 0.15 + μ(gl) × 120 + 0.8
- **Power** = Σ(nfin) × 12.5 × VT_penalty + 5.0
- **Delay** = 180 / (Σ(nfin) × 0.9 + ε) + μ(gl) × 800

Where VT_penalty depends on threshold voltage variant (vt0/vt1/vt2).

## SPF Handler Interface

```python
from src.spf_handler import SPFSession, run_simulation

# Example: User implements simulation
class MySPFSession(SPFSession):
    def run_simulation(self, combo_params):
        # 1. Apply parameters to SPF template
        modified_spf = self.apply_combination(combo_params)
        # 2. Run simulation tool
        output = self.execute_simulation(modified_spf)
        # 3. Parse results
        return self.parse_ppa_output(output)
```

If no SPF path is configured, the built-in PPA evaluator is used automatically.