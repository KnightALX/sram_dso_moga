#!/usr/bin/env python3
"""
SRAM DSO-MOGA: Main optimization script.

Execution Flow:
    Step 1: Parse YAML → Calculate combinations (LHS or full grid)
    Step 2: SPF Hack + Simulation (USER-IMPLEMENTED)
    Step 3: Collect PPA results
    Step 4: NSGA optimization

Usage:
    # Using built-in analytical PPA model
    python run_moga.py --config config/sram_config_v2.yaml

    # Using external simulation (if configured)
    python run_moga.py --config config/sram_config_v2.yaml --simulation

    # Custom parameters
    python run_moga.py --config config/sram_config_v2.yaml --pop-size 100 --n-gen 100

    # Run with dashboard
    python run_moga.py --config config/sram_config_v2.yaml --dashboard

    # With verbose logging
    python run_moga.py --config config/sram_config_v2.yaml --verbose --log

For Step 2/3 (SPF simulation), see:
    - src/spf_handler.py: SPF manipulation interface
    - src/fitness_collector.py: PPA result collection
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config, load_config
from evaluator import create_evaluator, PPAEvaluator
from nsga import NSGA, AlgorithmConfig, Individual
from fitness_collector import FitnessCollector, CombinationGenerator
from exporter import export_all


def setup_logging(log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger('sram_dso_moga')
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def step1_parse_and_calculate(config: Config, logger: logging.Logger) -> list:
    """
    Step 1: Parse YAML configuration and calculate parameter combinations.

    Args:
        config: Config object
        logger: Logger instance

    Returns:
        List of parameter combination dictionaries
    """
    logger.info("=" * 60)
    logger.info("Step 1: Parsing YAML and Calculating Combinations")
    logger.info("=" * 60)

    generator = CombinationGenerator(config.to_dict())

    total_combos = generator.total_combinations
    max_combo = generator.max_combo

    logger.info(f"Total design space: {total_combos} combinations")
    logger.info(f"Max before sampling: {max_combo}")

    # Determine sampling strategy
    if total_combos <= max_combo:
        logger.info(f"Using full enumeration ({total_combos} combinations)")
        combinations = generator.generate_all()
    else:
        combo_count = config.to_dict().get('sampling', {}).get('combo_count', 200)
        logger.info(f"Using LHS sampling ({combo_count} samples from {total_combos})")
        combinations = generator.generate_sampled(combo_count)

    logger.info(f"Generated {len(combinations)} parameter combinations")

    return combinations


def step2_spf_hack_and_simulate(combinations: list, config: Config,
                                 output_dir: Path, logger: logging.Logger) -> dict:
    """
    Step 2: SPF netlist hack and simulation.

    USER-IMPLEMENTED:
    This step is where you integrate your circuit simulation flow.
    The framework provides interfaces in src/spf_handler.py for you to implement.

    Args:
        combinations: List of parameter combinations
        config: Config object
        output_dir: Output directory for simulation files
        logger: Logger instance

    Returns:
        Dictionary mapping combination index to PPA results
    """
    logger.info("=" * 60)
    logger.info("Step 2: SPF Hack + Simulation")
    logger.info("=" * 60)

    spf_path = config.spf_path
    sim_config = config.to_dict().get('simulation', {})

    if not spf_path or spf_path == '/path/to/your/sense_amp.spf':
        logger.info("No SPF path configured - skipping external simulation")
        logger.info("Use built-in analytical PPA model")
        return {}

    if not sim_config.get('enabled', False):
        logger.info("Simulation not enabled - skipping external simulation")
        logger.info("Use built-in analytical PPA model")
        return {}

    # =====================================================================
    # USER IMPLEMENTATION REQUIRED
    # =====================================================================
    #
    # In this section, you would implement your actual circuit simulation:
    #
    # 1. Load your SPF netlist as template
    # 2. For each combination:
    #    a. Substitute device parameters (W, L, NF) based on combo
    #    b. Run HSPICE/Spectre simulation
    #    c. Parse output for power, delay
    #    d. Calculate area from device sizes
    #
    # The framework provides SPFSession class in src/spf_handler.py
    # as a starting point for your implementation.
    #
    # Example implementation structure:
    #
    # from spf_handler import HSPICESession  # Your implementation
    #
    # session = HSPICESession(spf_path, sim_config)
    #
    # results = {}
    # for i, combo in enumerate(combinations):
    #     logger.info(f"Simulating combination {i+1}/{len(combinations)}")
    #
    #     # Apply parameters to SPF
    #     modified_spf = session.apply_combination(combo)
    #
    #     # Run simulation
    #     stdout, stderr = session.run_simulation(modified_spf, output_dir / f'sim_{i}')
    #
    #     # Parse results
    #     ppa = session.parse_ppa_output(stdout, stderr)
    #     if ppa:
    #         results[i] = ppa
    #
    # return results
    # =====================================================================

    logger.info("SPF simulation interface ready - implement in src/spf_handler.py")
    logger.info("For now, using analytical PPA model as fallback")

    return {}


def step3_collect_ppa(config: Config, logger: logging.Logger) -> callable:
    """
    Step 3: Set up PPA result collection.

    Creates a fitness collector that:
    - Uses external simulation results if available
    - Falls back to analytical PPA model otherwise

    Args:
        config: Config object
        logger: Logger instance

    Returns:
        Fitness evaluator function
    """
    logger.info("=" * 60)
    logger.info("Step 3: Setting Up PPA Collection")
    logger.info("=" * 60)

    ppa_evaluator = create_evaluator(config.to_dict())
    fitness_collector = FitnessCollector(config.to_dict(), ppa_evaluator.evaluate)

    mode = "external simulation + analytical" if fitness_collector.use_simulation else "analytical only"
    logger.info(f"PPA evaluation mode: {mode}")

    return fitness_collector.evaluate


def step4_run_nsga(config: Config, fitness_fn: callable,
                   output_dir: Path, logger: logging.Logger) -> dict:
    """
    Step 4: Run NSGA optimization.

    Args:
        config: Config object
        fitness_fn: Fitness evaluation function
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Optimization results dictionary
    """
    logger.info("=" * 60)
    logger.info("Step 4: Running NSGA Optimization")
    logger.info("=" * 60)

    # Get tunables
    tunables = config.get_tunables()
    n_combinations = config.get_total_combinations()

    logger.info(f"Configuration: {config.top_name}")
    logger.info(f"Design space: {n_combinations} combinations")
    logger.info(f"Active bundles: {config.active_bundles}")
    logger.info(f"Tunables: {len(tunables)} parameters")

    # Algorithm config
    algo_cfg = config.algorithm
    nsga_config = AlgorithmConfig(
        pop_size=algo_cfg.get('pop_size', 80),
        n_gen=algo_cfg.get('n_gen', 60),
        seed=algo_cfg.get('seed', 42),
        crossover_prob=algo_cfg.get('crossover', {}).get('prob', 0.9),
        sbx_eta=algo_cfg.get('crossover', {}).get('eta', 15.0),
        mutation_prob=algo_cfg.get('mutation', {}).get('prob', 0.15),
        pm_eta=algo_cfg.get('mutation', {}).get('eta', 20.0),
        tournament_size=algo_cfg.get('selection', {}).get('tournament_size', 2),
        use_nsga3=algo_cfg.get('name', 'NSGA-II') == 'NSGA-III',
        n_partitions=algo_cfg.get('n_partitions', 12),
    )

    # Initialize NSGA
    optimizer = NSGA(
        config=nsga_config,
        tunables=tunables,
        evaluator=fitness_fn,
        logger=logger,
    )

    # Run optimization
    algo_name = 'NSGA-III' if nsga_config.use_nsga3 else 'NSGA-II'
    logger.info(f"Starting {algo_name}...")
    logger.info(f"Population: {nsga_config.pop_size}, Generations: {nsga_config.n_gen}")

    start_time = time.time()
    results = optimizer.evolve()
    elapsed = time.time() - start_time

    logger.info(f"Optimization completed in {elapsed:.2f}s")
    logger.info(f"Pareto solutions found: {len(results)}")

    # Get full results
    full_results = optimizer.get_results()

    return full_results


def save_results(results: dict, config: Config, output_dir: Path,
                 logger: logging.Logger) -> None:
    """Save optimization results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add config to results for export
    results['config'] = config.to_dict()

    # Export all formats
    paths = export_all(results, output_dir)

    for fmt, path in paths.items():
        logger.info(f"{fmt.upper()} saved to {path}")


def run_dashboard(results: dict, config: Config,
                  output_dir: Path, port: int = 8050) -> None:
    """Run interactive dashboard."""
    from dashboard import create_dashboard

    logger = logging.getLogger('sram_dso_moga')
    logger.info(f"Starting dashboard on port {port}...")

    # Convert results for dashboard (handle numpy types)
    dashboard_results = {
        'pareto_solutions': results.get('pareto_solutions', []),
        'pareto_objectives': [[float(x) for x in obj] for obj in results.get('pareto_objectives', [])],
        'history': results.get('history', []),
        'n_pareto': results.get('n_pareto', 0),
    }

    dash = create_dashboard(dashboard_results, config.to_dict(), output_dir)
    dash.run(port=port, debug=True)


def main():
    parser = argparse.ArgumentParser(
        description='SRAM DSO-MOGA: Multi-Objective Genetic Algorithm Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', '-c', required=True,
                       help='YAML configuration file path')
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Output directory (default: ./results)')
    parser.add_argument('--pop-size', type=int,
                       help='Override population size')
    parser.add_argument('--n-gen', type=int,
                       help='Override number of generations')
    parser.add_argument('--seed', type=int,
                       help='Override random seed')
    parser.add_argument('--dashboard', action='store_true',
                       help='Run interactive dashboard after optimization')
    parser.add_argument('--dashboard-port', type=int, default=8050,
                       help='Dashboard port (default: 8050)')
    parser.add_argument('--simulation', action='store_true',
                       help='Force enable external simulation mode')
    parser.add_argument('--log', action='store_true',
                       help='Save log to file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'moga.log' if args.log else None
    logger = setup_logging(log_file, log_level)

    logger.info("=" * 60)
    logger.info("SRAM DSO-MOGA v2.0")
    logger.info("=" * 60)

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")

        # Apply overrides
        if args.pop_size:
            config._data['algorithm']['pop_size'] = args.pop_size
            logger.info(f"Overriding pop_size: {args.pop_size}")
        if args.n_gen:
            config._data['algorithm']['n_gen'] = args.n_gen
            logger.info(f"Overriding n_gen: {args.n_gen}")
        if args.seed:
            config._data['algorithm']['seed'] = args.seed
            logger.info(f"Overriding seed: {args.seed}")
        if args.simulation:
            if 'simulation' not in config._data:
                config._data['simulation'] = {}
            config._data['simulation']['enabled'] = True
            logger.info("Enabling external simulation mode")

        # =================================================================
        # Execution Flow
        # =================================================================

        # Step 1: Parse YAML and calculate combinations
        combinations = step1_parse_and_calculate(config, logger)

        # Step 2: SPF hack and simulation (if configured)
        # Returns dict of pre-computed PPA results keyed by combination index
        sim_results = step2_spf_hack_and_simulate(combinations, config, output_dir, logger)

        # Step 3: Setup PPA collection
        fitness_fn = step3_collect_ppa(config, logger)

        # Step 4: Run NSGA optimization
        results = step4_run_nsga(config, fitness_fn, output_dir, logger)

        # =================================================================

        # Save results
        save_results(results, config, output_dir, logger)

        # Run dashboard if requested
        if args.dashboard:
            run_dashboard(results, config, output_dir, args.dashboard_port)
        else:
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("Pareto Optimal Solutions:")
            logger.info("=" * 60)
            for i, (sol, obj) in enumerate(zip(
                    results['pareto_solutions'][:5],
                    results['pareto_objectives'][:5]
            )):
                logger.info(f"  Solution {i+1}: Area={obj[0]:.3f}, Power={obj[1]:.2f}, Delay={obj[2]:.2f}")

        logger.info("\nOptimization completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()