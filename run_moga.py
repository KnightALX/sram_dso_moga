#!/usr/bin/env python3
"""
SRAM DSO-MOGA: Main optimization script.

Usage:
    python run_moga.py --config config/sram_config_v2.yaml
    python run_moga.py --config config/sram_config_v2.yaml --pop-size 100 --n-gen 100
    python run_moga.py --config config/sram_config_v2.yaml --dashboard  # Run interactive dashboard
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

from config import load_config, Config
from evaluator import create_evaluator, PPAEvaluator
from nsga import NSGA, AlgorithmConfig, NSGA as NSGAOptimizer


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


def run_optimization(config: Config, logger: logging.Logger) -> dict:
    """Run the MOGA optimization."""
    start_time = time.time()

    # Create evaluator
    ppa_eval = create_evaluator(config.to_dict())
    evaluator = ppa_eval.evaluate

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
        use_nsga3=algo_cfg.get('name', 'NSGA-II') == 'NSGA-III',
    )

    # Initialize NSGA
    optimizer = NSGA(
        config=nsga_config,
        tunables=tunables,
        evaluator=evaluator,
        logger=logger,
    )

    # Run optimization
    logger.info(f"Starting NSGA-{('II' if not nsga_config.use_nsga3 else 'III')}...")
    logger.info(f"Population: {nsga_config.pop_size}, Generations: {nsga_config.n_gen}")

    results = optimizer.evolve()

    elapsed = time.time() - start_time
    logger.info(f"Optimization completed in {elapsed:.2f}s")
    logger.info(f"Pareto solutions found: {len(results)}")

    # Get full results
    full_results = optimizer.get_results()

    return full_results


def save_results(results: dict, config: Config, output_dir: Path, logger: logging.Logger):
    """Save optimization results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'results.json'
    # Convert numpy types for JSON serialization
    json_results = {
        'pareto_solutions': results['pareto_solutions'],
        'pareto_objectives': [[float(x) for x in obj] for obj in results['pareto_objectives']],
        'history': results['history'],
        'n_pareto': results['n_pareto'],
        'config': config.to_dict(),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {json_path}")

    # Save CSV
    csv_path = output_dir / 'pareto_solutions.csv'
    import pandas as pd
    df = pd.DataFrame(results['pareto_solutions'])
    obj_df = pd.DataFrame(results['pareto_objectives'], columns=['Area', 'Power', 'Delay'])
    df = pd.concat([df, obj_df], axis=1)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")


def run_dashboard(results: dict, config: Config, output_dir: Path, port: int = 8050):
    """Run interactive dashboard."""
    from dashboard import create_dashboard

    logger = logging.getLogger('sram_dso_moga')
    logger.info(f"Starting dashboard on port {port}...")

    dash = create_dashboard(results, config.to_dict(), output_dir)
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

        # Run optimization
        results = run_optimization(config, logger)

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
            for i, (sol, obj) in enumerate(zip(results['pareto_solutions'][:5], results['pareto_objectives'][:5])):
                logger.info(f"  Solution {i+1}: Area={obj[0]:.3f}, Power={obj[1]:.2f}, Delay={obj[2]:.2f}")

        logger.info("\n✅ Optimization completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()