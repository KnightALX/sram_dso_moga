#!/usr/bin/env python3
"""
SRAM DSO-MOGA: Main optimization script.

Execution Flow (4 Steps):
    Step 1: Parse YAML → Calculate combinations (LHS or full grid)
    Step 2: SPF Hack + Simulation submission (USER-IMPLEMENTED)
    Step 3: Poll PPA results from simulation batch
    Step 4: NSGA optimization

Usage:
    # Run all steps sequentially (default)
    python run_moga.py --config config/sram_config_v2.yaml

    # Run individual steps (for long-running simulations)
    python run_moga.py --config config.yaml --step 1      # Generate combinations only
    python run_moga.py --config config.yaml --step 2      # Generate SPFs for submission
    python run_moga.py --config config.yaml --step 3      # Poll and collect results
    python run_moga.py --config config.yaml --step 4      # Run optimization

    # Resume from a specific step
    python run_moga.py --config config.yaml --resume-from 3

    # With results polling for external simulations
    python run_moga.py --config config.yaml --step 3 --poll-interval 60 --poll-timeout 7200

For Step 2/3 (SPF simulation), see:
    - src/spf_handler.py: SPF manipulation interface
    - src/fitness_collector.py: PPA result collection
"""

from __future__ import annotations

import argparse
import json
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


# =====================================================================
# Checkpoint Management
# =====================================================================

class CheckpointManager:
    """Manages checkpoints for resumable execution."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, data: dict) -> None:
        """Save checkpoint for a step."""
        path = self.checkpoint_dir / f'step_{step}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.getLogger('sram_dso_moga').info(f"Checkpoint saved: {path}")

    def load(self, step: int) -> dict:
        """Load checkpoint for a step."""
        path = self.checkpoint_dir / f'step_{step}.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def exists(self, step: int) -> bool:
        """Check if checkpoint exists."""
        return (self.checkpoint_dir / f'step_{step}.json').exists()

    def clear(self, step: int = None) -> None:
        """Clear checkpoint(s)."""
        if step is None:
            for p in self.checkpoint_dir.glob('step_*.json'):
                p.unlink()
        else:
            (self.checkpoint_dir / f'step_{step}.json').unlink(missing_ok=True)


def setup_logging(log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with file and console handlers."""
    logger = logging.getLogger('sram_dso_moga')
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =====================================================================
# Step Implementations
# =====================================================================

def step1_parse_and_calculate(config: Config, output_dir: Path,
                               logger: logging.Logger,
                               checkpoint: CheckpointManager) -> list:
    """Step 1: Parse YAML configuration and calculate parameter combinations."""
    logger.info("=" * 60)
    logger.info("Step 1: Parsing YAML and Calculating Combinations")
    logger.info("=" * 60)

    # Check for existing checkpoint
    cached = checkpoint.load(1)
    if cached:
        logger.info("Found checkpoint from previous Step 1 run")
        combinations = cached.get('combinations', [])
        logger.info(f"Loaded {len(combinations)} combinations from checkpoint")
        return combinations

    generator = CombinationGenerator(config.to_dict())

    total_combos = generator.total_combinations
    max_combo = generator.max_combo

    logger.info(f"Total design space: {total_combos} combinations")
    logger.info(f"Max before sampling: {max_combo}")

    if total_combos <= max_combo:
        logger.info(f"Using full enumeration ({total_combos} combinations)")
        combinations = generator.generate_all()
    else:
        combo_count = config.to_dict().get('sampling', {}).get('combo_count', 200)
        logger.info(f"Using LHS sampling ({combo_count} samples from {total_combos})")
        combinations = generator.generate_sampled(combo_count)

    logger.info(f"Generated {len(combinations)} parameter combinations")

    # Save checkpoint
    checkpoint.save(1, {
        'combinations': combinations,
        'total_combos': total_combos,
        'sampled_count': len(combinations),
        'timestamp': time.time(),
    })

    return combinations


def step2_spf_hack_and_submit(combinations: list, config: Config,
                               output_dir: Path, logger: logging.Logger,
                               checkpoint: CheckpointManager) -> dict:
    """
    Step 2: Generate modified SPFs and submit to simulation batch.

    USER-IMPLEMENTED: See src/spf_handler.py for implementation guidelines.
    """
    logger.info("=" * 60)
    logger.info("Step 2: SPF Hack + Simulation Submission")
    logger.info("=" * 60)

    spf_path = config.spf_path
    sim_config = config.to_dict().get('simulation', {})

    if not spf_path or spf_path == '/path/to/your/sense_amp.spf':
        logger.info("No SPF path configured - skipping Step 2")
        return {'mode': 'none'}

    if not sim_config.get('enabled', False):
        logger.info("Simulation not enabled - skipping Step 2")
        return {'mode': 'none'}

    # Check for existing checkpoint
    cached = checkpoint.load(2)
    if cached:
        logger.info("Found checkpoint from previous Step 2 run")
        logger.info("To re-run Step 2, clear checkpoint first: --clear-checkpoint 2")
        return cached.get('submission_info', {})

    # USER IMPLEMENTATION REQUIRED in src/spf_handler.py
    # See example implementation in spf_handler.py
    logger.info("SPF simulation submission interface ready")
    logger.info("Implement your SPF hack and batch submission in src/spf_handler.py")
    logger.info("For now, using analytical PPA model as fallback")

    return {'mode': 'none'}


def step3_collect_ppa_results(config: Config, output_dir: Path,
                              logger: logging.Logger, checkpoint: CheckpointManager,
                              poll_interval: int = 60, poll_timeout: int = 7200) -> dict:
    """Step 3: Poll for PPA results from simulation batch."""
    logger.info("=" * 60)
    logger.info("Step 3: Collecting PPA Results")
    logger.info("=" * 60)

    sim_config = config.to_dict().get('simulation', {})
    use_external = sim_config.get('enabled', False)

    if not use_external:
        logger.info("External simulation not enabled - using analytical PPA model")
        return {'mode': 'analytical'}

    # Check for Step 2 checkpoint
    step2_data = checkpoint.load(2)
    if not step2_data:
        logger.error("Step 2 checkpoint not found. Run --step 2 first.")
        return {'mode': 'error', 'error': 'No Step 2 checkpoint'}

    submission_info = step2_data.get('submission_info', {})
    jobs = submission_info.get('jobs', [])

    if not jobs:
        logger.error("No jobs found in Step 2 checkpoint")
        return {'mode': 'error', 'error': 'No jobs to poll'}

    logger.info(f"Polling for {len(jobs)} simulation jobs...")
    logger.info(f"Poll interval: {poll_interval}s, Timeout: {poll_timeout}s")

    # USER IMPLEMENTATION REQUIRED: See src/spf_handler.py
    # Implement result polling from your batch system
    logger.info("Result polling interface ready - implement in src/spf_handler.py")

    return {'mode': 'placeholder', 'jobs': jobs}


def step4_run_nsga(config: Config, fitness_fn, output_dir: Path,
                   logger: logging.Logger, checkpoint: CheckpointManager) -> dict:
    """Step 4: Run NSGA optimization."""
    logger.info("=" * 60)
    logger.info("Step 4: Running NSGA Optimization")
    logger.info("=" * 60)

    tunables = config.get_tunables()
    n_combinations = config.get_total_combinations()

    logger.info(f"Configuration: {config.top_name}")
    logger.info(f"Design space: {n_combinations} combinations")
    logger.info(f"Active bundles: {config.active_bundles}")
    logger.info(f"Tunables: {len(tunables)} parameters")

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

    optimizer = NSGA(
        config=nsga_config,
        tunables=tunables,
        evaluator=fitness_fn,
        logger=logger,
    )

    algo_name = 'NSGA-III' if nsga_config.use_nsga3 else 'NSGA-II'
    logger.info(f"Starting {algo_name}...")
    logger.info(f"Population: {nsga_config.pop_size}, Generations: {nsga_config.n_gen}")

    start_time = time.time()
    results = optimizer.evolve()
    elapsed = time.time() - start_time

    logger.info(f"Optimization completed in {elapsed:.2f}s")
    logger.info(f"Pareto solutions found: {len(results)}")

    full_results = optimizer.get_results()

    # Save checkpoint
    checkpoint.save(4, {
        'pareto_solutions': full_results.get('pareto_solutions', []),
        'pareto_objectives': full_results.get('pareto_objectives', []),
        'history': full_results.get('history', []),
        'n_pareto': full_results.get('n_pareto', 0),
        'elapsed': elapsed,
        'timestamp': time.time(),
    })

    return full_results


# =====================================================================
# Main Entry Point
# =====================================================================

def run_pipeline(start_step: int, config: Config, args: argparse.Namespace,
                logger: logging.Logger) -> dict:
    """Run the optimization pipeline from start_step to step 4."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = CheckpointManager(output_dir)

    # Clear checkpoints if requested
    if args.clear_checkpoint is not None:
        checkpoint.clear(args.clear_checkpoint if isinstance(args.clear_checkpoint, int) else None)

    results = {}
    combinations = []
    use_external_sim = False

    # Step 1: Generate combinations
    if start_step <= 1:
        combinations = step1_parse_and_calculate(config, output_dir, logger, checkpoint)
        results['step1'] = {'combinations': combinations}
    else:
        cached = checkpoint.load(1)
        combinations = cached.get('combinations', []) if cached else []

    # Step 2: Generate SPFs and submit
    if start_step <= 2:
        submission_info = step2_spf_hack_and_submit(
            combinations, config, output_dir, logger, checkpoint
        )
        results['step2'] = {'submission_info': submission_info}
        use_external_sim = submission_info.get('mode') == 'hpc'

    # Step 3: Collect PPA results
    if start_step <= 3:
        ppa_results = step3_collect_ppa_results(
            config, output_dir, logger, checkpoint,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout
        )
        results['step3'] = {'ppa_results': ppa_results}
        use_external_sim = ppa_results.get('mode') == 'hpc'

    # Step 4: Run NSGA optimization
    if start_step <= 4:
        ppa_evaluator = create_evaluator(config.to_dict())

        # Use external results or analytical model
        if use_external_sim:
            logger.info("Using external simulation results")
            step3_data = checkpoint.load(3)
            external_results = step3_data.get('ppa_results', {}).get('results', {})

            def external_fitness_fn(combo_params):
                # Hash-based lookup (customize for your indexing)
                idx = hash(str(sorted(combo_params.items()))) % max(1, len(external_results))
                r = external_results.get(idx, {})
                return [
                    r.get('area', float('inf')),
                    r.get('power', float('inf')),
                    r.get('delay', float('inf')),
                ]
            fitness_fn = external_fitness_fn
        else:
            logger.info("Using analytical PPA model")
            fitness_fn = ppa_evaluator.evaluate

        nsga_results = step4_run_nsga(config, fitness_fn, output_dir, logger, checkpoint)
        results['step4'] = nsga_results

    return results


def save_final_results(results: dict, config: Config, output_dir: Path,
                       logger: logging.Logger) -> None:
    """Save final optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    nsga_results = results.get('step4', {})

    if isinstance(nsga_results, dict) and 'pareto_solutions' in nsga_results:
        nsga_results['config'] = config.to_dict()
        paths = export_all(nsga_results, output_dir)
        for fmt, path in paths.items():
            logger.info(f"{fmt.upper()} saved to {path}")
    else:
        with open(output_dir / 'step_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description='SRAM DSO-MOGA: Multi-Objective Genetic Algorithm Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', '-c', required=True,
                       help='YAML configuration file path')
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Output directory (default: ./results)')

    # Step control (mutually exclusive)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                           help='Run a specific step only')
    step_group.add_argument('--resume-from', type=int, choices=[1, 2, 3, 4],
                           help='Resume from a specific step')

    # Checkpoint control
    parser.add_argument('--clear-checkpoint', nargs='?', const=True, type=int,
                       help='Clear checkpoint(s). Use --clear-checkpoint 2 for step 2, or no value for all')

    # Simulation polling options
    parser.add_argument('--poll-interval', type=int, default=60,
                       help='Seconds between polling attempts (default: 60)')
    parser.add_argument('--poll-timeout', type=int, default=7200,
                       help='Max seconds to wait for results (default: 7200 = 2h)')

    # Algorithm overrides
    parser.add_argument('--pop-size', type=int, help='Override population size')
    parser.add_argument('--n-gen', type=int, help='Override number of generations')
    parser.add_argument('--seed', type=int, help='Override random seed')

    # Dashboard and logging
    parser.add_argument('--dashboard', action='store_true',
                       help='Run interactive dashboard after optimization')
    parser.add_argument('--dashboard-port', type=int, default=8050,
                       help='Dashboard port (default: 8050)')
    parser.add_argument('--log', action='store_true', help='Save log to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

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

        # Determine start step
        if args.step:
            start_step = args.step
            logger.info(f"Running Step {start_step} only")
        elif args.resume_from:
            start_step = args.resume_from
            logger.info(f"Resuming from Step {start_step}")
        else:
            start_step = 1
            logger.info("Running all steps sequentially")

        # Run pipeline
        results = run_pipeline(start_step, config, args, logger)

        # Save final results
        save_final_results(results, config, output_dir, logger)

        # Run dashboard if full pipeline
        if args.dashboard and not args.step and not args.resume_from:
            from dashboard import create_dashboard

            nsga_results = results.get('step4', {})
            if isinstance(nsga_results, dict) and 'pareto_solutions' in nsga_results:
                dashboard_results = {
                    'pareto_solutions': nsga_results.get('pareto_solutions', []),
                    'pareto_objectives': [[float(x) for x in obj]
                                         for obj in nsga_results.get('pareto_objectives', [])],
                    'history': nsga_results.get('history', []),
                    'n_pareto': nsga_results.get('n_pareto', 0),
                }
                dash = create_dashboard(dashboard_results, config.to_dict(), output_dir)
                logger.info(f"Starting dashboard on port {args.dashboard_port}...")
                dash.run(port=args.dashboard_port, debug=True)
        elif args.dashboard:
            logger.warning("Dashboard only available when running full pipeline")

        logger.info("\nExecution completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()