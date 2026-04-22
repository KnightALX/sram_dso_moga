#!/usr/bin/env python3
"""
SRAM DSO-MOGA: Multi-Stage Optimization with Two-Phase Simulation

Workflow:
    Stage 1 (Coarse Search):
        Step 1: Generate full design space combinations (LHS or full grid)
        Step 2: SPF hack → Batch simulation submission
        Step 3: Collect all PPA results
        Step 4: NSGA coarse search → Pareto front

    Stage 2 (Fine Search):
        Step 1': Generate fine samples around Stage 1 Pareto front
        Step 2': SPF hack → Fine batch simulation
        Step 3': Collect fine PPA results
        Step 4': NSGA fine search → Final Pareto front

Usage:
    # Stage 1: Coarse search
    python run_moga.py --config config.yaml --stage 1 --pop-size 100 --n-gen 50

    # Stage 2: Fine search (uses Stage 1 results)
    python run_moga.py --config config.yaml --stage 2 --pop-size 50 --n-gen 30

    # Run both stages sequentially
    python run_moga.py --config config.yaml --stage 1 --stage 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config, load_config
from evaluator import create_evaluator
from nsga import NSGA, AlgorithmConfig
from fitness_collector import CombinationGenerator
from exporter import export_all


# =====================================================================
# Stage Manager
# =====================================================================

class StageManager:
    """Manages multi-stage optimization workflow."""

    STAGE_DIRS = {
        1: 'stage1',
        2: 'stage2',
    }

    def __init__(self, base_output_dir: Path):
        self.base_dir = base_output_dir
        self.checkpoint_dir = base_output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_stage_dir(self, stage: int) -> Path:
        """Get output directory for a stage."""
        d = self.base_dir / self.STAGE_DIRS[stage]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_stage_checkpoint(self, stage: int, key: str):
        """Load checkpoint data for a stage."""
        path = self.checkpoint_dir / f'stage{stage}_{key}.json'
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def save_stage_checkpoint(self, stage: int, key: str, data: dict):
        """Save checkpoint data for a stage."""
        path = self.checkpoint_dir / f'stage{stage}_{key}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def stage_exists(self, stage: int, key: str) -> bool:
        """Check if stage checkpoint exists."""
        return (self.checkpoint_dir / f'stage{stage}_{key}.json').exists()

    def save_population_checkpoint(self, stage: int, gen: int,
                                   population: List["Individual"]) -> None:
        """
        Save population state at a specific generation for crash recovery.

        Args:
            stage: Stage number (1 or 2)
            gen: Generation number
            population: List of Individual objects
        """
        path = self.checkpoint_dir / f'stage{stage}_pop_gen{gen}.json'
        data = {
            'gen': gen,
            'stage': stage,
            'individuals': [
                {
                    'genes': ind.genes.tolist(),
                    'objectives': [float(x) for x in ind.objectives],
                    'rank': int(ind.rank),
                    'crowding_distance': float(ind.crowding_distance),
                }
                for ind in population
            ]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_population_checkpoint(self, stage: int, gen: int):
        """
        Load population checkpoint from a specific generation.

        Args:
            stage: Stage number (1 or 2)
            gen: Generation number

        Returns:
            Tuple of (gen, population) or None if checkpoint not found
        """
        path = self.checkpoint_dir / f'stage{stage}_pop_gen{gen}.json'
        if not path.exists():
            return None

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        from nsga import Individual
        population = [
            Individual(
                genes=np.array(ind_data['genes']),
                objectives=np.array(ind_data['objectives']),
                rank=ind_data['rank'],
                crowding_distance=ind_data['crowding_distance'],
            )
            for ind_data in data['individuals']
        ]
        return data['gen'], population

    def get_latest_population_gen(self, stage: int) -> Optional[int]:
        """Find the latest saved generation for a stage."""
        checkpoint_files = list(self.checkpoint_dir.glob(f'stage{stage}_pop_gen*.json'))
        if not checkpoint_files:
            return None

        gens = []
        for f in checkpoint_files:
            try:
                name = f.stem  # stage1_pop_gen50
                gen = int(name.split('_gen')[1])
                gens.append(gen)
            except (ValueError, IndexError):
                continue

        return max(gens) if gens else None

    def get_combined_results(self) -> Dict[int, dict]:
        """Load results from all completed stages."""
        combined = {}
        for stage in [1, 2]:
            if self.stage_exists(stage, 'ppa_results'):
                combined[stage] = self.get_stage_checkpoint(stage, 'ppa_results')
        return combined


def setup_logging(log_file: Path = None, level: int = logging.INFO) -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger('sram_dso_moga')
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =====================================================================
# Combination Generation (Supports Coarse and Fine Modes)
# =====================================================================

class CombinationGeneratorV2(CombinationGenerator):
    """Enhanced combination generator with coarse and fine modes."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config

    def generate_coarse(self, n_samples: int = None) -> List[dict]:
        """Generate combinations covering full design space."""
        if n_samples is None:
            n_samples = self.config.get('sampling', {}).get('combo_count', 200)

        if self.total_combinations <= self.max_combo:
            return self.generate_all()
        else:
            return self.generate_sampled(n_samples)

    def generate_fine(self, pareto_solutions: List[dict],
                     n_samples_per_solution: int = 10,
                     parameter_variation: float = 0.2) -> List[dict]:
        """
        Generate fine samples around Pareto solutions.

        For each Pareto solution, generate variations by slightly perturbing
        each parameter within a percentage of its current value.

        Args:
            pareto_solutions: List of Pareto-optimal solutions
            n_samples_per_solution: Number of samples per Pareto solution
            parameter_variation: Fractional variation (0.2 = +/-20%)

        Returns:
            List of fine-tuned parameter combinations
        """
        import random
        import itertools

        fine_combinations = []

        for sol in pareto_solutions:
            # Get tunable parameter names
            tunables = self._extract_tunable_names()

            # Generate parameter variations
            for _ in range(n_samples_per_solution):
                variant = sol.copy()

                for param_name in tunables:
                    if param_name not in variant:
                        continue

                    original_value = variant[param_name]

                    # Handle string parameters (VT options)
                    if isinstance(original_value, str):
                        if '_vt' in param_name:
                            # For VT options, randomly switch to adjacent variant
                            # Keep original with 60% probability, switch with 40%
                            if random.random() < 0.4:
                                _, vt_options = self._get_tunable_options(param_name)
                                if len(vt_options) > 1:
                                    current_idx = vt_options.index(original_value)
                                    # Switch to adjacent or random different option
                                    if current_idx == 0:
                                        variant[param_name] = vt_options[1]
                                    elif current_idx == len(vt_options) - 1:
                                        variant[param_name] = vt_options[current_idx - 1]
                                    else:
                                        variant[param_name] = vt_options[current_idx + 1] if random.random() < 0.5 else vt_options[current_idx - 1]
                            # else keep original
                        continue

                    # Handle numeric parameters (GL, nfin)
                    if isinstance(original_value, (int, float)):
                        variation = abs(original_value) * parameter_variation
                        perturbation = random.uniform(-variation, variation)
                        new_value = original_value + perturbation

                        # Clamp to valid range based on tunable options
                        _, options = self._get_tunable_options(param_name)
                        if options:
                            new_value = max(min(options), min(max(options), new_value))
                            # Preserve integer type for nfin
                            if isinstance(options[0], int):
                                new_value = int(round(new_value))

                        variant[param_name] = new_value

                fine_combinations.append(variant)

        return fine_combinations

    def _get_tunable_options(self, param_name: str) -> tuple:
        """Get options tuple for a parameter name from self.tunables."""
        for name, options in self.tunables:
            if name == param_name:
                return (name, options)
        return (param_name, [])

    def _extract_tunable_names(self) -> List[str]:
        """Extract all tunable parameter names (vt, gl, nfin for each device)."""
        names = []
        for group in self.config.get('groups', []):
            if group.get('bundle_flag') not in set(self.config.get('active_bundles', [])):
                continue
            for device in group.get('devices', []):
                base = device['name']
                for opt_suffix in ['vt_options', 'gl_options', 'nfin_options']:
                    if opt_suffix in device:
                        names.append(f"{base}_{opt_suffix.rsplit('_', 1)[0]}")
        return names


# =====================================================================
# SPF Handler Interface (User Implements)
# =====================================================================

def submit_batch_simulation(combinations: List[dict],
                           config: Config,
                           output_dir: Path,
                           logger: logging.Logger,
                           stage: int) -> dict:
    """
    Submit batch simulation for combinations.

    USER IMPLEMENTATION REQUIRED:
    =============================
    This function should:
    1. Generate modified SPFs for each combination
    2. Submit to batch simulation system (HSPICE/Spectre)
    3. Return submission info with job IDs

    For now, returns placeholder - implement in src/spf_handler.py
    """
    logger.info(f"[Stage {stage}] Batch simulation submission interface")
    logger.info(f"[Stage {stage}] {len(combinations)} combinations to simulate")

    sim_enabled = config.to_dict().get('simulation', {}).get('enabled', False)
    spf_path = config.spf_path

    if not sim_enabled or not spf_path or spf_path == '/path/to/your/sense_amp.spf':
        logger.info(f"[Stage {stage}] Using analytical PPA model (no external simulation)")
        return {
            'mode': 'analytical',
            'total': len(combinations),
        }

    # USER IMPLEMENTATION REQUIRED:
    # Implement your batch submission logic here
    #
    # Example structure:
    # jobs = []
    # for i, combo in enumerate(combinations):
    #     modified_spf = apply_combination(combo, spf_template)
    #     job_id = submit_to_hpc(modified_spf, output_dir / f'job_{i}')
    #     jobs.append({'combo_idx': i, 'job_id': job_id, 'status': 'pending'})
    #
    # return {'mode': 'hpc', 'jobs': jobs, 'total': len(combinations)}

    return {
        'mode': 'analytical',
        'total': len(combinations),
    }


def poll_and_collect_results(config: Config,
                            output_dir: Path,
                            logger: logging.Logger,
                            stage: int,
                            submission_info: dict,
                            poll_interval: int = 60,
                            poll_timeout: int = 7200) -> dict:
    """
    Poll simulation results and collect PPA values.

    USER IMPLEMENTATION REQUIRED:
    =============================
    This function should:
    1. Poll job status from batch system
    2. When complete, parse PPA from output
    3. Handle failed jobs gracefully
    4. Return {combo_idx: {area, power, delay}}

    For now, returns placeholder - implement in src/spf_handler.py
    """
    logger.info(f"[Stage {stage}] Result collection interface")
    logger.info(f"[Stage {stage}] Poll interval: {poll_interval}s, Timeout: {poll_timeout}s")

    if submission_info.get('mode') == 'analytical':
        logger.info(f"[Stage {stage}] Analytical mode - no polling needed")
        return {'mode': 'analytical'}

    # USER IMPLEMENTATION REQUIRED:
    # Implement your polling logic here
    #
    # Example structure:
    # results = {}
    # pending = {j['combo_idx']: j for j in submission_info['jobs']}
    #
    # while pending and elapsed < poll_timeout:
    #     for combo_idx, job in list(pending.items()):
    #         status = check_job_status(job['job_id'])
    #         if status == 'completed':
    #             results[combo_idx] = parse_ppa_output(job['output_dir'])
    #             del pending[combo_idx]
    #         elif status == 'failed':
    #             results[combo_idx] = None  # Mark as failed
    #             del pending[combo_idx]
    #
    #     if pending:
    #         time.sleep(poll_interval)
    #
    # return {'mode': 'hpc', 'results': results}

    return {'mode': 'placeholder'}


# =====================================================================
# PPA Evaluator with Combined Results Support
# =====================================================================

class CombinedPPAEvaluator:
    """
    PPA evaluator that queries from combined simulation results.
    Falls back to analytical model for missing combinations.
    """

    def __init__(self, analytical_evaluator, combined_results: Dict[int, dict]):
        """
        Args:
            analytical_evaluator: PPAEvaluator instance for fallback
            combined_results: {stage: {combo_idx: {area, power, delay}}}
        """
        self.analytical = analytical_evaluator
        self.combined = combined_results
        self._build_lookup_index()

    def _build_lookup_index(self):
        """Build index for fast lookup by parameter combination."""
        # This would need a hash-based lookup for exact matches
        pass

    def evaluate(self, combo_params: dict) -> List[float]:
        """
        Evaluate PPA for a combination.

        Priority:
        1. Stage 2 results (finest)
        2. Stage 1 results
        3. Analytical model fallback
        """
        # Try to find in Stage 2 results first (most refined)
        for stage in sorted(self.combined.keys(), reverse=True):
            results = self.combined[stage].get('results', {})
            # User implements exact lookup here
            # For now, fall through to analytical

        # Fallback to analytical model
        return self.analytical.evaluate(combo_params)


# =====================================================================
# Main Pipeline
# =====================================================================

def run_stage(stage: int, config: Config, stage_mgr: StageManager,
             args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Run a single stage."""
    stage_dir = stage_mgr.get_stage_dir(stage)
    logger.info("=" * 60)
    logger.info(f"STAGE {stage}: {'Coarse Search' if stage == 1 else 'Fine Search'}")
    logger.info("=" * 60)

    # Determine which step to start from
    start_step = 1

    results = {}

    # =================================================================
    # Step 1: Generate combinations
    # =================================================================
    if start_step <= 1:
        logger.info(f"[Stage {stage}] Step 1: Generating combinations")

        if stage == 1:
            # Coarse: Full design space
            generator = CombinationGeneratorV2(config.to_dict())
            coarse_samples = config.to_dict().get('sampling', {}).get('coarse_samples', 200)
            combinations = generator.generate_coarse(n_samples=coarse_samples)
            logger.info(f"[Stage {stage}] Generated {len(combinations)} coarse combinations")

            # Save checkpoint
            stage_mgr.save_stage_checkpoint(stage, 'combinations', {
                'combinations': combinations,
                'mode': 'coarse',
                'timestamp': time.time(),
            })

        else:
            # Fine: Around Stage 1 Pareto front
            stage1_checkpoint = stage_mgr.get_stage_checkpoint(1, 'pareto_solutions')
            if not stage1_checkpoint:
                logger.error("[Stage 2] Stage 1 Pareto solutions not found!")
                logger.error("[Stage 2] Run --stage 1 first")
                return results

            pareto_solutions = stage1_checkpoint.get('pareto_solutions', [])
            logger.info(f"[Stage 2] Loading {len(pareto_solutions)} Pareto solutions from Stage 1")

            # Generate fine variations around each Pareto solution
            generator = CombinationGeneratorV2(config.to_dict())
            fine_samples_per = config.to_dict().get('sampling', {}).get('fine_samples_per_solution', 10)
            variation_pct = config.to_dict().get('sampling', {}).get('parameter_variation', 0.2)

            combinations = generator.generate_fine(
                pareto_solutions,
                n_samples_per_solution=fine_samples_per,
                parameter_variation=variation_pct
            )
            logger.info(f"[Stage 2] Generated {len(combinations)} fine combinations")

            # Save checkpoint
            stage_mgr.save_stage_checkpoint(stage, 'combinations', {
                'combinations': combinations,
                'mode': 'fine',
                'based_on': len(pareto_solutions),
                'timestamp': time.time(),
            })

        results['step1'] = {'combinations': combinations}

    # =================================================================
    # Step 2: Submit batch simulation
    # =================================================================
    if start_step <= 2:
        logger.info(f"[Stage {stage}] Step 2: Batch simulation submission")

        submission_info = submit_batch_simulation(
            combinations, config, stage_dir, logger, stage
        )
        results['step2'] = {'submission_info': submission_info}

        stage_mgr.save_stage_checkpoint(stage, 'submission', submission_info)

    # =================================================================
    # Step 3: Collect PPA results
    # =================================================================
    if start_step <= 3:
        logger.info(f"[Stage {stage}] Step 3: Collecting PPA results")

        ppa_results = poll_and_collect_results(
            config, stage_dir, logger, stage,
            submission_info,
            poll_interval=args.poll_interval,
            poll_timeout=args.poll_timeout
        )
        results['step3'] = {'ppa_results': ppa_results}

        stage_mgr.save_stage_checkpoint(stage, 'ppa_results', ppa_results)

    # =================================================================
    # Step 4: Run NSGA optimization
    # =================================================================
    if start_step <= 4:
        logger.info(f"[Stage {stage}] Step 4: NSGA optimization")

        ppa_evaluator = create_evaluator(config.to_dict())

        # Determine if using external simulation results
        use_external = ppa_results.get('mode') == 'hpc'

        if use_external:
            logger.info(f"[Stage {stage}] Using external simulation results")
            # Would need to implement lookup from combined results
            fitness_fn = ppa_evaluator.evaluate
        else:
            logger.info(f"[Stage {stage}] Using analytical PPA model")
            fitness_fn = ppa_evaluator.evaluate

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
        )

        tunables = config.get_tunables()

        optimizer = NSGA(
            config=nsga_config,
            tunables=tunables,
            evaluator=fitness_fn,
            logger=logger,
        )

        # Register checkpoint callback to save population after each generation
        def save_checkpoint(gen, population):
            stage_mgr.save_population_checkpoint(stage, gen, population)

        optimizer.config.checkpoint_fn = save_checkpoint

        start_time = time.time()
        pareto_front = optimizer.evolve()
        elapsed = time.time() - start_time

        logger.info(f"[Stage {stage}] Optimization completed in {elapsed:.2f}s")
        logger.info(f"[Stage {stage}] Pareto solutions found: {len(pareto_front)}")

        full_results = optimizer.get_results()

        # Save Pareto solutions for Stage 2
        stage_mgr.save_stage_checkpoint(stage, 'pareto_solutions', full_results)

        results['step4'] = full_results

    return results


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""

    # Main parser (just dispatches to subcommands)
    parser = argparse.ArgumentParser(
        description='SRAM DSO-MOGA: Two-Phase Multi-Objective Genetic Algorithm Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =================================================================
    # Common arguments (used by multiple subcommands)
    # =================================================================
    def add_config_arg(p: argparse.ArgumentParser, required=True):
        p.add_argument('--config', '-c',
                     required=required,
                     help='YAML configuration file')

    # =================================================================
    # 'run' subcommand - Execute optimization
    # =================================================================
    parser_run = subparsers.add_parser('run',
        help='Run optimization (Stage 1, Stage 2, or both)',
        description='Execute the two-phase MOGA optimization workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both stages (default)
  python run_moga.py run --config config.yaml

  # Run Stage 1 only (coarse search)
  python run_moga.py run --config config.yaml --stage 1

  # Run Stage 2 only (fine search, requires Stage 1 results)
  python run_moga.py run --config config.yaml --stage 2

  # Override algorithm parameters
  python run_moga.py run --config config.yaml --pop-size 100 --n-gen 80

  # Override sampling parameters
  python run_moga.py run --config config.yaml --coarse-samples 300 --fine-samples 20

  # Save log to file
  python run_moga.py run --config config.yaml --log
        """
    )

    # Common args for run
    add_config_arg(parser_run)
    parser_run.add_argument('--output-dir', '-o', default='./results',
                          help='Output directory (default: ./results)')
    parser_run.add_argument('--log', action='store_true',
                          help='Save log to file')
    parser_run.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose logging (DEBUG level)')

    # Stage selection
    stage_group = parser_run.add_mutually_exclusive_group()
    stage_group.add_argument('--stage', type=int, choices=[1, 2],
                            help='Run specific stage only (1=coarse, 2=fine)')
    stage_group.add_argument('--both', action='store_true',
                            help='Run both stages sequentially')

    # Sampling parameters
    parser_run.add_argument('--coarse-samples', type=int,
                           help='Number of samples for coarse search (Stage 1)')
    parser_run.add_argument('--fine-samples', type=int,
                           help='Number of fine samples per Pareto solution (Stage 2)')
    parser_run.add_argument('--variation', type=float, default=0.2,
                           help='Parameter variation fraction for fine search (default: 0.2 = +/-20%%)')

    # Simulation/HPC parameters
    parser_run.add_argument('--poll-interval', type=int, default=60,
                           help='Simulation polling interval in seconds (default: 60)')
    parser_run.add_argument('--poll-timeout', type=int, default=7200,
                           help='Simulation polling timeout in seconds (default: 7200)')

    # Algorithm parameters
    algo_group = parser_run.add_argument_group('algorithm',
        'Algorithm hyperparameters (override config file)')
    algo_group.add_argument('--pop-size', type=int,
                            help='Population size')
    algo_group.add_argument('--n-gen', type=int,
                            help='Number of generations')
    algo_group.add_argument('--seed', type=int,
                            help='Random seed for reproducibility')

    parser_run.set_defaults(func=_handle_run)

    # =================================================================
    # 'resume' subcommand - Resume from checkpoint
    # =================================================================
    parser_resume = subparsers.add_parser('resume',
        help='Resume optimization from checkpoint',
        description='Resume a crashed or interrupted optimization from the latest checkpoint')

    add_config_arg(parser_resume)
    parser_resume.add_argument('--output-dir', '-o', default='./results',
                              help='Output directory (default: ./results)')
    parser_resume.add_argument('--log', action='store_true',
                              help='Save log to file')
    parser_resume.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose logging (DEBUG level)')
    parser_resume.add_argument('--stage', type=int, choices=[1, 2], required=True,
                              help='Stage to resume (1 or 2)')
    parser_resume.add_argument('--gen', type=int,
                              help='Generation to resume from (default: latest)')

    parser_resume.set_defaults(func=_handle_resume)

    # =================================================================
    # 'export' subcommand - Export results
    # =================================================================
    parser_export = subparsers.add_parser('export',
        help='Export results to various formats',
        description='Export optimization results to JSON, CSV, or HTML dashboard')

    add_config_arg(parser_export)
    parser_export.add_argument('--results-dir', '-r', default='./results',
                              help='Results directory (default: ./results)')
    parser_export.add_argument('--stage', type=int, choices=[1, 2], default=2,
                              help='Which stage results to export (default: 2)')
    parser_export.add_argument('--format', '-f', nargs='+',
                              choices=['json', 'csv', 'html', 'all'],
                              default=['all'],
                              help='Export format(s)')

    parser_export.set_defaults(func=_handle_export)

    # =================================================================
    # 'dashboard' subcommand - Launch interactive dashboard
    # =================================================================
    parser_dash = subparsers.add_parser('dashboard',
        help='Launch interactive dashboard server',
        description='Start an interactive Plotly Dash dashboard to visualize optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch dashboard with Stage 2 results
  python run_moga.py dashboard --config config.yaml

  # Launch on custom port
  python run_moga.py dashboard --config config.yaml --port 8080

  # Use results from a different directory
  python run_moga.py dashboard --config config.yaml --results-dir ./my_results
        """
    )

    add_config_arg(parser_dash)
    parser_dash.add_argument('--results-dir', '-r', default='./results',
                           help='Results directory (default: ./results)')
    parser_dash.add_argument('--stage', type=int, choices=[1, 2], default=2,
                           help='Which stage results to display (default: 2)')
    parser_dash.add_argument('--port', type=int, default=8050,
                           help='Dashboard server port (default: 8050)')
    parser_dash.add_argument('--host', default='0.0.0.0',
                           help='Dashboard server host (default: 0.0.0.0)')
    parser_dash.add_argument('--debug', action='store_true',
                           help='Enable Dash debug mode')

    parser_dash.set_defaults(func=_handle_dashboard)

    # =================================================================
    # 'info' subcommand - Show configuration info
    # =================================================================
    parser_info = subparsers.add_parser('info',
        help='Show configuration and design space info',
        description='Display configuration summary and design space statistics')

    add_config_arg(parser_info)
    parser_info.add_argument('--show-tunables', action='store_true',
                             help='Show all tunable parameters and their options')

    parser_info.set_defaults(func=_handle_info)

    return parser


def _handle_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'run' subcommand."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    output_dir = Path(args.output_dir)
    stage_mgr = StageManager(output_dir)
    logger = setup_logging(
        output_dir / 'moga.log' if args.log else None,
        log_level
    )

    logger.info("=" * 60)
    logger.info("SRAM DSO-MOGA v2.1 (Two-Phase Optimization)")
    logger.info("=" * 60)

    try:
        # Load config
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")

        # Apply sampling overrides
        if args.coarse_samples:
            if 'sampling' not in config._data:
                config._data['sampling'] = {}
            config._data['sampling']['coarse_samples'] = args.coarse_samples

        if args.fine_samples:
            if 'sampling' not in config._data:
                config._data['sampling'] = {}
            config._data['sampling']['fine_samples_per_solution'] = args.fine_samples

        if args.variation:
            if 'sampling' not in config._data:
                config._data['sampling'] = {}
            config._data['sampling']['parameter_variation'] = args.variation

        # Apply algorithm overrides
        if args.pop_size:
            config._data['algorithm']['pop_size'] = args.pop_size
        if args.n_gen:
            config._data['algorithm']['n_gen'] = args.n_gen
        if args.seed:
            config._data['algorithm']['seed'] = args.seed

        # Run stages
        if args.stage:
            results = run_stage(args.stage, config, stage_mgr, args, logger)
        elif args.both:
            logger.info("Running Stage 1 + Stage 2 sequentially")
            results_s1 = run_stage(1, config, stage_mgr, args, logger)
            results_s2 = run_stage(2, config, stage_mgr, args, logger)
            results = {'stage1': results_s1, 'stage2': results_s2}
        else:
            # Default: run both stages
            logger.info("No --stage specified, running both stages")
            results_s1 = run_stage(1, config, stage_mgr, args, logger)
            results_s2 = run_stage(2, config, stage_mgr, args, logger)
            results = {'stage1': results_s1, 'stage2': results_s2}

        # Save final results
        if 'stage2' in results and 'step4' in results['stage2']:
            final_results = results['stage2']['step4']
            final_results['config'] = config.to_dict()
            export_all(final_results, output_dir / 'stage2')
            logger.info("Final results saved to stage2/")

        logger.info("\nOptimization completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def _handle_resume(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'resume' subcommand."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    output_dir = Path(args.output_dir)
    stage_mgr = StageManager(output_dir)
    logger = setup_logging(
        output_dir / 'moga.log' if args.log else None,
        log_level
    )

    # Find checkpoint to resume from
    if args.gen:
        gen_to_resume = args.gen
    else:
        gen_to_resume = stage_mgr.get_latest_population_gen(args.stage)

    if gen_to_resume is None:
        logger.error(f"No checkpoint found for Stage {args.stage}")
        logger.error("Run 'python run_moga.py run --config config.yaml --stage 1' first")
        sys.exit(1)

    logger.info(f"Found checkpoint: Stage {args.stage}, Generation {gen_to_resume}")

    # Load checkpoint
    checkpoint_data = stage_mgr.load_population_checkpoint(args.stage, gen_to_resume)
    if checkpoint_data:
        loaded_gen, population = checkpoint_data
        logger.info(f"Loaded {len(population)} individuals from generation {loaded_gen}")

    # TODO: Implement full resume logic
    # - Load config
    # - Set start_step to resume generation
    # - Continue NSGA evolution from checkpoint
    logger.warning("Full resume functionality is under development")
    logger.info("For now, you can re-run with the same config and --stage to start fresh")


def _handle_export(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'export' subcommand."""
    results_dir = Path(args.results_dir)
    stage_dir = results_dir / f'stage{args.stage}'

    results_file = stage_dir / 'results.json'
    if not results_file.exists():
        print(f"Error: Results not found at {results_file}")
        print(f"Run 'python run_moga.py run --config config.yaml --stage {args.stage}' first")
        sys.exit(1)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Import here to avoid circular imports
    from src.exporter import export_json, export_csv
    from src.dashboard import create_dashboard
    from src.config import load_config

    config = load_config(args.config)

    formats = args.format if args.format != ['all'] else ['json', 'csv', 'html']

    print("\n" + "=" * 60)
    print(f"Exporting Stage {args.stage} Results")
    print("=" * 60)

    if 'json' in formats:
        json_path = stage_dir / 'results.json'
        export_json(results, json_path)
        print(f"  JSON: {json_path.resolve()}")

    if 'csv' in formats:
        csv_path = stage_dir / 'pareto_solutions.csv'
        export_csv(
            results.get('pareto_solutions', []),
            results.get('pareto_objectives', []),
            csv_path
        )
        print(f"  CSV:  {csv_path.resolve()}")

    if 'html' in formats:
        # Create static dashboard HTML
        try:
            dash = create_dashboard(results, config.to_dict(), stage_dir)
            html_path = dash.save_html(stage_dir / 'dashboard.html')
            print(f"  HTML: {html_path.resolve()}")
        except Exception as e:
            print(f"  HTML: Failed - {e}")

    print(f"\nExport complete for Stage {args.stage} results")
    print(f"Output directory: {stage_dir.resolve()}")


def _handle_dashboard(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'dashboard' subcommand."""
    results_dir = Path(args.results_dir)
    stage_dir = results_dir / f'stage{args.stage}'

    results_file = stage_dir / 'results.json'
    if not results_file.exists():
        print(f"Error: Results not found at {results_file}")
        print(f"Run 'python run_moga.py run --config config.yaml --stage {args.stage}' first")
        sys.exit(1)

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    from src.dashboard import create_dashboard
    config = load_config(args.config)

    print("=" * 60)
    print("SRAM DSO-MOGA Interactive Dashboard")
    print("=" * 60)
    print(f"Results: {results_file}")
    print(f"Stage: {args.stage}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print()

    dash = create_dashboard(results, config.to_dict(), stage_dir)
    dash.run(port=args.port, host=args.host, debug=args.debug)


def _handle_info(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle the 'info' subcommand."""
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SRAM DSO-MOGA Configuration Info")
    print("=" * 60)

    # Basic info
    cfg_dict = config.to_dict()
    print(f"\nDesign: {cfg_dict.get('top_name', 'Unknown')}")
    print(f"Algorithm: {cfg_dict.get('algorithm', {}).get('name', 'NSGA-II')}")

    # Algorithm settings
    algo = cfg_dict.get('algorithm', {})
    print(f"\nAlgorithm Settings:")
    print(f"  Population Size: {algo.get('pop_size', 80)}")
    print(f"  Generations: {algo.get('n_gen', 60)}")
    print(f"  Seed: {algo.get('seed', 42)}")
    print(f"  Crossover Prob: {algo.get('crossover', {}).get('prob', 0.9)}")
    print(f"  Mutation Prob: {algo.get('mutation', {}).get('prob', 0.15)}")

    # Sampling settings
    sampling = cfg_dict.get('sampling', {})
    print(f"\nSampling Settings:")
    print(f"  Coarse Samples: {sampling.get('coarse_samples', 200)}")
    print(f"  Fine Samples Per Solution: {sampling.get('fine_samples_per_solution', 10)}")
    print(f"  Parameter Variation: {sampling.get('parameter_variation', 0.2) * 100}%")

    # Design space
    tunables = config.get_tunables()
    total_combos = config.get_total_combinations()
    print(f"\nDesign Space:")
    print(f"  Tunable Parameters: {len(tunables)}")
    print(f"  Total Combinations: {total_combos:,}")

    if args.show_tunables:
        print(f"\nTunable Parameters:")
        for name, options in tunables:
            print(f"  {name}: {len(options)} options -> {options}")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    args.func(args, parser)


if __name__ == '__main__':
    main()


