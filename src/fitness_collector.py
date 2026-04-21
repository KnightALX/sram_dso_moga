"""
SRAM DSO-MOGA: Fitness Collector

Collects PPA results from simulations or uses fallback analytical models.
Handles batch collection, error handling, and result aggregation.

Usage:
    collector = FitnessCollector(config, evaluator)
    fitness = collector.evaluate(combo_params)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import numpy as np


logger = logging.getLogger(__name__)


class FitnessCollector:
    """
    Collects fitness (PPA) values from simulation or analytical model.

    Supports two modes:
    1. External simulation: Uses SPF handler for real circuit simulation
    2. Analytical model: Uses built-in PPA evaluator when simulation unavailable

    When external simulation is used, results are cached to avoid re-running
    identical combinations.
    """

    def __init__(self, config: Dict[str, Any], evaluator: Callable):
        """
        Initialize fitness collector.

        Args:
            config: Configuration dictionary
            evaluator: Analytical PPA evaluator function (fallback)
        """
        self.config = config
        self.evaluator = evaluator
        self.use_simulation = False

        # Check if simulation is configured
        spf_path = config.get('spf_path')
        sim_config = config.get('simulation', {})
        if spf_path and sim_config.get('enabled', False):
            self.use_simulation = True
            logger.info("Using external simulation mode")
        else:
            logger.info("Using analytical PPA model")

        # Import here to avoid circular imports
        try:
            from .spf_handler import create_spf_session
            self.spf_session = create_spf_session(config)
        except Exception as e:
            logger.warning(f"Failed to create SPF session: {e}")
            self.spf_session = None

        # Cache for simulation results
        self._result_cache: Dict[str, Dict[str, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, combo_params: Dict[str, Any], output_dir: Optional[Path] = None) -> List[float]:
        """
        Evaluate PPA for a given combination.

        Args:
            combo_params: Parameter combination dictionary
            output_dir: Directory for simulation outputs (if using external sim)

        Returns:
            [area, power, delay] objectives
        """
        if self.use_simulation and self.spf_session is not None:
            result = self._evaluate_simulation(combo_params, output_dir)
            if result is not None:
                return [result['area'], result['power'], result['delay']]

        # Fallback to analytical model
        return self.evaluator(combo_params)

    def _evaluate_simulation(self, combo_params: Dict[str, Any],
                             output_dir: Optional[Path]) -> Optional[Dict[str, float]]:
        """
        Run external simulation and parse results.

        Args:
            combo_params: Parameter combination
            output_dir: Output directory

        Returns:
            PPA dict or None if simulation failed
        """
        # Create cache key from combo_params
        cache_key = self._make_cache_key(combo_params)

        # Check cache
        if cache_key in self._result_cache:
            self._cache_hits += 1
            return self._result_cache[cache_key]

        self._cache_misses += 1

        # Run simulation
        try:
            from .spf_handler import run_simulation
            result = run_simulation(combo_params, self.config, output_dir or Path('./sim_output'))

            if result is not None:
                self._result_cache[cache_key] = result
                return result
            else:
                logger.warning("Simulation returned None, using analytical model")
                return None

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return None

    def _make_cache_key(self, combo_params: Dict[str, Any]) -> str:
        """Create a hashable cache key from combo_params."""
        # Sort items for consistent ordering
        sorted_items = sorted(combo_params.items())
        return str(sorted_items)

    def evaluate_batch(self, combo_list: List[Dict[str, Any]],
                       output_dir: Optional[Path] = None) -> List[List[float]]:
        """
        Evaluate PPA for a batch of combinations.

        Args:
            combo_list: List of parameter combinations
            output_dir: Output directory for simulations

        Returns:
            List of [area, power, delay] objective arrays
        """
        results = []
        for i, combo in enumerate(combo_list):
            try:
                ppa = self.evaluate(combo, output_dir)
                results.append(ppa)
            except Exception as e:
                logger.error(f"Failed to evaluate combo {i}: {e}")
                # Use worst-case values as fallback
                results.append([float('inf'), float('inf'), float('inf')])

        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': self._cache_hits + self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }

    def clear_cache(self) -> None:
        """Clear the simulation result cache."""
        self._result_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


def create_fitness_collector(config: Dict[str, Any],
                             evaluator: Callable) -> FitnessCollector:
    """
    Factory function to create fitness collector.

    Args:
        config: Configuration dictionary
        evaluator: Analytical PPA evaluator function

    Returns:
        FitnessCollector instance
    """
    return FitnessCollector(config, evaluator)


class CombinationGenerator:
    """
    Generates parameter combinations from YAML configuration.

    Supports:
    - Full enumeration (when total combinations <= max_combo)
    - LHS sampling (when combinations exceed max_combo)
    - Boundary + midpoint + LHS hybrid sampling
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize combination generator.

        Args:
            config: Configuration dictionary with groups and active_bundles
        """
        self.config = config
        self.tunables = self._extract_tunables()
        self.total_combinations = self._calculate_total()
        self.max_combo = config.get('max_combo', 10000)

    def _extract_tunables(self) -> List[tuple]:
        """Extract (param_name, options_list) from config."""
        tunables = []
        active_bundles = set(self.config.get('active_bundles', []))

        for group in self.config.get('groups', []):
            if group.get('bundle_flag') not in active_bundles:
                continue

            for device in group.get('devices', []):
                base = device['name']
                tunables.append((f"{base}_vt", device.get('vt_options', [])))
                tunables.append((f"{base}_gl", device.get('gl_options', [])))
                tunables.append((f"{base}_nfin", device.get('nfin_options', [])))

        return tunables

    def _calculate_total(self) -> int:
        """Calculate total number of possible combinations."""
        total = 1
        for _, options in self.tunables:
            total *= len(options)
        return total

    def generate_all(self) -> List[Dict[str, Any]]:
        """
        Generate all possible combinations.

        Returns:
            List of parameter dictionaries
        """
        import itertools

        options_lists = [opts for _, opts in self.tunables]
        param_names = [name for name, _ in self.tunables]

        combinations = []
        for combo in itertools.product(*options_lists):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def generate_sampled(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Generate sampled combinations using LHS + boundary + midpoint.

        Args:
            n_samples: Target number of samples

        Returns:
            List of sampled parameter dictionaries
        """
        import random

        options_lists = [opts for _, opts in self.tunables]
        param_names = [name for name, _ in self.tunables]
        n_dims = len(options_lists)

        sampled = []

        # Add boundary combinations (all min, all max)
        bounds = [[min(o), max(o)] for o in options_lists]
        for corner in itertools.product(*bounds):
            sampled.append(dict(zip(param_names, corner)))

        # Add midpoint
        mid = []
        for opts in options_lists:
            mid.append(opts[len(opts) // 2])
        sampled.append(dict(zip(param_names, mid)))

        # Generate LHS samples
        n_lhs = max(0, n_samples - len(sampled))
        if n_lhs > 0:
            random_indices = np.random.randint(0, 9999, size=(n_lhs, n_dims))
            for row in random_indices:
                # Sort to ensure LHS property
                sorted_row = np.argsort(row)
                combo = tuple(options_lists[d][sorted_row[d] % len(options_lists[d])]
                              for d in range(n_dims))
                sampled.append(dict(zip(param_names, combo)))

        return sampled[:n_samples]

    def generate(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate combinations based on configuration.

        Args:
            n_samples: Target sample count (from combo_count in config)

        Returns:
            List of parameter dictionaries
        """
        if n_samples is None:
            # Get from config
            n_samples = self.config.get('sampling', {}).get('combo_count', 200)

        if self.total_combinations <= self.max_combo:
            # Full enumeration
            return self.generate_all()
        else:
            # Sampled
            return self.generate_sampled(n_samples)


# Add missing import at top of file
import itertools