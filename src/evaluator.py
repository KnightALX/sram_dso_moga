"""
SRAM DSO-MOGA: PPA Evaluation models.

Provides evaluation functions for Area, Power, Delay objectives
based on device parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class PPAEvaluator:
    """
    SRAM Peripheral PPA (Power, Performance, Area) evaluator.

    Uses simplified analytical models based on device parameters:
    - nfin (number of fins): affects drive strength, leakage, area
    - gl (gate length): affects switching speed, leakage
    - vt (threshold voltage): affects speed vs power tradeoff
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ppa_config = config.get('ppa_model', {})
        self._init_coefficients()

    def _init_coefficients(self) -> None:
        """Initialize PPA model coefficients from config."""
        # Area model: A = nfin_sum * nfin_coef + gl_mean * gl_coef + base
        area_cfg = self.ppa_config.get('area', {})
        self.area_nfin_coef = area_cfg.get('nfin_coef', 0.15)
        self.area_gl_coef = area_cfg.get('gl_coef', 120.0)
        self.area_base = area_cfg.get('base', 0.8)

        # Power model: P = nfin_sum * nfin_coef * vt_penalty + base
        power_cfg = self.ppa_config.get('power', {})
        self.power_nfin_coef = power_cfg.get('nfin_coef', 12.5)
        self.power_base = power_cfg.get('base', 5.0)
        self.vt_penalty = power_cfg.get('vt_penalty', {'vt0': 1.0, 'vt1': 0.8, 'vt2': 1.2})

        # Delay model: D = 180/(nfin_sum * nfin_coef + eps) + gl_mean * gl_coef
        delay_cfg = self.ppa_config.get('delay', {})
        self.delay_nfin_coef = delay_cfg.get('nfin_coef', 0.9)
        self.delay_gl_coef = delay_cfg.get('gl_coef', 800.0)

    def evaluate(self, config: Dict[str, Any]) -> List[float]:
        """
        Evaluate PPA for a given device configuration.

        Args:
            config: Dictionary of {param_name: value} for all tunables

        Returns:
            [area, power, delay] objectives
        """
        area = self._calc_area(config)
        power = self._calc_power(config)
        delay = self._calc_delay(config)

        return [area, power, delay]

    def _calc_area(self, config: Dict[str, Any]) -> float:
        """Calculate area from configuration."""
        nfin_sum = self._get_nfin_sum(config)
        gl_mean = self._get_gl_mean(config)
        return nfin_sum * self.area_nfin_coef + gl_mean * self.area_gl_coef + self.area_base

    def _calc_power(self, config: Dict[str, Any]) -> float:
        """Calculate power from configuration."""
        nfin_sum = self._get_nfin_sum(config)
        vt_penalty = self._get_vt_penalty(config)
        return nfin_sum * self.power_nfin_coef * vt_penalty + self.power_base

    def _calc_delay(self, config: Dict[str, Any]) -> float:
        """Calculate delay from configuration."""
        nfin_sum = self._get_nfin_sum(config)
        gl_mean = self._get_gl_mean(config)
        eps = 1e-6  # Avoid division by zero
        return 180.0 / (nfin_sum * self.delay_nfin_coef + eps) + gl_mean * self.delay_gl_coef

    def _get_nfin_sum(self, config: Dict[str, Any]) -> float:
        """Sum of all nfin values."""
        nfin_sum = 0.0
        for key, val in config.items():
            if key.endswith('_nfin') and isinstance(val, (int, float)):
                nfin_sum += val
        return nfin_sum

    def _get_gl_mean(self, config: Dict[str, Any]) -> float:
        """Mean of all gl (gate length) values."""
        gl_vals = []
        for key, val in config.items():
            if key.endswith('_gl') and isinstance(val, (int, float)):
                gl_vals.append(val)
        return np.mean(gl_vals) if gl_vals else 0.025

    def _get_vt_penalty(self, config: Dict[str, Any]) -> float:
        """Calculate VT penalty factor."""
        total_penalty = 0.0
        count = 0
        for key, val in config.items():
            if key.endswith('_vt') and isinstance(val, str):
                penalty = self.vt_penalty.get(val, 1.0)
                total_penalty += penalty
                count += 1
        return total_penalty / count if count > 0 else 1.0

    def get_objective_names(self) -> List[str]:
        """Return objective names in order."""
        return ['Area', 'Power', 'Delay']

    def get_objective_labels(self) -> List[str]:
        """Return display labels for objectives."""
        return ['Area (um²)', 'Power (uW)', 'Delay (ps)']

    def get_directions(self) -> List[str]:
        """Return optimization directions ('minimize' or 'maximize')."""
        return ['minimize', 'minimize', 'minimize']


def create_evaluator(config: Dict[str, Any]) -> PPAEvaluator:
    """Factory function to create evaluator from config."""
    return PPAEvaluator(config)