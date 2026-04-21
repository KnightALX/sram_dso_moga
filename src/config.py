"""
SRAM DSO-MOGA: Configuration management module.

Handles YAML configuration loading, validation, and schema checking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class Config:
    """Configuration container with validation and accessors."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(data)

    def _validate(self) -> None:
        """Validate required fields and structure."""
        required = ['top_name', 'algorithm', 'objectives', 'groups', 'active_bundles']
        for key in required:
            if key not in self._data:
                raise ConfigValidationError(f"Missing required field: {key}")

        # Validate algorithm section
        algo = self._data.get('algorithm', {})
        for field in ['pop_size', 'n_gen']:
            if field not in algo:
                raise ConfigValidationError(f"Missing algorithm.{field}")

        # Validate at least one objective
        objectives = self._data.get('objectives', [])
        if not objectives:
            raise ConfigValidationError("At least one objective is required")

        # Validate active_bundles reference existing groups
        active = set(self._data.get('active_bundles', []))
        group_flags = {g['bundle_flag'] for g in self._data.get('groups', [])}
        unknown = active - group_flags
        if unknown:
            raise ConfigValidationError(f"Unknown bundle_flags: {unknown}")

        # Validate each group has devices
        for group in self._data.get('groups', []):
            devices = group.get('devices', [])
            if not devices:
                raise ConfigValidationError(f"Group '{group.get('bundle_flag')}' has no devices")
            for dev in devices:
                for opt_key in ['vt_options', 'gl_options', 'nfin_options']:
                    if opt_key not in dev or not dev[opt_key]:
                        raise ConfigValidationError(
                            f"Device '{dev.get('name')}' missing or empty '{opt_key}'"
                        )

    @property
    def top_name(self) -> str:
        return self._data.get('top_name', 'unknown')

    @property
    def spf_path(self) -> Optional[str]:
        return self._data.get('spf_path')

    @property
    def algorithm(self) -> Dict[str, Any]:
        return self._data.get('algorithm', {})

    @property
    def pop_size(self) -> int:
        return self.algorithm.get('pop_size', 80)

    @property
    def n_gen(self) -> int:
        return self.algorithm.get('n_gen', 60)

    @property
    def seed(self) -> int:
        return self.algorithm.get('seed', 42)

    @property
    def algo_name(self) -> str:
        return self.algorithm.get('name', 'NSGA-II')

    @property
    def objectives(self) -> List[Dict[str, Any]]:
        return [o for o in self._data.get('objectives', []) if o.get('enabled', True)]

    @property
    def active_bundles(self) -> List[str]:
        return self._data.get('active_bundles', [])

    @property
    def groups(self) -> List[Dict[str, Any]]:
        return self._data.get('groups', [])

    @property
    def active_groups(self) -> List[Dict[str, Any]]:
        """Groups filtered by active_bundles."""
        active = set(self.active_bundles)
        return [g for g in self.groups if g.get('bundle_flag') in active]

    @property
    def ppa_model(self) -> Dict[str, Any]:
        return self._data.get('ppa_model', {})

    @property
    def output_dir(self) -> Path:
        return Path(self._data.get('output', {}).get('dir', './results'))

    def get_tunables(self) -> List[Tuple[str, List[Any]]]:
        """Get list of (parameter_name, options) tuples for active devices."""
        tunables = []
        for group in self.active_groups:
            for dev in group.get('devices', []):
                base = dev['name']
                tunables.append((f"{base}_vt", dev.get('vt_options', [])))
                tunables.append((f"{base}_gl", dev.get('gl_options', [])))
                tunables.append((f"{base}_nfin", dev.get('nfin_options', [])))
        return tunables

    def get_total_combinations(self) -> int:
        """Calculate total design space size."""
        tunables = self.get_tunables()
        total = 1
        for _, opts in tunables:
            total *= len(opts)
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Return raw configuration dictionary."""
        return self._data.copy()


def load_config(path: str | Path) -> Config:
    """Convenience function to load configuration."""
    return Config.from_yaml(path)