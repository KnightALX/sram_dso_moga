"""
SRAM DSO-MOGA: Configuration and Evaluator Tests

Run with: python -m pytest tests/test_config_eval.py -v
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config, ConfigValidationError, load_config
from evaluator import PPAEvaluator, create_evaluator


class TestConfig:
    """Test configuration management."""

    def test_load_valid_config(self):
        """Test loading a valid config file."""
        config_data = {
            'top_name': 'test_design',
            'algorithm': {
                'pop_size': 50,
                'n_gen': 30,
                'seed': 123,
            },
            'objectives': [
                {'name': 'area', 'direction': 'minimize', 'enabled': True},
                {'name': 'power', 'direction': 'minimize', 'enabled': True},
            ],
            'groups': [
                {
                    'bundle_flag': 'test_group',
                    'devices': [
                        {
                            'name': 'test_device',
                            'vt_options': ['vt0', 'vt1'],
                            'gl_options': [0.020, 0.025],
                            'nfin_options': [2, 4, 6],
                        }
                    ]
                }
            ],
            'active_bundles': ['test_group'],
        }

        config = Config(config_data)
        assert config.top_name == 'test_design'
        assert config.pop_size == 50
        assert config.n_gen == 30
        assert config.seed == 123

    def test_missing_required_field(self):
        """Test that missing required fields raise error."""
        config_data = {
            'top_name': 'test',
            # Missing algorithm, objectives, groups, active_bundles
        }

        with pytest.raises(ConfigValidationError) as exc:
            Config(config_data)

        assert 'Missing required field' in str(exc.value)

    def test_unknown_bundle_flag(self):
        """Test that unknown bundle flags raise error."""
        config_data = {
            'top_name': 'test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {'bundle_flag': 'group1', 'devices': [
                    {'name': 'd1', 'vt_options': ['a'], 'gl_options': [0.02], 'nfin_options': [2]}
                ]}
            ],
            'active_bundles': ['unknown_group'],
        }

        with pytest.raises(ConfigValidationError) as exc:
            Config(config_data)

        assert 'Unknown bundle_flags' in str(exc.value)

    def test_empty_devices(self):
        """Test that empty device list raises error."""
        config_data = {
            'top_name': 'test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {'bundle_flag': 'group1', 'devices': []}  # Empty!
            ],
            'active_bundles': ['group1'],
        }

        with pytest.raises(ConfigValidationError) as exc:
            Config(config_data)

        assert 'no devices' in str(exc.value)

    def test_empty_options(self):
        """Test that empty option arrays raise error."""
        config_data = {
            'top_name': 'test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {'bundle_flag': 'group1', 'devices': [
                    {'name': 'd1', 'vt_options': [], 'gl_options': [0.02], 'nfin_options': [2]}
                ]}
            ],
            'active_bundles': ['group1'],
        }

        with pytest.raises(ConfigValidationError) as exc:
            Config(config_data)

        assert "missing or empty 'vt_options'" in str(exc.value)

    def test_get_tunables(self):
        """Test getting tunable parameters."""
        config_data = {
            'top_name': 'test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {
                    'bundle_flag': 'group1',
                    'devices': [
                        {'name': 'dev1', 'vt_options': ['vt0', 'vt1'],
                         'gl_options': [0.02], 'nfin_options': [2, 4]},
                    ]
                }
            ],
            'active_bundles': ['group1'],
        }

        config = Config(config_data)
        tunables = config.get_tunables()

        assert len(tunables) == 3  # vt, gl, nfin for one device
        assert tunables[0] == ('dev1_vt', ['vt0', 'vt1'])
        assert tunables[1] == ('dev1_gl', [0.02])
        assert tunables[2] == ('dev1_nfin', [2, 4])

    def test_get_total_combinations(self):
        """Test combination count calculation."""
        config_data = {
            'top_name': 'test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {
                    'bundle_flag': 'group1',
                    'devices': [
                        {'name': 'dev1',
                         'vt_options': ['vt0', 'vt1', 'vt2'],  # 3
                         'gl_options': [0.020, 0.025],         # 2
                         'nfin_options': [2, 4, 6, 8]},       # 4
                    ]
                }
            ],
            'active_bundles': ['group1'],
        }

        config = Config(config_data)
        total = config.get_total_combinations()

        # 3 * 2 * 4 = 24 combinations
        assert total == 24

    def test_from_yaml_file(self):
        """Test loading config from YAML file."""
        config_data = {
            'top_name': 'yaml_test',
            'algorithm': {'pop_size': 50, 'n_gen': 30},
            'objectives': [{'name': 'area', 'enabled': True}],
            'groups': [
                {
                    'bundle_flag': 'g1',
                    'devices': [
                        {'name': 'd1', 'vt_options': ['a'], 'gl_options': [0.02], 'nfin_options': [2]}
                    ]
                }
            ],
            'active_bundles': ['g1'],
        }

        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.top_name == 'yaml_test'
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')


class TestPPAEvaluator:
    """Test PPA evaluator."""

    def test_basic_evaluation(self):
        """Test basic PPA evaluation."""
        config_data = {
            'ppa_model': {
                'area': {'nfin_coef': 0.15, 'gl_coef': 120.0, 'base': 0.8},
                'power': {'nfin_coef': 12.5, 'vt_penalty': {'vt0': 1.0, 'vt1': 0.8}, 'base': 5.0},
                'delay': {'nfin_coef': 0.9, 'gl_coef': 800.0},
            }
        }

        evaluator = PPAEvaluator(config_data)
        cfg = {
            'dev1_nfin': 4,
            'dev2_nfin': 2,
            'dev1_gl': 0.025,
            'dev2_gl': 0.022,
            'dev1_vt': 'vt0',
            'dev2_vt': 'vt1',
        }

        result = evaluator.evaluate(cfg)

        assert len(result) == 3
        assert result[0] > 0  # Area should be positive
        assert result[1] > 0  # Power should be positive
        assert result[2] > 0  # Delay should be positive

    def test_default_coefficients(self):
        """Test that missing coefficients use defaults."""
        evaluator = PPAEvaluator({})

        cfg = {
            'dev1_nfin': 4,
            'dev1_gl': 0.025,
            'dev1_vt': 'vt0',
        }

        result = evaluator.evaluate(cfg)

        # Should not crash, should return valid numbers
        assert len(result) == 3
        assert all(r > 0 for r in result)

    def test_empty_config(self):
        """Test evaluation with empty config (no devices)."""
        evaluator = PPAEvaluator({})

        cfg = {}  # No device parameters

        result = evaluator.evaluate(cfg)

        # Should use defaults for missing values
        assert len(result) == 3
        # Area will use default gl_mean = 0.025
        assert result[0] > 0

    def test_vt_penalty(self):
        """Test that VT penalty affects power calculation."""
        evaluator = PPAEvaluator({
            'ppa_model': {
                'power': {'nfin_coef': 10.0, 'vt_penalty': {'vt0': 1.0, 'vt1': 0.5}, 'base': 0},
            }
        })

        cfg_low_vt = {'dev1_nfin': 2, 'dev1_vt': 'vt0'}
        cfg_high_vt = {'dev1_nfin': 2, 'dev1_vt': 'vt1'}

        power_low = evaluator.evaluate(cfg_low_vt)[1]
        power_high = evaluator.evaluate(cfg_high_vt)[1]

        # vt1 has lower penalty (0.5 vs 1.0), so power should be lower
        assert power_low > power_high

    def test_create_evaluator_factory(self):
        """Test evaluator factory function."""
        config = {'ppa_model': {}}
        evaluator = create_evaluator(config)

        assert isinstance(evaluator, PPAEvaluator)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])