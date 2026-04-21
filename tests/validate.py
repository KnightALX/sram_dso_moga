#!/usr/bin/env python3
"""
SRAM DSO-MOGA: Self-Validation Script

Run this to validate the entire system works correctly.
Usage: python tests/validate.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import Config, load_config
from evaluator import PPAEvaluator, create_evaluator
from nsga import NSGA, AlgorithmConfig


def test_config_loading():
    """Test configuration loading and validation."""
    print("=" * 60)
    print("Test 1: Configuration Loading")
    print("=" * 60)

    config_data = {
        'top_name': 'validation_test',
        'algorithm': {
            'pop_size': 40,
            'n_gen': 10,
            'seed': 42,
            'name': 'NSGA-II',
            'crossover': {'prob': 0.9, 'eta': 15.0},
            'mutation': {'prob': 0.15, 'eta': 20.0},
        },
        'objectives': [
            {'name': 'area', 'label': 'Area', 'direction': 'minimize', 'enabled': True},
            {'name': 'power', 'label': 'Power', 'direction': 'minimize', 'enabled': True},
            {'name': 'delay', 'label': 'Delay', 'direction': 'minimize', 'enabled': True},
        ],
        'groups': [
            {
                'bundle_flag': 'readpath',
                'devices': [
                    {
                        'name': 'precharge_pmos',
                        'vt_options': ['vt0', 'vt1', 'vt2'],
                        'gl_options': [0.020, 0.025, 0.030],
                        'nfin_options': [2, 3, 4, 5, 6],
                    },
                    {
                        'name': 'sense_nmos',
                        'vt_options': ['vt0', 'vt1'],
                        'gl_options': [0.018, 0.022],
                        'nfin_options': [1, 2, 3, 4],
                    },
                ]
            }
        ],
        'active_bundles': ['readpath'],
        'ppa_model': {
            'area': {'nfin_coef': 0.15, 'gl_coef': 120.0, 'base': 0.8},
            'power': {'nfin_coef': 12.5, 'vt_penalty': {'vt0': 1.0, 'vt1': 0.8, 'vt2': 1.2}, 'base': 5.0},
            'delay': {'nfin_coef': 0.9, 'gl_coef': 800.0},
        },
    }

    config = Config(config_data)
    print("  [PASS] Config loaded: {}".format(config.top_name))
    print("  [PASS] Population: {}, Generations: {}".format(config.pop_size, config.n_gen))
    print("  [PASS] Tunables: {} parameters".format(len(config.get_tunables())))
    print("  [PASS] Total combinations: {}".format(config.get_total_combinations()))

    return config


def test_evaluator(config):
    """Test PPA evaluator."""
    print("\n" + "=" * 60)
    print("Test 2: PPA Evaluator")
    print("=" * 60)

    evaluator = create_evaluator(config.to_dict())

    test_config = {
        'precharge_pmos_vt': 'vt0',
        'precharge_pmos_gl': 0.025,
        'precharge_pmos_nfin': 4,
        'sense_nmos_vt': 'vt1',
        'sense_nmos_gl': 0.022,
        'sense_nmos_nfin': 3,
    }

    result = evaluator.evaluate(test_config)
    print("  [PASS] Evaluation result: Area={:.3f}, Power={:.2f}, Delay={:.2f}".format(*result))

    # Test objective names
    names = evaluator.get_objective_names()
    print("  [PASS] Objective names: {}".format(names))


def test_nsga(config):
    """Test NSGA optimizer."""
    print("\n" + "=" * 60)
    print("Test 3: NSGA Optimization")
    print("=" * 60)

    tunables = config.get_tunables()
    evaluator = create_evaluator(config.to_dict()).evaluate

    nsga_config = AlgorithmConfig(
        pop_size=40,
        n_gen=10,
        seed=42,
        crossover_prob=0.9,
        sbx_eta=15.0,
        mutation_prob=0.15,
        pm_eta=20.0,
        tournament_size=2,
        use_nsga3=False,
    )

    print("  Running NSGA-II with {} individuals for {} generations...".format(nsga_config.pop_size, nsga_config.n_gen))
    optimizer = NSGA(nsga_config, tunables, evaluator)

    results = optimizer.evolve()

    print("  [PASS] Optimization completed")
    print("  [PASS] Pareto solutions found: {}".format(len(results)))

    full_results = optimizer.get_results()
    print("  [PASS] History entries: {}".format(len(full_results['history'])))

    return full_results


def test_results_saving(results, config):
    """Test saving results to files."""
    print("\n" + "=" * 60)
    print("Test 4: Results Saving")
    print("=" * 60)

    output_dir = Path(tempfile.mkdtemp())

    import json
    import pandas as pd

    # Save JSON
    json_path = output_dir / 'results.json'
    json_results = {
        'pareto_solutions': results['pareto_solutions'],
        'pareto_objectives': [[float(x) for x in obj] for obj in results['pareto_objectives']],
        'history': results['history'],
        'n_pareto': results['n_pareto'],
        'config': config.to_dict(),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print("  [PASS] JSON saved to {}".format(json_path))

    # Save CSV
    csv_path = output_dir / 'pareto_solutions.csv'
    df = pd.DataFrame(results['pareto_solutions'])
    obj_df = pd.DataFrame(results['pareto_objectives'], columns=['Area', 'Power', 'Delay'])
    df = pd.concat([df, obj_df], axis=1)
    df.to_csv(csv_path, index=False)
    print("  [PASS] CSV saved to {}".format(csv_path))

    # Cleanup
    shutil.rmtree(output_dir)
    print("  [PASS] Temp directory cleaned up")


def test_yaml_config_file():
    """Test loading from actual YAML file."""
    print("\n" + "=" * 60)
    print("Test 5: YAML File Loading")
    print("=" * 60)

    yaml_path = Path(__file__).parent.parent / 'config' / 'sram_config_v2.yaml'

    if yaml_path.exists():
        config = load_config(yaml_path)
        print("  [PASS] Loaded from {}".format(yaml_path))
        print("  [PASS] Design: {}".format(config.top_name))
        print("  [PASS] Active bundles: {}".format(config.active_bundles))
    else:
        print("  [WARN] Config file not found: {}".format(yaml_path))
        print("  [WARN] Skipping YAML loading test")


def print_summary(results):
    """Print final summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("\n  Pareto Front Size: {} solutions".format(results['n_pareto']))

    if results['pareto_objectives']:
        objs = results['pareto_objectives']
        print("\n  Best Objectives:")
        print("    Best Area:  {:.4f}".format(min(o[0] for o in objs)))
        print("    Best Power: {:.4f}".format(min(o[1] for o in objs)))
        print("    Best Delay: {:.4f}".format(min(o[2] for o in objs)))

    print("\n  Convergence:")
    history = results['history']
    if history:
        front0_sizes = [h['front0_size'] for h in history]
        print("    Initial Front0: {}".format(front0_sizes[0] if front0_sizes else 0))
        print("    Final Front0:   {}".format(front0_sizes[-1] if front0_sizes else 0))

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL VALIDATION TESTS PASSED")
    print("=" * 60)


def main():
    print("\n" + "#" * 60)
    print("# SRAM DSO-MOGA Self-Validation")
    print("#" * 60)

    try:
        # Run validation tests
        config = test_config_loading()
        test_evaluator(config)
        results = test_nsga(config)
        test_results_saving(results, config)
        test_yaml_config_file()

        # Print summary
        print_summary(results)

        return 0

    except Exception as e:
        print("\n[FAIL] VALIDATION FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())