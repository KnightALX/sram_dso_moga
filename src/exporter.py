"""
SRAM DSO-MOGA: Results Export Module

Exports optimization results to various formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


def export_json(results: Dict[str, Any], output_path: Path) -> None:
    """
    Export results to JSON format.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    def make_serializable(obj):
        """Convert numpy and other non-standard types to native Python types."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    serializable_results = make_serializable(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def export_csv(pareto_solutions: List[Dict[str, Any]],
               pareto_objectives: List[List[float]],
               output_path: Path) -> None:
    """
    Export Pareto solutions to CSV.

    Args:
        pareto_solutions: List of parameter configurations
        pareto_objectives: List of [area, power, delay] arrays
        output_path: Output file path
    """
    df_params = pd.DataFrame(pareto_solutions)
    df_obj = pd.DataFrame(pareto_objectives, columns=['Area', 'Power', 'Delay'])
    df = pd.concat([df_params, df_obj], axis=1)

    df.to_csv(output_path, index=False)


def export_summary(results: Dict[str, Any], output_path: Path) -> None:
    """
    Export a human-readable summary report.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SRAM DSO-MOGA Optimization Summary")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    config = results.get('config', {})
    lines.append(f"Design: {config.get('top_name', 'Unknown')}")
    lines.append(f"Algorithm: {config.get('algorithm', {}).get('name', 'NSGA-II')}")
    lines.append(f"Population: {config.get('algorithm', {}).get('pop_size', 'N/A')}")
    lines.append(f"Generations: {config.get('algorithm', {}).get('n_gen', 'N/A')}")
    lines.append("")

    # Results
    n_pareto = results.get('n_pareto', 0)
    lines.append(f"Pareto Solutions Found: {n_pareto}")
    lines.append("")

    # Best values
    objectives = results.get('pareto_objectives', [])
    if objectives:
        areas = [o[0] for o in objectives if len(o) >= 1]
        powers = [o[1] for o in objectives if len(o) >= 2]
        delays = [o[2] for o in objectives if len(o) >= 3]

        lines.append("Best Objectives:")
        lines.append(f"  Best Area:  {min(areas):.4f}")
        lines.append(f"  Best Power: {min(powers):.4f}")
        lines.append(f"  Best Delay: {min(delays):.4f}")
        lines.append("")

    # Convergence
    history = results.get('history', [])
    if history:
        front0_sizes = [h.get('front0_size', 0) for h in history]
        lines.append(f"Convergence: Front0 size {front0_sizes[0]} -> {front0_sizes[-1]}")
        lines.append("")

    lines.append("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_all(results: Dict[str, Any], output_dir: Path,
               include_summary: bool = True) -> Dict[str, Path]:
    """
    Export all result formats.

    Args:
        results: Results dictionary
        output_dir: Output directory
        include_summary: Whether to generate summary report

    Returns:
        Dictionary mapping format name to output file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # JSON
    json_path = output_dir / 'results.json'
    export_json(results, json_path)
    paths['json'] = json_path

    # CSV
    csv_path = output_dir / 'pareto_solutions.csv'
    export_csv(
        results.get('pareto_solutions', []),
        results.get('pareto_objectives', []),
        csv_path
    )
    paths['csv'] = csv_path

    # Summary
    if include_summary:
        summary_path = output_dir / 'summary.txt'
        export_summary(results, summary_path)
        paths['summary'] = summary_path

    return paths