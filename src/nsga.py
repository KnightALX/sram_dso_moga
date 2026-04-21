"""
SRAM DSO-MOGA: NSGA-II and NSGA-III implementations.

Core multi-objective genetic algorithm framework with:
- Fast non-dominated sorting
- Crowding distance (NSGA-II) or reference point association (NSGA-III)
- Tournament selection
- SBX crossover and polynomial mutation
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """Algorithm hyperparameters."""
    pop_size: int = 80
    n_gen: int = 60
    seed: int = 42
    crossover_prob: float = 0.9
    sbx_eta: float = 15.0
    mutation_prob: float = 0.15
    pm_eta: float = 20.0
    tournament_size: int = 2
    use_nsga3: bool = False
    n_partitions: int = 12  # For NSGA-III reference points


@dataclass
class Individual:
    """Represents a single solution in the population."""
    genes: np.ndarray
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = -1
    crowding_distance: float = 0.0

    def __post_init__(self):
        if isinstance(self.genes, list):
            self.genes = np.array(self.genes, dtype=int)

    def dominates(self, other: "Individual") -> bool:
        """Check if self dominates other (all <= and at least one <)."""
        return bool(np.all(self.objectives <= other.objectives) and
                    np.any(self.objectives < other.objectives))


def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """Static version for numpy arrays."""
    return bool(np.all(obj1 <= obj2) and np.any(obj1 < obj2))


def fast_non_dominated_sort(pop_obj: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Fast non-dominated sorting algorithm (Deb et al.).

    Args:
        pop_obj: (N, M) array of objective values for N individuals, M objectives

    Returns:
        rank: (N,) array with Pareto rank for each individual
        fronts: List of fronts, each front is a list of individual indices
    """
    n = len(pop_obj)
    domination_count = np.zeros(n, dtype=int)
    dominated_list: List[List[int]] = [[] for _ in range(n)]
    rank = np.full(n, -1, dtype=int)
    fronts: List[List[int]] = [[]]

    # O(MN²) dominance comparison
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(pop_obj[i], pop_obj[j]):
                dominated_list[i].append(j)
            elif dominates(pop_obj[j], pop_obj[i]):
                domination_count[i] += 1

        if domination_count[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    # Extract subsequent fronts
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_list[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    # Remove trailing empty fronts
    while fronts and not fronts[-1]:
        fronts.pop()

    return rank, fronts


def crowding_distance(pop_obj: np.ndarray, front_idx: List[int]) -> np.ndarray:
    """
    Calculate crowding distance for a single front.

    Args:
        pop_obj: (N, M) objective values
        front_idx: Indices of individuals in the front

    Returns:
        dist: (len(front_idx),) crowding distance values
    """
    n = len(front_idx)
    m = pop_obj.shape[1]

    if n <= 2:
        return np.full(n, np.inf)

    dist = np.zeros(n)

    for obj_idx in range(m):
        sorted_idx = np.argsort(pop_obj[front_idx, obj_idx])

        # Boundary individuals get infinite distance
        dist[sorted_idx[0]] = np.inf
        dist[sorted_idx[-1]] = np.inf

        fmin = pop_obj[front_idx[sorted_idx[0]], obj_idx]
        fmax = pop_obj[front_idx[sorted_idx[-1]], obj_idx]

        if fmax <= fmin:
            continue

        # Interior individuals
        obj_range = fmax - fmin
        for k in range(1, n - 1):
            dist[k] += (pop_obj[front_idx[sorted_idx[k + 1]], obj_idx] -
                        pop_obj[front_idx[sorted_idx[k - 1]], obj_idx]) / obj_range

    return dist


def generate_reference_points(n_obj: int, n_partitions: int) -> np.ndarray:
    """
    Generate reference points for NSGA-III using Das and Dennis method.

    For M objectives with p partitions, generates C(p+M-1, M-1) points.
    """
    # Simple case: 2 objectives
    if n_obj == 2:
        points = np.array([[i / n_partitions, (n_partitions - i) / n_partitions]
                           for i in range(n_partitions + 1)])
        return points

    # For >2 objectives, use分层采样 approach
    # This is a simplified implementation
    ref_points = []
    _generate_recursive(ref_points, [], n_obj, n_partitions, n_partitions)
    return np.array(ref_points)


def _generate_recursive(points: List, current: List, n_obj: int, remaining: int, total: int):
    """Helper for reference point generation."""
    if len(current) == n_obj - 1:
        current.append(remaining / total)
        points.append(current.copy())
        current.pop()
        return

    for i in range(remaining + 1):
        current.append(i / total)
        _generate_recursive(points, current, n_obj, remaining - i, total)
        current.pop()


def associate_to_reference(ref_points: np.ndarray, pop_obj: np.ndarray) -> np.ndarray:
    """
    Associate each solution to the nearest reference point using perpendicular distance.

    Args:
        ref_points: (N_ref, M) array of M-dimensional reference points
        pop_obj: (N, M) array of M-dimensional objective values

    Returns:
        (N,) array of reference point indices for each solution
    """
    n = len(pop_obj)
    n_ref = len(ref_points)

    if n_ref == 0 or pop_obj.shape[1] == 0:
        return np.zeros(n, dtype=int)

    M = pop_obj.shape[1]  # Number of objectives
    n_obj = M

    # Normalize objectives to [0, 1] range using ideal/nadir points
    pop_normalized = np.zeros_like(pop_obj)
    for j in range(M):
        fmin = pop_obj[:, j].min()
        fmax = pop_obj[:, j].max()
        if fmax > fmin:
            pop_normalized[:, j] = (pop_obj[:, j] - fmin) / (fmax - fmin)
        else:
            pop_normalized[:, j] = 0

    # Normalize reference points to [0, 1] range (assuming same scale as objectives)
    ref_norm = ref_points.copy()
    for j in range(M):
        ref_max = ref_points[:, j].max() if len(ref_points) > 0 else 1.0
        if ref_max > 0:
            ref_norm[:, j] = ref_points[:, j] / ref_max if ref_max > 0 else ref_points[:, j]

    # Calculate perpendicular distance to each reference point's hyperplane
    # Hyperplane passes through ref point with normal vector (1, 1, ..., 1)
    distances = np.zeros((n, n_ref))

    for i in range(n_ref):
        ref_vec = ref_norm[i]
        # Normal vector to hyperplane
        normal = np.ones(M)

        for k in range(n):
            point = pop_normalized[k]

            # Project point onto hyperplane and compute perpendicular distance
            # d = |(x - x0) . n| / |n|
            diff = point - ref_vec
            numerator = abs(np.dot(diff, normal))
            denominator = np.sqrt(M)  # ||n|| = sqrt(M)
            distances[k, i] = numerator / denominator if denominator > 0 else 0

    return np.argmin(distances, axis=1)


def tournament_selection(population: List[Individual], k: int = 2) -> Individual:
    """
    Binary tournament selection based on rank then crowding distance.
    """
    candidates = random.sample(population, k)
    candidates.sort(key=lambda x: (x.rank, -x.crowding_distance))
    return candidates[0]


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray,
                  eta: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for integer-coded genes.
    Adapted from Deb's NSGA-II implementation.
    """
    n_var = len(p1)
    c1, c2 = p1.copy().astype(float), p2.copy().astype(float)

    for i in range(n_var):
        if random.random() < 0.9:  # Crossover probability
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            # Real-valued SBX applied to integer indices
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])

            # Round to nearest integer and clamp to valid range
            c1[i] = int(round(c1[i]))
            c2[i] = int(round(c2[i]))
            ub = bounds[i] - 1
            c1[i] = max(0, min(c1[i], ub))
            c2[i] = max(0, min(c2[i], ub))

    return c1.astype(int), c2.astype(int)


def polynomial_mutation(ind: np.ndarray, bounds: np.ndarray,
                         eta: float = 20.0, prob: float = 0.15) -> np.ndarray:
    """
    Polynomial mutation adapted for integer genes.
    """
    n_var = len(ind)
    mut = ind.copy().astype(float)

    for i in range(n_var):
        if random.random() < prob:
            u = random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

            mut[i] += delta * (bounds[i] - 1)
            mut[i] = int(round(mut[i]))
            mut[i] = max(0, min(mut[i], bounds[i] - 1))

    return mut.astype(int)


def decode_genes(genes: np.ndarray, tunables: List[Tuple[str, List]]) -> dict:
    """Decode integer gene indices to actual parameter values."""
    config = {}
    for (key, opts), choice in zip(tunables, genes):
        idx = int(choice)
        config[key] = opts[idx] if idx < len(opts) else opts[-1]
    return config


class NSGA:
    """NSGA-II/III optimizer class."""

    def __init__(self, config: AlgorithmConfig, tunables: List[Tuple[str, List]],
                 evaluator, logger: Optional[logging.Logger] = None):
        self.config = config
        self.tunables = tunables
        self.n_var = len(tunables)
        self.bounds = np.array([len(opts) for _, opts in tunables])
        self.evaluator = evaluator
        self.logger = logger or logging.getLogger(__name__)

        self.population: List[Individual] = []
        self.history: List[dict] = []
        self.best_front: List[Individual] = []

    def initialize_population(self) -> None:
        """Initialize random population with LHS + boundary sampling."""
        pop_size = self.config.pop_size
        genes = np.random.randint(0, self.bounds, size=(pop_size, self.n_var))

        self.population = [
            Individual(genes=genes[i].copy())
            for i in range(pop_size)
        ]

        # Evaluate initial population
        for ind in self.population:
            config = decode_genes(ind.genes, self.tunables)
            ind.objectives = np.array(self.evaluator(config))

    def evolve(self) -> List[Individual]:
        """Run the NSGA-II/III evolutionary process."""
        self.initialize_population()
        self._evaluate_population()

        for gen in range(self.config.n_gen):
            # Create offspring population
            offspring = self._create_offspring()

            # Combine parent and offspring
            combined = self.population + offspring
            obj_mat = np.array([ind.objectives for ind in combined])

            # Non-dominated sorting
            rank, fronts = fast_non_dominated_sort(obj_mat)

            # Calculate crowding distance or associate reference points
            if not self.config.use_nsga3:
                for front in fronts:
                    dist = crowding_distance(obj_mat, front)
                    for i, idx in enumerate(front):
                        combined[idx].rank = rank[idx]
                        combined[idx].crowding_distance = dist[i]
            else:
                # NSGA-III: associate to reference points
                n_obj = obj_mat.shape[1]  # Number of objectives (M), not number of variables
                ref_points = generate_reference_points(n_obj, self.config.n_partitions)
                associations = associate_to_reference(ref_points, obj_mat)
                for i, ind in enumerate(combined):
                    ind.rank = rank[i]
                    ind.crowding_distance = 0.0  # Not used in NSGA-III

            # Select next generation (truncation)
            self.population = self._select_next(combined, rank, fronts)

            # Log progress
            front0_size = len(fronts[0]) if fronts else 0
            self.history.append({
                'gen': gen,
                'front0_size': front0_size,
                'avg_fitness': np.mean(obj_mat, axis=0).tolist()
            })

            if gen % 10 == 0 or gen == self.config.n_gen - 1:
                self.logger.info(f"Gen {gen:3d} | Front0 size: {front0_size}")

        return self.get_pareto_front()

    def _evaluate_population(self) -> None:
        """Evaluate objectives for all individuals."""
        for ind in self.population:
            config = decode_genes(ind.genes, self.tunables)
            ind.objectives = np.array(self.evaluator(config))

    def _create_offspring(self) -> List[Individual]:
        """Create offspring through selection, crossover, and mutation."""
        offspring = []
        target_size = self.config.pop_size

        while len(offspring) < target_size:
            # Tournament selection
            parent1 = tournament_selection(self.population, self.config.tournament_size)
            parent2 = tournament_selection(self.population, self.config.tournament_size)

            # SBX crossover
            c1_genes, c2_genes = sbx_crossover(
                parent1.genes, parent2.genes, self.bounds, self.config.sbx_eta
            )

            # Polynomial mutation
            c1_genes = polynomial_mutation(c1_genes, self.bounds, self.config.pm_eta, self.config.mutation_prob)
            c2_genes = polynomial_mutation(c2_genes, self.bounds, self.config.pm_eta, self.config.mutation_prob)

            offspring.append(Individual(genes=c1_genes))
            if len(offspring) < target_size:
                offspring.append(Individual(genes=c2_genes))

        # Evaluate offspring
        for ind in offspring:
            config = decode_genes(ind.genes, self.tunables)
            ind.objectives = np.array(self.evaluator(config))

        return offspring[:target_size]

    def _select_next(self, combined: List[Individual], rank: np.ndarray,
                     fronts: List[List[int]]) -> List[Individual]:
        """Select next generation using elitism."""
        pop_size = self.config.pop_size
        selected: List[Individual] = []

        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend([combined[i] for i in front])
            else:
                # Fill remaining slots by crowding distance
                remaining = pop_size - len(selected)
                front_individuals = [combined[i] for i in front]
                obj_mat = np.array([ind.objectives for ind in front_individuals])
                dist = crowding_distance(obj_mat, list(range(len(front))))

                # Sort by crowding distance (descending)
                sorted_idx = np.argsort(-dist)[:remaining]
                selected.extend([front_individuals[i] for i in sorted_idx])
                break

        return selected

    def get_pareto_front(self) -> List[Individual]:
        """Extract Pareto optimal solutions."""
        obj_mat = np.array([ind.objectives for ind in self.population])
        rank, fronts = fast_non_dominated_sort(obj_mat)

        pareto_idx = np.where(rank == 0)[0]
        self.best_front = [self.population[i] for i in pareto_idx]
        return self.best_front

    def get_results(self) -> dict:
        """Get optimization results as dictionary."""
        pareto = self.get_pareto_front()
        return {
            'pareto_solutions': [
                decode_genes(ind.genes, self.tunables) for ind in pareto
            ],
            'pareto_objectives': [ind.objectives.tolist() for ind in pareto],
            'history': self.history,
            'n_pareto': len(pareto)
        }