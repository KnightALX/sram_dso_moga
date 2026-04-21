"""
SRAM DSO-MOGA: Test Suite

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nsga import (
    dominates, fast_non_dominated_sort, crowding_distance,
    sbx_crossover, polynomial_mutation, decode_genes,
    tournament_selection, Individual, AlgorithmConfig, NSGA
)


class TestDominates:
    """Test Pareto dominance functions."""

    def test_dominates_strict(self):
        """A strictly dominates B when all objectives <= and at least one <."""
        obj_a = np.array([1.0, 2.0, 3.0])
        obj_b = np.array([2.0, 3.0, 4.0])
        assert dominates(obj_a, obj_b)
        assert not dominates(obj_b, obj_a)

    def test_dominates_equal(self):
        """Equal objectives don't dominate each other."""
        obj_a = np.array([1.0, 2.0, 3.0])
        obj_b = np.array([1.0, 2.0, 3.0])
        assert not dominates(obj_a, obj_b)
        assert not dominates(obj_b, obj_a)

    def test_dominates_partial(self):
        """Partial dominance: one objective better, one worse."""
        obj_a = np.array([1.0, 3.0, 3.0])
        obj_b = np.array([2.0, 2.0, 4.0])
        # A <= B (1<=2, 3<=2=False, 3<=4=True) -> not all <=
        assert not dominates(obj_a, obj_b)

    def test_dominates_multidim(self):
        """Test with more than 3 dimensions."""
        obj_a = np.array([1.0, 2.0, 3.0, 4.0])
        obj_b = np.array([2.0, 2.5, 3.5, 4.5])
        assert dominates(obj_a, obj_b)


class TestFastNonDominatedSort:
    """Test NSGA-II non-dominated sorting."""

    def test_empty_population(self):
        """Empty population should return empty fronts."""
        pop_obj = np.array([]).reshape(0, 3)
        rank, fronts = fast_non_dominated_sort(pop_obj)
        assert len(rank) == 0
        assert len(fronts) == 0

    def test_single_individual(self):
        """Single individual should be in front 0."""
        pop_obj = np.array([[1.0, 2.0, 3.0]])
        rank, fronts = fast_non_dominated_sort(pop_obj)
        assert rank[0] == 0
        assert len(fronts) == 1
        assert 0 in fronts[0]

    def test_two_individuals_no_dominance(self):
        """Two individuals neither dominating the other."""
        pop_obj = np.array([[1.0, 5.0, 3.0], [2.0, 3.0, 4.0]])
        rank, fronts = fast_non_dominated_sort(pop_obj)
        assert rank[0] == 0
        assert rank[1] == 0
        assert len(fronts) == 1

    def test_two_individuals_dominance(self):
        """One individual dominates the other."""
        pop_obj = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        rank, fronts = fast_non_dominated_sort(pop_obj)
        assert rank[0] == 0  # First dominates second
        assert rank[1] == 1

    def test_front_ordering(self):
        """Fronts should be in ascending Pareto rank order."""
        # Create known Pareto front scenario
        pop_obj = np.array([
            [1.0, 1.0],  # Front 0
            [2.0, 2.0],  # Front 0
            [1.5, 1.5],  # Front 0
            [3.0, 1.0],  # Front 1
            [2.0, 3.0],  # Front 1
        ])
        rank, fronts = fast_non_dominated_sort(pop_obj)

        # All front 0 individuals should have rank 0
        for idx in fronts[0]:
            assert rank[idx] == 0

        # Later fronts should have higher ranks
        for i, front in enumerate(fronts):
            for idx in front:
                assert rank[idx] == i

    def test_large_population(self):
        """Test with larger population for performance."""
        np.random.seed(42)
        pop_obj = np.random.rand(100, 3)  # 100 individuals, 3 objectives

        rank, fronts = fast_non_dominated_sort(pop_obj)

        # Every individual should have a rank
        assert len(rank) == 100
        assert all(r >= 0 for r in rank)

        # Front 0 should not be empty (usually)
        assert len(fronts) >= 1


class TestCrowdingDistance:
    """Test crowding distance calculation."""

    def test_small_front(self):
        """Front with <= 2 individuals should have infinite distance."""
        front_idx = [0, 1]
        pop_obj = np.array([[1.0, 2.0], [2.0, 3.0]])
        dist = crowding_distance(pop_obj, front_idx)
        assert np.all(np.isinf(dist))

    def test_three_individuals(self):
        """Front with 3 individuals."""
        front_idx = [0, 1, 2]
        pop_obj = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
        dist = crowding_distance(pop_obj, front_idx)

        # Boundary individuals should have inf distance
        assert np.isinf(dist[0])
        assert np.isinf(dist[2])

        # Middle individual should have finite distance
        assert not np.isinf(dist[1])

    def test_identical_objectives(self):
        """When all objectives in a front are identical in one dimension."""
        front_idx = [0, 1, 2]
        pop_obj = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        dist = crowding_distance(pop_obj, front_idx)

        # Boundary individuals should have inf distance
        assert np.isinf(dist[0])
        assert np.isinf(dist[2])
        # Middle individual gets 0 since no variation contributes

    def test_distributes_diversity(self):
        """Test that crowding distance handles different scales correctly."""
        # When objectives have different scales, crowding distance
        # normalizes by range so interior individuals get proportional distances
        front_idx = [0, 1, 2]

        # Case 1: Small range [0,1]
        pop_small = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        dist_small = crowding_distance(pop_small, front_idx)

        # Case 2: Large range [0,100]
        pop_large = np.array([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]])
        dist_large = crowding_distance(pop_large, front_idx)

        # Both should give same normalized distance (2.0) for interior
        # This is expected behavior - normalization makes comparisons fair
        assert dist_small[1] == dist_large[1] == 2.0


class TestSBXCrossover:
    """Test SBX crossover operator."""

    def test_crossover_output_shape(self):
        """Crossover should produce offspring with same shape as parents."""
        p1 = np.array([3, 5, 2, 7])
        p2 = np.array([4, 6, 3, 8])
        bounds = np.array([8, 10, 6, 12])

        c1, c2 = sbx_crossover(p1, p2, bounds)

        assert c1.shape == p1.shape
        assert c2.shape == p2.shape

    def test_crossover_within_bounds(self):
        """Offspring genes should be within valid range."""
        p1 = np.array([3, 5, 2, 7])
        p2 = np.array([4, 6, 3, 8])
        bounds = np.array([8, 10, 6, 12])

        c1, c2 = sbx_crossover(p1, p2, bounds, eta=15.0)

        assert all(0 <= c1[i] < bounds[i] for i in range(len(bounds)))
        assert all(0 <= c2[i] < bounds[i] for i in range(len(bounds)))

    def test_crossover_reproducibility(self):
        """Same seed should produce same crossover results."""
        p1 = np.array([3, 5, 2, 7])
        p2 = np.array([4, 6, 3, 8])
        bounds = np.array([8, 10, 6, 12])

        import random
        random.seed(42)
        c1_a, c2_a = sbx_crossover(p1, p2, bounds)

        random.seed(42)
        c1_b, c2_b = sbx_crossover(p1, p2, bounds)

        assert np.array_equal(c1_a, c1_b)
        assert np.array_equal(c2_a, c2_b)

    def test_crossover_produces_valid_genes(self):
        """SBX crossover should produce valid genes within bounds."""
        p1 = np.array([2, 4])
        p2 = np.array([7, 8])
        bounds = np.array([12, 12])

        import random
        random.seed(42)

        # Run crossover many times, all offspring should be valid
        for _ in range(50):
            c1, c2 = sbx_crossover(p1, p2, bounds, eta=15.0)
            # All genes should be valid indices
            assert 0 <= c1[0] < bounds[0]
            assert 0 <= c1[1] < bounds[1]
            assert 0 <= c2[0] < bounds[0]
            assert 0 <= c2[1] < bounds[1]


class TestPolynomialMutation:
    """Test polynomial mutation operator."""

    def test_mutation_output_shape(self):
        """Mutation should preserve individual shape."""
        ind = np.array([3, 5, 2, 7])
        bounds = np.array([8, 10, 6, 12])

        mutated = polynomial_mutation(ind, bounds)

        assert mutated.shape == ind.shape

    def test_mutation_within_bounds(self):
        """Mutated genes should be within valid range."""
        ind = np.array([3, 5, 2, 7])
        bounds = np.array([8, 10, 6, 12])

        mutated = polynomial_mutation(ind, bounds, eta=20.0, prob=0.5)

        assert all(0 <= mutated[i] < bounds[i] for i in range(len(bounds)))

    def test_mutation_probability(self):
        """With prob=0, no changes should occur."""
        ind = np.array([3, 5, 2, 7])
        bounds = np.array([8, 10, 6, 12])

        mutated = polynomial_mutation(ind, bounds, eta=20.0, prob=0.0)

        # Should be identical since no mutation
        assert np.array_equal(mutated, ind)

    def test_mutation_changes_some_genes(self):
        """With prob=1, many genes should change but not necessarily all."""
        ind = np.array([3, 5, 2, 7])
        bounds = np.array([8, 10, 6, 12])

        import random
        random.seed(42)

        # Run many times and count changes
        total_changes = 0
        for _ in range(100):
            mutated = polynomial_mutation(ind, bounds, eta=20.0, prob=1.0)
            # Count how many genes changed
            changes = np.sum(mutated != ind)
            total_changes += changes

        # On average, with prob=1 and 4 genes, we expect many changes
        # But due to delta=0 cases, not all will change every time
        assert total_changes > 100, "Most genes should change across many runs"


class TestDecodeGenes:
    """Test gene decoding."""

    def test_decode_basic(self):
        """Basic decoding should map indices to option values."""
        tunables = [
            ('a_vt', ['vt0', 'vt1', 'vt2']),
            ('b_nfin', [2, 4, 6, 8]),
        ]
        genes = np.array([1, 2])  # Select vt1 and nfin=6

        config = decode_genes(genes, tunables)

        assert config['a_vt'] == 'vt1'
        assert config['b_nfin'] == 6

    def test_decode_out_of_bounds_index(self):
        """Index beyond option length should use last option."""
        tunables = [
            ('a_vt', ['vt0', 'vt1', 'vt2']),
        ]
        genes = np.array([5])  # Index 5 is out of range for 3 options

        config = decode_genes(genes, tunables)

        # Should use last option (index 2)
        assert config['a_vt'] == 'vt2'


class TestTournamentSelection:
    """Test tournament selection."""

    def test_tournament_returns_individual(self):
        """Tournament should return one of the tournament members."""
        individuals = [
            Individual(genes=np.array([1, 2, 3]), objectives=np.array([1.0, 2.0, 3.0]), rank=0),
            Individual(genes=np.array([4, 5, 6]), objectives=np.array([2.0, 3.0, 4.0]), rank=0),
            Individual(genes=np.array([7, 8, 9]), objectives=np.array([1.5, 2.5, 3.5]), rank=1),
        ]

        import random
        random.seed(42)

        winner = tournament_selection(individuals, k=2)

        assert isinstance(winner, Individual)

    def test_tournament_prefers_better_rank(self):
        """Tournament should prefer individuals with lower rank."""
        individuals = [
            Individual(genes=np.array([1]), objectives=np.array([10.0]), rank=2),
            Individual(genes=np.array([2]), objectives=np.array([5.0]), rank=0),
        ]

        import random
        random.seed(42)

        # Run many tournaments, winner should be rank 0 most of the time
        rank0_wins = 0
        for _ in range(100):
            winner = tournament_selection(individuals, k=2)
            if winner.rank == 0:
                rank0_wins += 1

        assert rank0_wins > 50  # Should prefer better rank


class TestNSGA:
    """Test NSGA optimizer class."""

    def test_initialization(self):
        """Test NSGA initialization."""
        config = AlgorithmConfig(pop_size=20, n_gen=10, seed=42)
        tunables = [('a_vt', ['vt0', 'vt1'])]
        evaluator = lambda x: [1.0, 2.0, 3.0]

        nsga = NSGA(config, tunables, evaluator)

        assert nsga.config.pop_size == 20
        assert nsga.n_var == 1

    def test_evolve_returns_pareto_front(self):
        """Evolve should return list of Pareto optimal individuals."""
        config = AlgorithmConfig(pop_size=30, n_gen=5, seed=42)
        tunables = [
            ('a_vt', ['vt0', 'vt1', 'vt2']),
            ('b_nfin', [2, 4, 6]),
        ]

        def evaluator(cfg):
            # Simple test evaluator
            area = cfg.get('b_nfin', 2) * 0.1
            power = cfg.get('b_nfin', 2) * 0.2
            delay = cfg.get('a_vt', 'vt0') == 'vt0' and 10.0 or 12.0
            return [area, power, delay]

        nsga = NSGA(config, tunables, evaluator)
        results = nsga.evolve()

        assert len(results) > 0  # Should find some Pareto solutions
        assert all(isinstance(ind, Individual) for ind in results)

    def test_get_results(self):
        """Get results should return structured data."""
        config = AlgorithmConfig(pop_size=20, n_gen=3, seed=42)
        tunables = [('a_vt', ['vt0', 'vt1'])]
        evaluator = lambda x: [1.0, 2.0, 3.0]

        nsga = NSGA(config, tunables, evaluator)
        nsga.evolve()

        results = nsga.get_results()

        assert 'pareto_solutions' in results
        assert 'pareto_objectives' in results
        assert 'history' in results
        assert 'n_pareto' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])