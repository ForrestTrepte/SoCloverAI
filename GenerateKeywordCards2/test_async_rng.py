import asyncio
from asyncio import TaskGroup

import pytest

from .async_rng import AsyncRng, LocalRng


class TestAsyncRng:
    def test_unwrap_creates_local_rng(self) -> None:
        rng = AsyncRng.create(seed=42)
        local = rng.unwrap()
        assert isinstance(local, LocalRng)

    def test_unwrap_twice_fails(self) -> None:
        rng = AsyncRng.create(seed=42)
        rng.unwrap()
        with pytest.raises(AssertionError, match="unwrap.*already called"):
            rng.unwrap()


class TestDeterminism:
    def test_same_seed_produces_same_results(self) -> None:
        """Same seed produces identical sequences at both parent and child levels."""
        rng1 = AsyncRng.create(seed=42).unwrap()
        rng2 = AsyncRng.create(seed=42).unwrap()

        # Parents match
        assert rng1.randint(0, 1000) == rng2.randint(0, 1000)
        assert rng1.random() == rng2.random()

        # Children also match (fork uses parent RNG deterministically)
        child1 = rng1.fork().unwrap()
        child2 = rng2.fork().unwrap()
        assert child1.randint(0, 10000) == child2.randint(0, 10000)
        assert child1.random() == child2.random()

    def test_different_seed_different_results(self) -> None:
        rng1 = AsyncRng.create(seed=42).unwrap()
        rng2 = AsyncRng.create(seed=43).unwrap()

        # Different seeds should (almost certainly) produce different results
        results1 = [rng1.randint(0, 10000) for _ in range(10)]
        results2 = [rng2.randint(0, 10000) for _ in range(10)]
        assert results1 != results2

    def test_sibling_forks_are_independent(self) -> None:
        """Multiple forks from the same parent have different sequences."""
        rng = AsyncRng.create(seed=42).unwrap()

        child_rng1 = rng.fork().unwrap()
        child_rng2 = rng.fork().unwrap()

        results1 = [child_rng1.randint(0, 10000) for _ in range(10)]
        results2 = [child_rng2.randint(0, 10000) for _ in range(10)]
        assert results1 != results2


class TestLocalRngMethods:
    def test_choice(self) -> None:
        rng = AsyncRng.create(seed=42).unwrap()

        items = ["a", "b", "c", "d", "e"]
        result = rng.choice(items)
        assert result in items

    def test_shuffle(self) -> None:
        rng = AsyncRng.create(seed=42).unwrap()

        items = [1, 2, 3, 4, 5]
        original = items.copy()
        rng.shuffle(items)
        # Should be shuffled (same elements, likely different order)
        assert set(items) == set(original)

    def test_randint(self) -> None:
        rng = AsyncRng.create(seed=42).unwrap()

        for _ in range(100):
            result = rng.randint(10, 20)
            assert 10 <= result <= 20

    def test_random(self) -> None:
        rng = AsyncRng.create(seed=42).unwrap()

        for _ in range(100):
            result = rng.random()
            assert 0.0 <= result < 1.0

    def test_sample(self) -> None:
        rng = AsyncRng.create(seed=42).unwrap()

        population = list(range(100))
        sample = rng.sample(population, 10)
        assert len(sample) == 10
        assert len(set(sample)) == 10  # No duplicates
        assert all(item in population for item in sample)


class TestLegacyMode:
    def test_legacy_shares_same_random_instance(self) -> None:
        rng = AsyncRng.create(seed=42, legacy=True).unwrap()

        child_rng = rng.fork().unwrap()

        # Both should share the same underlying Random
        assert rng._rng is child_rng._rng

    def test_non_legacy_has_independent_random_instances(self) -> None:
        rng = AsyncRng.create(seed=42, legacy=False).unwrap()

        child_rng = rng.fork().unwrap()

        # Should be different Random instances
        assert rng._rng is not child_rng._rng


class TestTaskBoundAssertions:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy", [False, True])
    async def test_cross_task_usage(self, legacy: bool) -> None:
        """Using a LocalRng in a different task fails in normal mode, succeeds in legacy mode."""
        rng = AsyncRng.create(seed=42, legacy=legacy).unwrap()

        async def use_in_other_task() -> int:
            return rng.randint(0, 100)

        if legacy:
            # Legacy mode: cross-task usage allowed
            async with TaskGroup() as tg:
                task = tg.create_task(use_in_other_task())
            assert isinstance(task.result(), int)
        else:
            # Normal mode: cross-task usage raises assertion
            with pytest.raises(ExceptionGroup) as exc_info:
                async with TaskGroup() as tg:
                    tg.create_task(use_in_other_task())
            assert len(exc_info.value.exceptions) == 1
            assert isinstance(exc_info.value.exceptions[0], AssertionError)
            assert "Use fork() to pass to other tasks" in str(
                exc_info.value.exceptions[0]
            )

    @pytest.mark.asyncio
    async def test_forked_rng_works_in_child_task(self) -> None:
        """Passing an AsyncRng to a child task allows safe cross-task RNG usage."""
        rng = AsyncRng.create(seed=42).unwrap()

        async def use_in_other_task(async_rng: AsyncRng) -> int:
            local_rng = async_rng.unwrap()
            return local_rng.randint(0, 100)

        async with TaskGroup() as tg:
            task1 = tg.create_task(use_in_other_task(rng.fork()))
            task2 = tg.create_task(use_in_other_task(rng.fork()))

        assert isinstance(task1.result(), int)
        assert isinstance(task2.result(), int)


class TestParallelDeterminism:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("legacy", [False, True])
    async def test_parallel_tasks_deterministic(self, legacy: bool) -> None:
        """Verify that parallel execution produces deterministic results (unless legacy mode)."""

        # Only import random here for varying the sleeps. The rest of this file should be testing the
        # AsyncRng, not using the standard library random.
        import random

        async def run_with_seed(seed: int) -> list[list[int]]:
            rng = AsyncRng.create(seed=seed, legacy=legacy).unwrap()

            async def child_task(async_rng: AsyncRng) -> list[int]:
                local_rng = async_rng.unwrap()
                results = []
                for _ in range(10):
                    await asyncio.sleep(random.random() * 0.01)
                    results.append(local_rng.randint(0, 10000))
                return results

            # Fork all before creating tasks (deterministic seed generation)
            forks = [rng.fork() for _ in range(5)]

            async with TaskGroup() as tg:
                tasks = [tg.create_task(child_task(f)) for f in forks]

            # Results are in task creation order, not completion order
            return [t.result() for t in tasks]

        # Run 10 trials with the same seed
        all_results = [await run_with_seed(42) for _ in range(10)]

        if legacy:
            # Legacy mode: shared RNG causes non-deterministic results
            assert not all(r == all_results[0] for r in all_results)
        else:
            # Normal mode: forked RNGs produce deterministic results
            assert all(r == all_results[0] for r in all_results)
