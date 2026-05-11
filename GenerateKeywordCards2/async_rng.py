import asyncio
from random import Random
from typing import Sequence, TypeVar

T = TypeVar("T")


class AsyncRng:
    """
    A token for passing RNG state to async functions.
    Must call unwrap() exactly once to get a usable LocalRng.
    All async functions that receive RNG state from their parent must use an AsyncRng parameter to receive their state.
    """

    def __init__(self, seed: int, legacy_rng: Random | None = None):
        self._seed = seed
        self._legacy_rng = legacy_rng
        self._unwrapped = False

    @staticmethod
    def create(seed: int, legacy: bool = False) -> "AsyncRng":
        """Create an initial AsyncRng to pass to the first async function."""
        legacy_rng = Random(seed) if legacy else None
        return AsyncRng(seed, legacy_rng)

    def unwrap(self) -> "LocalRng":
        """Convert to a usable LocalRng. Call exactly once at start of receiving function."""
        assert not self._unwrapped, "AsyncRng.unwrap() already called"
        self._unwrapped = True

        if self._legacy_rng is not None:
            return LocalRng(self._legacy_rng, legacy=True)
        return LocalRng(Random(self._seed), legacy=False)


class LocalRng:
    """
    A random number generator bound to a specific async task.
    Use fork() to create AsyncRng tokens for child tasks.
    This pattern ensures a deterministic set of RNG operations even if execution order changes due to async timing differences.
    """

    def __init__(self, rng: Random, legacy: bool):
        self._rng = rng
        self._legacy = legacy
        self._bound_task_id: int | None = self._get_current_task_id()

    @staticmethod
    def _get_current_task_id() -> int | None:
        try:
            task = asyncio.current_task()
            return id(task) if task else None
        except RuntimeError:
            return None  # Not in async context

    def _assert_same_task(self) -> None:
        if self._legacy:
            return
        current = self._get_current_task_id()
        assert current == self._bound_task_id, (
            f"LocalRng used in task {current}, but bound to {self._bound_task_id}. "
            f"Use fork() to pass to other tasks."
        )

    def fork(self) -> AsyncRng:
        """Create an AsyncRng to pass to a child task."""
        self._assert_same_task()
        if self._legacy:
            return AsyncRng(0, legacy_rng=self._rng)
        return AsyncRng(self._rng.randint(0, 2**32 - 1))

    def choice(self, seq: Sequence[T]) -> T:
        self._assert_same_task()
        return self._rng.choice(seq)

    def shuffle(self, x: list[T]) -> None:
        self._assert_same_task()
        self._rng.shuffle(x)

    def randint(self, a: int, b: int) -> int:
        self._assert_same_task()
        return self._rng.randint(a, b)

    def random(self) -> float:
        self._assert_same_task()
        return self._rng.random()

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        self._assert_same_task()
        return self._rng.sample(population, k)
