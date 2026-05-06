import asyncio
from types import TracebackType
from typing import Any, Coroutine


class CompatTaskGroup:
    """A simple compatibility shim for asyncio.TaskGroup. In Python 3.14, TaskGroup offers the eager_start parameter, but
    I encountered a problem where, in my Jupyter notebook, only the first cell has an asyncio.current_task(). The means that
    the standard TaskGroup creation fails with 'RuntimeError: TaskGroup <TaskGroup> cannot determine the parent task'.

    This shim can be used as a replacement for TaskGroup in such environments. It does not support all features of TaskGroup,
    but it provides basic task management and cancellation on exceptions."""

    # CONSIDER: Report this as an issue in Python 3.14, IPython, Jupyter, VS Code or some combination thereof.
    # CONSIDER: Remove this class once the issue is resolved.

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> "CompatTaskGroup":
        return self

    def create_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str | None = None,
        eager_start: bool | None = None,
    ) -> asyncio.Task[Any]:
        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if eager_start is not None:
            kwargs["eager_start"] = eager_start

        # Pass loop explicitly to work around a possible bug in Python 3.12's C implementation
        # of asyncio.Task. According to Claude's theory (uncomfirmed) when eager_start=True,
        # Task_init uses a local `loop` variable # that is never assigned from FutureObj_init,
        # so it remains NULL and calling loop.is_running() crashes with:
        #   AttributeError: 'NoneType' has no attribute 'is_running'
        # Passing loop= directly avoids this code path.
        # CONSIDER: Can `loop=asyncio.get_running_loop()` be removed once the Python 3.12 bug is fixed?
        task = asyncio.Task(coro, loop=asyncio.get_running_loop(), **kwargs)
        self._tasks.append(task)
        return task

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            return None

        try:
            await asyncio.gather(*self._tasks)
        except Exception:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            raise
