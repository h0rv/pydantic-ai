from __future__ import annotations

from typing import Any

from pydantic_ai.exceptions import UserError


class BEAMNotAvailableError(UserError):
    """Raised when code expects to run inside a BEAM worker but the `erlang` module is not available."""


def get_erlang() -> Any:
    """Import and return the `erlang` module injected by `erlang_python` at runtime."""
    try:
        import erlang  # type: ignore[import-not-found]
    except ImportError:
        raise BEAMNotAvailableError(
            'The `erlang` module is not available. This code must run inside a BEAM worker via `erlang_python`.'
        ) from None
    else:
        return erlang


def erlang_call(function_name: str, *args: Any) -> Any:
    """Synchronous reentrant call into Erlang via the injected `erlang` module."""
    erlang = get_erlang()
    return erlang.call(function_name, *args)


def in_beam_worker() -> bool:
    """Detect whether the current process is running inside a BEAM worker."""
    try:
        import erlang  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    else:
        return True
