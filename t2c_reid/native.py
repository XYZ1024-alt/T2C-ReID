"""Versioned access to the mandatory Rust extension."""

from __future__ import annotations

try:
    from t2c_reid import _native as native_extension
except ImportError as exc:
    raise ImportError(
        "the mandatory t2c_reid Rust extension is unavailable; install Rust 1.85+ "
        "and the platform linker, then rebuild with `uv sync`"
    ) from exc

EXPECTED_NATIVE_ABI_VERSION = 2


def _validate_native_extension() -> tuple[str, int]:
    try:
        version, abi_version = native_extension.native_version()
    except (AttributeError, TypeError) as exc:
        raise ImportError(
            "t2c_reid._native is missing its ABI metadata; rebuild the package with `uv sync`"
        ) from exc
    if int(abi_version) != EXPECTED_NATIVE_ABI_VERSION:
        raise ImportError(
            "t2c_reid native ABI mismatch: "
            f"Python expects {EXPECTED_NATIVE_ABI_VERSION}, extension provides {abi_version}; "
            "rebuild the package with `uv sync`"
        )
    return str(version), int(abi_version)


NATIVE_VERSION, NATIVE_ABI_VERSION = _validate_native_extension()
