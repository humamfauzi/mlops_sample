"""Build provenance for the running artifact.

``tools/stamp_build.py`` writes ``_buildinfo.py`` immediately before a
PyInstaller build, so a frozen binary carries a fixed record of the commit it
was built from. When running from a source checkout that file is absent and we
fall back to asking git directly.

Consumers:
  * ``GET /health``      -> ``build`` block, so a deployed server can be
                            checked against the commit it is supposed to be
  * ``server/launcher.py`` -> logged on startup
  * ``train_module --version``
"""
from __future__ import annotations

import subprocess

try:  # generated at build time; absent in a plain source checkout
    from _buildinfo import BUILD_TIME, GIT_SHA, VERSION  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised by running from source
    VERSION = "0.1.0"

    def _git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ("git", *args), capture_output=True, text=True, timeout=5, check=False,
            )
        except Exception:
            return ""
        return proc.stdout.strip() if proc.returncode == 0 else ""

    _sha = _git("rev-parse", "HEAD") or "unknown"
    if _sha != "unknown" and _git("status", "--porcelain"):
        _sha += "-dirty"
    GIT_SHA = _sha
    BUILD_TIME = "source"

__all__ = ["VERSION", "GIT_SHA", "BUILD_TIME", "describe"]


def describe() -> dict:
    """Return the provenance record embedded in this artifact."""
    return {"version": VERSION, "git_sha": GIT_SHA, "build_time": BUILD_TIME}
