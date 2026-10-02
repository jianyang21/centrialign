"""Shared guardrails. Every workflow calls guard() right before a destructive side effect."""
import os
import re
from pathlib import Path
from langgraph.types import interrupt

# Folders destructive actions may touch (their contents, never the folder itself).
# Resolved so junctions/OneDrive redirects compare correctly.
ALLOWED_ROOTS = [(Path.home() / d).resolve() for d in
                 ("Documents", "Downloads", "Desktop", "OneDrive/Documents", "OneDrive/Desktop")]


def resolve(path) -> Path | None:
    """Absolute, fully resolved path, or None. Expands ~ and %VARS%; refuses relative,
    root-relative (/Windows) and drive-relative (C:Windows) paths instead of guessing a base."""
    try:
        p = Path(os.path.expandvars(os.path.expanduser(str(path))))
        if p.drive.startswith(("\\\\", "//")):  # network share: never allowed, and resolve() would hang on it
            return None
        return p.resolve() if p.is_absolute() else None  # resolve() collapses '..' and follows links
    except (OSError, ValueError):
        return None


def is_allowed(path) -> bool:
    p = resolve(path)
    return p is not None and any(root in p.parents for root in ALLOWED_ROOTS)  # strictly inside a root


def guard(action: str, *paths) -> str | None:
    """None = allowed and confirmed by the user. Otherwise the reason the action was blocked.
    Must run inside a graph node: it pauses the graph with interrupt() to ask the user."""
    bad = [str(p) for p in paths if not is_allowed(p)]
    if bad:
        return f"refused: outside allowed folders: {', '.join(bad)}"
    targets = ", ".join(str(resolve(p)) for p in paths) or "none detected (not checked by code)"
    answer = interrupt(f"About to: {action}\nTargets: {targets}\nProceed? (yes/no)")
    if str(answer).strip().lower() not in ("y", "yes"):
        return "declined by user"
    return None


# ponytail: placeholder-only; real workflows pass guard() the exact path their tool will touch.
# Delete with the placeholders. Where a path-like token starts: C:/x, C:x, ~, \\unc, .., /x, %VAR%, $env:X, $HOME
PATH_START = re.compile(r"""(?<![^\s"'(])(?:[A-Za-z]:(?=\S)|~|\\\\|\.\.|/|%\w+%|\$env:\w+|\$[A-Za-z_]\w*)""")


def path_readings(task: str) -> list[str]:
    """Free text can't say where a path with spaces ends, so return every reading of each path:
    cut at each following space, and run to the end. guard() then has to accept all of them."""
    out = []
    for m in PATH_START.finditer(task):
        rest = task[m.start():]
        ends = [i for i, c in enumerate(rest) if c.isspace()] + [len(rest)]
        for i in ends:
            r = rest[:i].rstrip("\"').,;")
            if r and r not in out:
                out.append(r)
    return out
