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


def check_paths(*paths) -> str | None:
    """Code-only check, never asks. None = every path is inside the allowed folders."""
    bad = [str(p) for p in paths if not is_allowed(p)]
    return f"refused: outside allowed folders: {', '.join(bad)}" if bad else None


def guard(action: str, *paths) -> str | None:
    """For destructive tools only (delete, move, send, pay, change settings): path check, then ask.
    None = allowed and confirmed by the user. Otherwise the reason the action was blocked.
    Must run inside a graph node: it pauses the graph with interrupt() to ask the user."""
    if (blocked := check_paths(*paths)):
        return blocked
    head = f"About to: {action}\n"
    if paths:
        head += f"Targets: {', '.join(str(resolve(p)) for p in paths)}\n"
    answer = str(interrupt(f"{head}Proceed? (yes / no / or say what to change)")).strip()
    if answer.lower() in ("y", "yes"):
        return None
    return "declined by user" + (f": {answer}" if answer.lower() not in ("", "n", "no") else "")


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
