"""Version-compat helpers for the two pinned transformers stacks.

Session A/C run transformers 5.x; Session B runs ``>=4.33.2,<5``. The two
generations renamed several ``TrainingArguments``/``Trainer`` keywords, and not
every release keeps a working alias — a verbatim kwarg can therefore either emit
a ``FutureWarning`` or raise ``TypeError`` at launch (i.e. a dead 12h Kaggle run).

:func:`supported_kwargs` renames what it can against the *installed* signature
and reports the rest, so version drift degrades to a log line instead of a
crash. It is deliberately small and dependency-free so both stacks can import it
before torch/transformers are loaded.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# Deprecated/renamed keyword -> the name current releases accept.
#   evaluation_strategy -> eval_strategy            (transformers >= 4.46)
#   generation_max_length -> generation_max_new_tokens
#   tokenizer -> processing_class                   (Trainer-like constructors >= 4.46)
ALIASES: Dict[str, str] = {
    "evaluation_strategy": "eval_strategy",
    "generation_max_length": "generation_max_new_tokens",
    "tokenizer": "processing_class",
}


def _parameters(target: Any) -> Optional[set]:
    """Return the accepted parameter names of ``target``, or ``None`` if unknown."""
    fn = target.__init__ if inspect.isclass(target) else target
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / C extensions
        return None
    return set(params)


def supported_kwargs(
    target: Any,
    kwargs: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Rename/filter ``kwargs`` to what ``target`` accepts in this environment.

    Args:
        target: A class (its ``__init__`` is inspected) or a callable.
        kwargs: Keyword arguments about to be passed to ``target``.
        aliases: Overrides for :data:`ALIASES` (old name -> new name).

    Returns:
        ``(kept, dropped)``: the arguments that are safe to pass, and the names
        that were silently unsupported (callers should log these loudly; they
        mean the installed version renamed or removed a setting).
    """
    params = _parameters(target)
    if params is None or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(
            target.__init__ if inspect.isclass(target) else target
        ).parameters.values()
    ):
        return dict(kwargs), []

    mapping = dict(ALIASES)
    if aliases:
        mapping.update(aliases)
    # Reverse mapping (new name -> old name) so new-spelling kwargs degrade
    # gracefully on an old stack (e.g. ``processing_class`` -> ``tokenizer``,
    # ``eval_strategy`` -> ``evaluation_strategy``) instead of being dropped.
    # Only added where safe: the reverse target must not itself be a key in
    # ``mapping`` (i.e. it is genuinely the "other" spelling).
    reverse: Dict[str, str] = {}
    for _old, _new in mapping.items():
        if _new not in mapping and _new not in reverse:
            reverse[_new] = _old

    kept: Dict[str, Any] = {}
    dropped: List[str] = []
    for name, value in kwargs.items():
        if name in params:
            kept[name] = value
            continue
        replacement = mapping.get(name)
        if replacement and replacement in params:
            kept[replacement] = value
            continue
        fallback = reverse.get(name)
        if fallback and fallback in params:
            kept[fallback] = value
            continue
        dropped.append(name)
    if dropped:
        _logger.warning(
            "supported_kwargs dropped unsupported kwargs for %s: %s",
            getattr(target, "__name__", target),
            dropped,
        )
    return kept, dropped
