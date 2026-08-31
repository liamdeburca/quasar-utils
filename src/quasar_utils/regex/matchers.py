from collections.abc import Callable, Iterable, Iterator
from re import Pattern, fullmatch
from re import compile as re_compile

from quasar_typing.re import Pattern_, StrPattern


def match(
    pattern: StrPattern | Callable[[str], bool],
    candidates: Iterable[str],
) -> Iterator[bool]:
    """
    Return a boolean iterator indicating whether each candidate matches the 
    given pattern.
    """
    if isinstance(pattern, str):
        pattern = re_compile(pattern)

    if isinstance(pattern, Pattern):
        def match_func(candidate: str) -> bool:
            return fullmatch(pattern, candidate) is not None
    else:
        match_func = pattern

    yield from map(match_func, candidates)


def find_matches(
    pattern: StrPattern | Callable[[str], bool],
    candidates: Iterable[str],
) -> Iterator[str]:
    """
    Return an iterator of candidates that match the given pattern.
    """
    _match = match(pattern, candidates)
    yield from (
        candidate 
        for candidate, is_match in zip(candidates, _match) 
        if is_match
    )


def match_multiple(
    patterns: tuple[StrPattern | Pattern_, ...],
    candidates: Iterable[str],
) -> Iterator[bool]:
    """
    Return a boolean iterator indicating whether each candidate matches any of
    the given patterns.
    """
    patterns = tuple(
        re_compile(p) 
            if isinstance(p, str)
            else p
        for p in patterns
    )
    def match_func(candidate: str) -> bool:
        return any(
            fullmatch(p, candidate) is not None 
            for p in patterns
        )

    yield from match(match_func, candidates)


def find_matches_multiple(
    patterns: tuple[StrPattern, ...],
    candidates: Iterable[str],
) -> Iterator[str]:
    """
    Return an iterator of candidates that match any of the given patterns.
    """
    _match = match_multiple(patterns, candidates)
    yield from (
        candidate 
        for candidate, is_match in zip(candidates, _match) 
        if is_match
    )