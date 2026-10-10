from re import Pattern
from re import compile as re_compile
from typing import Self


class PatternBuilder:
    """Utility class for constructing re.Pattern instances easily."""

    def __init__(self, pattern: str):
        self.pattern: str = pattern

    def __str__(self) -> str:
        return self.pattern

    def __repr__(self) -> str:
        return f"PatternBuilder(pattern={self.pattern!r})"

    def prefix(self, prefix: str) -> Self:
        self.pattern = prefix + self.pattern
        return self

    def suffix(self, suffix: str) -> Self:
        self.pattern += suffix
        return self

    def suffixAny(self) -> Self:
        """
        Add a suffix that matches any character.
        """
        return self.suffix(".")

    def suffixAnyInt(self) -> Self:
        """
        Add a suffix that matches any single digit integer.
        """
        return self.suffix("[0-9]")

    def prefixAny(self) -> Self:
        """
        Add a prefix that matches any character.
        """
        return self.prefix(".")

    def suffixChars(self, *chars: str) -> Self:
        """
        Add a suffix that matches any character in the provided selection.
        """
        return self.suffix(f"[{''.join(chars)}]")

    def prefixChars(self, *chars: str) -> Self:
        """
        Add a prefix that matches any character in the provided selection.
        """
        return self.prefix(f"[{''.join(chars)}]")

    def suffixDigitRange(self, start: int, end: int) -> Self:
        """
        Add a suffix that matches digits in the specified range.
        """
        return self.suffix(f"[{start}-{end}]")

    def prefixDigitRange(self, start: int, end: int) -> Self:
        """
        Add a prefix that matches digits in the specified range.
        """
        return self.prefix(f"[{start}-{end}]")

    def suffixNonCapturingStart(self) -> Self:
        return self.suffix("(?:")

    def suffixNonCapturingEnd(self, optional: bool = False) -> Self:
        self.pattern += ")"
        if optional:
            self.pattern += "?"
        return self

    def suffixContinuing(self) -> Self:
        """
        Add a suffix that allows for continuation of the pattern.
        """
        return self.suffix(r"+")

    def compile(self) -> Pattern:
        """
        Compile the pattern string to a re.Pattern.

        Returns:
            A compiled regular expression pattern.
        """
        return re_compile(self.pattern)
    