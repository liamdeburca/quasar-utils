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

    def suffixAny(self) -> Self:
        """
        Add a suffix that matches any character.
        """
        self.pattern += "."
        return self

    def suffixAnyInt(self) -> Self:
        """
        Add a suffix that matches any single digit integer.
        """
        self.pattern += r"\d"
        return self

    def prefixAny(self) -> Self:
        """
        Add a prefix that matches any character.
        """
        self.pattern = "." + self.pattern
        return self

    def suffixChars(self, *chars: str) -> Self:
        """
        Add a suffix that matches any character in the provided selection.
        """
        self.pattern += f"[{''.join(chars)}]"
        return self

    def prefixChars(self, *chars: str) -> Self:
        """
        Add a prefix that matches any character in the provided selection.
        """
        self.pattern = f"[{''.join(chars)}]" + self.pattern
        return self

    def suffixDigitRange(self, start: int, end: int) -> Self:
        """
        Add a suffix that matches digits in the specified range.
        """
        self.pattern += f"[{start}-{end}]"
        return self

    def prefixDigitRange(self, start: int, end: int) -> Self:
        """
        Add a prefix that matches digits in the specified range.
        """
        self.pattern = f"[{start}-{end}]" + self.pattern
        return self

    def compile(self) -> Pattern:
        """
        Compile the pattern string to a re.Pattern.

        Returns:
            A compiled regular expression pattern.
        """
        return re_compile(self.pattern)
    