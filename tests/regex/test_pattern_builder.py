import re

import pytest

from quasar_utils.regex.pattern_builder import PatternBuilder


@pytest.mark.parametrize("base,expected_pattern", [
    ("test", r"test\d"),
    ("word", r"word\d"),
    ("abc", r"abc\d"),
])
def test_suffix_any_int_pattern(base: str, expected_pattern: str) -> None:
    """Test that suffixAnyInt adds \\d to the pattern."""
    builder = PatternBuilder(base)
    builder.suffixAnyInt()
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,expected_pattern", [
    ("test", "test."),
    ("word", "word."),
    ("abc", "abc."),
])
def test_suffix_any_pattern(base: str, expected_pattern: str) -> None:
    """Test that suffixAny adds . to the pattern."""
    builder = PatternBuilder(base)
    builder.suffixAny()
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,expected_pattern", [
    ("test", ".test"),
    ("word", ".word"),
    ("abc", ".abc"),
])
def test_prefix_any_pattern(base: str, expected_pattern: str) -> None:
    """Test that prefixAny adds . to the beginning of the pattern."""
    builder = PatternBuilder(base)
    builder.prefixAny()
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base", [
    "test",
    "word",
    "abc",
])
def test_compile_returns_pattern(base: str) -> None:
    """Test that compile returns a re.Pattern instance."""
    builder = PatternBuilder(base)
    pattern = builder.compile()
    assert isinstance(pattern, re.Pattern)


@pytest.mark.parametrize("base,test_string,should_match", [
    ("test", "test5", True),
    ("test", "test", False),
    ("test", "testa", False),
    ("word", "word0", True),
    ("word", "word9", True),
    ("abc", "abc10", False),
])
def test_suffix_any_int_matching(
    base: str,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that compiled patterns with suffixAnyInt match correctly."""
    pattern = PatternBuilder(base).suffixAnyInt().compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,test_string,should_match", [
    ("test", "testa", True),
    ("test", "test5", True),
    ("test", "test", False),
    ("word", "wordx", True),
    ("abc", "abc!", True),
])
def test_suffix_any_matching(
    base: str,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that compiled patterns with suffixAny match correctly."""
    pattern = PatternBuilder(base).suffixAny().compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,test_string,should_match", [
    ("test", "atest", True),
    ("test", "5test", True),
    ("test", "test", False),
    ("word", "xword", True),
    ("abc", "!abc", True),
])
def test_prefix_any_matching(
    base: str,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that compiled patterns with prefixAny match correctly."""
    pattern = PatternBuilder(base).prefixAny().compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,methods", [
    ("test", ["suffixAnyInt", "prefixAny"]),
    ("word", ["prefixAny", "suffixAny"]),
    ("abc", ["suffixAny", "suffixAnyInt"]),
])
def test_method_chaining(base: str, methods: list[str]) -> None:
    """Test that methods can be chained by returning self."""
    builder = PatternBuilder(base)
    for method_name in methods:
        method = getattr(builder, method_name)
        result = method()
        assert result is builder


@pytest.mark.parametrize("base,expected_pattern", [
    ("test", ".test\\d"),
    ("word", ".word."),
    ("abc", ".abc\\d"),
])
def test_chained_operations_pattern(base: str, expected_pattern: str) -> None:
    """Test that chained operations produce the expected pattern."""
    builder = PatternBuilder(base)
    if expected_pattern == ".test\\d":
        builder.prefixAny().suffixAnyInt()
    elif expected_pattern == ".word.":
        builder.prefixAny().suffixAny()
    elif expected_pattern == ".abc\\d":
        builder.prefixAny().suffixAnyInt()
    
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,chars,expected_pattern", [
    ("test", "abc", r"test[abc]"),
    ("word", "xyz", r"word[xyz]"),
    ("abc", "0123", r"abc[0123]"),
])
def test_suffix_chars_pattern(base: str, chars: str, expected_pattern: str) -> None:
    """Test that suffixChars adds character class to suffix."""
    builder = PatternBuilder(base)
    builder.suffixChars(chars)
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,chars,expected_pattern", [
    ("test", "abc", r"[abc]test"),
    ("word", "xyz", r"[xyz]word"),
    ("abc", "0123", r"[0123]abc"),
])
def test_prefix_chars_pattern(base: str, chars: str, expected_pattern: str) -> None:
    """Test that prefixChars adds character class to prefix."""
    builder = PatternBuilder(base)
    builder.prefixChars(chars)
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,chars,test_string,should_match", [
    ("test", "abc", "testa", True),
    ("test", "abc", "testb", True),
    ("test", "abc", "testc", True),
    ("test", "abc", "testd", False),
    ("test", "abc", "test", False),
    ("word", "xyz", "wordx", True),
    ("word", "xyz", "wordq", False),
])
def test_suffix_chars_matching(
    base: str,
    chars: str,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that suffixChars matches correctly."""
    pattern = PatternBuilder(base).suffixChars(chars).compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,chars,test_string,should_match", [
    ("test", "abc", "atest", True),
    ("test", "abc", "btest", True),
    ("test", "abc", "ctest", True),
    ("test", "abc", "dtest", False),
    ("test", "abc", "test", False),
    ("word", "xyz", "xword", True),
    ("word", "xyz", "qword", False),
])
def test_prefix_chars_matching(
    base: str,
    chars: str,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that prefixChars matches correctly."""
    pattern = PatternBuilder(base).prefixChars(chars).compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,start,end,expected_pattern", [
    ("test", 0, 9, r"test[0-9]"),
    ("word", 1, 5, r"word[1-5]"),
    ("abc", 3, 7, r"abc[3-7]"),
])
def test_suffix_digit_range_pattern(
    base: str,
    start: int,
    end: int,
    expected_pattern: str,
) -> None:
    """Test that suffixDigitRange adds digit range to suffix."""
    builder = PatternBuilder(base)
    builder.suffixDigitRange(start, end)
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,start,end,expected_pattern", [
    ("test", 0, 9, r"[0-9]test"),
    ("word", 1, 5, r"[1-5]word"),
    ("abc", 3, 7, r"[3-7]abc"),
])
def test_prefix_digit_range_pattern(
    base: str,
    start: int,
    end: int,
    expected_pattern: str,
) -> None:
    """Test that prefixDigitRange adds digit range to prefix."""
    builder = PatternBuilder(base)
    builder.prefixDigitRange(start, end)
    assert builder.pattern == expected_pattern


@pytest.mark.parametrize("base,start,end,test_string,should_match", [
    ("test", 0, 9, "test0", True),
    ("test", 0, 9, "test5", True),
    ("test", 0, 9, "test9", True),
    ("test", 0, 9, "testa", False),
    ("word", 1, 5, "word1", True),
    ("word", 1, 5, "word0", False),
    ("word", 1, 5, "word9", False),
])
def test_suffix_digit_range_matching(
    base: str,
    start: int,
    end: int,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that suffixDigitRange matches correctly."""
    pattern = PatternBuilder(base).suffixDigitRange(start, end).compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)


@pytest.mark.parametrize("base,start,end,test_string,should_match", [
    ("test", 0, 9, "0test", True),
    ("test", 0, 9, "5test", True),
    ("test", 0, 9, "9test", True),
    ("test", 0, 9, "atest", False),
    ("word", 1, 5, "1word", True),
    ("word", 1, 5, "0word", False),
    ("word", 1, 5, "9word", False),
])
def test_prefix_digit_range_matching(
    base: str,
    start: int,
    end: int,
    test_string: str,
    should_match: bool,
) -> None:
    """Test that prefixDigitRange matches correctly."""
    pattern = PatternBuilder(base).prefixDigitRange(start, end).compile()
    match = pattern.fullmatch(test_string)
    assert should_match ^ (match is None)
