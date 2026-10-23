from pydantic import validate_call
from quasar_typing.literals import LawName, MapName


@validate_call
def func1(law_name: LawName) -> str:
    return law_name

@validate_call
def func2(map_name: MapName) -> str:
    return map_name


def test_law_name_validation():
    assert func1("CCM89") == func1("ccm89")
    assert func1("O94") == func1("o94")


def test_map_name_validation():
    assert func2("SFD") == func2("sfd")
    assert func2("CSFD") == func2("csfd")

