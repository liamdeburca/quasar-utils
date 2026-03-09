from pydantic import validate_call
from quasar_typing.pathlib import AbsoluteFileLike
from quasar_typing.misc.logger import Logger_

def check_if_comment(string:str):
    return string[0] == '#'

def trim_line(line:str):
    out = []
    for string in line.strip().split():
        if check_if_comment(string): break
        out.append(string)
    return out

@validate_call(validate_return=False)
def get_lines_from_file(
    hdr: str,
    path: AbsoluteFileLike,
    logger: Logger_ | None = None,
) -> list[list[str]]:

    if logger is not None:
        logger.debug(f"Reading block '{hdr}' of path '{path}':")

    with open(path) as f:
        all_lines = [
            line \
            for line in map(trim_line, f.readlines()) \
            if len(line) > 0
        ]

    if logger is not None:
        logger.debug(f">>> Found {len(all_lines)} lines in total.")        

    lines = []
    active = False
    for line in all_lines:
        if line[0] == hdr.upper():
            active = True
            continue

        if active and line[0].isupper():
            break
        elif active and len(line) >= 2:
            lines.append(line)

    if logger is not None:
        logger.debug(f">>> Found {len(lines)} relevant lines.")

    return lines