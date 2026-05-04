from typing import Literal
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase, digits
from random import choice

def get_yes_or_no(
    prompt: str,
    default: str = 'y',
    max_tries: int = 10,
) -> Literal['y', 'n']:
    """
    Prompts the user for a yes or no answer until a valid response is given,
    or until the maximum number of tries is reached. If the maximum number of
    tries is reached without a valid input, the default answer is returned.
    """
    prefix = None
    tries = 1
    while tries <= max_tries:

        if tries == max_tries:
            prefix = f"Last try! Will default to '{default}'." + '\n' + prefix

        user_input = input(prompt if prefix is None else prefix + '\n' + prompt)
        if not user_input: user_input = default

        if (len(user_input) == 1) and (user_input.lower() in 'yn'):
            return user_input.lower()
        
        prefix = "Please type either 'y', 'n', or the enter key ({})! [{}/{}]" \
            .format(default, tries, max_tries)
        
        tries += 1

    return default

def get_alternative_dir(
    prompt: str,
    directory: str | Path,
    max_tries: int = 10,
) -> Path:
    """
    Prompts the user to input an alternative directory name until a valid,
    non-existing directory name is provided, or until the maximum number of
    tries is reached. If the maximum number of tries is reached without a valid
    input, a random directory name is generated.
    """
    if isinstance(directory, str):
        directory: Path = Path(directory)
    
    prefix = None

    tries = 1
    while tries <= max_tries:

        if tries == max_tries:
            prefix = "Last try! Will randomise name if failed again!\n" + prefix

        user_input: str = \
            input(prompt if prefix is None else prefix + '\n' + prompt)

        new_path: Path = directory / user_input
        if not new_path.exists(): return new_path
        
        prefix = f"Please try again! [{tries}/{max_tries}]"
        tries += 1

    random_name = ''.join(
        choice(ascii_lowercase + ascii_uppercase + digits) \
        for _ in range(8)
    ) + '_out'

    return directory / random_name