from argparse import ArgumentParser
from pathlib import Path

from quasar_typing.pathlib import AnyAbsoluteJSONPath, AnyAbsoluteYAMLPath

from quasar_utils.setup import Info


def info_to_json(path: AnyAbsoluteJSONPath) -> None:
    info = Info()
    info.to_json(path)

def info_to_yaml(path: AnyAbsoluteYAMLPath) -> None:
    info = Info()
    info.to_yaml(path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Create a template info.json or info.yaml file.")
    parser.add_argument(
        "-p", "--path", 
        type=str, 
        required=True, 
        help="Path to the output file (info.json or info.yaml)."
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force overwrite if the file already exists."
    )
    args = parser.parse_args()

    output_path = Path(args.path).resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"The file {output_path} already exists. Use --force to overwrite."
        )
    
    if output_path.suffix == ".json":
        info_to_json(output_path)
    elif output_path.suffix == ".yaml":
        info_to_yaml(output_path)
    else:
        raise ValueError("Output path must end with .json or .yaml")
