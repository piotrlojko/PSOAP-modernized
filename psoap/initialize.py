"""
Initialize a new working directory for PSOAP inference.

Usage::

    psoap-initialize --model SB2
    psoap-initialize --check
"""

import sys
import importlib.resources
import psoap


def main():
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Initialize a new directory for PSOAP inference.")
    parser.add_argument(
        "--check", action="store_true",
        help="Check whether the package was installed properly.")
    parser.add_argument(
        "--model", choices=["SB2", "ST2", "ST3"],
        help="Which type of model to use: SB2 (double-lined binary), "
             "ST2 (double-lined tertiary), or ST3 (triple-lined tertiary).")
    args = parser.parse_args()

    if args.check:
        print("PSOAP successfully installed and linked.")
        print("Using Python Version", sys.version)
        sys.exit()
    else:
        # ST2 and ST3 are hooks — fall back to SB2 config if no dedicated one
        model = args.model or "SB2"
        config_name = "config.{}.yaml".format(model)

        try:
            ref = importlib.resources.files("psoap").joinpath("data/" + config_name)
            shutil.copy(str(ref), "config.yaml")
        except (FileNotFoundError, TypeError):
            # Fall back for Python < 3.9 or missing resource
            import pkg_resources
            config = pkg_resources.resource_filename("psoap", "data/" + config_name)
            shutil.copy(config, "config.yaml")

        print("Copied config file for {} model to config.yaml".format(model))
        print("Edit config.yaml to point to your spectra_list file and "
              "adjust orbital/GP parameters.")


if __name__ == "__main__":
    main()
