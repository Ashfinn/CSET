# Copyright 2022-2023 Met Office and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CSET: Convective Scale Evaluation Tool
"""

import argparse
import logging
from pathlib import Path
import os


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="cset", description="Convective Scale Evaluation Tool"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase output verbosity, may be specified multiple times",
    )
    # https://docs.python.org/3/library/argparse.html#sub-commands
    subparsers = parser.add_subparsers(title="subcommands", dest="subparser")

    # Run operator chain
    parser_bake = subparsers.add_parser("bake", help="run a recipe file")
    parser_bake.add_argument("input_file", type=Path, help="input file to read")
    parser_bake.add_argument("output_file", type=Path, help="output file to write")
    parser_bake.add_argument(
        "recipe_file",
        type=Path,
        nargs="?",
        help="recipe file to execute. If omitted reads from CSET_RECIPE environment variable",
        default=os.getenv("CSET_RECIPE"),
    )
    parser_bake.set_defaults(func=_bake_command)

    parser_graph = subparsers.add_parser("graph", help="visualise a recipe file")
    parser_graph.add_argument(
        "recipe",
        type=Path,
        nargs="?",
        help="recipe file to read. If omitted reads from CSET_RECIPE environment variable",
        default=os.getenv("CSET_RECIPE"),
    )
    parser_graph.add_argument(
        "-o",
        "--output_path",
        type=Path,
        nargs="?",
        help="file in which to save the graph image, otherwise uses a temporary file. When specified the file is not automatically opened",
        default=None,
    )
    parser_graph.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="include operator arguments in output",
    )
    parser_graph.set_defaults(func=_render_graph)

    parser_unpack = subparsers.add_parser(
        "unpack-recipes", help="unpack default recipe files to a folder"
    )
    parser_unpack.add_argument(
        "recipe_dir",
        type=Path,
        nargs="?",
        help='directory to save recipes. If omitted creates "recipes" directory in $PWD',
        default=None,
    )
    parser_unpack.set_defaults(func=_unpack_recipes)

    args = parser.parse_args()

    # Logging verbosity
    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)

    if args.subparser:
        args.func(args)
    else:
        parser.print_help()


def _bake_command(args):
    from CSET.operators import execute_recipe

    execute_recipe(args.recipe_file, args.input_file, args.output_file)


def _render_graph(args):
    from CSET.graph import save_graph

    save_graph(
        args.recipe,
        args.output_path,
        auto_open=not bool(args.output_path),
        detailed=args.detailed,
    )


def _unpack_recipes(args):
    from CSET.recipes import unpack

    if not args.recipe_dir:
        args.recipe_dir = Path.cwd().joinpath("recipes")
    unpack(args.recipe_dir)
