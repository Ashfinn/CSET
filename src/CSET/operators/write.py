# Copyright 2022 Met Office and contributors.
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

"""Operators for writing various types of files to disk."""

from pathlib import Path
from typing import Union

import iris
import iris.cube

from CSET._common import get_recipe_metadata, slugify


def write_cube_to_nc(
    cube: Union[iris.cube.Cube, iris.cube.CubeList], filename: Path = None, **kwargs
) -> str:
    """Write a cube to a NetCDF file.

    This operator expects an iris cube object that will then be passed to MET
    for further processing.

    Arguments
    ---------
    cube: iris.cube.Cube | iris.cube.CubeList
        Data to save.
    filename: Path, optional
        Path to save the cubes too. Defaults to the recipe title + .nc

    Returns
    -------
    Cube | CubeList
        The inputted cube(list) (so further operations can be applied)
    """
    if filename is None:
        filename = slugify(get_recipe_metadata().get("title", "Untitled"))
    # Ensure that output filename is a Path with a .nc suffix
    filename = Path(filename).with_suffix(".nc")
    # Save the file as nc compliant (iris should handle this)
    iris.save(cube, filename, zlib=True)
    return cube
