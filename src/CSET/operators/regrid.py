# Copyright 2024 Met Office and contributors.
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

"""Operators to regrid cubes."""


import iris
import iris.cube
import numpy as np


def regrid_onto_cube(
    incube: iris.cube.Cube, targetcube: iris.cube.Cube, regridmethod: str, **kwargs
) -> iris.cube.Cube:
    """Regrid a cube, projecting onto a target cube.

    Cube must have minimum 2 dimensions, but can have more dimensions.

    Arguments
    ----------
    incube: Cube
        An iris cube of the data to regrid. As a minimum, it needs to be 2D with a latitude,
        longitude coordinates.
    targetcube: Cube
        An iris cube of the data to regrid onto. It needs to be 2D with a latitude,
        longitude coordinate, though I think more dims are acceptable and are ignored.
    regridmethod: str
        Method used to regrid onto, etc. Linear will use iris.analysis.Linear()

    Returns
    -------
    iris.cube.Cube
        An iris cube of the data that has been regridded.

    Raises
    ------
    ValueError
        If a unique x/y coordinate cannot be found
    NotImplementedError
        If the cubes grid, or the method for regridding, is not yet supported.
    """
    # Usual names for spatial coordinates
    X_COORD_NAMES = ["longitude", "grid_longitude", "projection_x_coordinate", "x"]
    Y_COORD_NAMES = ["latitude", "grid_latitude", "projection_y_coordinate", "y"]

    # Get a list of coordinate names for the cube
    coord_names = [coord.name() for coord in incube.coords()]

    # Check which x-coordinate we have, if any
    x_coords = [coord for coord in coord_names if coord in X_COORD_NAMES]
    if len(x_coords) != 1:
        raise ValueError("Could not identify a unique x-coordinate in cube")
    x_coord = incube.coord(x_coords[0])

    # Check which y-coordinate we have, if any
    y_coords = [coord for coord in coord_names if coord in Y_COORD_NAMES]
    if len(y_coords) != 1:
        raise ValueError("Could not identify a unique y-coordinate in cube")
    y_coord = incube.coord(y_coords[0])

    # List of supported grids - check if it is compatible
    supported_grids = [iris.coord_systems.GeogCS]
    if type(incube.coord(x_coord).coord_system) not in supported_grids:
        raise NotImplementedError(
            f"Does not currently support {incube.coord(x_coord).coord_system} regrid method"
        )
    elif type(incube.coord(y_coord).coord_system) not in supported_grids:
        raise NotImplementedError(
            f"Does not currently support {incube.coord(y_coord).coord_system} regrid method"
        )

    if regridmethod == "Linear":
        return incube.regrid(targetcube, iris.analysis.Linear())
    else:
        raise NotImplementedError(
            f"Does not currently support {regridmethod} regrid method"
        )


def regrid_onto_xyspacing(
    incube: iris.cube.Cube, xspacing: int, yspacing: int, regridmethod: str, **kwargs
) -> iris.cube.Cube:
    """Regrid cube onto a set x,y spacing.

    Regrid cube using specified x,y spacing, which is performed linearly.

    Parameters
    ----------
    incube: Cube
        An iris cube of the data to regrid. As a minimum, it needs to be 2D with a latitude,
        longitude coordinates.
    xspacing: integer
        Spacing of points in longitude direction (could be degrees, meters etc.)
    yspacing: integer
        Spacing of points in latitude direction (could be degrees, meters etc.)
    regridmethod: str
        Method used to regrid onto, etc. Linear will use iris.analysis.Linear()

    Returns
    -------
    cube_rgd: Cube
        An iris cube of the data that has been regridded.

    Raises
    ------
    ValueError
        If a unique x/y coordinate cannot be found
    NotImplementedError
        If the cubes grid, or the method for regridding, is not yet supported.
    """
    # Usual names for spatial coordinates
    X_COORD_NAMES = ["longitude", "grid_longitude", "projection_x_coordinate", "x"]
    Y_COORD_NAMES = ["latitude", "grid_latitude", "projection_y_coordinate", "y"]

    # Get a list of coordinate names for the cube
    coord_names = [coord.name() for coord in incube.coords()]

    # Check which x-coordinate we have, if any
    x_coords = [coord for coord in coord_names if coord in X_COORD_NAMES]
    if len(x_coords) != 1:
        raise ValueError("Could not identify a unique x-coordinate in cube")
    x_coord = incube.coord(x_coords[0])

    # Check which y-coordinate we have, if any
    y_coords = [coord for coord in coord_names if coord in Y_COORD_NAMES]
    if len(y_coords) != 1:
        raise ValueError("Could not identify a unique y-coordinate in cube")
    y_coord = incube.coord(y_coords[0])

    # List of supported grids - check if it is compatible
    supported_grids = [iris.coord_systems.GeogCS]
    if type(incube.coord(x_coord).coord_system) not in supported_grids:
        raise NotImplementedError(
            f"Does not currently support {incube.coord(x_coord).coord_system} regrid method"
        )
    elif type(incube.coord(y_coord).coord_system) not in supported_grids:
        raise NotImplementedError(
            f"Does not currently support {incube.coord(y_coord).coord_system} regrid method"
        )

    # Get axis
    lat, lon = incube.coord(y_coord), incube.coord(x_coord)

    # Get bounds
    lat_min, lon_min = lat.points.min(), lon.points.min()
    lat_max, lon_max = lat.points.max(), lon.points.max()

    # Generate new mesh
    latout = np.arange(lat_min, lat_max, yspacing)
    lonout = np.arange(lon_min, lon_max, xspacing)

    if regridmethod == "Linear":
        cube_rgd = incube.interpolate(
            [(y_coord, latout), (x_coord, lonout)], iris.analysis.Linear()
        )
    else:
        raise NotImplementedError(
            f"Does not currently support {regridmethod} regrid method"
        )

    return cube_rgd
