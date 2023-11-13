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

"""Test collapse operators."""

import pytest

from CSET.operators import collapse, constraints, filters, read


def test_collapse():
    """Reduces dimension of cube."""
    cubes = read.read_cubes("tests/test_data/air_temp.nc")
    constraint = constraints.combine_constraints(
        constraints.generate_stash_constraint("m01s03i236"),
        a=constraints.generate_cell_methods_constraint([]),
    )
    cube = filters.filter_cubes(cubes, constraint)

    # Test collapsing a single coordinate.
    collapsed_cube = collapse.collapse(cube, "time", "MEAN")
    expected_cube = (
        "<iris 'Cube' of air_temperature / (K) (grid_latitude: 17; grid_longitude: 13)>"
    )
    assert repr(collapsed_cube) == expected_cube

    # Test collapsing two coordinates.
    collapsed_cube = collapse.collapse(
        cube, ["grid_latitude", "grid_longitude"], "MEAN"
    )
    expected_cube = "<iris 'Cube' of air_temperature / (K) (time: 3)>"
    assert repr(collapsed_cube) == expected_cube


def test_collapse_percentile():
    """Reduce dimension of a cube with a PERCENTILE aggregation."""
    cubes = read.read_cubes("tests/test_data/air_temp.nc")
    constraint = constraints.combine_constraints(
        constraints.generate_stash_constraint("m01s03i236"),
        a=constraints.generate_cell_methods_constraint([]),
    )
    cube = filters.filter_cubes(cubes, constraint)

    with pytest.raises(ValueError):
        collapse.collapse(cube, "time", "PERCENTILE")

    # Test collapsing a single coordinate.
    collapsed_cube = collapse.collapse(
        cube, "time", "PERCENTILE", additional_percent=75
    )
    expected_cube = (
        "<iris 'Cube' of air_temperature / (K) (grid_latitude: 17; grid_longitude: 13)>"
    )
    assert repr(collapsed_cube) == expected_cube
