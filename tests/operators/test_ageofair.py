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

"""Tests age of air operators."""

import numpy as np

import CSET.operators.ageofair as aoa_operators


def test_calc_dist():
    """
    Test distance calculated from calc_dist in age of air calculation.

    Allow a tolerance of 20km (TBD, expect some error).
    """
    # London and Johannesburg coordinates to 2 decimal places.
    london_coords = (51.51, -0.13)
    johanbg_coords = (-26.21, 28.03)

    dist = aoa_operators.calc_dist(london_coords, johanbg_coords)
    actual_distance = 9068670  # Air line according to Google?!

    assert np.allclose(dist, actual_distance, rtol=1e-06, atol=20000)
