#!/usr/bin/env python3

"""Tests for convection diagnostics."""

import iris
import numpy as np

import CSET.operators.convection as convection


def test_cape_ratio():
    """Compare with precalculated ratio."""
    precalculated = iris.load_cube("tests/test_data/convection/ECFlagB.nc")
    precalculated_2 = iris.load_cube("tests/test_data/convection/ECFlagB_2.nc")
    SBCAPE = iris.load_cube("tests/test_data/convection/SBCAPE.nc")
    MUCAPE = iris.load_cube("tests/test_data/convection/MUCAPE.nc")
    MUCIN = iris.load_cube("tests/test_data/convection/MUCIN.nc")
    assert np.allclose(
        convection.cape_ratio(SBCAPE, MUCAPE, MUCIN).data, precalculated.data
    )
    assert np.allclose(
        convection.cape_ratio(SBCAPE, MUCAPE, MUCIN, MUCIN_thresh=-1.5).data,
        precalculated_2.data,
    )


def test_inflow_layer_properties():
    """Compare with precalculated properties."""
    precalculated = iris.load_cube("tests/test_data/convection/ECFlagD.nc")
    EIB = iris.load_cube("tests/test_data/convection/EIB.nc")
    BLheight = iris.load_cube("tests/test_data/convection/BLheight.nc")
    Orography = iris.load_cube("tests/test_data/convection/Orography.nc")
    inflow_layer_properties = convection.inflow_layer_properties(
        EIB, BLheight, Orography
    )
    assert np.allclose(inflow_layer_properties.data, precalculated.data)
