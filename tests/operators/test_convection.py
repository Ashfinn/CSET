#!/usr/bin/env python3

"""Tests for convection diagnostics."""

import iris
import numpy as np

import CSET.operators.convection as convection


def test_cape_ratio():
    """Compare with precalculated ratio."""
    SBCAPE = iris.load_cube("tests/test_data/convection/SBCAPE.nc")
    MUCAPE = iris.load_cube("tests/test_data/convection/MUCAPE.nc")
    MUCIN = iris.load_cube("tests/test_data/convection/MUCIN.nc")
    cape_75 = convection.cape_ratio(SBCAPE, MUCAPE, MUCIN)
    precalculated_75 = iris.load_cube("tests/test_data/convection/ECFlagB.nc")
    assert np.allclose(cape_75.data, precalculated_75.data, atol=1e-5, equal_nan=True)
    # TODO: Test data clobbered by -75, so disabled until regenerated.
    # It seems to be passing for some reason...? Maybe too tolerant?
    cape_1p5 = convection.cape_ratio(SBCAPE, MUCAPE, MUCIN, MUCIN_thresh=-1.5)
    precalculated_1p5 = iris.load_cube("tests/test_data/convection/ECFlagB_2.nc")
    assert np.allclose(cape_1p5.data, precalculated_1p5.data, atol=1e-5, equal_nan=True)


def test_inflow_layer_properties():
    """Compare with precalculated properties."""
    EIB = iris.load_cube("tests/test_data/convection/EIB.nc")
    BLheight = iris.load_cube("tests/test_data/convection/BLheight.nc")
    Orography = iris.load_cube("tests/test_data/convection/Orography.nc")
    inflow_layer_properties = convection.inflow_layer_properties(
        EIB, BLheight, Orography
    )
    precalculated = iris.load_cube("tests/test_data/convection/ECFlagD.nc")
    assert np.allclose(inflow_layer_properties.data, precalculated.data)
