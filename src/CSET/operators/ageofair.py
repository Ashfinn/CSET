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

"""Processing for the age of air operator."""

import datetime
import multiprocessing
import os
from functools import partial
from math import atan2, cos, radians, sin, sqrt

import iris
import numpy as np
from scipy.ndimage import gaussian_filter

iris.FUTURE.datum_support = True


def calc_dist(coord_1, coord_2):
    """Haversine distance in meters."""
    # Approximate radius of earth in km
    R = 6378.0

    lat1 = radians(coord_1[0])
    lon1 = radians(coord_1[1])
    lat2 = radians(coord_2[0])
    lon2 = radians(coord_2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance * 1000


def aoa_core(
    x_arr,
    y_arr,
    z_arr,
    g_arr,
    lats,
    lons,
    dt,
    plev_idx,
    timeunit,
    cyclic,
    tmpdir,
    lon_pnt,
):
    """
    Part of the multiprocessing capability.

    Requires access
    to the full array to do the back trajectory, so there is some scaling of increase
    no of cores and increased mem demands. For short forecasts, or small latitude arrays,
    there is not much benefit from multicore (more overhead from IO).

    Once complete, save latitude row to tmpdir as a ndarray.
    """
    # If already run, skip processing.
    if os.path.exists(tmpdir + "/aoa_frag_" + str(lon_pnt).zfill(4) + ".npy"):
        print("Already done", lon_pnt, ", skipping")
        return None

    # Initialise empty array to store age of air for this latitude strip.
    ageofair_local = np.zeros((x_arr.shape[0], x_arr.shape[2]))
    print("Working on ", lon_pnt)

    # Ignore leadtime 0 as this is trivial.
    for leadtime in range(1, x_arr.shape[0]):
        # Initialise leadtime slice with current leadtime.
        ageofair_local[leadtime, :] = leadtime * dt
        for lat_pnt in range(0, x_arr.shape[2]):
            # Gridpoint initialised as within lam by construction
            outside_lam = False

            # If final column, look at dist from prev column, otherwise look at next column.
            if lon_pnt == len(lons) - 1:
                ew_spacing = calc_dist(
                    (lats[lat_pnt], lons[lon_pnt]), (lats[lat_pnt], lons[lon_pnt - 1])
                )
            else:
                ew_spacing = calc_dist(
                    (lats[lat_pnt], lons[lon_pnt]), (lats[lat_pnt], lons[lon_pnt + 1])
                )

            # If final row, look at dist from row column, otherwise look at next row.
            if lat_pnt == len(lats) - 1:
                ns_spacing = calc_dist(
                    (lats[lat_pnt], lons[lon_pnt]), (lats[lat_pnt - 1], lons[lon_pnt])
                )
            else:
                ns_spacing = calc_dist(
                    (lats[lat_pnt], lons[lon_pnt]), (lats[lat_pnt + 1], lons[lon_pnt])
                )

            # Go through past timeslices
            for n in range(0, leadtime):
                # First step back, so we use i,j coords to find out parcel location
                # in terms of array point
                if n == 0:
                    x = lon_pnt
                    y = lat_pnt
                    z = plev_idx

                # Only seek preceding wind if its inside domain..
                if not outside_lam:
                    # Get vector profile at current time - nearest whole gridpoint.
                    u = x_arr[leadtime - n, int(z), int(y), int(x)]
                    v = y_arr[leadtime - n, int(z), int(y), int(x)]
                    w = z_arr[leadtime - n, int(z), int(y), int(x)]
                    g = g_arr[leadtime - n, int(z), int(y), int(x)]

                    # First, compute horizontal displacement using inverse of horizontal vector
                    # Convert m/s to m/[samplingrate]h, then m -> deg, then deg -> model gridpoints
                    # TODO: assume 1 degree is 111km displacement.
                    # Convert m/s to grid boxes per unit time.
                    if timeunit == "hour":
                        du = ((u * 60 * 60 * dt) / ew_spacing) * -1.0
                        dv = ((v * 60 * 60 * dt) / ns_spacing) * -1.0
                        dz = (w * 60 * 60 * dt) * -1.0

                    # Get column of geopot height.
                    g_col = g_arr[(leadtime - n), :, int(y), int(x)]

                    # New geopotential height of parcel - store 'capacity' between timesteps as vertical motions smaller.
                    if n == 0:
                        new_g = g + dz
                        pre_g = new_g
                    else:
                        new_g = pre_g + dz

                    # Calculate which geopot level is closest to new geopot level.
                    z = np.argmin(np.abs(g_col - new_g))

                    # Update x,y location based on displacement. Z already updated
                    x = x + du
                    y = y + dv

                    # If it is now outside domain, then save age and dont process further with outside lam flag
                    # Support cyclic domains like K-SCALE, where x coord out of domain gets moved through dateline.
                    if cyclic:
                        if (
                            x <= -1
                        ):  # as for example -0.3 would still be in domain, but x_arr.shape-0.3 would result in index error
                            x = x_arr.shape[3] + x  # wrap back around dateline
                        elif x >= x_arr.shape[3]:
                            x = x_arr.shape[3] - x
                    else:
                        if x < 0 or x >= x_arr.shape[3]:
                            ageofair_local[leadtime, lat_pnt] = n * dt
                            outside_lam = True

                    if y < 0 or y >= x_arr.shape[2]:
                        ageofair_local[leadtime, lat_pnt] = n * dt
                        outside_lam = True

    # Save 3d array containing age of air
    np.save(tmpdir + "/aoa_frag_" + str(lon_pnt).zfill(4) + ".npy", ageofair_local)


def compute_ageofair(
    cubelist, plev, includevertical=True, cyclic=False, tmpdir="/tmp/", multicore=None
):
    """
    Compute back trajectories for a particular timeslice of a forecast.

    Arguments:
        cubelist: an iris.CubeList object, containing 4 cubes (u, v, w component of wind, with geopotential
                  height. All cubes 4D, structured time, pressure, latitude, longitude. Assumed correct
                  ordering of arrays. Expects data to already be regridded onto a standard 0p5 degree grid.
        plev: pressure level  to evaluate the back trajectory on (etc 700 or 70000) depending on hPa or Pa.
        includevertical: If True, use vertical velocity vector to compute back trajectories. If False, then
                         assume flow laminar (set vertical velocity to zero).
        tmpdir: Location of a temporary directory that can be used to write intermediate files, as used
                when multiprocessing.
        multicore: If set as None, then run on a single thread. Otherwise, split over number of multicores.
    """
    # Filter cubelist and pull out required cubes (sometimes multiple with same name, need to check)
    for cube in cubelist:
        if cube.standard_name == "x_wind":
            x_wind = cube
        elif cube.standard_name == "y_wind":
            y_wind = cube
        elif cube.standard_name == "upward_air_velocity":
            z_wind = cube
        elif cube.standard_name == "geopotential_height":
            geopot = cube

    # Check that all cubes are of same dimension (some might differ slightly, extra y point often)
    if not x_wind.shape == y_wind.shape == z_wind.shape == geopot.shape:
        quit("Cubes not same dimlen! Quitting...")

    # Get time units and assign for later
    if str(x_wind.coord("time").units) == "hours since 1970-01-01 00:00:00":
        timeunit = "hour"
    else:
        quit("Unsupported time base! Quitting...")

    # Make data non-lazy to speed up code.
    print("Making data non-lazy...")
    x_arr = x_wind.data
    y_arr = y_wind.data
    z_arr = z_wind.data
    g_arr = geopot.data

    # Smooth vertical velocity if using, to 2sigma (standard for 0.5 degree).
    # TODO could this be a user input, or should it scale with resolution?
    if includevertical:
        print("Smoothing vertical velocity...")
        for t in range(0, z_arr.shape[0]):
            for p in range(0, z_arr.shape[1]):
                z_arr[t, p, :, :] = gaussian_filter(
                    z_arr[t, p, :, :], [2, 2], mode="nearest"
                )
    else:
        z_arr[:] = 0

    # Get time spacing of cube -
    dt = x_wind.coord("time").points[1:] - x_wind.coord("time").points[:-1]
    if np.all(dt == dt[0]):
        dt = dt[0]
    else:
        quit("Time not monotonically increasing, not supported...")

    # Get coord points
    lats = x_wind.coord("latitude").points
    lons = x_wind.coord("longitude").points
    time = x_wind.coord("time").points

    # Get array index for user specified pressure level.
    try:
        plev_idx = np.where(x_wind.coord("pressure").points == plev)[0][0]
    except IndexError:
        print("Can't find plev ", str(plev), " in ", x_wind.coord("pressure").points)
        quit()

    # Initialise cube containing age of air.
    ageofair_cube = iris.cube.Cube(
        np.zeros((len(time), len(lats), len(lons))),
        long_name="age_of_air",
        dim_coords_and_dims=[
            (x_wind.coord("time"), 0),
            (x_wind.coord("latitude"), 1),
            (x_wind.coord("longitude"), 2),
        ],
    )

    print("STARTING AOA DIAG...")
    start = datetime.datetime.now()
    if multicore is not None:
        # Multiprocessing on each longitude slice
        # TODO: there was an error where 719 was passed to x idx.
        pool = multiprocessing.Pool(multicore)
        func = partial(
            aoa_core,
            np.copy(x_arr),
            np.copy(y_arr),
            np.copy(z_arr),
            np.copy(g_arr),
            lats,
            lons,
            dt,
            plev_idx,
            timeunit,
            cyclic,
            tmpdir,
        )
        pool.map(func, range(0, x_wind.shape[3]))
    else:
        # Single core - better for debugging.
        for i in range(0, x_wind.shape[3]):
            aoa_core(
                x_arr,
                y_arr,
                z_arr,
                g_arr,
                lats,
                lons,
                dt,
                plev_idx,
                timeunit,
                cyclic,
                tmpdir,
                i,
            )

    # Verbose for time taken to run, and collate tmp ndarrays into final cube, and return
    print("AOA DIAG DONE, took", (datetime.datetime.now() - start).total_seconds(), "s")
    for i in range(0, x_wind.shape[3]):
        ageofair_cube.data[:, :, i] = np.load(
            tmpdir + "/aoa_frag_" + str(i).zfill(4) + ".npy"
        )
        os.remove(tmpdir + "/aoa_frag_" + str(i).zfill(4) + ".npy")

    return ageofair_cube
