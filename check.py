#! /usr/bin/python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


# Table row	|  matching file
# Data	| (this is the DINO reference)
# No Constraint	| generated_restart_no_C_1_4deg.nc
# Constraint | 	generated_C1_restart.nc


def read_numpy(mesh_mask, np_data,  depth=1500):
    """
    Read numpy file and convert to xarray DataArray.

    This operates on low resolution npy data generated from GORCE spinup
    Shape is (batch, lev, lat, lon) : e.g. (8, 36, 199, 62)
    """
    # print("Dtype:", toce.dtype)

    # print("Min:", np.nanmin(toce))
    # print("Max:", np.nanmax(toce))
    # print("Mean:", np.nanmean(toce))

    # print("numpy vertical levels: ", toce.shape[1])
    # # print("Sample slice (level 0):")
    # print("numpy shape: ", toce[0].shape)
    mesh_mask = mesh_mask.rename({"y": "nav_lat", "x": "nav_lon"})

    toce_da = xr.DataArray(
        # toce.mean(axis=0),
        toce[0],
        dims=("nav_lev", "nav_lat", "nav_lon"),
        coords={
            "nav_lev": mesh_mask["nav_lev"],  # 1D depth index
            "nav_lat": (("nav_lat", "nav_lon"), mesh_mask["nav_lat"].values),
            "nav_lon": (("nav_lat", "nav_lon"), mesh_mask["nav_lon"].values),
        },
        name="toce",
        attrs={
            "long_name": "Conservative temperature (generated)",
            "units": "degC",  # or whatever is correct for your file
        },
    )

    # rebuild toce_da from mesh_mask.e3t_0 if needed
    data = mesh_mask.e3t_0.copy()
    data[:] = toce[0]

    ## this is the fix.
    # data = data.assign_coords(
    #     {
    #         "nav_lat": (("nav_lat", "nav_lon"), mesh_mask["nav_lat"].values),
    #         "nav_lon": (("nav_lat", "nav_lon"), mesh_mask["nav_lon"].values),
    #     }
    # )

    return BWbox(data, mesh_mask, depth).item(), BWbox(toce_da, mesh_mask, depth).item()


def BWbox_error(thetao, file_mask, depth_box=3000):
    """
    Metric Extraction Function : Bottom Water Box Temperature.

    Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below
    3000m or beyond 30 degrees of latitude North and South.

    ________________________________________________ _Surface
    | . . . . |__________________________| . . . . |_500m
    | . . . . |                          | . . . . |
    | . . . . |        Deep Water        | . . . . |
    | . . . . |__________________________| . . . . |_3000m
    | . . . . . . . . Bottom Water . . . . . . . . |
    |______________________________________________|_Bottom
    S        30S           Eq.          30N        N

    Figure : Schematic Representation of the Bottom Water box used in this metric.

    Unit : °C

    Input :
       -  thetao    : xarray.DataArray
       -  file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files
    """
    t_BW = thetao.where(1 - (thetao.nav_lev < depth_box) * (abs(thetao.y) < 30))

    # Computing Area Weights from Mask over Box
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    area_BW = (
        e1t
        * e2t
        * e3t
        * tmask.where(1 - (thetao.nav_lev < depth_box) * (abs(thetao.y) < 30))
    )

    # Returning Average Temperature on Box
    return (t_BW * area_BW).sum(dim=["y", "x", "nav_lev"]) / area_BW.sum(
        dim=["y", "x", "nav_lev"]
    )


def BWbox(thetao, file_mask, depth_box=3000):
    """
    Metric Extraction Function : Bottom Water Box Temperature.

    Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below
    3000m or beyond 30 degrees of latitude North and South.

    ________________________________________________ _Surface
    | . . . . |__________________________| . . . . |_500m
    | . . . . |                          | . . . . |
    | . . . . |        Deep Water        | . . . . |
    | . . . . |__________________________| . . . . |_3000m
    | . . . . . . . . Bottom Water . . . . . . . . |
    |______________________________________________|_Bottom
    S        30S           Eq.          30N        N

    Figure : Schematic Representation of the Bottom Water box used in this metric.

    Unit : °C

    Input :
       -  thetao    : xarray.DataArray
       -  file_mask : xarray.Dataset
    Output :
       - np.float32 or np.float64 depending on recording precision of simulation files
    """
    t_BW = thetao.where(1 - (thetao.nav_lev < depth_box) * (abs(thetao.nav_lat) < 30))

    # Computing Area Weights from Mask over Box
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    area_BW = (
        e1t
        * e2t
        * e3t
        * tmask.where(1 - (thetao.nav_lev < depth_box) * (abs(thetao.nav_lat) < 30))
    )

    # Returning Average Temperature on Box
    return (t_BW * area_BW).sum(dim=["nav_lat", "nav_lon", "nav_lev"]) / area_BW.sum(
        dim=["nav_lat", "nav_lon", "nav_lev"]
    )


def visualise_BWbox(restart_C1):
    # find deepest level that has any ocean
    T = restart_C1.tn  # (nav_lev, nav_lat, nav_lon)
    tmask = mesh_mask.tmask.squeeze()
    depth = T.nav_lev
    lat = restart_C1.nav_lat
    lat_1d = T.nav_lat.isel(nav_lon=0)  # shape (nav_lat,)

    BW_3d = 1 - ((depth < 3000) & (abs(lat) < 30))  # boolean mask

    BW_3d = BW_3d & (tmask == 1)

    T_zon = T.mean(dim="nav_lon")

    # reattach coords
    T_zon = T_zon.assign_coords(nav_lat=("nav_lat", lat_1d.values))

    BW_zon = BW_3d.any(dim="nav_lon")  # zonal 2D mask (depth × lat)

    BW_zon = BW_zon.assign_coords(nav_lat=("nav_lat", lat_1d.values))

    deepest_ocean_lev = np.where((tmask == 1).any(dim=("nav_lat", "nav_lon")))[0].max()
    print("Deepest ocean level index:", deepest_ocean_lev)
    print("Depth there:", float(depth.isel(nav_lev=deepest_ocean_lev)))

    # see if BW is ever true below that
    print(
        "Any BW below that depth?",
        bool(BW_3d.isel(nav_lev=slice(deepest_ocean_lev + 1, None)).any()),
    )
    # plot temperature
    _, ax = plt.subplots(figsize=(8, 4))
    T_zon.plot(x="nav_lat", y="nav_lev", cmap="coolwarm", ax=ax)

    # overlay BW region
    BW_zon.astype(int).plot.contour(
        x="nav_lat",
        y="nav_lev",
        levels=[0.5],  # contour where mask == 1
        colors="yellow",
        linewidths=2,
    )

    # BW_zon.astype(int).plot.contour(x="nav_lat",y="nav_lev",levels=[0.5],colors="yellow",linewidths=2)

    ax.invert_yaxis()
    ax.set_title("Zonally averaged T with U-shaped Bottom Water box")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    plt.show()


def DWbox(thetao, file_mask, depth_box=1500):
    """Compute the average temperature in the deep water box defined in region."""
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    condition = (
        (thetao.nav_lev < depth_box)
        * (thetao.nav_lev > 500)
        * (abs(thetao.nav_lat) < 30)
    )
    t_DW = thetao.where(condition)

    # Computing Area Weights from Mask over Box
    area_DW = e1t * e2t * e3t * tmask.where(condition)

    # Returning Average Temperature on Box
    return (t_DW * area_DW).sum(dim=["nav_lat", "nav_lon", "nav_lev"]) / area_DW.sum(
        dim=["nav_lat", "nav_lon", "nav_lev"]
    )


def DWbox_error(thetao, file_mask, depth_box=1500):
    """Compute the average temperature in the deep water box defined in region."""
    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    condition = (
        (thetao.nav_lev < depth_box) * (thetao.nav_lev > 500) * (abs(thetao.y) < 30)
    )
    t_DW = thetao.where(condition)

    # Computing Area Weights from Mask over Box
    area_DW = e1t * e2t * e3t * tmask.where(condition)

    # Returning Average Temperature on Box
    return (t_DW * area_DW).sum(dim=["y", "x", "nav_lev"]) / area_DW.sum(
        dim=["y", "x", "nav_lev"]
    )


# path_to_restart_ref = "./Gorce-data/restart_files_1_4deg/reference_restart_file/DINO_12960000_restart.nc"
# path_to_restart_ref_david = "./Gorce-data/restart_files_1_4deg/restart_C2_David/DINO_11232000_restart.nc"
# restart_ref = xr.load_dataset(path_to_restart_ref)
# restart_ref_david = xr.load_dataset(path_to_restart_ref_david)

if __name__ == "__main__":
    vis = False
    restart = False
    depth = 1500
    error = False
    check_numpy = True

    # Using metrics on restart file in 'correct' way. Only if restart=True
    if restart:
        mesh_mask_path = "./Gorce-data/Dinonline/restart0/mesh_mask.nc"
        mesh_mask = xr.load_dataset(mesh_mask_path)
        mesh_mask = mesh_mask.rename({"y": "nav_lat", "x": "nav_lon"})

        restart_C1 = xr.load_dataset(
            "./Gorce-data/restart_files_1_4deg/gen3/generated_C1_restart.nc"
        )
        restart_C1 = restart_C1.rename({"y": "nav_lat", "x": "nav_lon"})
        restart_C1 = restart_C1.rename({"lat": "nav_lat", "lon": "nav_lon"})

        temperature = restart_C1.tn

        print(
            "Bottom water temperature:",
            BWbox(temperature, mesh_mask, depth_box=depth).item(),
        )
        print(
            "Deep water temperature:",
            DWbox(temperature, mesh_mask).item(),
        )

        if vis:
            visualise_BWbox(restart_C1)

        # print(restart_C1.nav_lat)

    # Using error version of metrics
    if error:
        mesh_mask_path = "./Gorce-data/Dinonline/restart0/mesh_mask.nc"
        mesh_mask = xr.load_dataset(mesh_mask_path)
        restart_C1 = xr.load_dataset(
            "./Gorce-data/restart_files_1_4deg/gen3/generated_C1_restart.nc"
        )
        print(BWbox_error(restart_C1.tn, mesh_mask, depth_box=depth))
        print(DWbox_error(restart_C1.tn, mesh_mask, depth_box=depth))

    # Numpy
    if check_numpy:
        path = Path("./Gorce-data/generated_npy_files/no_constraint_clean/toce.npy")
        mask_path = Path( "./DINO-Fusion/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc")
        toce = np.load(path)
        mesh_mask = xr.load_dataset(mask_path)
        state_bug, state_cor = read_numpy(mesh_mask, toce, depth=1500)
        # print("Type:", type(toce))
        print("Shape:", toce.shape)
        print(f"Numpy file: {path.name}")
        print(f"Mesh mask name: {mask_path.name}")
        print(f"Bottom water temperature: Bug: {state_bug}, Correct: {state_cor}")

    # from utils import write_metric_table_to_readme
    # write_metric_table_to_readme(readme_path="README.md", depth_box=depth)
    # generate markdown table with all results in README.md:
