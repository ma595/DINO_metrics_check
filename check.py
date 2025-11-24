#! /usr/bin/python
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Table row	|  matching file
# Data	| (this is the DINO reference)
# No Constraint	| generated_restart_no_C_1_4deg.nc
# Constraint | 	generated_C1_restart.nc

# def temperature_BWbox_metric(thetao, file_mask, depth_box=3000):
#     depth = thetao.nav_lev
#     lat   = thetao.nav_lat

#     # Deep Water region (to be excluded)
#     deep_trop = (depth < depth_box) & (abs(lat) < 30)

#     # Bottom Water = complement of deep_trop
#     bw_mask = (~deep_trop) & (file_mask.tmask.squeeze() == 1)

#     e1t = file_mask.e1t.squeeze()
#     e2t = file_mask.e2t.squeeze()
#     e3t = file_mask.e3t_0.squeeze()

#     volume = (e1t * e2t * e3t).where(bw_mask)
#     t_BW   = thetao.where(bw_mask)

#     num = (t_BW * volume).sum(dim=["nav_lat", "nav_lon", "nav_lev"])
#     den = volume.sum(dim=["nav_lat", "nav_lon", "nav_lev"])

#     return num / den

def read_numpy(depth=1500):
    # Path to your file
    path = Path("./Gorce-data/generated_npy_files/no_constraint_clean/toce.npy")

    mask_path = Path("/Users/matt/work/nemo/spinup-evaluation/check/DINO-Fusion/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc")

    # Load the array
    toce = np.load(path)

    mesh_mask = xr.load_dataset(mask_path)

    # # Print basic information
    # print("Type:", type(toce))
    # print("Shape:", toce.shape)
    # print("Dtype:", toce.dtype)

    # # Inspect min/max/mean
    # print("Min:", np.nanmin(toce))
    # print("Max:", np.nanmax(toce))
    # print("Mean:", np.nanmean(toce))

    # # Optionally inspect a slice
    # print("numpy vertical levels: ", toce.shape[1])
    # # print("Sample slice (level 0):")
    # print("numpy shape: ", toce[0].shape)
    mesh_mask = mesh_mask.rename({"y" : "nav_lat", "x" : "nav_lon"})
    # print(mesh_mask)

    # Build DataArray with same dims as mesh_mask
    toce_da = xr.DataArray(
        toce.mean(axis=0),
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

    print(temperature_BWbox_metric(toce_da, mesh_mask, depth))


      


def temperature_BWbox_metric(thetao,   file_mask, depth_box=3000):
    """
        Metric Extraction Function :
        Average Temperature in a U-shaped "Bottom Water" box corresponding to waters below 3000m or beyond 30 degrees of latitude North and South.

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

    breakpoint()
    t_BW=thetao.where(1-(thetao.nav_lev<depth_box)*(abs(thetao.nav_lat)<30))

    # Computing Area Weights from Mask over Box
    e1t=file_mask.e1t.squeeze()
    e2t=file_mask.e2t.squeeze()
    e3t=file_mask.e3t_0.squeeze()
    tmask=file_mask.tmask.squeeze()
    area_BW=e1t*e2t*e3t*tmask.where(1-(thetao.nav_lev<depth_box)*(abs(thetao.nav_lat)<30))

    #Returning Average Temperature on Box
    return ((t_BW*area_BW).sum(dim=["nav_lat","nav_lon","nav_lev"])/area_BW.sum(dim=["nav_lat","nav_lon","nav_lev"]))


def visualise_BWbox(restart_C1):


# find deepest level that has any ocean
    T = restart_C1.tn                  # (nav_lev, nav_lat, nav_lon)
    tmask = mesh_mask.tmask.squeeze()
    depth = T.nav_lev
    lat   = restart_C1.nav_lat
    lat_1d = T.nav_lat.isel(nav_lon=0)   # shape (nav_lat,)
    

    # --- U-shaped BW box condition ---
    BW_3d = 1 - ((depth < 3000) & (abs(lat) < 30))    # boolean mask

    # also require ocean
    BW_3d = BW_3d & (tmask == 1)

    # zonal averages
    T_zon  = T.mean(dim="nav_lon")

    
    # reattach coords
    T_zon = T_zon.assign_coords(nav_lat=("nav_lat", lat_1d.values))

    BW_zon = BW_3d.any(dim="nav_lon")    # zonal 2D mask (depth × lat)

    BW_zon = BW_zon.assign_coords(nav_lat=("nav_lat", lat_1d.values))

    deepest_ocean_lev = np.where((tmask == 1).any(dim=("nav_lat","nav_lon")))[0].max()
    print("Deepest ocean level index:", deepest_ocean_lev)
    print("Depth there:", float(depth.isel(nav_lev=deepest_ocean_lev)))

    # see if BW is ever true below that
    print("Any BW below that depth?",
          bool(BW_3d.isel(nav_lev=slice(deepest_ocean_lev+1, None)).any()))
    # plot temperature
    fig, ax = plt.subplots(figsize=(8,4))
    T_zon.plot(
        x="nav_lat", y="nav_lev",
        cmap="coolwarm", ax=ax
    )

    # overlay BW region
    BW_zon.astype(int).plot.contour(
            x="nav_lat",
            y="nav_lev",
            levels=[0.5],         # contour where mask == 1
            colors="yellow",
            linewidths=2
            )

    # OR if you prefer an outline
    # BW_zon.astype(int).plot.contour(x="nav_lat",y="nav_lev",levels=[0.5],colors="yellow",linewidths=2)

    ax.invert_yaxis()
    ax.set_title("Zonally averaged T with U-shaped Bottom Water box")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Depth (m)")
    plt.show()




def temperature_DWbox_metric(thetao, file_mask, depth_box=1500):
    """
    Compute the average temperature in the deep water box defined as the region
    """

    e1t = file_mask.e1t.squeeze()
    e2t = file_mask.e2t.squeeze()
    e3t = file_mask.e3t_0.squeeze()
    tmask = file_mask.tmask.squeeze()
    condition = (
        (thetao.nav_lev < depth_box) * (thetao.nav_lev > 500) * (abs(thetao.nav_lat) < 30)
    )
    t_DW = thetao.where(condition)

    # Computing Area Weights from Mask over Box
    area_DW = e1t * e2t * e3t * tmask.where(condition)

    # Returning Average Temperature on Box
    return (t_DW * area_DW).sum(dim=["nav_lat", "nav_lon", "nav_lev"]) / area_DW.sum(
        dim=["nav_lat", "nav_lon", "nav_lev"]
    )


# path_to_restart_ref = "../spinup-data/Gorce-data/restart_files_1_4deg/reference_restart_file/DINO_12960000_restart.nc"
# path_to_restart_ref_david = "../spinup-data/Gorce-data/restart_files_1_4deg/restart_C2_David/DINO_11232000_restart.nc"
# restart_ref = xr.load_dataset(path_to_restart_ref)
# restart_ref_david = xr.load_dataset(path_to_restart_ref_david)

restart = False
depth = 1500

if restart:
    mesh_mask_path = "./Gorce-data/Dinonline/restart0/mesh_mask.nc"
    mesh_mask = xr.load_dataset(mesh_mask_path)
    mesh_mask = mesh_mask.rename({'y':'nav_lat','x':'nav_lon'})

    restart_C1 = xr.load_dataset("./Gorce-data/restart_files_1_4deg/gen3/generated_C1_restart.nc")
    restart_C1 = restart_C1.rename({"y" : "nav_lat", "x" : "nav_lon"})
    restart_C1 = restart_C1.rename({"lat" : "nav_lat", "lon" : "nav_lon"})

    temperature = restart_C1.tn

    print(
        "Bottom water temperature:",
        temperature_BWbox_metric(temperature, mesh_mask, depth_box=depth).item(),
    )
    print(
        "Deep water temperature:",
        temperature_DWbox_metric(temperature, mesh_mask).item(),
    )
        
    visualise_BWbox(restart_C1)

    print(restart_C1.nav_lat)

read_numpy(depth)
