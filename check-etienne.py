#! /usr/bin/python
import xarray as xr

def temperature_BWbox_metric(thetao, file_mask, depth_box=1500):
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

mesh_mask_path = "./Gorce-data/Dinonline/restart0/mesh_mask.nc"
mesh_mask = xr.load_dataset(mesh_mask_path)
mesh_mask = mesh_mask.rename({'y':'nav_lat','x':'nav_lon'})

restart_C1 = xr.load_dataset("./Gorce-data/restart_files_1_4deg/gen3/generated_C1_restart.nc")
restart_C1 = restart_C1.rename({"y" : "nav_lat", "x" : "nav_lon"})
restart_C1 = restart_C1.rename({"lat" : "nav_lat", "lon" : "nav_lon"})

# temperature_BWbox_metric(restart_C1.tn, mesh_mask_new, 3000)

temperature = restart_C1.tn
depth = 1500

print(
    "Bottom water temperature:",
    temperature_BWbox_metric(temperature, mesh_mask, depth_box=1500).item(),
)
print(
    "Deep water temperature:",
    temperature_DWbox_metric(temperature, mesh_mask, depth_box=1500).item(),
)
