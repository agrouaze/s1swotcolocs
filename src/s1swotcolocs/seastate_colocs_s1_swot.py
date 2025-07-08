import collections

import s1ifr
import os
import argparse
import pandas as pd
import sys
import glob
import logging
import xarray as xr
from tqdm import tqdm
from scipy import spatial
import numpy as np
import alphashape
import geopandas as gpd
from shapely.geometry import MultiPoint
import datetime
from shapely import wkt
from slcl1butils.utils import xndindex
from s1ifr import paths_safe_product_family
from s1swotcolocs.utils import get_conf_content
from s1swotcolocs.pickup_best_swot_file import (
    check_if_latest_version
)

DEFAULT_IFREMER_S1_VERSION_L1B = ["A17", "A18", "A21", "A23", "A16", "A15"]
DEFAULT_SWOT_VARIABLES = [
    "latitude",
    "longitude",
    "swh_karin",
]  # Level2 # ,'swh_karin_qual','sig0_karin_uncert'

app_logger = logging.getLogger(__file__)


def get_original_sar_filepath_from_metacoloc(metacolocds, cpt) -> (list, dict):
    """

    :param metacolocds:
    :param cpt: defaultdict
    :return:
        fullpath_iw_slc_safes: str
    """
    fullpath_iw_slc_safe = None
    basesafe = metacolocds["sar_safe_name"].values
    fullpath_iw_slc_safes = []
    for basesafe in basesafe:
        app_logger.info("basesafe : %s", basesafe)
        orignal_s1 = s1ifr.get_path_from_base_safe.get_path_from_base_safe(
            basesafe, archive_name="datawork"
        )
        orignal_s1_bis = s1ifr.get_path_from_base_safe.get_path_from_base_safe(
            basesafe, archive_name="scale"
        )
        if os.path.exists(orignal_s1) is True or os.path.exists(orignal_s1_bis) is True:
            if os.path.exists(orignal_s1) is True:
                fullpath_iw_slc_safe = orignal_s1
            else:
                fullpath_iw_slc_safe = orignal_s1_bis
            if os.path.exists(fullpath_iw_slc_safe):
                fullpath_iw_slc_safes.append(fullpath_iw_slc_safe)
            else:
                cpt["safe_iw_slc_not_found"] += 1
                # fullpath_iw_slc_safes.append(None)
        else:
            cpt["safe_iw_slc_not_found"] += 1

    return fullpath_iw_slc_safes, cpt


def get_L2WAV_S1_IW_path(fullpath_s1_iw_slc, version_L1B=None):
    """

    :param fullpath_s1_iw_slc: str
    :param version_L1B: list of str
    :return:
    """
    if version_L1B is None:
        version_L1B = DEFAULT_IFREMER_S1_VERSION_L1B
    path_l2wav_sar = None
    df = pd.DataFrame({"L1_SLC": [fullpath_s1_iw_slc]})
    app_logger.info("df : %s", df)
    newdf = paths_safe_product_family.get_products_family(df, l1bversions=version_L1B)
    if len(newdf["L2_WAV_E12"]) > 0:
        path_l2wav_sar = newdf["L2_WAV_E12"].values[0]
    elif len(newdf["L2_WAV_E11"]) > 0:
        path_l2wav_sar = newdf["L2_WAV_E11"].values[0]

    return path_l2wav_sar


def read_swot_windwave_l2_file(metacolocds, confpath,cpt=None) -> (xr.Dataset, str):
    """

    :param metacolocds: xr.Dataset
    :param confpath: str
    :param cpt: collections.defaultdict
    :return:
        dsswotl2: xr.Dataset or None SWOT Level-2 WindWave AVISO product
    """
    dsswotl2 = None
    pathswotl2final = None
    baseswotl3 = metacolocds["filepath_swot"].values[0].item()
    app_logger.debug("baseswotl3 %s", baseswotl3)
    conf = get_conf_content(confpath)
    dir_swot_l3 = conf["SWOT_L3_AVISO_DIR"]
    assert os.path.exists(dir_swot_l3)

    date_swot_dt = datetime.datetime.strptime(baseswotl3.split("_")[7], "%Y%m%dT%H%M%S")
    full_path_swot = os.path.join(
        dir_swot_l3,
        date_swot_dt.strftime("%Y"),
        date_swot_dt.strftime("%j"),
        baseswotl3,
    )
    if os.path.exists(full_path_swot):
        dir_swot_l2 = conf["SWOT_L2_AVISO_DIR"]
        pattern_l2 = os.path.join(
            dir_swot_l2,
            baseswotl3.replace("Expert", "WindWave")
            .replace("L3", "L2")
            .split("_v")[0][0:-1]
            + "*.nc",
        )
        app_logger.debug("pattern_l2 %s", pattern_l2)
        lst_nc = glob.glob(pattern_l2)
        if len(lst_nc) > 0:
            pathswotl2final = lst_nc[0]
            is_the_latest_version, true_latest_file = check_if_latest_version(file_to_check=pathswotl2final, all_available_files=lst_nc)
            if true_latest_file is not None:
                cpt['swot_file_change_for_latest'] += 1
                pathswotl2final = true_latest_file
            else:
                cpt['swot_file_direct_pickup_latest'] += 1
            dsswotl2 = xr.open_dataset(pathswotl2final).load()
    return dsswotl2, pathswotl2final, cpt


def create_empty_coloc_res(indexes_sar_grid) -> xr.Dataset:
    """

    :param indexes_sar_grid: dict with tile_line and tile_sample keys
    :return:
        empty_dummy_condensated_swot_coloc: xr.Dataset filled with NaN variables
    """
    empty_dummy_condensated_swot_coloc = xr.Dataset()
    # fcts = {'mean': np.nanmean,
    #             'med': np.nanmedian,
    #             'std': np.nanstd}
    fcts = {"mean": np.mean, "med": np.median, "std": np.std}
    for vv in DEFAULT_SWOT_VARIABLES:
        for fct in fcts:
            empty_dummy_condensated_swot_coloc["%s_%s" % (vv, fct)] = xr.DataArray(
                np.nan,
                attrs={
                    "description": "%s of %s variable from SWOT product" % (fct, vv)
                },
            )
    empty_dummy_condensated_swot_coloc = (
        empty_dummy_condensated_swot_coloc.assign_coords(indexes_sar_grid)
    )
    empty_dummy_condensated_swot_coloc = empty_dummy_condensated_swot_coloc.expand_dims(
        ["tile_line", "tile_sample"]
    )
    return empty_dummy_condensated_swot_coloc


def s1swot_core_tile_coloc(
    lontile, lattile, treeswot, radius_coloc, dsswot, indexes_sar, cpt
) -> xr.Dataset:
    """

    Args:
        lontile: float longitude of a SAR point
        lattile: float latitude of a SAR point
        treeswot: scipy.spatial SWOT
        radius_coloc: float
        dsswot: xr.Dataset SWOT data
        indexes_sar: tile_line and tile_sample indexes to be able to reconstruct the SAR swath

    Returns:
        condensated_swot: xr.Dataset
    """
    neighbors = treeswot.query_ball_point([lontile, lattile], r=radius_coloc)
    indices = []
    condensated_swot = xr.Dataset()
    for oneneighbor in neighbors:
        index_original_shape_swot_num_lines, index_original_shape_swot_num_pixels = (
            np.unravel_index(oneneighbor, dsswot["longitude"].shape)
        )
        indices.append(
            (index_original_shape_swot_num_lines, index_original_shape_swot_num_pixels)
        )
    subset = [dsswot.isel(num_lines=i, num_pixels=j) for i, j in indices]
    if len(subset) > 0:
        swotclosest = xr.concat(subset, dim="points")

        # number of points in the radius
        condensated_swot["nb_SWOT_points"] = xr.DataArray(len(swotclosest["points"]))
        # variables wind/ std / mean /median
        fcts = {"mean": np.nanmean, "med": np.nanmedian, "std": np.nanstd}

        for vv in DEFAULT_SWOT_VARIABLES:
            for fct in fcts:
                if np.isfinite(swotclosest[vv].values).any():
                    if np.isfinite(swotclosest[vv].values).sum() == 1:
                        if fct == "std":
                            valval = np.nan
                        else:
                            valval = swotclosest[vv].values
                    else:
                        # many pts I use quality flag SWOT
                        if vv == "swk_karin":
                            maskswotswhqual = (
                                swotclosest["swh_karin_qual"].values == 0
                            ) & (swotclosest["rain_flag"].values == 0)
                        else:
                            maskswotswhqual = True
                        valval = fcts[fct](swotclosest[vv].values[maskswotswhqual])

                else:
                    valval = (
                        np.nan
                    )  # to avoid RuntimeWarning Degrees of freedom <= 0 for slice
                condensated_swot["%s_%s" % (vv, fct)] = xr.DataArray(
                    valval,
                    attrs={
                        "description": "%s of %s variable from SWOT points within a %f deg radius after swh_karin_qual=0 and rain_flag=0 filtering"
                        % (fct, vv, radius_coloc)
                    },
                )
                condensated_swot["%s_%s" % (vv, fct)].attrs.update(
                    swotclosest[vv].attrs
                )
        condensated_swot = condensated_swot.assign_coords(indexes_sar)
        condensated_swot = condensated_swot.expand_dims(["tile_line", "tile_sample"])
        cpt["tile_with_SWOT_neighbors"] += 1
    else:
        app_logger.debug("one tile without SWOT neighbors")
        cpt["tile_without_SWOT_neighbors"] += 1
        condensated_swot = create_empty_coloc_res(indexes_sar_grid=indexes_sar)
    return condensated_swot, cpt


def loop_on_each_sar_tiles(
    dssar, dsswotl2, radius_coloc, full_path_swot, cpt
) -> (xr.Dataset, collections.defaultdict):
    """

    :param dssar:
    :param radius_coloc: float
    :param full_path_swot: str
    :param cpt: defaultdict
    :return:
    """

    lonswot = dsswotl2["longitude"].values.ravel()
    lonswot = (lonswot + 180) % 360 - 180
    latswot = dsswotl2["latitude"].values.ravel()
    maskswot = np.isfinite(lonswot) & np.isfinite(latswot)
    app_logger.info(
        "nb NaN in the 2km SWOT grid: %i/%i",
        len(latswot) - maskswot.sum(),
        len(latswot),
    )
    points = np.c_[lonswot[maskswot], latswot[maskswot]]
    treeswot = spatial.KDTree(points)

    all_tiles_colocs = []
    gridsarL2 = {
        d: k
        for d, k in dssar.sizes.items()
        # if d in ["burst", "tile_sample", "tile_line"]
        if d in ["tile_sample", "tile_line"]
    }
    all_tile_cases = [i for i in xndindex(gridsarL2)]
    for x in tqdm(range(len(all_tile_cases))):
        # for ii in tqdm(range(len(sardf['lat_centroid_sar']))):
        # lontile = sardf['lon_centroid_sar'].iloc[ii]
        # lattile = sardf['lat_centroid_sar'].iloc[ii]

        i = all_tile_cases[x]
        lontile = dssar["longitude"][i].values
        lattile = dssar["latitude"][i].values

        if np.isfinite(lontile) and np.isfinite(lattile):
            tile_swot_condensated_at_SAR_point, cpt = s1swot_core_tile_coloc(
                lontile,
                lattile,
                treeswot,
                radius_coloc,
                dsswot=dsswotl2,
                indexes_sar=i,
                cpt=cpt,
            )
        else:
            cpt["tile_sar_with_corrupted_geolocation"] += 1
            tile_swot_condensated_at_SAR_point = create_empty_coloc_res(
                indexes_sar_grid=i
            )

        all_tiles_colocs.append(tile_swot_condensated_at_SAR_point)
    consolidated_all_tiles_colocs = []
    for uu in all_tiles_colocs:
        if "dim_0" in uu.dims:
            app_logger.debug("variables with dim_0 : %s", uu)
        else:
            consolidated_all_tiles_colocs.append(uu)
    ds_colocation_grd = xr.merge(
        consolidated_all_tiles_colocs,
    )
    # xr.combine_by_coords(all_tiles_colocs)
    ds_l1c = xr.merge([dssar, ds_colocation_grd])
    ds_l1c.attrs["SWOT_L3_data"] = full_path_swot
    return ds_l1c, cpt


def save_sea_state_coloc_file(colocds, fpath_out, cpt):
    """

    :param colocds:
    :param fpath_out:
    :param cpt:
    :return:
    """
    if os.path.exists(fpath_out):
        app_logger.info("remove the existing file")
        os.remove(fpath_out)
        cpt["file_replaced"] += 1
    else:
        app_logger.debug("file does not exist -> brand-new file on disk")
        cpt["new_file"] += 1
    if not os.path.exists(os.path.dirname(fpath_out)):
        os.makedirs(os.path.dirname(fpath_out), mode=0o775)
    colocds.to_netcdf(fpath_out, engine="h5netcdf")
    os.chmod(fpath_out, 0o664)
    app_logger.info("coloc file created : %s", fpath_out)
    return cpt


def associate_sar_and_swot_seastate_params(
    metacolocpath, confpath, groupsar="intraburst", overwrite=True, outputdir=None
):
    """

    :param metacolocpath: str (e.g. .../coloc_SWOT_L3_Sentinel-1_IW_20240729T172147.nc)
    :param confpath: str
    :param groupsar: str intraburst or interburst
    :param outputdir: path where to store the output sea state coloc files (.nc), will superseed the config file
    :return:
    """
    app_logger.info("SAR grid : %s", groupsar)
    cpt = collections.defaultdict(int)
    conf = get_conf_content(confpath)
    if outputdir is None:
        outputdir = conf["HOST_SEASTATE_COLOC_OUTPUT_DIR"]
    mode = os.path.basename(metacolocpath).split("_")[4]  # IW or EW"
    metacolocds = xr.open_dataset(metacolocpath, engine="h5netcdf")
    if "filepath_swot" not in metacolocds:
        if "filepath_swot" in metacolocds.attrs:
            metacolocds["filepath_swot"] = xr.DataArray(
                np.tile(metacolocds.attrs["filepath_swot"], len(metacolocds["coloc"])),
                dims="coloc",
            )
        else:
            raise KeyError("filepath_swot not present in metacolocds")

    SWOT_start_piece = datetime.datetime.strptime(
        os.path.basename(metacolocpath).split("_")[5].replace(".nc", ""),
        "%Y%m%dT%H%M%S",
    )
    year = SWOT_start_piece.strftime("%Y")
    month = SWOT_start_piece.strftime("%m")
    day = SWOT_start_piece.strftime("%d")
    fullpath_iw_slc_safes, cpt = get_original_sar_filepath_from_metacoloc(
        metacolocds, cpt=cpt
    )
    app_logger.info("nb IW SAFE found at Ifremer: %i", len(fullpath_iw_slc_safes))
    for iw_slc_safe in fullpath_iw_slc_safes:
        cpt["total_safe_SAR_tested"] += 1
        app_logger.info("treat : %s", iw_slc_safe)
        dsswotl2, pathswotl2final, cpt = read_swot_windwave_l2_file(
            metacolocds, confpath=confpath,cpt=cpt
        )
        if dsswotl2 is not None:
            path_l2wav_sar_safe = get_L2WAV_S1_IW_path(
                iw_slc_safe, version_L1B=DEFAULT_IFREMER_S1_VERSION_L1B
            )
            for subswath_sar in ["iw1", "iw2", "iw3"]:
                cpt["total_suswath_sar_tested"] += 1
                fpath_out = os.path.join(
                    outputdir,
                    mode,
                    "%s" % year,
                    "%s" % month,
                    "%s" % day,
                    os.path.basename(metacolocpath).replace(
                        "coloc_",
                        "seastate_coloc_%s_"
                        % (
                            os.path.basename(iw_slc_safe).replace(
                                ".SAFE", "-" + subswath_sar
                            )
                        ),
                    ),
                )
                if os.path.exists(fpath_out) and overwrite is False:
                    app_logger.info("coloc file already exists: %s", fpath_out)
                else:
                    pattern_sar = os.path.join(
                        path_l2wav_sar_safe, "l2*" + subswath_sar + "*.nc"
                    )
                    lst_nc_sar = glob.glob(pattern_sar)
                    if len(lst_nc_sar) > 0:
                        fsar = lst_nc_sar[0]
                        dssar = xr.open_dataset(fsar, group=groupsar).load()
                        polygon_sar_swubswath = wkt.loads(dssar.attrs["footprint"])
                        delta_bound = 1.9  # deg
                        ymin = dssar["latitude"].values.ravel().min() - delta_bound
                        ymax = dssar["latitude"].values.ravel().max() + delta_bound
                        xmin = dssar["longitude"].values.ravel().min() - delta_bound
                        xmax = dssar["longitude"].values.ravel().max() + delta_bound
                        # subset SWOT dataset
                        subswot = dsswotl2.where(
                            (dsswotl2["latitude"] >= ymin)
                            & (dsswotl2["latitude"] <= ymax)
                            & (dsswotl2["longitude"] >= xmin)
                            & (dsswotl2["longitude"] <= xmax),
                            drop=True,
                        )
                        lonswot = subswot["longitude"].values.ravel()
                        lonswot[lonswot > 180] += -360.0
                        points = np.column_stack(
                            (lonswot, subswot["latitude"].values.ravel())
                        )
                        # Create a MultiPoint object
                        multi_point = MultiPoint(points)

                        # Get the convex hull (smallest polygon enclosing all points)
                        # polygon = multi_point.convex_hull
                        gdfswot = gpd.GeoDataFrame(geometry=list(multi_point.geoms))
                        tolerance_simplification = 0.1
                        alpha_shape_swot = alphashape.alphashape(
                            gdfswot, alpha=tolerance_simplification
                        )
                        if alpha_shape_swot.intersects(polygon_sar_swubswath):
                            l2c_ds, cpt = loop_on_each_sar_tiles(
                                dssar,
                                dsswotl2,
                                radius_coloc=conf["RADIUS_COLOC"],
                                full_path_swot=pathswotl2final,
                                cpt=cpt,
                            )

                            cpt = save_sea_state_coloc_file(
                                colocds=l2c_ds, fpath_out=fpath_out, cpt=cpt
                            )
                        else:
                            app_logger.info(
                                "subswath %s SAR not intersecting this SWOT swath",
                                subswath_sar,
                            )
                            cpt["subswath_not_intersecting_swot"] += 1
                    else:
                        app_logger.info(
                            "SAR IW Level-2 WAV product %s is not available : %s",
                            pattern_sar,
                        )
                        cpt["SAR_IW_L2WAV_not_available"] += 1
        else:
            app_logger.info("SWOT Level-2 is not available.")
            cpt["SWOT_L2_not_available"] += 1
    app_logger.info("counter : %s", cpt)


def parse_args():
    parser = argparse.ArgumentParser(description="S1SWOTswhcoloc")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        required=False,
        help="overwrite existing coloc file [default: False]",
    )
    parser.add_argument(
        "--metacolocfile", required=True, help="full path of meta coloc file"
    )
    parser.add_argument("--confpath", required=True, help="full path of config file")
    parser.add_argument(
        "--outputdir",
        required=True,
        help="directory where to store output netCDF files, path will be completed by mypath/IW/YYYY/MM/DD/filename.nc",
    )
    parser.add_argument(
        "--groupsar",
        required=False,
        choices=["intraburst", "interburst"],
        default="intraburst",
        help="intraburst or interburst [default=intraburst]",
    )
    args = parser.parse_args()
    return args


def main():
    """

    treat a meta coloc file SWOT-s1 data to generate a sea state coloc file

    :return:
    """
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    date_format = "%d-%m-%Y %H:%M:%S"
    nouveau_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler_app = logging.StreamHandler(sys.stdout)
    console_handler_app.setFormatter(nouveau_formatter)

    # It's good practice to remove existing handlers, especially if main() might be called multiple times
    # Iterate over a slice [:] to avoid issues when modifying the list during iteration
    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)

    app_logger.addHandler(console_handler_app)
    app_logger.setLevel(log_level)
    app_logger.propagate = False  # <--- THIS IS THE KEY CHANGE

    associate_sar_and_swot_seastate_params(
        metacolocpath=args.metacolocfile,
        confpath=args.confpath,
        groupsar=args.groupsar,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
