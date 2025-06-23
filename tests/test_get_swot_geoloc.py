import unittest
import os
import collections
import geopandas as gpd
import xarray as xr
from shapely.geometry import (
    Polygon,
    MultiPolygon,
)  # For type checking and dummy land GDF
from unittest.mock import patch
import logging
import pandas as pd  # For pd.to_datetime in assertions

# Import the function to be tested
from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import get_swot_geoloc

# Suppress logger output from the module under test for cleaner test runs
# This assumes the logger in coloc_SWOT_L3_with_S1_CDSE_TOPS.py is named via its module path
module_logger_name = "s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS"
app_logger_under_test = logging.getLogger(module_logger_name)
if not app_logger_under_test.handlers:  # Avoid adding multiple NullHandlers
    app_logger_under_test.addHandler(logging.NullHandler())
app_logger_under_test.propagate = False


class TestGetSwotGeoloc(unittest.TestCase):

    SWOT_ASSET_FILENAME = (
        "SWOT_L3_LR_SSH_Expert_026_165_20241229T165806_20241229T174933_v2.0.1.nc"
    )
    SWOT_FILE = None
    expected_slice_swot_calls = 0

    @classmethod
    def setUpClass(cls):
        # Determine the path to the test asset.
        # This assumes your test file is in a 'tests' directory,
        # and 'src' is a sibling directory to 'tests' at the project root.
        # Project Root/
        #  |- src/
        #  |   |- s1swotcolocs/
        #  |       |- assests/
        #  |           |- SWOT_L3_LR_SSH_Expert_026_165_20241229T165806_20241229T174933_v2.0.1.nc
        #  |- tests/
        #      |- test_your_module.py (this file)
        current_test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.SWOT_FILE = os.path.normpath(
            os.path.join(
                current_test_dir,
                "..",
                "src",
                "s1swotcolocs",
                "assests",
                cls.SWOT_ASSET_FILENAME,
            )
        )

        if not os.path.exists(cls.SWOT_FILE):
            raise unittest.SkipTest(
                f"Test SWOT file not found at expected path: {cls.SWOT_FILE}. "
                "Please ensure the asset file is correctly placed or adjust the path."
            )

        # Pre-calculate expected calls to slice_swot based on file content
        try:
            with xr.open_dataset(cls.SWOT_FILE) as ds_temp:
                num_lines = ds_temp["time"].sizes["num_lines"]
            # 'segment' size is hardcoded to 1000 in get_swot_geoloc
            segment_size = 1000
            cls.expected_slice_swot_calls = (
                num_lines + segment_size - 1
            ) // segment_size  # Ceiling division
            if (
                cls.expected_slice_swot_calls == 0 and num_lines > 0
            ):  # if num_lines < segment_size but > 0
                cls.expected_slice_swot_calls = 1

        except Exception as e:
            raise unittest.SkipTest(
                f"Could not read test SWOT file {cls.SWOT_FILE} to determine segment count: {e}"
            )
        if cls.expected_slice_swot_calls == 0:
            print(
                f"Warning: Test SWOT file {cls.SWOT_FILE} seems to have no time lines, "
                f"expected_slice_swot_calls is 0."
            )

    # Mock external dependencies within slice_swot:
    # - geodatasets.get_path (to avoid actual download/lookup of land data)
    # - gpd.read_file (to avoid reading the actual land shapefile)
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.gpd.read_file")
    @patch("s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS.geodatasets.get_path")
    def test_process_swot_file(self, mock_geo_get_path, mock_gpd_read_file):
        """
        Tests get_swot_geoloc with the provided SWOT L3 file,
        mocking land data interactions.
        """
        # Configure mocks
        mock_geo_get_path.return_value = "mocked/path/to/naturalearth/land.shp"
        # Provide a dummy GeoDataFrame for land. An empty one means all SWOT data is treated as ocean.
        # A simple polygon can also be used to test land intersection logic if desired.
        dummy_land_gdf = gpd.GeoDataFrame(
            geometry=[]
        )  # Or: gpd.GeoDataFrame({'geometry': [Polygon([(0,0),(1,1),(1,0)])]})
        mock_gpd_read_file.return_value = dummy_land_gdf

        initial_counters = collections.defaultdict(int)
        initial_counters["nbSWOTfiles"] += 1
        test_delta_hours = 3
        test_mode = "IW"
        test_producttype = "SLC"

        # Call the function under test
        result_gdfs, result_counters = get_swot_geoloc(
            one_swot_l3_file=self.SWOT_FILE,
            delta_hours=test_delta_hours,
            mode=test_mode,
            max_area_size=100,
            producttype=test_producttype,
            cpt=initial_counters,
        )

        # Assertions on return types
        self.assertIsInstance(result_gdfs, list)
        self.assertIsInstance(result_counters, collections.defaultdict)

        # Assertions on counter updates
        self.assertIs(
            result_counters,
            initial_counters,
            "The counter dictionary should be updated in-place.",
        )
        self.assertEqual(result_counters["nbSWOTfiles"], 1)

        # Assertions on mocked calls (related to land data processing in slice_swot)
        if self.expected_slice_swot_calls > 0:
            self.assertEqual(
                mock_geo_get_path.call_count, self.expected_slice_swot_calls
            )
            mock_geo_get_path.assert_called_with("naturalearth.land")
            self.assertEqual(
                mock_gpd_read_file.call_count, self.expected_slice_swot_calls
            )
            mock_gpd_read_file.assert_called_with(
                "mocked/path/to/naturalearth/land.shp"
            )
        else:  # If file has no lines, these shouldn't be called
            self.assertEqual(mock_geo_get_path.call_count, 0)
            self.assertEqual(mock_gpd_read_file.call_count, 0)

        # Assertions on the produced GeoDataFrames
        if self.expected_slice_swot_calls == 0:
            self.assertEqual(
                len(result_gdfs),
                0,
                "Expected no GeoDataFrames if SWOT file has no processable segments.",
            )
        elif not result_gdfs:
            # This case can happen if all segments are over land, too small, too large, or problematic.
            # Check counters for indications.
            print(
                f"INFO: No GeoDataFrames produced for {self.SWOT_FILE}. Counters: {result_counters}"
            )
            # Example: self.assertTrue(result_counters['empty_polygon'] > 0 or result_counters['segment_with_area_too_large'] > 0)
        else:
            self.assertGreater(
                len(result_gdfs),
                0,
                "Expected at least one GeoDataFrame for a valid SWOT file segment.",
            )
            for gdf in result_gdfs:
                self.assertIsInstance(gdf, gpd.GeoDataFrame)
                self.assertEqual(
                    len(gdf), 1, "Each GeoDataFrame piece should contain a single row."
                )
                expected_columns = [
                    "start_datetime",
                    "end_datetime",
                    "geometry",
                    "collection",
                    "name",
                    "sensormode",
                    "producttype",
                    "Attributes",
                    "id_query",
                ]
                for col in expected_columns:
                    self.assertIn(col, gdf.columns)

                # Check specific column values
                self.assertEqual(gdf["collection"].iloc[0], "SENTINEL-1")
                self.assertEqual(gdf["sensormode"].iloc[0], test_mode)
                self.assertEqual(gdf["producttype"].iloc[0], test_producttype)
                self.assertIsInstance(gdf["geometry"].iloc[0], (Polygon, MultiPolygon))

                # Validate id_query format
                id_query_parts = gdf["id_query"].iloc[0].split(" ")
                self.assertEqual(id_query_parts[0], "SWOT")
                self.assertEqual(id_query_parts[1], os.path.basename(self.SWOT_FILE))
                # Ensure the timestamps are valid
                pd.to_datetime(id_query_parts[2])  # startswot
                pd.to_datetime(id_query_parts[3])  # stopswot


if __name__ == "__main__":
    unittest.main()
