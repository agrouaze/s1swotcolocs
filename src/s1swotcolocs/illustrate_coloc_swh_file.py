import panel as pn
import numpy as np
import geopandas as gpd
import hvplot.pandas  # noqa
import holoviews as hv
import xarray as xr
from shapely.geometry import Polygon
from scipy.stats import pearsonr
import os

# --- Configuration ---
hv.extension('bokeh')
pn.extension()


# --- 1. Data Loading and Helper Functions ---
def create_combined_gdf(fseastatecoloc):
    """
    Reads a NetCDF file and creates a single GeoDataFrame containing polygon geometries
    and all relevant data (SAR SWH, SWOT SWH) for linking.
    """
    # This function is correct and remains unchanged.
    ds_l2c = xr.open_dataset(fseastatecoloc)
    corner_lon, corner_lat = ds_l2c['corner_longitude'], ds_l2c['corner_latitude']
    ordered = [0, 1, 3, 2]
    records = []
    for i in range(ds_l2c.dims['tile_line']):
        for j in range(ds_l2c.dims['tile_sample']):
            lon_corners = corner_lon.isel(tile_line=i, tile_sample=j).values.ravel()[ordered]
            lat_corners = corner_lat.isel(tile_line=i, tile_sample=j).values.ravel()[ordered]
            poly = Polygon(list(zip(lon_corners, lat_corners)))
            records.append({
                'geometry': poly,
                'hs_sar': ds_l2c['hs_most_likely'].isel(tile_line=i, tile_sample=j).item(),
                'swh_swot': ds_l2c['swh_karin_mean'].isel(tile_line=i, tile_sample=j).item(),
                'hs_conf': ds_l2c['hs_conf'].isel(tile_line=i, tile_sample=j).item(),
            })
    combined_gdf = gpd.GeoDataFrame(records, crs='EPSG:4326')
    mask = np.isfinite(combined_gdf['swh_swot']) & np.isfinite(combined_gdf['hs_sar'])
    return combined_gdf[mask].reset_index(drop=True)


def calculate_statistics(gdf, x_col='swh_swot', y_col='hs_sar'):
    """Calculates statistics. Unchanged."""
    if gdf.empty:
        return {'N': 0, 'Bias': np.nan, 'RMSE': np.nan, 'Correlation (R)': np.nan, 'SI (%)': np.nan}
    x, y = gdf[x_col], gdf[y_col]
    difference = y - x
    stats = {
        'N': len(x),
        'Bias': np.mean(difference),
        'RMSE': np.sqrt(np.mean(difference ** 2)),
        'Correlation (R)': pearsonr(x, y)[0] if len(x) > 1 else np.nan,
        'SI (%)': (np.sqrt(np.mean(difference ** 2)) / np.mean(x)) * 100 if np.mean(x) != 0 else np.nan
    }
    return stats


# --- Main Application Function ---
def create_linked_dashboard(data_file):
    print("--- STEP 1: Reading and processing data file... ---")
    gdf = create_combined_gdf(data_file)
    print(f"--- Data loaded successfully. Found {len(gdf)} valid data points. ---")

    # --- Plotting Configuration ---
    vmin, vmax = (2, 6)
    cmap = 'viridis'
    width, height = (550, 500)

    print("--- STEP 2: Creating plot components... ---")
    # Manually create the stream we will use for the statistics pane.
    selection_stream = hv.streams.Selection1D()

    # Create the map plots (unchanged)
    polygons_sar = gdf.hvplot.polygons(geo=True, color='hs_sar', cmap=cmap, clim=(vmin, vmax),
                                       hover_cols=['hs_sar', 'swh_swot'], line_color='black', line_width=0.5,
                                       title='SAR S-1 IW VV', colorbar=True, width=width, height=height).opts(
        hv.opts.Polygons(tools=['box_select', 'lasso_select'], nonselection_alpha=0.2, selection_color='orange'))
    polygons_swot = gdf.hvplot.polygons(geo=True, color='swh_swot', cmap=cmap, clim=(vmin, vmax),
                                        hover_cols=['hs_sar', 'swh_swot'], line_color='black', line_width=0.5,
                                        title='SWOT KarIn', colorbar=True, width=width, height=height).opts(
        hv.opts.Polygons(tools=['box_select', 'lasso_select'], nonselection_alpha=0.2, selection_color='orange'))

    # Create a standard, interactive scatter plot.
    # This plot will be both visible and the source for our linking.
    scatter_plot = gdf.hvplot.scatter(
        x='swh_swot', y='hs_sar',
        hover_cols=['hs_sar', 'swh_swot']  # Hover tool now works!
    ).opts(
        hv.opts.Scatter(
            size=5,
            selection_color='orange',
            nonselection_alpha=0.2,
            tools=['box_select', 'lasso_select']
        )
    )

    print("--- STEP 3: Linking plots and attaching streams... ---")
    # Manually attach our stream to the scatter plot to get the selection indices.
    selection_stream.source = scatter_plot

    # Use the linker ONLY for visual highlighting between all plots.
    linker = hv.link_selections.instance()
    data_to_link = polygons_sar + polygons_swot + scatter_plot
    linked_layout = linker(data_to_link)

    # Extract the visually linked plots from the layout
    linked_sar = linked_layout[0, 0]
    linked_swot = linked_layout[0, 1]
    linked_scatter_plot = linked_layout[0, 2]

    print("--- STEP 4: Assembling final dashboard layout... ---")
    # Build the final scatter view by overlaying the identity line.
    identity_line = hv.Slope(1, 0).opts(color='red', line_width=1.5)

    final_scatter = (linked_scatter_plot * identity_line).opts(
        hv.opts.Overlay(
            width=width, height=height,
            xlabel='SWOT mean SWH [m]', ylabel='S-1 IW SWH most likely [m]',
            title='SWOT vs SAR SWH',
            show_grid=True,
            xlim=(0, 8), ylim=(0, 8)
        )
    )

    # Create the Dynamic Statistics Pane (unchanged)
    def create_stats_pane(index):
        subset_gdf = gdf if not index else gdf.iloc[index]
        stats = calculate_statistics(subset_gdf)
        title = "### Global Statistics" if not index else "### Selection Statistics"
        stats_text = f"""
        {title}
        | Metric          | Value      |
        |-----------------|------------|
        | **N points**    | {stats['N']}         |
        | **Bias (m)**    | {stats['Bias']:.3f}    |
        | **RMSE (m)**    | {stats['RMSE']:.3f}    |
        | **SI (%)**      | {stats['SI (%)']:.2f}    |
        | **Correlation** | {stats['Correlation (R)']:.3f}    |
        """
        return pn.pane.Markdown(stats_text, width=250, align='center')

    dynamic_stats_pane = pn.bind(create_stats_pane, index=selection_stream.param.index)

    # Assemble the Final Dashboard Layout
    background_tiles = hv.element.tiles.EsriImagery().opts(width=width, height=height)
    final_map_sar = background_tiles * linked_sar
    final_map_swot = background_tiles * linked_swot
    scatter_with_stats = pn.Column(final_scatter, dynamic_stats_pane)
    final_app = pn.Row(pn.Tabs(("SAR", final_map_sar), ("SWOT", final_map_swot)), scatter_with_stats)

    print("--- Dashboard object created successfully. Ready to be served. ---")
    return final_app


# --- This is the main execution block for a script ---
if __name__ == '__main__':
    # Define the path to your data file
    # Replace this with the actual path to your NetCDF file
    fseastatecoloc = "dummy_coloc.nc"

    if not os.path.exists(fseastatecoloc):
        print(f"ERROR: Data file not found at '{fseastatecoloc}'")
        # Here you could add a function call to create a dummy file if needed
    else:
        # Create the dashboard object
        app = create_linked_dashboard(fseastatecoloc)

        # This line marks the 'app' object as the one to be displayed
        # when you use the `panel serve` command.
        app.servable(title="Linked SWH Analysis Dashboard")
