# DATASETS: GDP, CMEMS/HYCOM, ERA5

This repo provides a synthetic dataset for instant testing and scripts for **real data** preparation.

## GDP Drifter Data (NOAA/AOML)
- Portal: https://www.aoml.noaa.gov/phod/gdp/data.php
- Hourly drifter product: https://www.aoml.noaa.gov/phod/gdp/hourly_data.php
- US Data Catalog entry: https://catalog.data.gov/dataset/hourly-location-current-velocity-and-temperature-collected-from-global-drifter-program-drifters1

You can download netCDF/CSV tracks, then subset drifters, clean QC flags, and resample to hourly  trajectories.

## CMEMS (Copernicus Marine Service) Surface Currents
- Access: https://marine.copernicus.eu/access-data
- Toolbox (CLI/Python): https://help.marine.copernicus.eu/en/articles/8286883-copernicus-marine-toolbox-api-get-original-files
- Subsetting guide: https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset
- Python client: https://pypi.org/project/copernicus-marine-client/

Typical variables: `uo`/`vo` (eastward/northward current). Use the Toolbox to subset by bounding box and time range covering your drifter tracks.

## HYCOM (Alternative/Complementary)
- Data server: https://www.hycom.org/dataserver
- OPeNDAP/THREDDS access: https://www.hycom.org/
- Example catalog entry: https://catalog.data.gov/dataset/nrl-hycomncoda-glbu0-08-expt-91-2-global-1-12-deg-2016-to-2018-at-depths

## ERA5 Winds (10m u/v)
- CDS dataset (single levels): https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
- ECMWF guide (example for 10m winds): https://confluence.ecmwf.int/display/CKB/ERA5%3A%2Blarge%2B10m%2Bwinds

### Collocation Pipeline
1. Spatially and temporally interpolate currents/winds to each drifter observation.
2. Build sequences of length `T` and forecast horizons (e.g., 24–72 hours).
3. Save as NPZ/Parquet: positions (lat, lon), optionally `u`, `v`, wind components, masks.
4. Train with `--data /path/to/npz_dir`.

See `scripts/preprocess_gdp.py` and `scripts/collocate_currents_winds.py` for runnable examples.
