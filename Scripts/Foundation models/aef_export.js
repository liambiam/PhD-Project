// =============================================================================
// aef_export.js  --  AlphaEarth (Satellite Embedding) export for the SDM trial
// Run this in the Google Earth Engine Code Editor (code.earthengine.google.com),
// then download the GeoTIFF from your Drive and point AEF_TIF at it in the R script.
//
// NOTE: confirm the dataset ID in the GEE catalog if this errors -- Google has
// used 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL' (64 bands A00..A63, 10 m, annual).
// =============================================================================

// ---- 1. Modelling extent -----------------------------------------------------
// Use the SAME extent the R script computes (occurrence bbox + BUFFER_DEG,
// capped to Africa). Print ext_mod from R and paste [lonMin, latMin, lonMax, latMax] here.
var lonMin = 28.0, latMin = -15.0, lonMax = 42.0, latMax = 5.0;   // <-- EDIT to match R
var region = ee.Geometry.Rectangle([lonMin, latMin, lonMax, latMax]);

// ---- 2. Mean embedding 2018-2022 --------------------------------------------
var col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filterDate('2018-01-01', '2023-01-01')   // 2018,2019,2020,2021,2022
            .filterBounds(region);

print('Images in window:', col.size());
var meanEmb = col.mean().clip(region).toFloat();      // 64-band mean surface state

Map.centerObject(region, 5);
Map.addLayer(meanEmb.select(['A01','A16','A09']), {min: -0.3, max: 0.3},
             'AEF (false colour)');

// ---- 3. Export at 1 km to Drive ---------------------------------------------
Export.image.toDrive({
  image: meanEmb,
  description: 'AEF_mean_2018_2022_EAfrica_1km',
  folder: 'GEE_exports',
  region: region,
  scale: 1000,            // 1 km, matching the R analysis grid
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
// After it finishes in the Tasks tab, download from Drive -> set AEF_TIF in R.
