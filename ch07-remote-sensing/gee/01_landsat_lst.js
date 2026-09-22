// ============================================================================
// Chapter 6 · Script 1 — Landsat 8/9 LST and spectral indices
// Google Earth Engine Code Editor  (https://code.earthengine.google.com)
// No installation. Free for research use after registration.
// ----------------------------------------------------------------------------
// EDIT THIS: draw your own polygon, or paste coordinates for your city.
var aoi = ee.Geometry.Rectangle([104.85, 11.48, 105.00, 11.62]);  // Phnom Penh
var YEAR = 2023;
// ----------------------------------------------------------------------------

// Cloud/shadow mask using the QA_PIXEL bitmask of Collection 2 Level 2
function maskL2(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)   // dilated cloud
      .and(qa.bitwiseAnd(1 << 3).eq(0))    // cloud
      .and(qa.bitwiseAnd(1 << 4).eq(0));   // cloud shadow
  // official Collection 2 Level 2 scale factors
  var sr = img.select('SR_B.').multiply(0.0000275).add(-0.2);
  var st = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15);
  return img.addBands(sr, null, true)
            .addBands(st.rename('LST_C'), null, true)
            .updateMask(mask);
}

var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(aoi)
    .filterDate(YEAR + '-01-01', YEAR + '-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .map(maskL2);

print('Images found:', col.size());   // if 0, widen the date range

var img = col.median().clip(aoi);

// Spectral indices
var ndvi = img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
var ndbi = img.normalizedDifference(['SR_B6','SR_B5']).rename('NDBI');
var ndwi = img.normalizedDifference(['SR_B3','SR_B5']).rename('NDWI');
var stack = img.addBands([ndvi, ndbi, ndwi]);

Map.centerObject(aoi, 12);
Map.addLayer(img, {bands:['SR_B4','SR_B3','SR_B2'], min:0, max:0.3}, 'True colour');
Map.addLayer(ndvi, {min:-0.2, max:0.8, palette:['white','green']}, 'NDVI');
Map.addLayer(img.select('LST_C'), {min:25, max:45,
             palette:['blue','yellow','red']}, 'LST (C)');

// Export the stack for local analysis with rs_toolkit.py
Export.image.toDrive({
  image: stack.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','LST_C'])
              .toFloat(),
  description: 'scene_' + YEAR,
  region: aoi, scale: 30, maxPixels: 1e9, crs: 'EPSG:4326'
});
