// ============================================================================
// Chapter 6 · Script 3 — Built-up expansion without your own classification
// GHSL (JRC) global built-up surface — use when you lack training samples.
// ============================================================================
var aoi = ee.Geometry.Rectangle([104.85, 11.48, 105.00, 11.62]);

var ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S');
var b2015 = ghsl.filter(ee.Filter.eq('system:index','2015')).first().clip(aoi);
var b2020 = ghsl.filter(ee.Filter.eq('system:index','2020')).first().clip(aoi);

var gain = b2020.subtract(b2015).rename('built_gain');

Map.centerObject(aoi, 12);
Map.addLayer(b2020, {min:0, max:10000, palette:['white','red']}, 'Built 2020');
Map.addLayer(gain.updateMask(gain.gt(500)),
             {min:0, max:5000, palette:['yellow','darkred']}, 'Gain 2015-2020');

// Built-up area in hectares
[b2015, b2020].forEach(function (im, i) {
  var ha = im.divide(10000).multiply(ee.Image.pixelArea()).divide(10000)
             .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                            scale: 100, maxPixels: 1e9});
  print(['Built-up ha 2015','Built-up ha 2020'][i], ha);
});

// Population (WorldPop) for density and exposure analysis
var pop = ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.eq('year', 2020)).mosaic().clip(aoi);
print('Population in AOI', pop.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: aoi, scale: 100, maxPixels: 1e9}));
