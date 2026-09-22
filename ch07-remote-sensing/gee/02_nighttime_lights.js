// ============================================================================
// Chapter 6 · Script 2 — VIIRS nighttime lights as an economic proxy
// Use where no reliable sub-national economic statistics exist.
// ============================================================================
var aoi = ee.Geometry.Rectangle([104.85, 11.48, 105.00, 11.62]);

// Monthly VIIRS, stray-light corrected
function annual(year) {
  return ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
      .filterDate(year + '-01-01', year + '-12-31')
      .select('avg_rad').median().clip(aoi).set('year', year);
}

var years = ee.List.sequence(2014, 2023);
var series = ee.ImageCollection(years.map(function (y) {
  return annual(ee.Number(y).format('%d'));
}));

// Mean radiance per year over the AOI -> a growth time series
var chart = ui.Chart.image.series({
  imageCollection: series, region: aoi,
  reducer: ee.Reducer.mean(), scale: 500, xProperty: 'year'
}).setOptions({title: 'Mean nighttime radiance', lineWidth: 2});
print(chart);

Map.centerObject(aoi, 11);
Map.addLayer(annual('2023'), {min:0, max:60, palette:['black','yellow','white']},
             'NTL 2023');

// Zonal aggregation to administrative units (join to Chapter 3 analysis)
// var adm = ee.FeatureCollection('users/YOUR_ID/communes');
// var stats = annual('2023').reduceRegions({
//   collection: adm, reducer: ee.Reducer.mean(), scale: 500});
// Export.table.toDrive({collection: stats, description: 'ntl_by_commune'});
