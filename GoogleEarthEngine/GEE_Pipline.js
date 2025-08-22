// =============================================================
// UNIFIED (Refined): Summer CMI + Winter SAVI/BSI (Per-year Tasks)
// - No prints; no getInfo; async evaluate + throttled task queue
// - Skips empty seasons; exports stable orchard polygons as GeoJSON
// - Filenames include crop label
// =============================================================

/*************** CONFIG ***************/
var COUNTY_NAME = 'Butte';        // e.g., 'Kern' or 'Fresno'
var STATEFP     = '06';
var CROP_CODE   = 75;            // Almond
var CROP_LABEL  = 'almonds';     // appears in filenames

var YEAR_START  = 2016;
var YEAR_END    = 2023;

var STABILITY_MIN_MATCH = 7;     // stable orchard threshold

// Summer QC
var MIN_IMAGES_PER_PIXEL = 3;
var CLOUD_PROB_THRESH = 40;      // s2cloudless 0–100

// Output folder on Drive
var DRIVE_FOLDER = 'GEE';

/*************** Throttled queue helper (prevents UI spikes) ***************/
function queueByChunks(items, chunkSize, delayMs, perItemFn) {
  var i = 0;
  function pump() {
    var end = Math.min(i + chunkSize, items.length);
    for (; i < end; i++) perItemFn(items[i]);
    if (i < items.length) ui.util.setTimeout(pump, delayMs);
  }
  pump();
}

/**************** AOI + Stable orchards (shared) ****************/
var counties = ee.FeatureCollection('TIGER/2018/Counties');
var aoi = counties.filter(ee.Filter.and(
  ee.Filter.eq('NAME', COUNTY_NAME),
  ee.Filter.eq('STATEFP', STATEFP)
)).geometry();

var csb = ee.FeatureCollection('projects/sat-io/open-datasets/USDA/CSB_1623')
            .filterBounds(aoi);

var yearsList = ee.List.sequence(YEAR_START, YEAR_END);

var withScores = csb.map(function(f) {
  var matches = yearsList.map(function(y) {
    y = ee.Number(y).int();
    var prop = ee.String('CDL').cat(y);
    var code = ee.Number(ee.Algorithms.If(f.get(prop), f.get(prop), 0));
    return code.eq(CROP_CODE);
  });
  var stabilityScore = ee.Number(matches.reduce(ee.Reducer.sum()));
  return f.set({ stability_score: stabilityScore, orch_id: f.id() });
});
var stable = withScores.filter(ee.Filter.gte('stability_score', STABILITY_MIN_MATCH));
var studyBounds = stable.geometry().bounds();

/**************** Export stable-orchard polygons as GeoJSON (once) ****************/
stable.size().evaluate(function(nStable) {
  if (nStable && nStable > 0) {
    var orchardPolys = stable.map(function(ft){
      var keep = ['orch_id','stability_score','ACRES','CNTY'];
      return ee.Feature(ft.geometry()).copyProperties(ft, keep);
    });
    var polyTag = COUNTY_NAME.replace(/\s/g, '') + '_' + CROP_LABEL + '_stable_orchards';
    Export.table.toDrive({
      collection: orchardPolys,
      description: polyTag,
      fileNamePrefix: polyTag,
      fileFormat: 'GeoJSON',
      folder: DRIVE_FOLDER
    });
  }
});

/**************** SUMMER helpers (CMI) ****************/
function joinCloudProb(col) {
  var clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');
  return ee.ImageCollection(ee.Join.saveFirst('clouds').apply({
    primary: col,
    secondary: clouds,
    condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
  }));
}
function maskWithS2Cloudless(img) {
  var cloudProb = ee.Image(img.get('clouds')).select('probability');
  var isCloud = cloudProb.gt(CLOUD_PROB_THRESH);
  var dark = img.select('B8').lt(3000);
  var mask = isCloud.or(dark).not();
  return img.updateMask(mask);
}
function addEVI2(img, scaleDiv) {
  var nir = img.select('B8').divide(scaleDiv);
  var red = img.select('B4').divide(scaleDiv);
  var evi2 = nir.subtract(red).multiply(2.5)
    .divide(nir.add(red.multiply(2.4)).add(1.0)).rename('EVI2');
  return img.addBands(evi2);
}
function s2Unified(start, end, geom) {
  var s2sr  = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(geom).filterDate(start, end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80));
  var s2toa = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                .filterBounds(geom).filterDate(start, end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80));

  var sr = joinCloudProb(s2sr).map(maskWithS2Cloudless)
           .map(function(i){ return addEVI2(i, 10000).select('EVI2'); });
  var toa = joinCloudProb(s2toa).map(maskWithS2Cloudless)
           .map(function(i){ return addEVI2(i, 10000).select('EVI2'); });

  return sr.merge(toa).sort('system:time_start');
}

/**************** WINTER helper (SAVI/BSI) ****************/
var PERIOD_DEFS = [
  {offS:'11-16', offE:'11-30', name:'p1', tp:1},
  {offS:'11-28', offE:'12-18', name:'p2', tp:2},
  {offS:'12-12', offE:'01-05', name:'p3', tp:3},
  {offS:'01-01', offE:'01-15', name:'p4', tp:4},
  {offS:'01-16', offE:'01-31', name:'p5', tp:5},
  {offS:'01-28', offE:'02-18', name:'p6', tp:6},
  {offS:'02-12', offE:'03-05', name:'p7', tp:7},
  {offS:'03-01', offE:'03-16', name:'p8', tp:8}
];

function saviBsiComposite(startD, endD, studyBounds) {
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(studyBounds)
            .filterDate(startD, endD)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .select(['B2','B4','B8','B11']);
  var hasImages = s2.size().gt(0);

  var comp = ee.Algorithms.If(
    hasImages,
    s2.map(function(i){ return i.divide(10000); }).median().clip(studyBounds),
    ee.Image([0,0,0,0]).rename(['B2','B4','B8','B11']).clip(studyBounds)
  );
  comp = ee.Image(comp);

  var nir  = comp.select('B8');
  var red  = comp.select('B4');
  var blue = comp.select('B2');
  var swir = comp.select('B11');

  var savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5).rename('SAVI');
  var bsi  = comp.expression('((sw+rd)-(ni+bl))/((sw+rd)+(ni+bl))', {sw: swir, rd: red, ni: nir, bl: blue})
                 .rename('BSI');

  var savi_fill = savi.gt(0.3).rename('SAVI_Fill'); // vegetation presence
  var bsi_fill  = bsi.lt(0.1).rename('BSI_Fill');   // low soil exposure

  var finalSavi     = ee.Algorithms.If(hasImages, savi,     savi.mask(0));
  var finalBsi      = ee.Algorithms.If(hasImages, bsi,      bsi.mask(0));
  var finalSaviFill = ee.Algorithms.If(hasImages, savi_fill, savi_fill.mask(0));
  var finalBsiFill  = ee.Algorithms.If(hasImages, bsi_fill,  bsi_fill.mask(0));

  return ee.Image(finalSavi)
    .addBands(ee.Image(finalBsi))
    .addBands(ee.Image(finalSaviFill))
    .addBands(ee.Image(finalBsiFill));
}

/**************** Per-year: SUMMER CMI export ****************/
function exportSummerCMI(y) {
  // y is plain JS number
  var useWide = y <= 2018;
  var start = ee.Date.fromYMD(y, useWide ? 5 : 6, 15);
  var end   = ee.Date.fromYMD(y, useWide ? 11 : 10, 15);

  var col = s2Unified(start, end, studyBounds);
  var size = col.size(); // server-side

  size.evaluate(function(n) {
    if (!n || n === 0) return; // skip empty season

    var count = col.count().rename('EVI2_count');
    var p90   = ee.Image(ee.Algorithms.If(
                  ee.Number(n).gt(0),
                  col.reduce(ee.Reducer.percentile([90])).rename('EVI2_p90'),
                  ee.Image.constant(-9999).rename('EVI2_p90')
                ));
    var valid = count.gte(MIN_IMAGES_PER_PIXEL).rename('valid');
    var img   = ee.Image.cat([p90.updateMask(valid), count, valid]);

    var reducer = ee.Reducer.mean().combine({reducer2: ee.Reducer.median(), sharedInputs: true});
    var stats = img.reduceRegions({
      collection: stable,
      reducer: reducer,
      scale: 10,
      tileScale: 16
    });

    // Tidy
    stats = stats.map(function(ft){
      var p90_mean   = ee.Algorithms.If(ft.get('EVI2_p90_mean'), ee.Number(ft.get('EVI2_p90_mean')), ee.Number(-9999));
      var p90_median = ee.Algorithms.If(ft.get('EVI2_p90_median'), ee.Number(ft.get('EVI2_p90_median')), ee.Number(-9999));
      var cnt_mean   = ee.Algorithms.If(ft.get('EVI2_count_mean'), ee.Number(ft.get('EVI2_count_mean')), ee.Number(0));
      var vfrac      = ee.Algorithms.If(ft.get('valid_mean'),      ee.Number(ft.get('valid_mean')),      ee.Number(0));
      return ft.set({
        summer_year: ee.Number(y),
        season: ee.String(String(y) + '_' + String(y + 1)),
        EVI2_p90_summer_mean: p90_mean,
        EVI2_p90_summer_median: p90_median,
        EVI2_summer_imgcount_mean: cnt_mean,
        valid_fraction: vfrac
      })
      .select(['orch_id','stability_score','summer_year','season',
               'EVI2_p90_summer_mean','EVI2_p90_summer_median',
               'EVI2_summer_imgcount_mean','valid_fraction']);
    });

    // Robust per-season z-score
    var good = stats.filter(ee.Filter.gt('EVI2_p90_summer_mean', -9999));
    var nGood = good.size();

    var meanStd = ee.Dictionary(ee.Algorithms.If(
      nGood.gte(2),
      good.reduceColumns(
        ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
        ['EVI2_p90_summer_mean']
      ),
      ee.Dictionary({'mean': null, 'stdDev': 0})
    ));
    var mu = ee.Number(meanStd.get('mean'));
    var sd = ee.Number(meanStd.get('stdDev'));

    var med = ee.Number(ee.Dictionary(ee.Algorithms.If(
      nGood.gt(0),
      good.reduceColumns(ee.Reducer.median(), ['EVI2_p90_summer_mean']),
      ee.Dictionary({'median': null})
    )).get('median'));

    var q25 = ee.Number(ee.Dictionary(ee.Algorithms.If(
      nGood.gt(0),
      good.reduceColumns(ee.Reducer.percentile([25]), ['EVI2_p90_summer_mean']),
      ee.Dictionary({'percentile_25': null})
    )).get('percentile_25'));

    var q75 = ee.Number(ee.Dictionary(ee.Algorithms.If(
      nGood.gt(0),
      good.reduceColumns(ee.Reducer.percentile([75]), ['EVI2_p90_summer_mean']),
      ee.Dictionary({'percentile_75': null})
    )).get('percentile_75'));

    var iqr = ee.Number(ee.Algorithms.If(q25.and(q75), q75.subtract(q25), 0));
    var robustSd = iqr.divide(1.349);

    var useRobust = sd.lte(0);
    var center    = ee.Number(ee.Algorithms.If(useRobust, med, mu));
    var scale     = ee.Number(ee.Algorithms.If(useRobust, robustSd, sd));
    scale         = ee.Number(ee.Algorithms.If(scale.lte(0), 1, scale));

    var withCMI = stats.map(function(ft){
      var val = ee.Number(ft.get('EVI2_p90_summer_mean'));
      var cmi = ee.Algorithms.If(val.gt(-9999), val.subtract(center).divide(scale), null);
      return ft.set({ CMI_season_z: cmi });
    });

    var baseTag = COUNTY_NAME.replace(/\s/g, '') + '_' + CROP_LABEL + '_' + y + '_CMI';
    Export.table.toDrive({
      collection: withCMI,
      description: baseTag,
      fileNamePrefix: baseTag,
      fileFormat: 'CSV',
      folder: DRIVE_FOLDER,
      selectors: [
        'orch_id','season','summer_year','stability_score',
        'EVI2_p90_summer_mean','EVI2_p90_summer_median',
        'EVI2_summer_imgcount_mean','valid_fraction','CMI_season_z'
      ]
    });
  });
}

/**************** Per-year: WINTER SAVI/BSI export ****************/
function exportWinterYear(y) {
  // y is plain JS number
  // Compute total images across all winter windows (server-side number)
  var totalImgs = PERIOD_DEFS.reduce(function(acc, d) {
    var sMonth = parseInt(d.offS.split('-')[0], 10);
    var eMonth = parseInt(d.offE.split('-')[0], 10);
    var sDay   = parseInt(d.offS.split('-')[1], 10);
    var eDay   = parseInt(d.offE.split('-')[1], 10);
    var sYear  = (sMonth >= 11) ? y : (y + 1);
    var eYear  = (eMonth >= 11) ? y : (y + 1);
    var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(studyBounds)
                .filterDate(ee.Date.fromYMD(sYear, sMonth, sDay),
                            ee.Date.fromYMD(eYear, eMonth, eDay))
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));
    return ee.Number(acc).add(s2.size());
  }, ee.Number(0));

  totalImgs.evaluate(function(n) {
    if (!n || n === 0) return; // skip empty winter

    var combos = PERIOD_DEFS.map(function(d) {
      var sMonth = parseInt(d.offS.split('-')[0], 10);
      var eMonth = parseInt(d.offE.split('-')[0], 10);
      var sDay   = parseInt(d.offS.split('-')[1], 10);
      var eDay   = parseInt(d.offE.split('-')[1], 10);
      var sYear  = (sMonth >= 11) ? y : (y + 1);
      var eYear  = (eMonth >= 11) ? y : (y + 1);
      return {
        season: y + '_' + (y + 1),
        tp: d.tp,
        startDate: ee.Date.fromYMD(sYear, sMonth, sDay),
        endDate:   ee.Date.fromYMD(eYear, eMonth, eDay)
      };
    });

    var allResults = ee.FeatureCollection(combos.map(function(c) {
      var img = saviBsiComposite(c.startDate, c.endDate, studyBounds);
      var reducer = ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});

      var stats = img.reduceRegions({
        collection: stable,
        reducer: reducer,
        scale: 10,
        tileScale: 16
      });

      return stats.map(function(ft) {
        return ft.set({
          season: c.season,
          time_period: c.tp,
          period_start_date: c.startDate.format('yyyy-MM-dd'),
          period_end_date:   c.endDate.format('yyyy-MM-dd'),
          orchard_savi_mean:    ee.Algorithms.If(ft.get('SAVI_mean'),       ft.get('SAVI_mean'),       0),
          orchard_savi_stddev:  ee.Algorithms.If(ft.get('SAVI_stdDev'),     ft.get('SAVI_stdDev'),     0),
          orchard_savi_fill:    ee.Algorithms.If(ft.get('SAVI_Fill_mean'),  ft.get('SAVI_Fill_mean'),  0),
          orchard_bsi_mean:     ee.Algorithms.If(ft.get('BSI_mean'),        ft.get('BSI_mean'),        0),
          orchard_bsi_stddev:   ee.Algorithms.If(ft.get('BSI_stdDev'),      ft.get('BSI_stdDev'),      0),
          orchard_bsi_fill:     ee.Algorithms.If(ft.get('BSI_Fill_mean'),   ft.get('BSI_Fill_mean'),   0)
        });
      });
    })).flatten();

    var exportColumns = [
      'orch_id', 'season', 'time_period', 'period_start_date', 'period_end_date',
      'orchard_savi_mean', 'orchard_savi_stddev', 'orchard_savi_fill',
      'orchard_bsi_mean', 'orchard_bsi_stddev', 'orchard_bsi_fill',
      'ACRES', 'CNTY', 'stability_score', 'startyear', 'endyear'
    ];
    var clean = allResults.select(exportColumns, null, false);

    var baseTag = COUNTY_NAME.replace(/\s/g, '') + '_' + CROP_LABEL + '_' + y + '_WINTER';
    Export.table.toDrive({
      collection: clean,
      description: baseTag,
      fileNamePrefix: baseTag,
      fileFormat: 'CSV',
      folder: DRIVE_FOLDER,
      selectors: exportColumns
    });
  });
}

/**************** Kick off: throttle task creation ****************/
yearsList.evaluate(function(yearsArray) {
  if (!yearsArray || yearsArray.length === 0) return;
  // e.g., 3 years queued every 250 ms; adjust if your UI is still touchy
  queueByChunks(yearsArray, 3, 250, function(y) {
    exportSummerCMI(y);
    exportWinterYear(y);
  });
});
