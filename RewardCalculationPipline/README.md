# PMTiles Documentation

## Overview

This directory contains tools and output for converting GeoJSON orchard data to PMTiles format for web mapping applications.

## Files

- `enhanced_orchards_to_pmtiles_fixed.sh` - Enhanced conversion script
- `enhanced_orchards.pmtiles` - Output PMTiles file (597KB)
- `FresnoAlmondOrchards.enhanced.geojson` - Processed GeoJSON with calculated properties
- `FresnoAlmondOrchards.geojson` - Original source data (8.4MB)

## PMTiles Properties

Each orchard feature contains these properties:

| Property | Type | Description | Example Values |
|----------|------|-------------|----------------|
| `orch_id` | string | Unique orchard identifier | "005600000000000031ca" |
| `county` | string | County name | "Fresno", "Madera", "Kings" |
| `dominant_crop` | string | Most frequent crop (2016-2023) | "Almonds" |
| `current_crop` | string | Current year crop (2023) | "Almonds", "Alfalfa" |
| `stability_score` | float | Confidence metric (1-8) | 7.0 |
| `acres` | float | Orchard area in acres | 5.38, 65.39 |

## Usage

### Conversion Script
```bash
./enhanced_orchards_to_pmtiles_fixed.sh FresnoAlmondOrchards.geojson
```

### Web Application Integration
```javascript
// Load PMTiles
const tilesUrl = './enhanced_orchards.pmtiles';

// County filtering example
map.setFilter('orchards', ['==', ['get', 'county'], 'Fresno']);

// Styling by dominant crop
map.setPaintProperty('orchards', 'fill-color', [
  'match',
  ['get', 'dominant_crop'],
  'Almonds', '#8B4513',
  'Alfalfa', '#32CD32',
  '#999999' // default
]);

// Size visualization by acres
map.setPaintProperty('orchards', 'fill-opacity', [
  'interpolate',
  ['linear'],
  ['get', 'acres'],
  0, 0.3,    // Small orchards (0 acres)
  50, 0.8    // Large orchards (50+ acres)
]);
```

## Data Summary

- **Total Features:** 3,637 orchards
- **Counties:** Fresno (99%), Madera, Kings, Tulare, Merced
- **Zoom Levels:** 0-10
- **Bounds:** -120.78°, 36.09° to -119.37°, 37.05°
- **File Size:** 597KB (compressed)

## Processing Notes

- Original GeoJSON contains 33+ properties per feature  
- Enhanced version calculates dominant crop from 8 years of data
- Includes original ACRES field for area-based visualizations and analysis
- Optimized for web performance while preserving essential attributes
- Handles crop misclassification years through dominant crop calculation