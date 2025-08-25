# Cover Crop Web App: Acres Calculation Fix

## Problem Description

The current web application's "percentage of cover cropped by acres" feature was not working, despite the PMTiles containing an `acres` attribute. The "% Acres Covered" statistic in the right panel consistently showed 0%, while the "% Cover Cropped" (count-based) statistic worked correctly.

## Root Cause Analysis

### Comparison Methodology

We compared the current non-working web app (`/Web_App/Data/webApp.html`) with a historical working version (`/HistroricalWorkingWebApp/deck_gl_polygon_dashboard_time_agnostic_fixed_v7_mvt2_PATCHED_v2.html`) to identify the discrepancies.

### Key Findings

The current web app had **sophisticated acres calculation infrastructure** but was missing the **data extraction mechanism**:

#### ✅ **Present in Current Version:**
- `updateStatsFromRows()` function (lines 1523-1580) with complete acres percentage calculation logic
- `areaById` Map for storing orchard areas
- UI elements for displaying "% Acres Covered" statistic
- Geodesic area calculation functions (`calculatePolygonArea`, `geodesicRingAreaSqm`)

#### ❌ **Missing in Current Version:**
1. **`indexFeaturesFromTile()` function** - Extracts acres data from PMTiles features
2. **`onTileLoad` callbacks** - Triggers data extraction when tiles load
3. **`sourceLayer: 'orchards'`** - Proper layer configuration for regular PMTiles

#### ✅ **Present in Working Historical Version:**
- Complete `indexFeaturesFromTile()` function (lines 513-547)
- Proper `onTileLoad` callbacks in both MVTLayers
- `sourceLayer: 'orchards'` configuration

## Identified Issues

### Issue 1: Missing Data Extraction Function
The current app lacked the `indexFeaturesFromTile()` function that:
- Reads `f.properties?.acres` from PMTiles features
- Falls back to geodesic calculation if no acres attribute exists
- Populates the `areaById` Map with orchard areas
- Updates existing row data with `area_acres` values

### Issue 2: Missing Tile Load Callbacks
Both MVTLayers (time-agnostic and regular) were missing `onTileLoad` callbacks to trigger the acres extraction when tiles are loaded.

### Issue 3: Missing Source Layer Configuration  
The regular PMTiles layer was missing the `sourceLayer: 'orchards'` configuration.

## Implementation Details

### Fix 1: Added `indexFeaturesFromTile()` Function
```javascript
function indexFeaturesFromTile(features = []) {
  for (const f of features) {
    const orchId = String(f.properties?.orch_id ?? f.properties?.ORCH_ID ?? f.properties?.id ?? '');
    if (!orchId) continue;

    // Centroid for viewport filtering
    if (!centroidsById.has(orchId)) {
      const c = polygonCentroid(f.geometry);
      if (c) centroidsById.set(orchId, c);
    }

    // Area for acreage stats
    if (!areaById.has(orchId)) {
      let area = f.properties?.acres ? Number(f.properties.acres) : 0;
      if (area <= 0) {
        if (f.geometry?.type === 'Polygon') area = calculatePolygonArea(f.geometry.coordinates);
        else if (f.geometry?.type === 'MultiPolygon') {
          for (const poly of f.geometry.coordinates) area += calculatePolygonArea(poly);
        }
      }
      if (area > 0) {
        areaById.set(orchId, area);
        const rows = rowsByOrch.get(orchId);
        if (rows) rows.forEach(r => r.area_acres = area);
      }
    }
  }
}
```

**Location:** Added after `calculatePolygonArea()` function around line 512.

### Fix 2: Added `onTileLoad` Callbacks

#### Time-Agnostic MVTLayer:
```javascript
onTileLoad: (tile) => {
  console.log('🧩 Time-agnostic tile loaded:', tile.index, 'features:', tile.data?.features?.length || 0);
  indexFeaturesFromTile(tile.data || []);
},
```

#### Regular PMTiles MVTLayer:
```javascript
onTileLoad: (tile) => {
  indexFeaturesFromTile(tile.data || []);
},
```

### Fix 3: Added Source Layer Configuration
Added `sourceLayer: 'orchards'` to the regular PMTiles layer configuration:

```javascript
const layer = new deck.MVTLayer({
  id: 'orchards-mvt',
  data: pmtilesUrl,
  sourceLayer: 'orchards',  // ← Added this line
  uniqueIdProperty: 'orch_id',
  // ... rest of configuration
});
```

## Data Flow

### Before Fix:
1. PMTiles load → No data extraction
2. `areaById` Map remains empty
3. `updateStatsFromRows()` calculates with `acres = 0` for all orchards
4. "% Acres Covered" always shows 0%

### After Fix:
1. PMTiles load → `onTileLoad` triggers
2. `indexFeaturesFromTile()` extracts acres from `f.properties?.acres`
3. `areaById` Map populated with actual orchard areas
4. `updateStatsFromRows()` calculates with real acre values
5. "% Acres Covered" shows correct percentage

## Verification

The fix enables the existing acres calculation infrastructure by:
- Populating `areaById` Map when tiles load
- Ensuring `row.area_acres` values are available for statistics
- Maintaining compatibility with existing `updateStatsFromRows()` logic

## Files Modified

- **Primary:** `/Web_App/Data/webApp.html`
  - Added `indexFeaturesFromTile()` function
  - Added `onTileLoad` callbacks to both MVTLayers  
  - Added `sourceLayer: 'orchards'` to regular PMTiles layer

## Technical Notes

- The fix preserves all existing functionality while adding the missing data extraction
- Uses both PMTiles `acres` attribute and fallback geodesic calculation
- Compatible with both time-agnostic and regular viewing modes
- No changes required to the existing statistics calculation logic

## Outcome

The "% Acres Covered" statistic should now display the correct percentage of total acres that are cover cropped, enabling proper analysis of cover crop adoption by land area rather than just orchard count.