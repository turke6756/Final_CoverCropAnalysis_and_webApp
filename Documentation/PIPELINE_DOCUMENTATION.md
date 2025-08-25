# Cover Crop Analysis Data Processing Pipeline

## Overview
This pipeline processes raw Google Earth Engine (GEE) data from multiple California counties, performs clustering analysis, converts data formats, and merges everything for web application deployment.

## Pipeline Steps (1-5)

### Prerequisites
- **Python Environment**: Ensure Python with required packages (pandas, scikit-learn, etc.)
- **Tippecanoe**: Install for PMTiles processing (`tile-join`, `tippecanoe`)
- **PMTiles CLI**: Install pmtiles command-line tool
- **jq**: JSON processing tool
- **Required Tools Check**:
  ```bash
  # Verify all tools are available
  python --version
  tippecanoe --version
  pmtiles --version
  jq --version
  ```

---

## Step 1: Process Raw County Data with Reward Pipeline

### Purpose
Process raw CSV data (CMI and WINTER) for each county to generate clustering analysis and reward calculations.

### Input Location
```
RAW_GEE_DATA/Almonds/Raw_Inputs/
├── Butte/
├── Colusa/
├── Fresno/
├── Glenn/
├── Kern/
├── Madera/
├── Merced/
├── Stanislaus/
└── Tehama/
```

### Expected Input Files per County
- **CMI Data**: `{County}_almonds_{Year}_CMI.csv` (2016-2023)
- **WINTER Data**: `{County}_almonds_{Year}_WINTER.csv` (2017-2023)
- **GeoJSON**: `{County}_almonds_stable_orchards.geojson` (untouched in this step)
  - *Exception*: Fresno uses `FresnoAlmondOrchards.geojson`

### Script Used
```bash
RewardCalculationPipline/Reward_Pipeline_and_Clustering.py
```

### Output Location
```
RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/
├── Butte/
├── Colusa/
├── Fresno/
├── Glenn/
├── Kern/
├── Madera/
├── Merced/
├── Stanislaus/
└── Tehama/
```

### Execution Instructions
1. **Create output directories**:
   ```bash
   mkdir -p "RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/{Butte,Colusa,Fresno,Glenn,Kern,Madera,Merced,Stanislaus,Tehama}"
   ```

2. **Process each county individually** (example for Butte):
   ```bash
   cd RewardCalculationPipline
   python Reward_Pipeline_and_Clustering.py \
     --csv-winter "../RAW_GEE_DATA/Almonds/Raw_Inputs/Butte/Butte_almonds_*_WINTER.csv" \
     --csv-cmi "../RAW_GEE_DATA/Almonds/Raw_Inputs/Butte/Butte_almonds_*_CMI.csv" \
     --out-ts "../RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/Butte/rewards_timeseries.csv" \
     --out-trends "../RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/Butte/reward_trends.csv" \
     [... additional output parameters ...]
   ```

3. **Repeat for all 9 counties** with appropriate paths

### Expected Outputs per County
- `rewards_timeseries.csv`
- `reward_trends.csv`
- `cluster_assignments_train.csv`
- `cluster_assignments_all.csv`
- `cluster_centroids.csv`
- `manifest.json`
- `streaks.csv`
- `attrs/` directory with yearly CSV files (2018-2023)
- Additional clustering analysis files

---

## Step 2: Convert CSV to JSON

### Purpose
Convert county-specific CSV outputs to JSON format for web application consumption.

### Script Used
```bash
RewardCalculationPipline/CSV_to_Json_Converter.py
```

### Modifications Made
The script was modified to accept command-line arguments:
- Added `--input_dir` and `--output_dir` parameters
- Added support for nested `attrs/` directory structure

### Output Location
```
RAW_GEE_DATA/Almonds/Pipline_Outputs/Json_Outputs/
├── Butte/
├── Colusa/
├── Fresno/
├── Glenn/
├── Kern/
├── Madera/
├── Merced/
├── Stanislaus/
└── Tehama/
```

### Execution Instructions
1. **Create output directories**:
   ```bash
   mkdir -p "RAW_GEE_DATA/Almonds/Pipline_Outputs/Json_Outputs/{Butte,Colusa,Fresno,Glenn,Kern,Madera,Merced,Stanislaus,Tehama}"
   ```

2. **Process each county** (example for Butte):
   ```bash
   cd RewardCalculationPipline
   python CSV_to_Json_Converter.py \
     "../RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/Butte" \
     "../RAW_GEE_DATA/Almonds/Pipline_Outputs/Json_Outputs/Butte"
   ```

3. **Repeat for all 9 counties**

### Files Converted
- `attrs/*.csv` → `attrs/*.json` (yearly attribute data)
- `streaks.csv` → `streaks.json`
- `manifest.json` → `manifest.json` (copied as-is)

---

## Step 3: Merge County JSONs for Web App

### Purpose
Merge all county-specific JSON files into unified files for web application access.

### Script Used
```bash
RewardCalculationPipline/Merge_Json_for_WebApp.py
```

### Modifications Made
Updated `discover_county_crop_folders()` function to handle `attrs/` subdirectory structure.

### Output Location
```
Web_App/Data/
```

### Execution Instructions
```bash
cd RewardCalculationPipline
python Merge_Json_for_WebApp.py \
  --input_root "../RAW_GEE_DATA/Almonds/Pipline_Outputs/Json_Outputs" \
  --output "../Web_App/Data"
```

### Expected Outputs
- `2017.json` (3 orchards - Stanislaus only)
- `2018.json` through `2023.json` (16,365 orchards each)
- `manifest.json` (unified configuration)
- `streaks.json` (7,464 orchards with streak data)
- `merge_report.json` (processing statistics)
- `merge_log.json` (merge tracking)

---

## Step 4: Convert GeoJSON to PMTiles

### Purpose
Convert county-specific GeoJSON orchard boundary files to PMTiles format for web mapping.

### Script Used
```bash
RewardCalculationPipline/GeoJson_to_PMTiles_Converter.sh
```

### Modifications Made
- Added county name extraction from filename
- Added `--output-dir` parameter support
- Updated output naming to use county names (e.g., `butte.pmtiles`)

### Input Files
- Standard naming: `{County}_almonds_stable_orchards.geojson`
- Fresno exception: `FresnoAlmondOrchards.geojson`

### Output Location
```
RAW_GEE_DATA/Almonds/Pipline_Outputs/pmTiles/
```

### Execution Instructions
Process each county individually:
```bash
cd RewardCalculationPipline

# Standard counties (8 counties)
./GeoJson_to_PMTiles_Converter.sh \
  "../RAW_GEE_DATA/Almonds/Raw_Inputs/Butte/Butte_almonds_stable_orchards.geojson" \
  --output-dir "../RAW_GEE_DATA/Almonds/Pipline_Outputs/pmTiles"

# Fresno (special naming)
./GeoJson_to_PMTiles_Converter.sh \
  "../RAW_GEE_DATA/Almonds/Raw_Inputs/Fresno/FresnoAlmondOrchards.geojson" \
  --output-dir "../RAW_GEE_DATA/Almonds/Pipline_Outputs/pmTiles"

# Repeat for all counties...
```

### Expected Outputs
- `{county}.pmtiles` (9 files: butte.pmtiles, colusa.pmtiles, etc.)
- `{county}.mbtiles` (intermediate format)
- Enhanced GeoJSON files (with crop analysis)

### Properties Included in PMTiles
- `orch_id` (string)
- `county` (string)
- `dominant_crop` (string)
- `current_crop` (string)
- `stability_score` (float)
- `acres` (float)

---

## Step 5: Merge PMTiles for Web App

### Purpose
Combine all county PMTiles into a single unified spatial dataset for the web application.

### Script Used
Manual process using tippecanoe tools (the provided shell script had compatibility issues)

### Output Location
```
Web_App/Data/california-orchards.pmtiles
```

### Execution Instructions
```bash
cd "RAW_GEE_DATA/Almonds/Pipline_Outputs/pmTiles"

# Merge MBTiles files (PMTiles direct merge had issues)
tile-join -f -pk -o california-orchards-merged.mbtiles *.mbtiles

# Convert to PMTiles and place in Web_App/Data
pmtiles convert california-orchards-merged.mbtiles \
  "/path/to/Web_App/Data/california-orchards.pmtiles"
```

### Expected Output
- `california-orchards.pmtiles` (~3.4 MB)
- Contains 16,514 orchard polygons
- Covers all 9 California counties
- Zoom levels 0-10 for web mapping

---

## Final Web App Data Structure

After completing all 5 steps, the `Web_App/Data/` directory should contain:

```
Web_App/Data/
├── 2017.json                    # 3 orchards (Stanislaus only)
├── 2018.json                    # 16,365 orchards (all counties)
├── 2019.json                    # 16,365 orchards (all counties)
├── 2020.json                    # 16,365 orchards (all counties)
├── 2021.json                    # 16,365 orchards (all counties)
├── 2022.json                    # 16,365 orchards (all counties)
├── 2023.json                    # 16,365 orchards (all counties)
├── california-orchards.pmtiles  # 16,514 spatial polygons
├── manifest.json                # Unified configuration
├── streaks.json                 # 7,464 orchards with streak data
├── merge_log.json               # Merge tracking
└── merge_report.json            # Merge statistics
```

## Pipeline Validation

### Data Counts Verification
- **Total Orchard Count**: 16,365 orchards in yearly JSON files
- **Spatial Polygon Count**: 16,514 polygons in PMTiles (slight difference due to processing)
- **Streak Data Count**: 7,464 orchards with cover crop streak information
- **Counties Covered**: 9 (Butte, Colusa, Fresno, Glenn, Kern, Madera, Merced, Stanislaus, Tehama)
- **Years Available**: 2017-2023 (7 years)

### File Size Expectations
- **Total JSON Size**: ~27 MB
- **PMTiles Size**: ~3.4 MB
- **Complete Dataset**: ~30 MB total

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all required tools are installed
2. **File Path Issues**: Use absolute paths when relative paths fail
3. **PMTiles Merge Issues**: Use MBTiles intermediate format if direct PMTiles merge fails
4. **Memory Issues**: Process counties individually rather than all at once for large datasets
5. **CMI Clustering Failure**: If all orchards default to "Mid" stratum instead of Young/Old, see CMI Clustering section below

### CMI Clustering Issues
**Problem**: All orchards assigned "Mid" stratum instead of proper Young/Old age classifications, causing incorrect `best_soft_norm` values.

**Root Cause**: Missing `--cmi-cluster` flag in Step 1 execution or stale intermediate files.

**Prevention Steps**:
1. **Clean Previous Outputs**: Remove all existing pipeline outputs before re-running
   ```bash
   rm -rf "RAW_GEE_DATA/Almonds/Pipline_Outputs/CSV_Outputs/*"
   ```

2. **Include CMI Clustering Flag**: Ensure `--cmi-cluster` is added to Step 1 execution for each county
   ```bash
   python Reward_Pipeline_and_Clustering.py \
     --csv-winter "path/to/county/WINTER/files.csv" \
     --csv-cmi "path/to/county/CMI/files.csv" \
     --cmi-cluster \  # ← CRITICAL: Enables age stratification
     --out-ts "output/path/rewards_timeseries.csv" \
     [... other parameters ...]
   ```

3. **Verification**: After each county execution, check:
   - `cmi_assignments.csv` contains "Young" and "Old" labels (not just "Mid")
   - `attrs/*.csv` files show varied stratum values (Young/Old distribution)
   - `best_soft_norm` values are varied (0.0 to 1.0 range, not uniform values)

**Testing**: If issues persist, test individual counties in isolation using the same parameters to verify clustering works correctly before running the full pipeline.

### File Naming Exceptions
- **Fresno GeoJSON**: Uses `FresnoAlmondOrchards.geojson` instead of standard naming
- **Stanislaus Data**: May have 2017 data (others typically start from 2018)

## Notes for Future Runs

1. **Input Data Updates**: When new years of data are available, update the year ranges in scripts
2. **County Additions**: If new counties are added, update all scripts to include them
3. **Web App Integration**: Ensure the web application is configured to read from `Web_App/Data/`
4. **Backup Strategy**: Consider backing up intermediate outputs in case of pipeline failures

## Performance Notes

- **Step 1**: Most time-consuming (clustering analysis)
- **Step 4**: Moderate time (GeoJSON processing)
- **Steps 2,3,5**: Relatively fast (data conversion/merging)
- **Total Pipeline Time**: Approximately 30-60 minutes depending on system performance

---

*Pipeline Documentation Generated: August 22, 2025*
*Last Updated: Initial Version*