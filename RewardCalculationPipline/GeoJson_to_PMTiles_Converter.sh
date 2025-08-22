#!/usr/bin/env bash
set -euo pipefail

# enhanced_orchards_to_pmtiles_fixed.sh  
# Convert orchard GeoJSON to PMTiles with county and crop analysis

# Defaults
LAYER_NAME="orchards"
MIN_ZOOM=""
MAX_ZOOM=""
COERCE_ID=false
MBTILES_OUT=""
PMTILES_OUT=""
OUTPUT_DIR="."

die() { echo >&2 "Error: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

# Parse args
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.geojson> [options...]"
  exit 1
fi

INPUT="$1"; shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer-name) LAYER_NAME="$2"; shift 2;;
    --minzoom)    MIN_ZOOM="$2"; shift 2;;
    --maxzoom)    MAX_ZOOM="$2"; shift 2;;
    --coerce-id)  COERCE_ID=true; shift;;
    --mbtiles)    MBTILES_OUT="$2"; shift 2;;
    --pmtiles)    PMTILES_OUT="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -f "$INPUT" ]] || die "Input not found: $INPUT"

# Extract county name from input filename
BASENAME=$(basename "$INPUT" .geojson)
if [[ "$BASENAME" == *"_almonds_stable_orchards" ]]; then
  # Standard naming: {County}_almonds_stable_orchards.geojson
  COUNTY=$(echo "$BASENAME" | sed 's/_almonds_stable_orchards$//' | tr '[:upper:]' '[:lower:]')
elif [[ "$BASENAME" == "FresnoAlmondOrchards" ]]; then
  # Special case for Fresno
  COUNTY="fresno"
else
  # Fallback: use the full basename
  COUNTY=$(echo "$BASENAME" | tr '[:upper:]' '[:lower:]')
fi

# Set default output filenames if not provided
if [[ -z "$MBTILES_OUT" ]]; then
  MBTILES_OUT="${OUTPUT_DIR}/${COUNTY}.mbtiles"
fi
if [[ -z "$PMTILES_OUT" ]]; then
  PMTILES_OUT="${OUTPUT_DIR}/${COUNTY}.pmtiles"
fi

# Check tools
need jq
need tippecanoe
need pmtiles

echo "==> Enhancing GeoJSON with crop analysis: $INPUT"

# Create enhanced GeoJSON with dominant crop analysis
ENHANCED="${INPUT%.geojson}.enhanced.geojson"

jq '.features = (.features | map(
  . as $feature |
  # Extract all crop years into array
  [$feature.properties.CROP16, $feature.properties.CROP17, $feature.properties.CROP18, 
   $feature.properties.CROP19, $feature.properties.CROP20, $feature.properties.CROP21, 
   $feature.properties.CROP22, $feature.properties.CROP23] as $crops |
  
  # Calculate dominant crop (most frequent)
  ($crops | group_by(.) | map({crop: .[0], count: length}) | sort_by(-.count) | .[0].crop) as $dominant_crop |
  
  # Create new properties with essential fields only
  $feature | .properties = {
    orch_id: $feature.properties.orch_id,
    county: $feature.properties.CNTY,
    dominant_crop: $dominant_crop,
    current_crop: $feature.properties.CROP23,
    stability_score: $feature.properties.stability_score,
    acres: $feature.properties.ACRES
  }
))' "$INPUT" > "$ENHANCED"

echo "    Enhanced file: $ENHANCED"
echo "    Added: county, dominant_crop, current_crop, stability_score, acres"

# Validate enhanced GeoJSON
TOTAL=$(jq '.features | length' "$ENHANCED")
echo "    Features: $TOTAL"

MISSING_ID=$(jq '[.features[] | has("properties") and (.properties|has("orch_id")) ] | map(select(.==false)) | length' "$ENHANCED")
if [[ "$MISSING_ID" != "0" ]]; then
  die "Validation failed: some features lack properties.orch_id"
fi

# Handle orch_id coercion if needed
if $COERCE_ID; then
  echo "    Coercing orch_id to string..."
  SANITIZED="${ENHANCED%.geojson}.sanitized.geojson"
  jq '.features = (.features | map(
      if (.properties.orch_id|type != "string")
      then .properties.orch_id = (.properties.orch_id|tostring)
      else . end
    ))' "$ENHANCED" > "$SANITIZED"
  ENHANCED="$SANITIZED"
fi

# Build tippecanoe args with multiple includes
TIPPE_ARGS=( 
  -o "$MBTILES_OUT" 
  -l "$LAYER_NAME" 
  --include=orch_id 
  --include=county
  --include=dominant_crop
  --include=current_crop
  --include=stability_score
  --include=acres
  --attribute-type=orch_id:string 
  --attribute-type=county:string
  --attribute-type=dominant_crop:string
  --attribute-type=current_crop:string
  --attribute-type=stability_score:float
  --attribute-type=acres:float
  --coalesce-densest-as-needed 
  --extend-zooms-if-still-dropping 
  --force 
)

if [[ -n "$MIN_ZOOM" ]]; then TIPPE_ARGS+=( -Z "$MIN_ZOOM" ); fi
if [[ -n "$MAX_ZOOM" ]]; then TIPPE_ARGS+=( -z "$MAX_ZOOM" ); fi
if [[ -z "$MIN_ZOOM" && -z "$MAX_ZOOM" ]]; then TIPPE_ARGS+=( -zg ); fi

echo "==> Running tippecanoe → MBTiles"
set -x
tippecanoe "${TIPPE_ARGS[@]}" "$ENHANCED"
set +x
[[ -f "$MBTILES_OUT" ]] || { echo "tippecanoe did not produce $MBTILES_OUT"; exit 3; }

echo "==> Converting MBTiles → PMTiles"  
set -x
pmtiles convert "$MBTILES_OUT" "$PMTILES_OUT"
set +x
[[ -f "$PMTILES_OUT" ]] || { echo "pmtiles did not produce $PMTILES_OUT"; exit 3; }

echo "==> PMTiles info"
pmtiles show "$PMTILES_OUT" | head -20 || true

echo ""
echo "✅ Done processing $COUNTY county."
echo "   - Enhanced  : $ENHANCED"
echo "   - MBTiles   : $MBTILES_OUT"  
echo "   - PMTiles   : $PMTILES_OUT"
echo ""
echo "Properties included: orch_id, county, dominant_crop, current_crop, stability_score, acres"