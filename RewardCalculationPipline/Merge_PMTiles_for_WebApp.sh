#!/bin/bash

# PMTiles Merger Script
# Merges multiple county PMTiles files into a single unified tileset
# Uses tippecanoe's tile-join command

set -e  # Exit on any error

# Configuration
OUTPUT_DIR="./merged_tiles"
OUTPUT_FILE="$OUTPUT_DIR/all_counties.pmtiles"
INPUT_PATTERN="*_county.pmtiles"  # Adjust this pattern to match your county files

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PMTiles Merger Script ===${NC}"

# Function to check if tippecanoe is installed
check_tippecanoe() {
    if ! command -v tile-join &> /dev/null; then
        echo -e "${RED}Error: tile-join (tippecanoe) is not installed or not in PATH${NC}"
        echo "Please install tippecanoe first"
        exit 1
    fi
    
    echo -e "${GREEN}✓ tile-join found${NC}"
}

# Function to create output directory
create_output_dir() {
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Created output directory: $OUTPUT_DIR${NC}"
    else
        echo -e "${YELLOW}Output directory already exists: $OUTPUT_DIR${NC}"
    fi
}

# Function to find input PMTiles files
find_input_files() {
    local files=()
    
    # Method 1: Look for files matching the pattern
    if ls $INPUT_PATTERN 2>/dev/null; then
        files=($(ls $INPUT_PATTERN))
        echo -e "${GREEN}Found PMTiles files matching pattern '$INPUT_PATTERN':${NC}"
        for file in "${files[@]}"; do
            echo "  - $file"
        done
    else
        echo -e "${YELLOW}No files found matching pattern '$INPUT_PATTERN'${NC}"
        
        # Method 2: Look for any PMTiles files
        local all_pmtiles=($(find . -maxdepth 1 -name "*.pmtiles" -type f))
        if [ ${#all_pmtiles[@]} -gt 0 ]; then
            echo -e "${YELLOW}Available PMTiles files:${NC}"
            for file in "${all_pmtiles[@]}"; do
                echo "  - $file"
            done
            
            # Ask user which files to merge
            echo -e "${BLUE}Enter the PMTiles files to merge (space-separated):${NC}"
            read -r -a files
        else
            echo -e "${RED}Error: No PMTiles files found in current directory${NC}"
            exit 1
        fi
    fi
    
    # Validate that files exist
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}Error: File not found: $file${NC}"
            exit 1
        fi
    done
    
    echo "${files[@]}"
}

# Function to merge PMTiles files
merge_pmtiles() {
    local input_files=("$@")
    
    if [ ${#input_files[@]} -lt 2 ]; then
        echo -e "${RED}Error: Need at least 2 files to merge${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Merging ${#input_files[@]} PMTiles files...${NC}"
    
    # Remove existing output file if it exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${YELLOW}Removing existing output file: $OUTPUT_FILE${NC}"
        rm "$OUTPUT_FILE"
    fi
    
    # Build tile-join command
    local cmd="tile-join"
    cmd="$cmd -f"          # Force overwrite if exists
    cmd="$cmd -pk"         # No tile size limit (prevents losing tiles)
    cmd="$cmd -o $OUTPUT_FILE"
    
    # Add all input files
    for file in "${input_files[@]}"; do
        cmd="$cmd $file"
    done
    
    echo -e "${BLUE}Running command:${NC}"
    echo "$cmd"
    echo
    
    # Execute the merge
    eval "$cmd"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully merged PMTiles files${NC}"
        echo -e "${GREEN}Output: $OUTPUT_FILE${NC}"
        
        # Show file size
        local size=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo -e "${GREEN}File size: $size${NC}"
    else
        echo -e "${RED}✗ Failed to merge PMTiles files${NC}"
        exit 1
    fi
}

# Function to inspect the merged file
inspect_merged_file() {
    if [ -f "$OUTPUT_FILE" ]; then
        echo -e "${BLUE}Inspecting merged PMTiles file...${NC}"
        pmtiles show "$OUTPUT_FILE"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}Starting PMTiles merge process...${NC}"
    
    # Check prerequisites
    check_tippecanoe
    
    # Create output directory
    create_output_dir
    
    # Find input files
    local input_files
    input_files=($(find_input_files))
    
    # Merge files
    merge_pmtiles "${input_files[@]}"
    
    # Inspect result
    inspect_merged_file
    
    echo -e "${GREEN}=== Merge Complete ===${NC}"
    echo -e "${GREEN}Merged PMTiles file: $OUTPUT_FILE${NC}"
}

# Handle command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "PMTiles Merger Script"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  --pattern     Set input file pattern (default: *_county.pmtiles)"
    echo "  --output      Set output file path (default: ./merged_tiles/all_counties.pmtiles)"
    echo
    echo "Examples:"
    echo "  $0                                    # Merge files matching default pattern"
    echo "  $0 --pattern '*.pmtiles'             # Merge all PMTiles files"
    echo "  $0 --output ./combined.pmtiles       # Set custom output file"
    echo
    echo "The script uses tippecanoe's tile-join command with the following options:"
    echo "  -f   Force overwrite existing output file"
    echo "  -pk  No tile size limit (prevents losing tiles)"
    exit 0
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pattern)
            INPUT_PATTERN="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main