#!/usr/bin/env python3
"""
Merge Script for Cover Crop Explorer Web App

Combines county-crop specific data folders into unified webapp structure.
Auto-discovers county/crop folders and merges JSON files by year.

Usage:
    python merge_for_webapp.py --input_root=data/ --output=webapp/data/
    python merge_for_webapp.py --input_root=data/ --output=webapp/data/ --incremental

Example folder structure:
    data/
    ├── fresno_almonds/
    │   ├── 2018.json
    │   ├── 2019.json
    │   ├── streaks.json
    │   └── orchards.pmtiles
    ├── kern_almonds/
    │   ├── 2018.json
    │   ├── 2019.json
    │   └── streaks.json
    └── tulare_walnuts/
        ├── 2018.json
        └── 2019.json
"""

import os
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import shutil

def discover_county_crop_folders(input_root):
    """Auto-discover county-crop folders and their contents."""
    folders = {}
    
    for folder_path in glob.glob(os.path.join(input_root, "*/")):
        folder_name = os.path.basename(folder_path.rstrip('/'))
        
        # Skip non-data folders
        if folder_name.startswith('.') or folder_name in ['webapp', 'output', 'temp']:
            continue
            
        # Discover JSON files in this folder
        json_files = {}
        pmtiles_files = []
        
        # Look for JSON files in root folder
        for file_path in glob.glob(os.path.join(folder_path, "*.json")):
            filename = os.path.basename(file_path)
            json_files[filename] = file_path
            
        # Look for JSON files in attrs subdirectory
        attrs_path = os.path.join(folder_path, "attrs")
        if os.path.exists(attrs_path):
            for file_path in glob.glob(os.path.join(attrs_path, "*.json")):
                filename = os.path.basename(file_path)
                json_files[filename] = file_path
            
        for file_path in glob.glob(os.path.join(folder_path, "*.pmtiles")):
            pmtiles_files.append(file_path)
        
        if json_files or pmtiles_files:
            folders[folder_name] = {
                'path': folder_path,
                'json_files': json_files,
                'pmtiles_files': pmtiles_files
            }
    
    return folders

def load_existing_merge_log(output_dir):
    """Load existing merge log to enable incremental merging."""
    log_path = os.path.join(output_dir, 'merge_log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return {'merged_folders': {}, 'last_merge': None, 'file_hashes': {}}

def calculate_file_hash(file_path):
    """Calculate simple hash for file change detection."""
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def merge_json_files(folders, output_dir, incremental=False):
    """Merge JSON files by filename across all county-crop folders."""
    
    # Load existing merge log for incremental mode
    merge_log = load_existing_merge_log(output_dir) if incremental else {'merged_folders': {}, 'file_hashes': {}}
    
    # Group files by filename
    files_by_name = defaultdict(list)
    
    print(f"🔍 Discovering files across {len(folders)} county-crop folders...")
    
    for folder_name, folder_info in folders.items():
        for filename, file_path in folder_info['json_files'].items():
            
            # Check if file changed (for incremental mode)
            file_hash = calculate_file_hash(file_path)
            file_key = f"{folder_name}/{filename}"
            
            if incremental and file_key in merge_log['file_hashes']:
                if merge_log['file_hashes'][file_key] == file_hash:
                    print(f"   ⏭️  Skipping {file_key} (unchanged)")
                    continue
            
            files_by_name[filename].append({
                'folder': folder_name,
                'path': file_path,
                'hash': file_hash
            })
            
            merge_log['file_hashes'][file_key] = file_hash
    
    print(f"\n📊 Found files to merge:")
    for filename, file_list in files_by_name.items():
        print(f"   {filename}: {len(file_list)} folders")
    
    # Merge each filename group
    merged_stats = {}
    
    for filename, file_list in files_by_name.items():
        print(f"\n🔄 Merging {filename}...")
        
        merged_data = {}
        source_folders = []
        
        for file_info in file_list:
            print(f"   📂 Reading {file_info['folder']}")
            
            with open(file_info['path'], 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if filename == 'manifest.json':
                # For manifest, merge arrays and take latest config
                if not merged_data:
                    merged_data = data.copy()
                else:
                    # Merge years arrays
                    if 'years' in data and 'years' in merged_data:
                        merged_data['years'] = sorted(list(set(merged_data['years'] + data['years'])))
            else:
                # For year JSONs and streaks, merge object keys
                if isinstance(data, dict):
                    merged_data.update(data)
                else:
                    print(f"   ⚠️  Warning: {filename} in {file_info['folder']} is not an object")
            
            source_folders.append(file_info['folder'])
        
        # Write merged file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, separators=(',', ':'))
        
        merged_stats[filename] = {
            'merged_count': len(merged_data) if isinstance(merged_data, dict) else 1,
            'source_folders': source_folders,
            'output_size_kb': round(os.path.getsize(output_path) / 1024, 1)
        }
        
        print(f"   ✅ Wrote {output_path} ({merged_stats[filename]['output_size_kb']} KB)")
    
    return merged_stats, merge_log

def merge_pmtiles_files(folders, output_dir):
    """Combine PMTiles files (placeholder - requires tippecanoe)."""
    pmtiles_files = []
    
    for folder_name, folder_info in folders.items():
        pmtiles_files.extend(folder_info['pmtiles_files'])
    
    if pmtiles_files:
        print(f"\n🗺️  Found {len(pmtiles_files)} PMTiles files:")
        for pmtiles_file in pmtiles_files:
            print(f"   📂 {pmtiles_file}")
        
        if len(pmtiles_files) == 1:
            # Single file - just copy
            output_path = os.path.join(output_dir, 'california-orchards.pmtiles')
            shutil.copy2(pmtiles_files[0], output_path)
            print(f"   ✅ Copied to {output_path}")
            return True
        else:
            # Multiple files - need tippecanoe merge
            print(f"   ⚠️  Multiple PMTiles files require tippecanoe merge:")
            print(f"   💡 Run: tippecanoe-tile-join -o {output_dir}/california-orchards.pmtiles {' '.join(pmtiles_files)}")
            return False
    
    return True

def generate_merge_report(folders, merged_stats, merge_log, output_dir):
    """Generate comprehensive merge report."""
    
    report = {
        'merge_timestamp': datetime.now().isoformat(),
        'total_folders_processed': len(folders),
        'folders_discovered': list(folders.keys()),
        'files_merged': merged_stats,
        'summary': {
            'total_orchards': 0,
            'counties_crops_included': list(folders.keys()),
            'years_available': [],
            'total_webapp_size_kb': 0
        }
    }
    
    # Calculate totals
    for filename, stats in merged_stats.items():
        report['summary']['total_webapp_size_kb'] += stats['output_size_kb']
        
        if filename.endswith('.json') and filename != 'manifest.json' and filename != 'streaks.json':
            # Assume year files contain orchard data
            report['summary']['total_orchards'] = max(
                report['summary']['total_orchards'], 
                stats['merged_count']
            )
        
        if filename.replace('.json', '').isdigit():
            report['summary']['years_available'].append(int(filename.replace('.json', '')))
    
    report['summary']['years_available'] = sorted(report['summary']['years_available'])
    
    # Update merge log
    merge_log['last_merge'] = report['merge_timestamp']
    merge_log['merged_folders'] = {name: True for name in folders.keys()}
    
    # Write reports
    report_path = os.path.join(output_dir, 'merge_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    log_path = os.path.join(output_dir, 'merge_log.json')
    with open(log_path, 'w') as f:
        json.dump(merge_log, f, indent=2)
    
    # Print summary
    print(f"\n📋 MERGE SUMMARY")
    print(f"   🗂️  Processed: {len(folders)} county-crop folders")
    print(f"   🏛️  Counties/Crops: {', '.join(folders.keys())}")
    print(f"   📅 Years: {report['summary']['years_available']}")
    print(f"   🌳 Total Orchards: {report['summary']['total_orchards']:,}")
    print(f"   💾 Total Size: {report['summary']['total_webapp_size_kb']:.1f} KB")
    print(f"   📊 Report: {report_path}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Merge county-crop data for webapp deployment')
    parser.add_argument('--input_root', required=True, help='Root directory containing county-crop folders')
    parser.add_argument('--output', required=True, help='Output directory for webapp data')
    parser.add_argument('--incremental', action='store_true', help='Only merge changed files')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_root):
        print(f"❌ Input root directory not found: {args.input_root}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"🚀 Starting webapp merge...")
    print(f"   📁 Input: {args.input_root}")
    print(f"   📁 Output: {args.output}")
    print(f"   🔄 Mode: {'Incremental' if args.incremental else 'Full'}")
    
    # Discover county-crop folders
    folders = discover_county_crop_folders(args.input_root)
    
    if not folders:
        print(f"❌ No county-crop folders found in {args.input_root}")
        return 1
    
    # Merge JSON files
    merged_stats, merge_log = merge_json_files(folders, args.output, args.incremental)
    
    # Handle PMTiles files
    pmtiles_success = merge_pmtiles_files(folders, args.output)
    
    # Generate report
    report = generate_merge_report(folders, merged_stats, merge_log, args.output)
    
    if not pmtiles_success:
        print(f"\n⚠️  PMTiles merge incomplete - see instructions above")
        return 1
    
    print(f"\n✅ Merge completed successfully!")
    print(f"   🌐 Deploy webapp from: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())