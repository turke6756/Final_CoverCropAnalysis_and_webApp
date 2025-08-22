#!/usr/bin/env python3
"""
Convert production-ready outputs from CSV to JSON format for web app deployment.

Based on requirements from Production/ConvertCSVtoJSON.md:
- Convert attrs/*.csv to attrs/*.json (object keyed by orch_id)
- Convert streaks.csv to streaks.json (object keyed by orch_id)  
- Copy manifest.json as-is
"""

import csv
import json
import os
import glob
import pathlib
import argparse

def convert_outputs(input_dir, output_dir):
    """Convert CSV outputs to JSON format for web app deployment."""
    
    # Set up paths
    root = pathlib.Path(input_dir)
    out = pathlib.Path(output_dir)
    
    # Create output directory structure
    (out / "attrs").mkdir(parents=True, exist_ok=True)
    
    print(f"Converting outputs from {root} to {out}")
    
    # 1) Convert attrs/*.csv -> attrs/*.json (object keyed by orch_id)
    print("\nConverting attrs CSV files...")
    attrs_files = sorted(glob.glob(str(root / "attrs" / "*.csv")))
    
    if not attrs_files:
        print("No attrs CSV files found!")
    else:
        for csv_path in attrs_files:
            year = pathlib.Path(csv_path).stem
            print(f"  Converting attrs/{year}.csv...")
            
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            
            # Convert to object keyed by orch_id
            out_obj = {}
            for r in rows:
                out_obj[r["orch_id"]] = {
                    "cluster_role_mapped": r["cluster_role_mapped"],
                    "best_soft_norm": float(r["best_soft_norm"]) if r["best_soft_norm"] else None,
                    "uncertain": r["uncertain"].lower() == "true",
                    "flipped_this_season": r["flipped_this_season"].lower() == "true",
                    "streak_len_current": int(float(r["streak_len_current"])) if r["streak_len_current"] else None,
                    "time_since_last_flip": int(float(r["time_since_last_flip"])) if r.get("time_since_last_flip") and r["time_since_last_flip"] else None,
                    "npts_total": int(float(r["npts_total"])) if r["npts_total"] else None,
                    "env_z_season": float(r["env_z_season"]) if r["env_z_season"] else None,
                    "stratum": r.get("stratum"),
                    "confidence_bucket": r.get("confidence_bucket"),
                }
            
            # Write JSON file
            json_path = out / "attrs" / f"{year}.json"
            with open(json_path, "w") as g:
                json.dump(out_obj, g, separators=(",", ":"))
            
            print(f"    ✓ Created {json_path} with {len(out_obj)} orchards")
    
    # 2) Convert streaks.csv -> streaks.json (object keyed by orch_id)
    print("\nConverting streaks.csv...")
    streaks_path = root / "streaks.csv"
    
    if not streaks_path.exists():
        print("  Warning: streaks.csv not found!")
    else:
        with open(streaks_path, newline="") as f:
            rows = list(csv.DictReader(f))
        
        # Convert to object keyed by orch_id
        streaks = {}
        for r in rows:
            streaks[r["orch_id"]] = {
                "mask": int(r["mask"]),
                "longest": int(r["longest"]),
                "years_active": int(r["years_active"]),
                "ever_cover": r["ever_cover"].lower() == "true",
                "never_cover": r["never_cover"].lower() == "true",
            }
        
        # Write JSON file
        streaks_json_path = out / "streaks.json"
        with open(streaks_json_path, "w") as g:
            json.dump(streaks, g, separators=(",", ":"))
        
        print(f"  ✓ Created {streaks_json_path} with {len(streaks)} orchards")
    
    # 3) Copy manifest.json as-is
    print("\nCopying manifest.json...")
    manifest_path = root / "manifest.json"
    
    if not manifest_path.exists():
        print("  Warning: manifest.json not found!")
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        manifest_out_path = out / "manifest.json"
        with open(manifest_out_path, "w") as g:
            json.dump(manifest, g, separators=(",", ":"))
        
        print(f"  ✓ Copied {manifest_out_path}")
    
    print(f"\n✅ Conversion complete! Output directory: {out}")
    print(f"\nGenerated files:")
    print(f"  manifest.json")
    print(f"  streaks.json")
    print(f"  attrs/")
    for json_file in sorted((out / "attrs").glob("*.json")):
        print(f"    {json_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV outputs to JSON format for web app deployment")
    parser.add_argument("input_dir", help="Input directory containing CSV files")
    parser.add_argument("output_dir", help="Output directory for JSON files")
    
    args = parser.parse_args()
    convert_outputs(args.input_dir, args.output_dir)