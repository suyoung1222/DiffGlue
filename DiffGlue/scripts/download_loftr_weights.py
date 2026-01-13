#!/usr/bin/env python3
"""
Download pretrained LoFTR weights for DiffGlue.

Usage:
    python scripts/download_loftr_weights.py

This downloads the outdoor_ds.ckpt weights from the official LoFTR repository.
The weights are used to initialize DiffGlue's backbone and coarse matching components.
"""

import os
import sys
from pathlib import Path

def download_loftr_weights(output_dir: str = None):
    """
    Download LoFTR pretrained weights from Google Drive.
    
    Args:
        output_dir: Directory to save weights. Defaults to scripts/models/matchers/LoFTR/weights/
    """
    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "models" / "matchers" / "LoFTR" / "weights"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LoFTR weight files and their Google Drive file IDs
    # From: https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf
    weights = {
        "outdoor_ds.ckpt": "1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY",
        "indoor_ds.ckpt": "1w1Qhea3WLRMS81Vod_k5rxS_GNRgIi-O",
    }
    
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown
    
    print("=" * 60)
    print("Downloading LoFTR pretrained weights")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    downloaded = []
    for filename, file_id in weights.items():
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            downloaded.append(filename)
            continue
        
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, str(output_path), quiet=False)
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ {filename} downloaded ({size_mb:.1f} MB)")
                downloaded.append(filename)
            else:
                print(f"✗ Failed to download {filename}")
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
    
    print()
    print("=" * 60)
    print(f"Downloaded {len(downloaded)}/{len(weights)} weight files")
    print("=" * 60)
    
    if "outdoor_ds.ckpt" in downloaded:
        print()
        print("To use LoFTR pretrained weights in DiffGlue, the config is already set up!")
        print("Default config now uses: model.matcher.loftr_pretrained=outdoor_ds.ckpt")
        print()
        print("The weights are located at:")
        print(f"  {output_dir / 'outdoor_ds.ckpt'}")
    
    return len(downloaded) == len(weights)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download LoFTR pretrained weights")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for weights (default: scripts/models/matchers/LoFTR/weights/)"
    )
    
    args = parser.parse_args()
    
    success = download_loftr_weights(args.output_dir)
    sys.exit(0 if success else 1)

