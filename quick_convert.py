#!/usr/bin/env python3
"""
Quick converter for metadata to NeMo manifest format
Usage: python quick_convert.py metadata.txt audio_folder
"""

import os
import json
import librosa
import sys
import random
from pathlib import Path


def convert_metadata(metadata_file, audio_dir):
    """Convert metadata to NeMo manifest format and split train/val"""
    print(f"🔄 Converting {metadata_file} with audio from {audio_dir}")
    
    # Read metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header if exists
    if lines[0].strip().lower().startswith('path'):
        lines = lines[1:]
    
    manifest_entries = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Parse: path|text
        parts = line.split('|', 1)
        if len(parts) != 2:
            print(f"⚠️ Skipping invalid line {i+1}: {line}")
            continue
            
        audio_filename, text = parts
        audio_path = os.path.join(audio_dir, audio_filename.strip())
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"⚠️ Audio not found: {audio_path}")
            continue
        
        # Get duration
        try:
            duration = librosa.get_duration(path=audio_path)
        except:
            print(f"⚠️ Cannot get duration: {audio_path}")
            continue
        
        # Create entry
        entry = {
            "audio_filepath": os.path.abspath(audio_path),
            "text": text.strip(),
            "duration": round(duration, 2),
            "offset": 0.0
        }
        
        manifest_entries.append(entry)
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1} files...")
    
    print(f"✅ Total valid entries: {len(manifest_entries)}")
    
    # Shuffle and split 95:5
    random.seed(42)
    random.shuffle(manifest_entries)
    
    train_size = int(len(manifest_entries) * 0.95)
    train_entries = manifest_entries[:train_size]
    val_entries = manifest_entries[train_size:]
    
    print(f"📊 Train: {len(train_entries)}, Val: {len(val_entries)}")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save train manifest
    with open("data/train_manifest.jsonl", 'w', encoding='utf-8') as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Save val manifest
    with open("data/val_manifest.jsonl", 'w', encoding='utf-8') as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Statistics
    total_duration = sum(e['duration'] for e in manifest_entries)
    avg_duration = total_duration / len(manifest_entries)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(manifest_entries)}")
    print(f"   Total duration: {total_duration/3600:.2f} hours")
    print(f"   Average duration: {avg_duration:.2f} seconds")
    print(f"   Train samples: {len(train_entries)}")
    print(f"   Val samples: {len(val_entries)}")
    
    print(f"\n✅ Manifests saved:")
    print(f"   📁 data/train_manifest.jsonl")
    print(f"   📁 data/val_manifest.jsonl")
    
    return len(manifest_entries)


def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_convert.py metadata.txt audio_folder")
        print("Example: python quick_convert.py metadata.txt ./audio")
        sys.exit(1)
    
    metadata_file = sys.argv[1]
    audio_dir = sys.argv[2]
    
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        sys.exit(1)
    
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    print("🎯 Quick Metadata Converter")
    print("=" * 40)
    
    total_samples = convert_metadata(metadata_file, audio_dir)
    
    if total_samples > 0:
        print("\n🎉 Conversion completed!")
        print("📋 Next steps:")
        print("   1. Check data/train_manifest.jsonl and data/val_manifest.jsonl")
        print("   2. Run training: python train.py")
    else:
        print("\n❌ No valid samples found!")


if __name__ == "__main__":
    main()
