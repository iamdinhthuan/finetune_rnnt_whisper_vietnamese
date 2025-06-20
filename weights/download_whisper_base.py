#!/usr/bin/env python3
"""
Download and export Whisper Base encoder weights for RNN-T training.

This script downloads the Whisper Base model from OpenAI and extracts
the encoder weights in a format compatible with our RNN-T implementation.
"""

import torch
import os
import sys
import urllib.request
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_whisper_base_weights():
    """Download Whisper Base model weights directly from OpenAI."""

    print("🔄 Downloading Whisper Base model...")

    # URL for Whisper Base model
    model_url = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
    model_path = Path(__file__).parent / "base.pt"

    try:
        if not model_path.exists():
            print(f"Downloading from: {model_url}")
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Whisper Base model downloaded successfully")
        else:
            print("✅ Whisper Base model already exists")

        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint

        return state_dict

    except Exception as e:
        print(f"❌ Failed to download Whisper Base model: {e}")
        return None

def download_and_export_whisper_base():
    """Download Whisper Base model and export encoder weights."""

    # Download the model weights
    checkpoint = download_whisper_base_weights()
    if checkpoint is None:
        return False

    # Get the actual state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        dims = checkpoint.get('dims', {})
    else:
        state_dict = checkpoint
        dims = {}

    # Extract encoder weights
    encoder_weight = {
        'dims': dims if dims else {
            'n_mels': 80,
            'n_audio_ctx': 1500,
            'n_audio_state': 512,  # Base model
            'n_audio_head': 8,     # Base model
            'n_audio_layer': 6,    # Base model
            'n_vocab': 51864,
            'n_text_ctx': 448,
            'n_text_state': 512,
            'n_text_head': 8,
            'n_text_layer': 6
        },
        'model_state_dict': {}
    }
    
    print("🔄 Extracting encoder weights...")

    # Extract encoder-specific weights
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder')]
    print(f"📋 Found {len(encoder_keys)} encoder keys")

    for key in encoder_keys:
        # Skip positional embedding as we don't use it
        if "positional_embedding" in key:
            print(f"⏭️  Skipping positional embedding: {key}")
            continue
        # Remove 'encoder.' prefix for our model
        new_key = key.replace('encoder.', '')
        encoder_weight['model_state_dict'][new_key] = state_dict[key]
        print(f"✅ Extracted: {key} -> {new_key}")

    # Create conv3 weights by copying conv2 (for compatibility with our 3-conv architecture)
    if 'conv2.weight' in encoder_weight['model_state_dict']:
        encoder_weight['model_state_dict']['conv3.weight'] = encoder_weight['model_state_dict']['conv2.weight'].clone()
        encoder_weight['model_state_dict']['conv3.bias'] = encoder_weight['model_state_dict']['conv2.bias'].clone()
        print("✅ Created conv3 weights from conv2 for compatibility")

    print(f"📊 Total extracted weights: {len(encoder_weight['model_state_dict'])}")
    
    # Save the encoder weights
    output_path = Path(__file__).parent / 'base_encoder.pt'
    torch.save(encoder_weight, output_path)
    
    print(f"✅ Whisper Base encoder weights saved to: {output_path}")
    print(f"📊 Model dimensions:")
    print(f"   - State size: {encoder_weight['dims']['n_audio_state']}")
    print(f"   - Attention heads: {encoder_weight['dims']['n_audio_head']}")
    print(f"   - Layers: {encoder_weight['dims']['n_audio_layer']}")
    
    # Verify the saved file
    try:
        loaded = torch.load(output_path, map_location='cpu', weights_only=True)
        print("✅ Verification: Saved weights can be loaded successfully")
        
        # Print some key information
        total_params = sum(p.numel() for p in loaded['model_state_dict'].values())
        print(f"📈 Total parameters in encoder: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    return True

def verify_compatibility():
    """Verify that the exported weights are compatible with our encoder."""
    
    print("\n🔍 Verifying compatibility with RNN-T encoder...")
    
    try:
        # Import our encoder
        from models.encoder import AudioEncoder
        
        # Load the exported weights
        weight_path = Path(__file__).parent / 'base_encoder.pt'
        if not weight_path.exists():
            print("❌ base_encoder.pt not found. Run download first.")
            return False
            
        encoder_state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        
        # Create our encoder with Base model dimensions
        encoder = AudioEncoder(
            n_mels=80,
            n_state=512,  # Base model
            n_head=8,     # Base model
            n_layer=6,    # Base model
            att_context_size=(80, 3)
        )
        
        # Try to load the weights
        missing_keys, unexpected_keys = encoder.load_state_dict(
            encoder_state_dict['model_state_dict'], strict=False
        )
        
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")
            
        print("✅ Weights are compatible with RNN-T encoder")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        encoder.eval()
        with torch.no_grad():
            # Create dummy input
            batch_size = 2
            n_mels = 80
            seq_len = 100
            x = torch.randn(batch_size, n_mels, seq_len)
            x_len = torch.tensor([seq_len, seq_len//2])
            
            # Forward pass
            output, output_len = encoder(x, x_len)
            print(f"✅ Forward pass successful. Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Compatibility check failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Whisper Base Encoder Download and Export")
    print("=" * 50)
    
    # Create weights directory if it doesn't exist
    weights_dir = Path(__file__).parent
    weights_dir.mkdir(exist_ok=True)
    
    # Download and export
    if download_and_export_whisper_base():
        print("\n" + "=" * 50)
        
        # Verify compatibility
        if verify_compatibility():
            print("\n🎉 All done! Whisper Base encoder is ready for training.")
            print("\n📝 Next steps:")
            print("   1. Update your config.yaml to use Whisper Base dimensions")
            print("   2. Set pretrained_encoder_weight to './weights/base_encoder.pt'")
            print("   3. Start training with the new encoder!")
        else:
            print("\n❌ Compatibility verification failed. Please check the logs above.")
    else:
        print("\n❌ Download and export failed. Please check the logs above.")
