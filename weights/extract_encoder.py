import torch

# Load the model
full_weight = torch.load('/kaggle/working/PhoWhisper-small/pytorch_model.bin')

# Initialize the encoder weight dictionary
encoder_weight = {'model_state_dict': {}}

# Extract encoder weights
encoder_params_count = 0
for key in full_weight.keys():
    # Only process encoder weights (they all start with 'model.encoder')
    if key.startswith('model.encoder'):
        # Skip positional embedding if needed, but it looks like we need to keep it
        # (no need to filter it out in this model architecture)

        # Remove the 'model.encoder.' prefix to keep only the relative path
        new_key = key.replace('model.encoder.', '')
        encoder_weight['model_state_dict'][new_key] = full_weight[key]
        encoder_params_count += 1

# Save the encoder weights
torch.save(encoder_weight, './phowhisper_small_encoder.pt')
print(f"Encoder weights saved to phowhisper_small_encoder.pt")
print(f"Extracted {encoder_params_count} encoder parameters")

# Print a sample of the keys to verify
print("\nSample of extracted keys:")
sample_keys = list(encoder_weight['model_state_dict'].keys())[:5]
for key in sample_keys:
    print(f"- {key}")