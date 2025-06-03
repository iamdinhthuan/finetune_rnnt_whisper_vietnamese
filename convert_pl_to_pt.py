import os
import torch
import argparse

def convert_pl_checkpoint_to_pt(pl_checkpoint_path, output_path):
    """
    Convert a PyTorch Lightning checkpoint to a standard PyTorch checkpoint.

    Args:
        pl_checkpoint_path: Path to the PyTorch Lightning checkpoint file
        output_path: Path to save the converted PyTorch checkpoint
    """
    # Load the PyTorch Lightning checkpoint
    checkpoint = torch.load(pl_checkpoint_path, map_location='cpu')

    # Create a new PyTorch checkpoint dictionary
    new_checkpoint = {
        'epoch': checkpoint.get('epoch', 0),
        'model_state_dict': checkpoint.get('state_dict', {}),
        'optimizer_state_dict': None,  # Will be initialized when training
        'scheduler_state_dict': None,  # Will be initialized when training
        'loss': checkpoint.get('val_loss', 0.0),
        'wer': checkpoint.get('val_wer', 0.0)
    }

    # Fix state dict keys by removing 'model.' prefix if it exists
    fixed_state_dict = {}
    for key, value in new_checkpoint['model_state_dict'].items():
        if key.startswith('model.'):
            fixed_state_dict[key[6:]] = value  # Remove 'model.' prefix
        else:
            fixed_state_dict[key] = value

    new_checkpoint['model_state_dict'] = fixed_state_dict

    # Save the converted checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_checkpoint, output_path)
    print(f"Converted checkpoint saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch Lightning checkpoint to standard PyTorch format")
    parser.add_argument("--input", type=str, required=True, help="Path to the PyTorch Lightning checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save the converted checkpoint")

    args = parser.parse_args()

    convert_pl_checkpoint_to_pt(args.input, args.output)


if __name__ == "__main__":
    main()