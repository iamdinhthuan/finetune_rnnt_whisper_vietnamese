import os

import torch
import torchaudio
from datasets import load_dataset
from datasets import load_dataset
from tqdm import tqdm

noise_datasets = {
    "fsdnoisy18k": "sps44/fsdnoisy18k-test",
}

save_root = "dataset/noise"
os.makedirs(save_root, exist_ok=True)

for name, hf_path in noise_datasets.items():
    print(f"Downloading {name} from {hf_path}...")
    dataset = load_dataset(hf_path, split="train")

    save_dir = os.path.join(save_root, name)
    os.makedirs(save_dir, exist_ok=True)

    for idx, sample in enumerate(tqdm(dataset, desc=f"Saving {name}")):
        audio_info = sample["audio"]
        waveform = torch.tensor(audio_info["array"]).unsqueeze(0)  # [1, T]
        sr = audio_info["sampling_rate"]

        filename = f"{idx:05d}.wav"
        out_path = os.path.join(save_dir, filename)

        # Chuẩn hóa về 16kHz nếu cần
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        torchaudio.save(out_path, waveform, sr)



