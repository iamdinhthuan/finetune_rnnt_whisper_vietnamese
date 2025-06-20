from audiomentations import (
    AddBackgroundNoise, AddGaussianNoise, Compose, OneOf, SomeOf,
    Gain, PitchShift, TimeStretch, Mp3Compression, Shift, PolarityInversion
)

from torch.utils.data import Dataset
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import numpy as np
import json
import librosa
import torch.nn.functional as F
import sentencepiece as spm
import os

from config import get_config

class AudioDataset(Dataset):
    def __init__(self,
                manifest_files, tokenizer_model_path, bg_noise_path=[], shuffle=False, augment=False,
                max_duration=15.1, min_duration=0.9, min_text_len=3, max_text_len=99999,
                config=None
                ):
        # Load config if not provided
        if config is None:
            config = get_config()
        self.config = config
        self.samples = []
        throw_away = 0
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                for line in tqdm(f, desc=f'Loading {manifest_file}'):
                    sample = json.loads(line)
                    if  sample['duration'] > max_duration or sample['duration'] < min_duration or \
                        len(sample['text'].strip()) < min_text_len or len(sample['text'].strip()) > max_text_len:
                        throw_away += 1
                        continue
                    self.samples.append(sample)
        logger.info(f"Throw away {throw_away} samples")

        if shuffle:
            np.random.shuffle(self.samples)

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        self.device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if augment:
            self.augmentation = Compose(  # chỉ 80 % sample được augment
                [
                    Gain(min_gain_db=-10, max_gain_db=6, p=0.8),

                    PitchShift(min_semitones=-4, max_semitones=4, p=0.3),

                    TimeStretch(min_rate=0.9, max_rate=1.1,
                                leave_length_unchanged=False, p=0.2),

                    Mp3Compression(p=0.3),

                    Shift(
                        min_shift=-0.2,
                        max_shift=0.2,
                        shift_unit="fraction",  # mặc định đã là "fraction", ghi rõ cho tường minh
                        rollover=False,  # giữ im lặng thay vì quấn đuôi audio (tùy ý)
                        p=0.5,
                    ),

                    # ---------- Noise ---------------------------------------
                    OneOf(
                        [
                            AddGaussianNoise(min_amplitude=0.001,
                                             max_amplitude=0.010,
                                             p=1.0),  # Gaussian luôn chạy khi nhánh này được chọn

                            AddBackgroundNoise(
                                sounds_path=[p for p in bg_noise_path if os.path.exists(p)],
                                min_snr_db=8.0,  # SNR 8–20 dB thực tế hơn
                                max_snr_db=20.0,
                                noise_transform=PolarityInversion(),
                                p=0.3  # hiếm hơn Gaussian
                            ),
                        ],
                        p=0.5  # chỉ 50 % sample có thêm noise
                    ),
                    # ---------------------------------------------------------
                ],
                p=0.8  # 20 % sample giữ nguyên
            )
        else:
            self.augmentation = lambda samples, sample_rate: samples
    
    def __len__(self):
        return len(self.samples)
    
    def mel_filters(self, device, n_mels: int) -> torch.Tensor:
        """
        load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
        Allows decoupling librosa dependency; saved using:

            np.savez_compressed(
                "mel_filters.npz",
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
            )
        """
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

        with np.load("./utils/mel_filters.npz", allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
    
    def log_mel_spectrogram(
        self, audio, n_mels, padding, device
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 and 128 are supported

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (n_mels, n_frames)
            A Tensor that contains the Mel spectrogram
        """

        if device is not None:
            audio = audio.to(device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(self.config.audio.n_fft).to(audio.device)
        stft = torch.stft(audio, self.config.audio.n_fft, self.config.audio.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sample_rate = librosa.load(
            sample['audio_filepath'],
            sr=self.config.audio.sample_rate,
            offset=sample['offset'],
            duration=sample['duration']
        )
        waveform = self.augmentation(samples=waveform, sample_rate=sample_rate)
        transcript_ids = self.tokenizer.encode_as_ids(sample['text'])

        waveform, transcript_ids = torch.from_numpy(waveform), torch.tensor(transcript_ids)
        melspec = self.log_mel_spectrogram(waveform, self.config.audio.n_mels, 0, self.device)

        return melspec, transcript_ids
    
def collate_fn(batch):
    mel, text_ids = zip(*batch)
    max_len = max(x.shape[-1] for x in mel)
    # mel have shape [n_mels, T]
    mel_input_lengths = torch.tensor([x.shape[-1] for x in mel])
    text_input_lengths = torch.tensor([len(x) for x in text_ids])
    mel_padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in mel]
    # text_ids have shape [T]
    config = get_config()
    text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=config.tokenizer.pad_token)
    return torch.stack(mel_padded), mel_input_lengths.int(), text_ids_padded, text_input_lengths.int()
