import os
import torch
import torch.nn.functional as F
import sentencepiece as spm
import librosa
import numpy as np
import threading
import queue
import time
from collections import deque
import argparse
import pyaudio
import webrtcvad
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer

from constants import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from constants import RNNT_BLANK, PAD, VOCAB_SIZE, TOKENIZER_MODEL_PATH, MAX_SYMBOLS
from constants import ATTENTION_CONTEXT_SIZE
from constants import N_STATE, N_LAYER, N_HEAD


class VoiceActivityDetector:
    """Voice Activity Detector sử dụng WebRTC VAD"""

    def __init__(self, aggressiveness=3, frame_duration_ms=30):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = SAMPLE_RATE
        self.frame_length = int(self.sample_rate * self.frame_duration_ms / 1000)

    def is_speech(self, audio_frame):
        """Kiểm tra frame có chứa giọng nói không"""
        if audio_frame.dtype != np.int16:
            audio_frame = (audio_frame * 32767).astype(np.int16)

        if len(audio_frame) != self.frame_length:
            return False

        return self.vad.is_speech(audio_frame.tobytes(), self.sample_rate)


class ContinuousAudioBuffer:
    """Buffer liên tục cho streaming audio"""

    def __init__(self, chunk_duration=0.5, overlap_duration=0.1):
        """
        Args:
            chunk_duration: Thời gian mỗi chunk để process (giây) - chỉ cho streaming mode
            overlap_duration: Thời gian overlap giữa các chunk (giây) - chỉ cho streaming mode
        """
        self.chunk_samples = int(chunk_duration * SAMPLE_RATE)
        self.overlap_samples = int(overlap_duration * SAMPLE_RATE)

        self.buffer = deque(maxlen=self.chunk_samples * 2)  # Buffer cho streaming mode

    def add_audio_streaming(self, audio_chunk):
        """Thêm audio và return chunk cho streaming mode"""
        # Thêm audio vào buffer
        for sample in audio_chunk:
            self.buffer.append(sample)

        # Kiểm tra xem đã đủ để tạo chunk mới chưa
        if len(self.buffer) >= self.chunk_samples:
            # Lấy chunk để process
            chunk = np.array(list(self.buffer)[:self.chunk_samples])

            # Shift buffer để chuẩn bị cho chunk tiếp theo (với overlap)
            shift_amount = self.chunk_samples - self.overlap_samples
            for _ in range(shift_amount):
                if len(self.buffer) > self.overlap_samples:
                    self.buffer.popleft()

            return chunk

        return None


class VADSegmentBuffer:
    """Buffer sử dụng VAD để cắt segment cho offline mode"""

    def __init__(self, max_duration=30.0, silence_duration=1.5, min_duration=0.5):
        """
        Args:
            max_duration: Thời gian tối đa của một segment (giây)
            silence_duration: Thời gian im lặng để kết thúc segment (giây)
            min_duration: Thời gian tối thiểu để coi là segment hợp lệ (giây)
        """
        self.max_samples = int(max_duration * SAMPLE_RATE)
        self.silence_samples = int(silence_duration * SAMPLE_RATE)
        self.min_samples = int(min_duration * SAMPLE_RATE)

        self.buffer = deque()
        self.silence_counter = 0
        self.has_speech = False

        self.vad = VoiceActivityDetector(aggressiveness=2)

    def add_audio(self, audio_chunk):
        """
        Thêm audio chunk và kiểm tra xem có segment hoàn chỉnh không

        Returns:
            tuple: (has_complete_segment, audio_segment)
        """
        # Thêm audio vào buffer
        for sample in audio_chunk:
            self.buffer.append(sample)

        # Kiểm tra buffer không quá dài
        while len(self.buffer) > self.max_samples:
            self.buffer.popleft()

        # Phân tích VAD cho chunk này
        chunk_has_speech = self._analyze_vad(audio_chunk)

        if chunk_has_speech:
            self.has_speech = True
            self.silence_counter = 0
        else:
            self.silence_counter += len(audio_chunk)

        # Kiểm tra điều kiện để trả về segment
        if self.has_speech and self.silence_counter >= self.silence_samples:
            # Đã có speech và im lặng đủ lâu
            if len(self.buffer) >= self.min_samples:
                # Segment đủ dài, trả về
                audio_segment = np.array(list(self.buffer))
                self._reset()
                return True, audio_segment

        # Kiểm tra buffer quá dài (force return)
        if len(self.buffer) >= self.max_samples and self.has_speech:
            audio_segment = np.array(list(self.buffer))
            self._reset()
            return True, audio_segment

        return False, None

    def _analyze_vad(self, audio_chunk):
        """Phân tích VAD cho audio chunk"""
        frame_size = self.vad.frame_length
        voice_frames = 0
        total_frames = 0

        for i in range(0, len(audio_chunk), frame_size):
            frame = audio_chunk[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                if self.vad.is_speech(frame):
                    voice_frames += 1

        # Coi là có speech nếu > 30% frames có voice
        return (voice_frames / max(total_frames, 1)) > 0.3 if total_frames > 0 else False

    def _reset(self):
        """Reset buffer sau khi trả về segment"""
        self.buffer.clear()
        self.silence_counter = 0
        self.has_speech = False

    def get_current_audio(self):
        """Lấy audio hiện tại trong buffer"""
        return np.array(list(self.buffer))


class StreamingMicrophone:
    """Microphone streaming liên tục"""

    def __init__(self, chunk_size=800):  # 50ms chunks
        self.chunk_size = chunk_size
        self.sample_rate = SAMPLE_RATE
        self.channels = 1
        self.format = pyaudio.paFloat32

        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False

        # VAD để detect voice activity
        self.vad = VoiceActivityDetector(aggressiveness=2)

    def start_stream(self):
        """Bắt đầu stream liên tục"""
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        self.stream.start_stream()

    def stop_stream(self):
        """Dừng stream"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback xử lý audio liên tục"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)

    def get_audio_chunk(self):
        """Lấy audio chunk từ queue"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def has_voice_activity(self, audio_chunk):
        """Kiểm tra có voice activity không"""
        frame_size = self.vad.frame_length
        voice_frames = 0
        total_frames = 0

        for i in range(0, len(audio_chunk), frame_size):
            frame = audio_chunk[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                if self.vad.is_speech(frame):
                    voice_frames += 1

        return voice_frames / max(total_frames, 1) > 0.3 if total_frames > 0 else False


class StreamingASR:
    """ASR với streaming mode và continuous recording"""

    def __init__(self, model_path=None, tokenizer_path=None, device=None,
                 streaming_mode=True, chunk_duration=0.5, append_mode=True,
                 silence_duration=1.5):
        """
        Args:
            streaming_mode: True để dùng online streaming, False để dùng offline mode với VAD
            chunk_duration: Thời gian mỗi chunk để process (giây) - chỉ cho streaming mode
            append_mode: True để append text, False để replace text
            silence_duration: Thời gian im lặng để kết thúc segment (giây) - chỉ cho offline mode
        """
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"🔧 Using device: {self.device}")

        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = TOKENIZER_MODEL_PATH
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

        # Load model
        print("📥 Loading ASR model...")
        self._load_model(model_path)
        self._load_mel_filters()

        # Setup streaming components
        self.streaming_mode = streaming_mode
        self.append_mode = append_mode
        self.microphone = StreamingMicrophone()

        if self.streaming_mode:
            # Streaming mode: sử dụng buffer theo thời gian
            self.audio_buffer = ContinuousAudioBuffer(chunk_duration=chunk_duration)
        else:
            # Offline mode: sử dụng VAD buffer
            self.vad_buffer = VADSegmentBuffer(
                max_duration=30.0,
                silence_duration=silence_duration,
                min_duration=0.5
            )

        # Initialize streaming states nếu dùng streaming mode
        if self.streaming_mode:
            self._init_streaming_states()

        self.is_running = False
        self.transcription_queue = queue.Queue()

    def _load_mel_filters(self):
        """Load mel filterbank matrix"""
        try:
            self.mel_filters_80 = np.load("./utils/mel_filters.npz", allow_pickle=False)["mel_80"]
            self.mel_filters_80 = torch.from_numpy(self.mel_filters_80)
        except FileNotFoundError:
            print("⚠️  Creating mel filters from librosa...")
            self.mel_filters_80 = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
            self.mel_filters_80 = torch.from_numpy(self.mel_filters_80)
            os.makedirs("./utils", exist_ok=True)
            np.savez_compressed("./utils/mel_filters.npz", mel_80=self.mel_filters_80.numpy())

    def _load_model(self, model_path):
        """Load model từ checkpoint"""
        if model_path is None:
            print("📥 Downloading model from huggingface hub...")
            model_path = hf_hub_download(
                repo_id="hkab/vietnamese-asr-model",
                filename="rnnt-latest.ckpt",
                subfolder="rnnt-whisper-small/80_3"
            )

        print(f"📂 Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        # Split weights
        encoder_weight = {}
        decoder_weight = {}
        joint_weight = {}

        for k, v in checkpoint['state_dict'].items():
            if 'alibi' in k:
                continue
            if 'encoder' in k:
                encoder_weight[k.replace('encoder.', '')] = v
            elif 'decoder' in k:
                decoder_weight[k.replace('decoder.', '')] = v
            elif 'joint' in k:
                joint_weight[k.replace('joint.', '')] = v

        # Initialize model components
        self.encoder = AudioEncoder(
            N_MELS,
            n_state=N_STATE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            att_context_size=ATTENTION_CONTEXT_SIZE
        )
        self.decoder = Decoder(vocab_size=VOCAB_SIZE + 1)
        self.joint = Jointer(vocab_size=VOCAB_SIZE + 1)

        # Load weights
        self.encoder.load_state_dict(encoder_weight, strict=False)
        self.decoder.load_state_dict(decoder_weight, strict=False)
        self.joint.load_state_dict(joint_weight, strict=False)

        # Move to device and eval mode
        self.encoder = self.encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()
        self.joint = self.joint.to(self.device).eval()

    def _init_streaming_states(self):
        """Initialize streaming states cho online mode"""
        self.audio_cache = torch.zeros(240, device=self.device)
        self.conv1_cache = torch.zeros(1, 80, 1, device=self.device)
        self.conv2_cache = torch.zeros(1, 768, 1, device=self.device)
        self.conv3_cache = torch.zeros(1, 768, 1, device=self.device)

        self.k_cache = torch.zeros(12, 1, ATTENTION_CONTEXT_SIZE[0], 768, device=self.device)
        self.v_cache = torch.zeros(12, 1, ATTENTION_CONTEXT_SIZE[0], 768, device=self.device)
        self.cache_len = torch.zeros(1, dtype=torch.int, device=self.device)

        self.hypothesis = [[None, None]]  # [label, state]
        self.seq_ids = []

    def log_mel_spectrogram(self, audio, n_mels=N_MELS, padding=0, streaming=False):
        """Compute log-Mel spectrogram"""
        audio = audio.to(self.device)

        if padding > 0:
            audio = F.pad(audio, (0, padding))

        window = torch.hann_window(N_FFT).to(audio.device)

        if not streaming:
            stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        else:
            stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, center=False, return_complex=True)

        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters_device = self.mel_filters_80.to(audio.device)
        mel_spec = mel_filters_device @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    @torch.no_grad()
    def _process_audio_chunk_offline(self, audio_chunk):
        """Process audio chunk using offline mode"""
        if len(audio_chunk) < 1600:
            return ""

        audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))

        # Compute mel spectrogram
        mels = self.log_mel_spectrogram(audio=audio_tensor, n_mels=N_MELS, padding=0, streaming=False)
        x = mels.reshape(1, *mels.shape)
        x_len = torch.tensor([x.shape[2]]).to(self.device)

        # Encoder forward
        enc_out, _ = self.encoder(x, x_len)

        # Greedy decoding
        hypothesis = [[None, None]]
        seq_enc_out = enc_out[0, :, :].unsqueeze(0)
        seq_ids = []

        for time_idx in range(seq_enc_out.shape[1]):
            current_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)
            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < MAX_SYMBOLS:
                if hypothesis[-1][0] is None:
                    last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=self.device)
                    last_seq_h_n = None
                else:
                    last_token = hypothesis[-1][0]
                    last_seq_h_n = hypothesis[-1][1]

                if last_seq_h_n is None:
                    current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                else:
                    current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)

                logits = self.joint(current_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]
                _, token_id = logits.max(0)
                token_id = token_id.detach().item()

                if token_id == RNNT_BLANK:
                    not_blank = False
                else:
                    symbols_added += 1
                    hypothesis.append([
                        torch.tensor([[token_id]], dtype=torch.long, device=self.device),
                        current_seq_h_n
                    ])
                    seq_ids.append(token_id)

        return self.tokenizer.decode(seq_ids)

    @torch.no_grad()
    def _process_audio_chunk_streaming(self, audio_chunk):
        """Process audio chunk using streaming mode"""
        chunk_size = HOP_LENGTH * 31 + N_FFT - (N_FFT - HOP_LENGTH)
        audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32)).to(self.device)

        # Prepare audio chunk with cache
        audio_chunk_with_cache = torch.cat([self.audio_cache, audio_tensor])

        if audio_chunk_with_cache.shape[0] < chunk_size + N_FFT - HOP_LENGTH:
            audio_chunk_with_cache = F.pad(
                audio_chunk_with_cache,
                (0, chunk_size + N_FFT - HOP_LENGTH - audio_chunk_with_cache.shape[0])
            )

        # Update audio cache
        self.audio_cache = audio_chunk_with_cache[-(N_FFT - HOP_LENGTH):]

        # Compute mel spectrogram
        x_chunk = self.log_mel_spectrogram(
            audio=audio_chunk_with_cache,
            n_mels=N_MELS,
            padding=0,
            streaming=True
        )
        x_chunk = x_chunk.reshape(1, *x_chunk.shape)

        if x_chunk.shape[-1] < 32:
            x_chunk = F.pad(x_chunk, (0, 32 - x_chunk.shape[-1]))

        # Encoder forward với caching
        x_chunk = torch.cat([self.conv1_cache, x_chunk], dim=2)
        self.conv1_cache = x_chunk[:, :, -1].unsqueeze(2)
        x_chunk = F.gelu(self.encoder.conv1(x_chunk))

        x_chunk = torch.cat([self.conv2_cache, x_chunk], dim=2)
        self.conv2_cache = x_chunk[:, :, -1].unsqueeze(2)
        x_chunk = F.gelu(self.encoder.conv2(x_chunk))

        x_chunk = torch.cat([self.conv3_cache, x_chunk], dim=2)
        self.conv3_cache = x_chunk[:, :, -1].unsqueeze(2)
        x_chunk = F.gelu(self.encoder.conv3(x_chunk))

        x_chunk = x_chunk.permute(0, 2, 1)

        # Attention với caching
        x_len = torch.tensor([x_chunk.shape[1]]).to(self.device)
        if self.k_cache is not None:
            x_len = x_len + ATTENTION_CONTEXT_SIZE[0]
            offset = torch.neg(self.cache_len) + ATTENTION_CONTEXT_SIZE[0]
        else:
            offset = None

        attn_mask = self.encoder.form_attention_mask_for_streaming(
            self.encoder.att_context_size, x_len, offset.to(self.device), self.device
        )

        if self.k_cache is not None:
            attn_mask = attn_mask[:, :, ATTENTION_CONTEXT_SIZE[0]:, :]

        # Process through transformer blocks
        new_k_cache = []
        new_v_cache = []
        for i, block in enumerate(self.encoder.blocks):
            x_chunk, layer_k_cache, layer_v_cache = block(
                x_chunk, mask=attn_mask, k_cache=self.k_cache[i], v_cache=self.v_cache[i]
            )
            new_k_cache.append(layer_k_cache)
            new_v_cache.append(layer_v_cache)

        enc_out = self.encoder.ln_post(x_chunk)

        # Update caches
        self.k_cache = torch.stack(new_k_cache, dim=0)
        self.v_cache = torch.stack(new_v_cache, dim=0)
        self.cache_len = torch.clamp(self.cache_len + ATTENTION_CONTEXT_SIZE[-1] + 1, max=ATTENTION_CONTEXT_SIZE[0])

        # Greedy decoding
        seq_enc_out = enc_out[0, :, :].unsqueeze(0)
        new_tokens = []

        for time_idx in range(seq_enc_out.shape[1]):
            current_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)
            not_blank = True
            symbols_added = 0

            while not_blank and symbols_added < MAX_SYMBOLS:
                if self.hypothesis[-1][0] is None:
                    last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=self.device)
                    last_seq_h_n = None
                else:
                    last_token = self.hypothesis[-1][0]
                    last_seq_h_n = self.hypothesis[-1][1]

                if last_seq_h_n is None:
                    current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                else:
                    current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)

                logits = self.joint(current_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]
                _, token_id = logits.max(0)
                token_id = token_id.detach().item()

                if token_id == RNNT_BLANK:
                    not_blank = False
                else:
                    symbols_added += 1
                    self.hypothesis.append([
                        torch.tensor([[token_id]], dtype=torch.long, device=self.device),
                        current_seq_h_n
                    ])
                    self.seq_ids.append(token_id)
                    new_tokens.append(token_id)

        # Return partial transcription nếu có tokens mới
        if new_tokens:
            return self.tokenizer.decode(new_tokens)
        return ""

    def _audio_processing_thread(self):
        """Thread xử lý audio liên tục"""
        print("🎯 Audio processing thread started")

        while self.is_running:
            # Lấy audio chunk từ microphone
            audio_chunk = self.microphone.get_audio_chunk()

            if audio_chunk is not None:
                if self.streaming_mode:
                    # Streaming mode: xử lý theo chunk thời gian cố định
                    chunk_to_process = self.audio_buffer.add_audio_streaming(audio_chunk)

                    if chunk_to_process is not None:
                        # Kiểm tra voice activity trước khi process
                        if self.microphone.has_voice_activity(chunk_to_process):
                            partial_text = self._process_audio_chunk_streaming(chunk_to_process)
                            if partial_text.strip():
                                self.transcription_queue.put(('partial', partial_text))
                else:
                    # Offline mode: sử dụng VAD để cắt segment
                    has_segment, segment_to_process = self.vad_buffer.add_audio(audio_chunk)

                    if has_segment and segment_to_process is not None:
                        print(
                            f"\n🎯 VAD detected complete segment ({len(segment_to_process) / SAMPLE_RATE:.1f}s), transcribing...")
                        partial_text = self._process_audio_chunk_offline(segment_to_process)
                        if partial_text.strip():
                            self.transcription_queue.put(('partial', partial_text))
                        else:
                            print("🔇 No speech detected in this segment")

            time.sleep(0.01)  # Small delay

    def _output_thread(self):
        """Thread để output kết quả liên tục"""
        print("📝 Output thread started")
        current_text = ""
        all_transcriptions = []

        while self.is_running:
            try:
                msg_type, text = self.transcription_queue.get(timeout=0.1)

                if msg_type == 'partial':
                    if self.streaming_mode:
                        # Streaming mode: always append new text
                        current_text += text
                        print(f"\r🔊 {current_text}", end="", flush=True)
                    else:
                        # Offline mode: check append_mode
                        if text.strip():
                            if self.append_mode:
                                # Append mode: tích lũy tất cả text
                                all_transcriptions.append(text.strip())
                                current_text = " ".join(all_transcriptions)
                            else:
                                # Replace mode: chỉ hiển thị text hiện tại
                                current_text = text.strip()
                            print(f"\r🔊 {current_text}", end="", flush=True)

                elif msg_type == 'final':
                    current_text = text
                    print(f"\n✅ Final: {current_text}")
                    if self.streaming_mode:
                        # Reset cho streaming mode
                        self._init_streaming_states()
                    current_text = ""
                    all_transcriptions = []

                elif msg_type == 'reset':
                    print(f"\n🔄 Transcription reset!")
                    current_text = ""
                    all_transcriptions = []
                    if self.streaming_mode:
                        self._init_streaming_states()
                    print("🔊 ", end="", flush=True)

            except queue.Empty:
                continue

    def start_continuous_recognition(self):
        """Bắt đầu continuous recognition"""
        if self.streaming_mode:
            mode_desc = f"Streaming (chunks: {self.audio_buffer.chunk_samples / SAMPLE_RATE:.1f}s)"
        else:
            mode_desc = f"Offline VAD (silence: {self.vad_buffer.silence_samples / SAMPLE_RATE:.1f}s)"

        print(f"🚀 Starting continuous recognition...")
        print(f"📊 Mode: {mode_desc}")
        print(f"📝 Output: {'Append' if self.append_mode else 'Replace'} mode")
        print("💡 Speak continuously. Press Ctrl+C to stop.")
        print("🔄 Press Enter during recognition to reset current transcription.")
        print("=" * 60)

        self.is_running = True
        self.microphone.start_stream()

        # Start threads
        audio_thread = threading.Thread(target=self._audio_processing_thread)
        output_thread = threading.Thread(target=self._output_thread)
        input_thread = threading.Thread(target=self._input_monitor_thread)

        audio_thread.daemon = True
        output_thread.daemon = True
        input_thread.daemon = True

        audio_thread.start()
        output_thread.start()
        input_thread.start()

        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping recognition...")
        finally:
            self.stop_recognition()

    def _input_monitor_thread(self):
        """Thread để monitor input từ user"""
        import sys
        import select

        while self.is_running:
            # Kiểm tra có input không (chỉ works trên Unix-like systems)
            try:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    input()  # Clear the input
                    self.transcription_queue.put(('reset', ''))
            except:
                # Fallback cho Windows - không có select
                time.sleep(0.1)
                pass

    def stop_recognition(self):
        """Dừng recognition"""
        self.is_running = False
        self.microphone.stop_stream()
        print("✅ Recognition stopped.")

        # Output final transcription nếu có
        if self.streaming_mode and self.seq_ids:
            final_text = self.tokenizer.decode(self.seq_ids)
            print(f"\n🎯 Final transcription: {final_text}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Continuous Vietnamese ASR with Microphone")

    parser.add_argument(
        "--model", type=str, default=None,
        help="Đường dẫn đến file model checkpoint"
    )

    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Đường dẫn đến tokenizer model"
    )

    parser.add_argument(
        "--device", type=str, default=None,
        help="Device để chạy model (cuda/cpu)"
    )

    parser.add_argument(
        "--streaming", action="store_true",
        help="Sử dụng streaming mode (online), mặc định là offline"
    )

    parser.add_argument(
        "--chunk-duration", type=float, default=0.5,
        help="Thời gian mỗi chunk để process cho streaming mode (giây)"
    )

    parser.add_argument(
        "--silence-duration", type=float, default=1.5,
        help="Thời gian im lặng để kết thúc segment cho offline mode (giây)"
    )

    parser.add_argument(
        "--no-append", action="store_true",
        help="Không append text, thay vào đó replace text mỗi chunk"
    )

    args = parser.parse_args()

    # Initialize StreamingASR
    asr = StreamingASR(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        streaming_mode=args.streaming,
        chunk_duration=args.chunk_duration,
        append_mode=not args.no_append,
        silence_duration=args.silence_duration
    )

    # Start continuous recognition
    asr.start_continuous_recognition()


if __name__ == "__main__":
    main()