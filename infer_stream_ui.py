import gradio as gr
import torch
import torch.nn.functional as F
import librosa
import sentencepiece as spm
from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer
from config import get_config
import numpy as np
from huggingface_hub import hf_hub_download
import threading
import queue
import time

# Load configuration
config = get_config()

# Tải model
trained_model_path = r"D:\train_model\whisper_rnnt\checkpoints\rnnt-latest.ckpt"  # Updated for base model
# Tải tokenizer
tokenizer_path = config.tokenizer.model_path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load(trained_model_path, map_location=device, weights_only=True)

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

# Khởi tạo models
encoder = AudioEncoder(
    config.audio.n_mels,
    n_state=config.model.whisper.n_state,
    n_head=config.model.whisper.n_head,
    n_layer=config.model.whisper.n_layer,
    att_context_size=config.model.attention_context_size
).to(device).eval()

decoder = Decoder(
    vocab_size=config.tokenizer.vocab_size + 1,
    embed_dim=config.model.whisper.n_state,
    hidden_dim=config.model.whisper.n_state
).to(device).eval()

joint = Jointer(
    encoder_dim=config.model.whisper.n_state,
    decoder_dim=config.model.whisper.n_state,
    vocab_size=config.tokenizer.vocab_size + 1
).to(device).eval()

encoder.load_state_dict(encoder_weight, strict=False)
decoder.load_state_dict(decoder_weight, strict=False)
joint.load_state_dict(joint_weight, strict=False)

# Load tokenizer
tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

def mel_filters(device, n_mels: int) -> torch.Tensor:
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    with np.load("./utils/mel_filters.npz", allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio, n_mels, padding, streaming, device):
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(config.audio.n_fft).to(audio.device)
    if not streaming:
        stft = torch.stft(audio, config.audio.n_fft, config.audio.hop_length, window=window, return_complex=True)
    else:
        stft = torch.stft(audio, config.audio.n_fft, config.audio.hop_length, window=window, center=False, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

class StreamingTranscriber:
    def __init__(self, encoder, decoder, joint, tokenizer, device='cpu'):
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.tokenizer = tokenizer
        self.device = device
        self.reset()
        
    def reset(self):
        """Reset tất cả cache cho phiên mới"""
        self.audio_cache = torch.zeros(240, device=self.device)
        self.conv1_cache = torch.zeros(1, 80, 1, device=self.device)
        self.conv2_cache = torch.zeros(1, config.model.whisper.n_state, 1, device=self.device)
        self.conv3_cache = torch.zeros(1, config.model.whisper.n_state, 1, device=self.device)
        self.k_cache = torch.zeros(config.model.whisper.n_layer, 1, config.model.attention_context_size[0], config.model.whisper.n_state, device=self.device)
        self.v_cache = torch.zeros(config.model.whisper.n_layer, 1, config.model.attention_context_size[0], config.model.whisper.n_state, device=self.device)
        self.cache_len = torch.zeros(1, dtype=torch.int, device=self.device)
        self.hypothesis = [[None, None]]
        self.seq_ids = []
        self.audio_buffer = torch.zeros(0, device=self.device)
        
    def process_chunk(self, audio_chunk, max_symbols=3):
        """Xử lý một chunk audio và trả về text được nhận diện"""
        # Thêm audio mới vào buffer
        self.audio_buffer = torch.cat([self.audio_buffer, audio_chunk])
        
        # Xử lý khi đủ kích thước chunk
        chunk_size = config.audio.hop_length * 31 + config.audio.n_fft - (config.audio.n_fft - config.audio.hop_length)
        new_text = ""
        
        while self.audio_buffer.shape[0] >= chunk_size + 240:
            # Lấy chunk để xử lý
            process_chunk = torch.cat([self.audio_cache, self.audio_buffer[:chunk_size]])
            self.audio_buffer = self.audio_buffer[chunk_size:]
            
            if process_chunk.shape[0] < config.audio.hop_length * 31 + config.audio.n_fft:
                process_chunk = F.pad(process_chunk, (0, config.audio.hop_length * 31 + config.audio.n_fft - process_chunk.shape[0]))

            self.audio_cache = process_chunk[-(config.audio.n_fft - config.audio.hop_length):]

            # Mel spectrogram
            x_chunk = log_mel_spectrogram(audio=process_chunk, n_mels=config.audio.n_mels, padding=0, streaming=True, device=self.device)
            x_chunk = x_chunk.reshape(1, *x_chunk.shape)
            
            if x_chunk.shape[-1] < 32:
                x_chunk = F.pad(x_chunk, (0, 32 - x_chunk.shape[-1]))
                
            # Convolutional layers với cache
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
            x_len = torch.tensor([x_chunk.shape[1]]).to(self.device)
            
            # Attention mask
            if self.k_cache is not None:
                x_len = x_len + config.model.attention_context_size[0]
                offset = torch.neg(self.cache_len) + config.model.attention_context_size[0]
            else:
                offset = None

            attn_mask = self.encoder.form_attention_mask_for_streaming(
                self.encoder.att_context_size, x_len, offset.to(self.device), self.device
            )

            if self.k_cache is not None:
                attn_mask = attn_mask[:, :, config.model.attention_context_size[0]:, :]
                
            # Encoder blocks
            new_k_cache = []
            new_v_cache = []
            for idx, block in enumerate(self.encoder.blocks):
                x_chunk, layer_k_cache, layer_v_cache = block(
                    x_chunk, mask=attn_mask, k_cache=self.k_cache[idx], v_cache=self.v_cache[idx]
                )
                new_k_cache.append(layer_k_cache)
                new_v_cache.append(layer_v_cache)
                
            enc_out = self.encoder.ln_post(x_chunk)
            self.k_cache = torch.stack(new_k_cache, dim=0)
            self.v_cache = torch.stack(new_v_cache, dim=0)
            self.cache_len = torch.clamp(
                self.cache_len + config.model.attention_context_size[-1] + 1, max=config.model.attention_context_size[0]
            )
            
            # Decode
            seq_enc_out = enc_out[0, :, :].unsqueeze(0)
            prev_len = len(self.seq_ids)
            
            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)
                not_blank = True
                symbols_added = 0
                
                while not_blank and symbols_added < max_symbols:
                    if self.hypothesis[-1][0] is None:
                        last_token = torch.tensor([[config.tokenizer.rnnt_blank]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = self.hypothesis[-1][0]
                        last_seq_h_n = self.hypothesis[-1][1]
                        
                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)
                        
                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]
                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()
                    
                    if token_id == config.tokenizer.rnnt_blank:
                        not_blank = False
                    else:
                        symbols_added += 1
                        self.hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=curent_seq_enc_out.device),
                            current_seq_h_n
                        ])
                        self.seq_ids.append(token_id)
            
            # Decode new text
            if len(self.seq_ids) > prev_len:
                new_ids = self.seq_ids[prev_len:]
                new_text += self.tokenizer.decode(new_ids)
                
        return new_text, self.tokenizer.decode(self.seq_ids)

# Khởi tạo transcriber
transcriber = StreamingTranscriber(encoder, decoder, joint, tokenizer, device)

def transcribe_streaming(audio, state):
    """Hàm xử lý audio streaming từ Gradio"""
    if state is None:
        state = {"full_text": "", "is_recording": True}
        transcriber.reset()
    
    if audio is None:
        return state["full_text"], state
    
    # Chuyển đổi audio về định dạng phù hợp
    sr, audio_data = audio
    
    # Resample nếu cần
    if sr != config.audio.sample_rate:
        audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sr, target_sr=config.audio.sample_rate)
    else:
        audio_data = audio_data.astype(np.float32)
    
    # Normalize audio
    audio_data = audio_data / np.abs(audio_data).max() if np.abs(audio_data).max() > 0 else audio_data
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_data).to(device)
    
    # Process chunk
    new_text, full_text = transcriber.process_chunk(audio_tensor)
    
    # Cập nhật state
    state["full_text"] = full_text
    
    return full_text, state

def clear_fn():
    """Reset transcriber khi clear"""
    transcriber.reset()
    return "", None

# Tạo Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Nhận diện giọng nói tiếng Việt Real-time")
    gr.Markdown("Nói vào microphone và xem kết quả nhận diện theo thời gian thực!")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                streaming=True,
                label="Microphone Input"
            )
        with gr.Column():
            output_text = gr.Textbox(
                label="Văn bản được nhận diện",
                lines=10,
                max_lines=20,
                interactive=False
            )
    
    state = gr.State()
    
    # Xử lý streaming
    audio_input.stream(
        transcribe_streaming,
        inputs=[audio_input, state],
        outputs=[output_text, state],
        show_progress=False
    )
    
    # Nút clear
    clear_btn = gr.Button("🗑️ Xóa và bắt đầu lại", variant="secondary")
    clear_btn.click(
        clear_fn,
        outputs=[output_text, state]
    )
    
    gr.Markdown("""
    ### Hướng dẫn sử dụng:
    1. Nhấn vào nút microphone để bắt đầu ghi âm
    2. Nói tiếng Việt rõ ràng vào microphone
    3. Văn bản sẽ được hiển thị real-time khi bạn nói
    4. Nhấn "Stop" để dừng ghi âm
    5. Nhấn "Xóa và bắt đầu lại" để reset
    
    **Lưu ý:** Model hoạt động tốt nhất với giọng nói rõ ràng và môi trường ít ồn.
    """)

if __name__ == "__main__":
    demo.launch(share=True)