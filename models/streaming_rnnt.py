import torch
import torch.nn as nn
import warprnnt_numba
import sentencepiece as spm
from jiwer import wer

from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer

from constants import (
    RNNT_BLANK, PAD, N_MELS, N_STATE, N_LAYER, N_HEAD,
    PRETRAINED_ENCODER_WEIGHT, MAX_SYMBOLS
)

class StreamingRNNT(nn.Module):
    def __init__(self, att_context_size, vocab_size, tokenizer_model_path):
        super().__init__()

        # Load pretrained encoder weights
        encoder_state_dict = torch.load(
            PRETRAINED_ENCODER_WEIGHT,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=True
        )

        # Create new keys 'conv3.weight', 'conv3.bias' that copy from 'conv2.weight', 'conv2.bias'
        # so that we don't have to initialize conv3 weights
        encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
        encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']

        # Initialize encoder
        self.encoder = AudioEncoder(
            n_mels=N_MELS,
            n_state=N_STATE,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            att_context_size=att_context_size
        )
        self.encoder.load_state_dict(encoder_state_dict['model_state_dict'], strict=False)

        # Initialize decoder and jointer
        self.decoder = Decoder(vocab_size=vocab_size + 1)
        self.joint = Jointer(vocab_size=vocab_size + 1)

        # Initialize tokenizer and loss function
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        self.loss = warprnnt_numba.RNNTLossNumba(
            blank=RNNT_BLANK, reduction="mean",
        )

    def forward(self, x, x_len, y=None, y_len=None):
        """
        Forward pass of the model.

        Args:
            x: Input mel spectrogram [batch, mels, time]
            x_len: Lengths of spectrograms [batch]
            y: Target text tokens [batch, max_len]
            y_len: Lengths of target text [batch]

        Returns:
            loss: RNN-T loss if y and y_len are provided
            enc_out, x_len: Encoder outputs if y and y_len are not provided
        """
        enc_out, x_len = self.encoder(x, x_len)

        if y is None or y_len is None:
            return enc_out, x_len

        # Add a blank token to the beginning of the target sequence (required by RNN-T)
        y_start = torch.cat([torch.full((y.shape[0], 1), RNNT_BLANK, dtype=torch.int).to(y.device), y], dim=1)
        dec_out, _ = self.decoder(y_start)
        logits = self.joint(enc_out, dec_out)

        # Calculate loss
        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()

        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        return loss

    def process_batch(self, batch):
        """Process a batch from the dataloader"""
        x, x_len, y, y_len = batch
        return x, x_len, y, y_len

    def greedy_decoding(self, x, x_len, max_symbols=MAX_SYMBOLS):
        """
        Greedy decoding for inference with optimization for short sentences.

        Args:
            x: Input mel spectrogram [batch, mels, time]
            x_len: Lengths of spectrograms [batch]
            max_symbols: Maximum number of symbols per timestep (default value from constants)

        Returns:
            all_sentences: List of decoded sentences
        """
        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []

        # Handle each sequence independently for easier implementation
        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None]]  # [label, state]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0)  # [1, T, D]
            seq_ids = []

            # Kiểm tra nếu sequence encoder ngắn (thường tương ứng với câu ngắn)
            # và điều chỉnh max_symbols tương ứng
            encoder_length = seq_enc_out.shape[1]
            # Tăng max_symbols cho câu ngắn
            adjusted_max_symbols = max_symbols
            if encoder_length < 10:  # Nếu encoder output ngắn (câu ngắn)
                adjusted_max_symbols = max_symbols + 2  # Tăng thêm 2 symbols
            elif encoder_length < 20:  # Câu hơi dài hơn một chút
                adjusted_max_symbols = max_symbols + 1  # Tăng thêm 1 symbol

            # Giảm thời gian giải mã ngưỡng blank cho câu ngắn
            blank_threshold = 0.0
            if encoder_length < 15:
                blank_threshold = -0.1  # Giảm nhẹ ngưỡng để có xu hướng tạo nhiều token hơn

            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1)  # [1, 1, D]

                not_blank = True
                symbols_added = 0

                while not_blank and (adjusted_max_symbols is None or symbols_added < adjusted_max_symbols):
                    # Khởi tạo token đầu tiên hoặc lấy token từ trạng thái trước
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[RNNT_BLANK]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    # Decode token
                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)

                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # [V + 1]

                    del current_seq_dec_out

                    # Áp dụng logic thay đổi ngưỡng cho câu ngắn
                    if encoder_length < 15 and symbols_added == 0:
                        # Nhân các giá trị non-blank lên một chút để khuyến khích phi-blank cho câu ngắn
                        non_blank_boost = 1.05
                        logits_except_blank = logits[:-1] * non_blank_boost
                        logits = torch.cat([logits_except_blank, logits[-1:]], dim=0)

                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()

                    del logits

                    # Nếu blank hoặc đã đạt max symbols, dừng thêm ký tự
                    if token_id == RNNT_BLANK:
                        not_blank = False
                    else:
                        symbols_added += 1
                        hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=curent_seq_enc_out.device),
                            current_seq_h_n
                        ])
                        seq_ids.append(token_id)

            # Giải mã thành văn bản
            decoded_text = self.tokenizer.decode(seq_ids)
            all_sentences.append(decoded_text)

        return all_sentences