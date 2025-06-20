import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# from warp_rnnt import rnnt_loss
import warprnnt_numba

import sentencepiece as spm
from jiwer import wer

from loguru import logger
import warnings
warnings.filterwarnings("ignore")
from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer

from config import get_config
from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR, CosineAnnealingWarmupLR
from utils.model_checkpoint import StepBasedModelCheckpoint

class StreamingRNNT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        encoder_state_dict = torch.load(
            config.paths.pretrained_encoder_weight,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
            weights_only=True
        )
        # Create new keys 'conv3.weight', 'conv3.bias' that copy from 'conv2.weight', 'conv2.bias' so that we don't have to initialize conv3 weights
        encoder_state_dict['model_state_dict']['conv3.weight'] = encoder_state_dict['model_state_dict']['conv2.weight']
        encoder_state_dict['model_state_dict']['conv3.bias'] = encoder_state_dict['model_state_dict']['conv2.bias']

        self.encoder = AudioEncoder(
            n_mels=config.audio.n_mels,
            n_state=config.model.whisper.n_state,
            n_head=config.model.whisper.n_head,
            n_layer=config.model.whisper.n_layer,
            att_context_size=config.model.attention_context_size
        )
        self.encoder.load_state_dict(encoder_state_dict['model_state_dict'], strict=False)

        self.decoder = Decoder(
            vocab_size=config.tokenizer.vocab_size + 1,
            embed_dim=config.model.whisper.n_state,
            hidden_dim=config.model.whisper.n_state
        )
        self.joint = Jointer(
            encoder_dim=config.model.whisper.n_state,
            decoder_dim=config.model.whisper.n_state,
            vocab_size=config.tokenizer.vocab_size + 1
        )

        self.tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer.model_path)

        # self.loss = torchaudio.transforms.RNNTLoss(reduction="mean") # RNNTLoss has bug with logits number of elements > 2**31
        self.loss = warprnnt_numba.RNNTLossNumba(
            blank=config.tokenizer.rnnt_blank, reduction="mean",
        )
        # Enhanced optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.training.optimizer.lr,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps,
            weight_decay=getattr(config.training.optimizer, 'weight_decay', 0.01)
        )
        
    
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L416
    def greedy_decoding(self, x, x_len, max_symbols=None):
        if max_symbols is None:
            max_symbols = 3  # Default from config

        enc_out, _ = self.encoder(x, x_len)
        all_sentences = []
        # greedy decoding, handle each sequence independently for easier implementation
        for batch_idx in range(enc_out.shape[0]):
            hypothesis = [[None, None]]  # [label, state]
            seq_enc_out = enc_out[batch_idx, :, :].unsqueeze(0) # [1, T, D]
            seq_ids = []

            for time_idx in range(seq_enc_out.shape[1]):
                curent_seq_enc_out = seq_enc_out[:, time_idx, :].unsqueeze(1) # 1, 1, D

                not_blank = True
                symbols_added = 0

                while not_blank and (max_symbols is None or symbols_added < max_symbols):
                    # In the first timestep, we initialize the network with RNNT Blank
                    # In later timesteps, we provide previous predicted label as input.
                    if hypothesis[-1][0] is None:
                        last_token = torch.tensor([[self.config.tokenizer.rnnt_blank]], dtype=torch.long, device=seq_enc_out.device)
                        last_seq_h_n = None
                    else:
                        last_token = hypothesis[-1][0]
                        last_seq_h_n = hypothesis[-1][1]

                    if last_seq_h_n is None:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token)
                    else:
                        current_seq_dec_out, current_seq_h_n = self.decoder(last_token, last_seq_h_n)
                    logits = self.joint(curent_seq_enc_out, current_seq_dec_out)[0, 0, 0, :]  # (B, T=1, U=1, V + 1)

                    del current_seq_dec_out

                    _, token_id = logits.max(0)
                    token_id = token_id.detach().item()  # K is the label at timestep t_s in inner loop, s >= 0.

                    del logits

                    if token_id == self.config.tokenizer.rnnt_blank:
                        not_blank = False
                    else:
                        symbols_added += 1
                        hypothesis.append([
                            torch.tensor([[token_id]], dtype=torch.long, device=curent_seq_enc_out.device),
                            current_seq_h_n
                        ])
                        seq_ids.append(token_id)
            all_sentences.append(self.tokenizer.decode(seq_ids))
        return all_sentences
    
    def process_batch(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        if batch_idx != 0 and batch_idx % 2000 == 0:
            all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
            all_true = []
            for i, y_i in enumerate(y):
                y_i = y_i.cpu().numpy().astype(int).tolist()
                y_i = y_i[:y_len[i]]
                all_true.append(self.tokenizer.decode_ids(y_i))

            for pred, true in zip(all_pred, all_true):
                logger.debug(f"Pred: {pred}")
                logger.debug(f"True: {true}")
            
            all_wer = wer(all_true, all_pred)
            self.log("train_wer", all_wer, prog_bar=False, on_step=True, on_epoch=False)

        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L185
        # Blank is also the start of sequence token and the sentence will start with blank; https://github.com/pytorch/audio/issues/3750
        y_start = torch.cat([torch.full((y.shape[0], 1), self.config.tokenizer.rnnt_blank, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()
        
        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        if batch_idx % 100 == 0:
            # Log training loss with format only two last digits
            self.log("train_loss", loss.detach().item(), prog_bar=True, on_step=True, on_epoch=False)
            # Log current learning rate
            self.log("lr", self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len = self.process_batch(batch)

        all_pred = self.greedy_decoding(x, x_len, max_symbols=3)
        all_true = []
        for i, y_i in enumerate(y):
            y_i = y_i.cpu().numpy().astype(int).tolist()
            y_i = y_i[:y_len[i]]
            all_true.append(self.tokenizer.decode_ids(y_i))
        
        all_wer = wer(all_true, all_pred)

        # ------------------CALCULATE LOSS------------------ 
        enc_out, x_len = self.encoder(x, x_len) # (B, T, Enc_dim)

        # Add a blank token to the beginning of the target sequence
        # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L185
        # Blank is also the start of sequence token and the sentence will start with blank; https://github.com/pytorch/audio/issues/3750
        y_start = torch.cat([torch.full((y.shape[0], 1), self.config.tokenizer.rnnt_blank, dtype=torch.int).to(y.device), y], dim=1).to(y.device)
        dec_out, _ = self.decoder(y_start) # (B, U, Dec_dim)
        logits = self.joint(enc_out, dec_out)

        input_lengths = x_len.int()
        target_lengths = y_len.int()
        targets = y.int()
        
        loss = self.loss(logits.to(torch.float32), targets, input_lengths, target_lengths)
        # ---------------------------------------------------

        if batch_idx % 1000 == 0:
            for pred, true in zip(all_pred, all_true):
                logger.debug(f"Pred: {pred}")
                logger.debug(f"True: {true}")

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_wer", all_wer, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    # def on_validation_end(self):
    #     # save the last checkpoint
    #     self.trainer.save_checkpoint(f"{LOG_DIR}/rnnt-latest.ckpt", weights_only=True)
    #     return super().on_validation_end()

    def on_train_epoch_end(self):
        # save the last checkpoint
        self.trainer.save_checkpoint(f"{self.config.paths.log_dir}/rnnt-latest.ckpt", weights_only=True)
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        # Choose scheduler based on config
        scheduler_type = getattr(self.config.training.scheduler, 'type', 'linear')

        if scheduler_type == 'cosine_annealing':
            scheduler = CosineAnnealingWarmupLR(
                self.optimizer,
                self.config.training.scheduler.warmup_steps,
                self.config.training.scheduler.total_steps,
                getattr(self.config.training.scheduler, 'min_lr_ratio', 0.1)
            )
        else:
            # Default to linear warmup + exponential decay
            scheduler = WarmupLR(
                self.optimizer,
                self.config.training.scheduler.warmup_steps,
                self.config.training.scheduler.total_steps,
                self.config.training.optimizer.min_lr
            )

        return (
            [self.optimizer],
            [{"scheduler": scheduler, "interval": "step"}],
        )

# Load configuration
config = get_config()

train_dataset = AudioDataset(
    manifest_files=config.dataset.train_manifest,
    bg_noise_path=config.dataset.bg_noise_paths,
    shuffle=True,
    augment=True,
    tokenizer_model_path=config.tokenizer.model_path,
    max_duration=config.dataset.max_duration,
    min_duration=config.dataset.min_duration,
    min_text_len=config.dataset.min_text_len,
    max_text_len=config.dataset.max_text_len
)

val_dataset = AudioDataset(
    manifest_files=config.dataset.val_manifest,
    shuffle=False,
    tokenizer_model_path=config.tokenizer.model_path,
    max_duration=config.dataset.max_duration,
    min_duration=config.dataset.min_duration,
    min_text_len=config.dataset.min_text_len,
    max_text_len=config.dataset.max_text_len
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    collate_fn=collate_fn,
    pin_memory=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    collate_fn=collate_fn,
    pin_memory=True
)

model = StreamingRNNT(config)

# Use custom step-based checkpoint callback
checkpoint_callback = StepBasedModelCheckpoint(
    dirpath=config.paths.log_dir,
    save_every_n_steps=config.model_saving.save_every_n_steps,
    keep_top_k=config.model_saving.keep_top_k,
    monitor_metric=config.model_saving.monitor_metric,
    mode=config.model_saving.mode,
    save_weights_only=config.model_saving.save_weights_only,
    filename_template=config.model_saving.filename_template,
    verbose=True
)

trainer = pl.Trainer(
    # profiler="simple",
    # max_steps=100,
    max_epochs=config.training.max_epochs,
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="bf16-mixed",
    strategy="ddp",
    callbacks=[checkpoint_callback],
    logger=pl.loggers.TensorBoardLogger(config.paths.log_dir),
    num_sanity_val_steps=0, # At the start, the model produced garbage predictions anyway. Should only be > 0 for testing
    val_check_interval=config.model_saving.save_every_n_steps,  # Run validation every N steps to get val_wer
    check_val_every_n_epoch=None,  # Disable epoch-based validation, use step-based instead

    # Enhanced training settings
    accumulate_grad_batches=getattr(config.training, 'accumulate_grad_batches', 1),
    gradient_clip_val=getattr(config.training, 'gradient_clip_val', None),
    gradient_clip_algorithm=getattr(config.training, 'gradient_clip_algorithm', 'norm')
)

if __name__ == "__main__":
    logger.info("🚀 Starting Whisper RNN-T Training")
    logger.info(f"📊 Model: Whisper Base ({config.model.whisper.n_state}d, {config.model.whisper.n_head}h, {config.model.whisper.n_layer}l)")
    logger.info(f"📁 Checkpoints: {config.paths.log_dir}")
    logger.info(f"💾 Save strategy: Every {config.model_saving.save_every_n_steps} steps, keep top {config.model_saving.keep_top_k} by {config.model_saving.monitor_metric}")

    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="/path/to/ckpt")
    trainer.fit(model, train_dataloader, val_dataloader)