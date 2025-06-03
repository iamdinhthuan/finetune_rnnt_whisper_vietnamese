# run.py
from models.encoder import AudioEncoder
from models.decoder import Decoder
from models.jointer import Jointer
from torch.utils.data import DataLoader
from train import StreamingRNNT
from utils.dataset import AudioDataset, collate_fn
from utils.scheduler import WarmupLR
from constants import *
import torch
import pytorch_lightning as pl
import multiprocessing

if __name__ == '__main__':
    # This is necessary for multiprocessing to work correctly on Windows
    multiprocessing.freeze_support()
    
    train_dataset = AudioDataset(
        manifest_files=TRAIN_MANIFEST,
        bg_noise_path=BG_NOISE_PATH,
        shuffle=True,
        augment=True,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )

    val_dataset = AudioDataset(
        manifest_files=VAL_MANIFEST,
        shuffle=False,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, collate_fn=collate_fn, pin_memory=True)

    model = StreamingRNNT(
        att_context_size=ATTENTION_CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        tokenizer_model_path=TOKENIZER_MODEL_PATH
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=LOG_DIR,
        filename='rnnt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min'
    )

    trainer = pl.Trainer(
        # profiler="simple",
        # max_steps=100,
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed",
        strategy="auto",
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(LOG_DIR),
        num_sanity_val_steps=0, # At the start, the model produced garbage predictions anyway. Should only be > 0 for testing
        check_val_every_n_epoch=2
    )


    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="/path/to/ckpt")
    trainer.fit(model, train_dataloader, val_dataloader)