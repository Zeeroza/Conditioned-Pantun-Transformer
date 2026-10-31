import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 1. Path Setup: Add sibling directories to sys.path to import modules
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, '2.MODEL'))
sys.path.append(os.path.join(BASE_DIR, '1.DATASET'))

from model import MiniTransformer, ModelConfig
from tokenization import PantunBPETokenizer, PantunDataset, PantunCollator

# 2. Default Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Data Paths
TRAIN_DATA_PATH = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'train.jsonl')
VAL_DATA_PATH = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'val.jsonl')
MASTER_DATA_PATH = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'master_data.jsonl')

CHECKPOINT_DIR = os.path.join(BASE_DIR, '3.TRAINING', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train(is_smoke_test=False):
    global MAX_EPOCHS
    if is_smoke_test:
        print("\n" + "="*50)
        print("🚀 SMOKE TEST MODE ACTIVATED")
        print("Will run only 1 epoch and 2 batches to verify pipeline.")
        print("="*50 + "\n")
        MAX_EPOCHS = 1

    print(f"Initializing tokenizer...")
    tokenizer = PantunBPETokenizer(base_encoding="gpt2")

    try:
        cfg = ModelConfig.from_config_py()
    except ImportError:
        print("config.py not found or invalid, using default ModelConfig.")
        cfg = ModelConfig()
    
    cfg.vocab_size = tokenizer.vocab_size

    print(f"Initializing model on {DEVICE}...")
    model = MiniTransformer(cfg).to(DEVICE)
    print(f"Model parameters: {model.num_params() / 1e6:.2f}M | Vocab size: {cfg.vocab_size}")

    train_path = TRAIN_DATA_PATH if os.path.exists(TRAIN_DATA_PATH) else MASTER_DATA_PATH
    val_path = VAL_DATA_PATH if os.path.exists(VAL_DATA_PATH) else MASTER_DATA_PATH
    print(f"Using Train Data: {train_path}\nUsing Val Data: {val_path}")

    train_dataset = PantunDataset(
        jsonl_path=train_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.block_size,
        is_training=True
    )
    val_dataset = PantunDataset(
        jsonl_path=val_path,
        tokenizer=tokenizer,
        max_seq_len=cfg.block_size,
        is_training=False
    )

    collator = PantunCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=-100)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    optimizer = model.configure_optimizers(lr=LEARNING_RATE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        batches_processed = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input_ids"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits, loss = model(idx=x, targets=y, ignore_index=-100)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batches_processed += 1

            if batch_idx % max(1, (100 if not is_smoke_test else 1)) == 0:
                print(f"Epoch {epoch}/{MAX_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            # Early break for smoke test
            if is_smoke_test and batches_processed >= 2:
                print("Smoke test: Reached 2 batches. Breaking training loop.")
                break

        avg_train_loss = train_loss / batches_processed
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x_val = batch["input_ids"].to(DEVICE)
                y_val = batch["labels"].to(DEVICE)
                _, loss = model(idx=x_val, targets=y_val, ignore_index=-100)
                val_loss += loss.item()
                val_batches += 1

                # Early break for smoke test
                if is_smoke_test and val_batches >= 2:
                    print("Smoke test: Reached 2 validation batches. Breaking validation loop.")
                    break
                
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        print(f"--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
        model.save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            optimizer_state_dict=optimizer.state_dict(),
            train_loss=avg_train_loss,
            val_loss=avg_val_loss
        )
        print(f"Checkpoint saved to {checkpoint_path}\n")

    if is_smoke_test:
        print("✅ SMOKE TEST PASSED: Full forward/backward pass and checkpointing successfully completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniTransformer for Pantun Generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick pipeline test with 1 epoch and 2 batches")
    args = parser.parse_args()

    train(is_smoke_test=args.smoke_test)