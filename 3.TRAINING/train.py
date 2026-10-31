import os
import sys
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =============================================================================
# Path Setup
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, '2.MODEL'))
sys.path.append(os.path.join(BASE_DIR, '1.DATASET'))

from model import MiniTransformer, ModelConfig
# PERBAIKAN 1: Import PantunCustomTokenizer yang baru
from tokenization import PantunCustomTokenizer, PantunDataset, PantunCollator

# =============================================================================
# Helper Functions
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_lr(step, args):
    """Cosine learning rate scheduler with linear warmup (Eq. 3 paper adaptation)."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    if step > args.max_steps:
        return args.min_lr
    decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)

@torch.no_grad()
def estimate_loss(model, dataloaders, eval_iters, device, device_type):
    """Evaluates training and validation loss accurately without dropout."""
    out = {}
    model.eval()
    for split, dl in dataloaders.items():
        losses = []
        for k, batch in enumerate(dl):
            if k >= eval_iters: break
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if isinstance(model, nn.DataParallel):
                    _, loss = model(idx=x, targets=y, ignore_index=-100)
                    loss = loss.mean()
                else:
                    _, loss = model(idx=x, targets=y, ignore_index=-100)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0
    model.train()
    return out

@torch.no_grad()
# PERBAIKAN 2: Format prompt sampel diekspansi menjadi Full Conditioning
def generate_sample(model, tokenizer, device, prompt_text="<|bos|><|tema|>cinta<|rima_a|>an<|rima_b|>ti<|isi|>"):
    """Sanity check qualitative generation during evaluation."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long).to(device)
    raw_model = model.module if hasattr(model, "module") else model
    out_ids = raw_model.generate(input_ids, max_new_tokens=80, top_k=50)
    generated_text = tokenizer.decode(out_ids[0].tolist())
    model.train()
    return generated_text

# =============================================================================
# Main Training Loop
# =============================================================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick pipeline test with 2 steps and 2 eval batches")
    args = parser.parse_args()

    # Apply smoke test overrides
    if args.smoke_test:
        print("\n" + "="*50)
        print("🚀 SMOKE TEST MODE ACTIVATED")
        print("Will run only 2 training steps and 2 eval iterations to verify the entire pipeline.")
        print("="*50 + "\n")
        args.max_steps = 2
        args.eval_interval = 1
        args.eval_iters = 2
        args.grad_accum_steps = 1
        args.warmup_steps = 1

    set_seed(args.seed)

    # 1. Device configuration
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[Init] Using device: {device}")

    # 2. Setup Tokenizer & Data
    # PERBAIKAN 3: Load Custom BPE Tokenizer yang sudah dilatih (Vocab 4000)
    tokenizer_path = os.path.join(BASE_DIR, '1.DATASET', 'pantun_tokenizer.json')
    tokenizer = PantunCustomTokenizer(vocab_size=4000)
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer tidak ditemukan di {tokenizer_path}! Jalankan 1.DATASET/tokenization.py terlebih dahulu.")
    
    tokenizer.load(tokenizer_path)
    
    cfg = ModelConfig.from_config_py() if os.path.exists(os.path.join(BASE_DIR, '2.MODEL', 'config.py')) else ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size

    train_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'train.txt')
    val_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'val.txt')

    for split_name, split_path in (("train", train_path), ("val", val_path)):
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"File data {split_name} tidak ditemukan: {split_path}\n"
                "Jalankan 1.DATASET/preprocessing.py lalu 1.DATASET/splitting.py terlebih dahulu."
            )

    train_dataset = PantunDataset(train_path, tokenizer, max_seq_len=cfg.block_size, is_training=True)
    val_dataset = PantunDataset(val_path, tokenizer, max_seq_len=cfg.block_size, is_training=False)

    if len(train_dataset) == 0:
        raise RuntimeError(f"Dataset train kosong atau formatnya tidak valid: {train_path}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"Dataset val kosong atau formatnya tidak valid: {val_path}")

    print(f"[Init] Train samples: {len(train_dataset):,} ({train_path})")
    print(f"[Init] Val samples  : {len(val_dataset):,} ({val_path})")
    
    collator = PantunCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=-100)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    dataloaders = {"train": train_loader, "val": val_loader}

    # Data Pipeline Unit Test (Point B)
    sample_batch = next(iter(train_loader))
    sx, sy = sample_batch["input_ids"], sample_batch["labels"]
    valid_mask = (sy[:, :-1] != -100) & (sx[:, 1:] != tokenizer.pad_token_id)
    assert torch.all(sy[:, :-1][valid_mask] == sx[:, 1:][valid_mask]), "Data Pipeline Causality Error: Targets are not shifted correctly!"

    # 3. Model & Optimizer
    model = MiniTransformer(cfg)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Init] Using {torch.cuda.device_count()} GPUs for DataParallel!")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)

    # 4. Resume Checkpoint Logic (Point H)
    ckpt_dir = os.path.join(BASE_DIR, '3.TRAINING', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    start_step = 0
    best_val_loss = float('inf')

    if args.resume and not args.smoke_test:
        last_ckpt = os.path.join(ckpt_dir, 'last.pt')
        if os.path.exists(last_ckpt):
            print(f"[Init] Resuming from {last_ckpt}")
            ckpt = torch.load(last_ckpt, map_location=device)
            raw_model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0) + 1
            best_val_loss = ckpt.get("best_val", float('inf'))
        else:
            print("[Warning] --resume passed but last.pt not found. Starting from scratch.")

    # 5. Training Loop
    print(f"[Init] Expected initial loss: ~{math.log(cfg.vocab_size):.2f}")
    train_iter = iter(train_loader)
    model.train()
    
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        # Update LR
        lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Micro-batch loop
        for micro_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if isinstance(model, nn.DataParallel):
                    _, loss = model(idx=x, targets=y, ignore_index=-100)
                    loss = loss.mean() / args.grad_accum_steps
                else:
                    _, loss = model(idx=x, targets=y, ignore_index=-100)
                    loss = loss / args.grad_accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Error] Loss is NaN/Inf at step {step}. Halting.")
                sys.exit(1)
                
            loss.backward()

        # Gradient Clipping & Optimizer Step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % max(1, (10 if not args.smoke_test else 1)) == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (args.batch_size * args.grad_accum_steps * cfg.block_size) / (dt if dt > 0 else 0.001)
            print(f"Step {step:4d} | Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {lr:.2e} | Norm: {grad_norm:.2f} | Tok/s: {tokens_per_sec:.0f}")

        # Evaluation & Checkpointing
        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            losses = estimate_loss(model, dataloaders, args.eval_iters, device, device_type)
            val_loss = losses['val']
            val_ppl = math.exp(min(val_loss, 20)) # Cap PPL for initial print safety
            print(f"\n--- EVAL STEP {step} ---")
            print(f"Train Loss: {losses['train']:.4f} | Val Loss: {val_loss:.4f} | Val PPL (BPE): {val_ppl:.2f}")
            print("Sample Gen:", generate_sample(model, tokenizer, device))
            print("------------------------\n")

            # Checkpointing
            raw_model.save_checkpoint(
                os.path.join(ckpt_dir, 'last.pt'),
                optimizer=optimizer.state_dict(),
                step=step,
                best_val=best_val_loss,
                val_loss=val_loss
            )
            
            if val_loss < best_val_loss and not args.smoke_test:
                best_val_loss = val_loss
                raw_model.save_checkpoint(
                    os.path.join(ckpt_dir, 'best.pt'),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    best_val=best_val_loss,
                    val_loss=val_loss
                )
                print(f"[Info] New best checkpoint saved with loss {best_val_loss:.4f}")

    if args.smoke_test:
        print("\n✅ SMOKE TEST PASSED: Full training loop, target shifting, loss calculation, evaluation, mixed-precision, and checkpoint saving executed successfully.\n")

if __name__ == "__main__":
    train()