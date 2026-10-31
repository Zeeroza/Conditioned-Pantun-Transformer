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
from tokenization import (
    PantunCharTokenizer,
    PantunDataset,
    PantunCollator,
    analyze_context_length,
)

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

EVAL_PROMPTS = {
    "full":     "<|bos|><|tema|>umum<|rima_a|>ng<|rima_b|>ar<|sampiran|>",
    "no_rhyme": "<|bos|><|tema|>umum<|sampiran|>",
    "polos":    "<|bos|><|sampiran|>",
}


@torch.no_grad()
def generate_sample(
    model,
    tokenizer,
    device,
    prompt_text=EVAL_PROMPTS["full"],
    max_new_tokens=200,
    top_k=15,
    block_size=256,
):
    """Sanity check kualitatif. Prompt HARUS diakhiri <|sampiran|>, sama seperti format training."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = model.training
    model.eval()
    try:
        prompt_ids = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Jangan melewati context window model.
        budget = min(max_new_tokens, block_size - len(prompt_ids))
        out_ids = raw_model.generate(input_ids, max_new_tokens=budget, top_k=top_k)

        # Asumsi: generate() mengembalikan prompt + hasil (gaya nanoGPT).
        new_ids = out_ids[0].tolist()[len(prompt_ids):]

        # Potong di <|eos|> pertama; sisanya hanya sampah yang tidak pernah dilihat saat training.
        if tokenizer.eos_token_id in new_ids:
            new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
    finally:
        if was_training:
            model.train()

    return prompt_text + tokenizer.decode(new_ids, skip_special_tokens=False)


def check_rhyme(generated: str, rima_a: str, rima_b: str):
    """
    Cek kasar pola a-b-a-b: larik 1 & 3 berakhiran rima_a, larik 2 & 4 berakhiran rima_b.
    Asumsi: 'rima' di datasetmu = akhiran huruf larik. Sesuaikan kalau definisinya beda.
    Mengembalikan jumlah larik yang cocok (0-4), atau None kalau larik < 4.
    """
    body = generated.split("<|sampiran|>", 1)[-1].replace("<|isi|>", "").strip().split("\n")
    if len(body) < 4:
        return None
    ends = [ln.strip().rstrip(" ,.;:!?") for ln in body[:4]]
    return sum([
        ends[0].endswith(rima_a), ends[2].endswith(rima_a),
        ends[1].endswith(rima_b), ends[3].endswith(rima_b),
    ])


def rhyme_hit_rate(model, tokenizer, device, block_size, n=16, rima_a="ng", rima_b="ar"):
    """Metrik objektif: dari n sampel (prompt 'full'), berapa persen larik yang rimanya cocok."""
    scores = []
    for _ in range(n):
        text = generate_sample(model, tokenizer, device, EVAL_PROMPTS["full"], block_size=block_size)
        scores.append(check_rhyme(text, rima_a, rima_b))
    valid = [s for s in scores if s is not None]
    format_ok = len(valid) / n
    hit = sum(valid) / (4 * len(valid)) if valid else 0.0
    return format_ok, hit

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
    parser.add_argument("--rhyme_samples", type=int, default=16, help="Jumlah sampel untuk metrik rima saat eval")
    parser.add_argument("--mask_conditioning_loss", action="store_true",
                        help="Training: loss hanya dihitung pada isi pantun (header kondisi tidak ikut dihitung)")
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
        args.grad_accum_steps = 1  # Simplify logic for smoke test
        args.warmup_steps = 1
        args.rhyme_samples = 2

    set_seed(args.seed)

    # 1. Device configuration
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[Init] Using device: {device}")

    # 2. Paths
    train_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'train.txt')
    val_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'val.txt')
    ckpt_dir = os.path.join(BASE_DIR, '3.TRAINING', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vocab_path = os.path.join(ckpt_dir, 'vocab_char.json')

    for split_name, split_path in (("train", train_path), ("val", val_path)):
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"File data {split_name} tidak ditemukan: {split_path}\n"
                "Jalankan 1.DATASET/preprocessing.py lalu 1.DATASET/splitting.py terlebih dahulu."
            )

    # 3. Tokenizer: saat resume, WAJIB pakai vocab lama agar id karakter tidak bergeser.
    if args.resume and os.path.exists(vocab_path):
        tokenizer = PantunCharTokenizer.load(vocab_path)
        print(f"[Init] Vocab dimuat dari {vocab_path}")
    else:
        tokenizer = PantunCharTokenizer.build_from_files([train_path, val_path])
        tokenizer.save(vocab_path)

    cfg = ModelConfig.from_config_py() if os.path.exists(os.path.join(BASE_DIR, '2.MODEL', 'config.py')) else ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size

    # Guard: kalau block_size < panjang pantun terpanjang, pantun akan terpotong dan
    # diberi <|eos|> paksa di tengah larik -> model belajar berhenti di tempat yang salah.
    stats = analyze_context_length(train_path, tokenizer)
    if cfg.block_size < stats["max_len"]:
        print(f"\n[WARNING] block_size={cfg.block_size} < max_len={stats['max_len']}. "
              f"Sebagian pantun akan terpotong! Naikkan block_size ke >= {stats['recommended_align8']}.\n")

    # 4. Datasets
    #    - train_dataset  : conditioning dropout aktif (khusus untuk training)
    #    - train_eval/val : selalu mode 'full' + loss hanya pada isi pantun, sehingga
    #                       train loss dan val loss pada estimate_loss() BISA dibandingkan.
    train_dataset = PantunDataset(
        train_path, tokenizer, max_seq_len=cfg.block_size, is_training=True,
        mask_conditioning_loss=args.mask_conditioning_loss,
    )
    train_eval_dataset = PantunDataset(
        train_path, tokenizer, max_seq_len=cfg.block_size, is_training=False,
        mask_conditioning_loss=True,
    )
    val_dataset = PantunDataset(
        val_path, tokenizer, max_seq_len=cfg.block_size, is_training=False,
        mask_conditioning_loss=True,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(f"Dataset train kosong atau formatnya tidak valid: {train_path}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"Dataset val kosong atau formatnya tidak valid: {val_path}")

    print(f"[Init] Train samples: {len(train_dataset):,} ({train_path})")
    print(f"[Init] Val samples  : {len(val_dataset):,} ({val_path})")
    print(f"[Init] Vocab size   : {cfg.vocab_size} | Block size: {cfg.block_size}")
    collator = PantunCollator(pad_token_id=tokenizer.pad_token_id, ignore_index=-100)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    dataloaders = {"train": train_eval_loader, "val": val_loader}

    # Data Pipeline Unit Test (Point B)
    sample_batch = next(iter(train_loader))
    sx, sy = sample_batch["input_ids"], sample_batch["labels"]
    valid_mask = (sy[:, :-1] != -100) & (sx[:, 1:] != tokenizer.pad_token_id)
    assert torch.all(sy[:, :-1][valid_mask] == sx[:, 1:][valid_mask]), "Data Pipeline Causality Error: Targets are not shifted correctly!"

    # 5. Model & Optimizer
    model = MiniTransformer(cfg)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"[Init] Using {torch.cuda.device_count()} GPUs for DataParallel!")
        model = nn.DataParallel(model)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(lr=args.lr, weight_decay=args.weight_decay)

    # 6. Resume Checkpoint Logic (Point H)
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

    # 7. Training Loop
    print(f"[Init] Expected initial loss: ~{math.log(cfg.vocab_size):.2f}")
    train_iter = iter(train_loader)
    model.train()

    log_interval = 1 if args.smoke_test else 10
    tokens_window = 0
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        # Update LR
        lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Micro-batch loop
        loss_accum = 0.0
        for micro_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            tokens_window += x.numel()

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
            loss_accum += loss.item()  # sudah dibagi grad_accum_steps -> jumlahnya = rata-rata

        # Gradient Clipping & Optimizer Step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = tokens_window / (dt if dt > 0 else 0.001)
            tokens_window = 0
            print(f"Step {step:4d} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Norm: {grad_norm:.2f} | Tok/s: {tokens_per_sec:.0f}")

        # Evaluation & Checkpointing
        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            losses = estimate_loss(model, dataloaders, args.eval_iters, device, device_type)
            val_loss = losses['val']
            val_ppl = math.exp(min(val_loss, 20))  # Cap PPL for initial print safety
            print(f"\n--- EVAL STEP {step} ---")
            print(f"Train Loss: {losses['train']:.4f} | Val Loss: {val_loss:.4f} | Val PPL (per-char): {val_ppl:.2f}")

            for name, prompt in EVAL_PROMPTS.items():
                sample = generate_sample(model, tokenizer, device, prompt, block_size=cfg.block_size)
                print(f"[{name}]\n{sample}\n")

            format_ok, hit = rhyme_hit_rate(model, tokenizer, device, cfg.block_size, n=args.rhyme_samples)
            print(f"Rhyme check (full prompt, n={args.rhyme_samples}): format 4 larik = {format_ok:.0%} | rima cocok = {hit:.0%}")
            print("------------------------\n")

            # Perbarui best_val SEBELUM menyimpan last.pt agar resume tidak membawa nilai usang.
            is_best = val_loss < best_val_loss and not args.smoke_test
            if is_best:
                best_val_loss = val_loss

            raw_model.save_checkpoint(
                os.path.join(ckpt_dir, 'last.pt'),
                optimizer=optimizer.state_dict(),
                step=step,
                best_val=best_val_loss,
                val_loss=val_loss
            )

            if is_best:
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