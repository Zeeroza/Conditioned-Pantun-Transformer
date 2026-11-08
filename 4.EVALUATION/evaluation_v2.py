"""evaluation.py - diselaraskan dengan train.py / tokenization.py"""
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("../2.MODEL")
sys.path.append("../1.DATASET")

from model import MiniTransformer
from tokenization import (
    PantunCustomTokenizer, PantunDataset, PantunCollator,
    load_pantun_file, extract_words, syllabify,
)

TOK_PATH = "../1.DATASET/pantun_syl_bpe.json"   # tokenizer yang SAMA dengan training
TRAIN_TXT = "../1.DATASET/processed_data/train.txt"
TEST_TXT = "../1.DATASET/processed_data/test.txt"
CKPT_DIR = "best-checkpoint"
CHECKPOINTS = ["best.pt"]
N_GEN = 200
SEED = 1337
MODES = {
    "greedy": dict(greedy=True),
    "sampling": dict(temperature=0.8, top_k=50),
}


# ---------- parsing & metrik ----------
def cut_and_parse(text):
    """Potong di <|end|>, ambil baris setelah <|isi|>, <|batas|> jadi newline (sama dgn train.py)."""
    text = text.split("<|end|>")[0]
    body = text.split("<|isi|>", 1)[-1].replace("<|batas|>", "\n")
    return [l.strip() for l in body.split("\n") if l.strip()]


def last_letter(line):
    letters = re.findall(r"[a-z]", line.lower())
    return letters[-1] if letters else ""


def rime_of(line):
    """Rime suku kata terakhir (vokal + koda), pakai syllabify yang sama dgn tokenizer."""
    words = extract_words(line)
    if not words:
        return ""
    syl = syllabify(words[-1])[-1]
    return re.sub(r"^[^aiueo]+", "", syl) or syl


def rime_pairs(lines):
    """(jumlah pasangan cocok 0-2, apakah ABAB penuh). Pasangan: (1,3) dan (2,4)."""
    if len(lines) < 4:
        return 0, False
    r = [rime_of(l) for l in lines[:4]]
    a = int(r[0] != "" and r[0] == r[2])
    b = int(r[1] != "" and r[1] == r[3])
    return a + b, bool(a and b)


def summarize(samples, train_lines, train_words):
    """samples: list of (lines, rima_a, rima_b)."""
    n = max(1, len(samples))
    c = dict(lab_hit=0, lab_tot=0, struct=0, pair=0, abab=0,
             syl_ok=0, lines=0, seen=0, words=0, oov=0)
    for lines, ra, rb in samples:
        c["struct"] += len(lines) == 4
        for l, tgt in zip(lines[:4], (ra, rb, ra, rb)):
            c["lab_tot"] += 1
            c["lab_hit"] += last_letter(l) == tgt
        p, full = rime_pairs(lines)
        c["pair"] += p
        c["abab"] += full
        for l in lines[:4]:
            s = sum(len(syllabify(w)) for w in extract_words(l))
            c["lines"] += 1
            c["syl_ok"] += 8 <= s <= 12
            c["seen"] += l in train_lines
        for w in extract_words(" ".join(lines)):
            c["words"] += 1
            c["oov"] += w not in train_words
    return {
        "label_rhyme": c["lab_hit"] / max(1, c["lab_tot"]),   # sama dgn metrik di train.py
        "pair_rime": c["pair"] / (2 * n),                     # rima asli (rime suku kata)
        "abab_rime": c["abab"] / n,                           # kedua pasangan cocok
        "struct_4line": c["struct"] / n,
        "syl_8_12": c["syl_ok"] / max(1, c["lines"]),
        "oov": c["oov"] / max(1, c["words"]),
        "seen_in_train": c["seen"] / max(1, c["lines"]),      # generate: copy/hafalan | REAL: duplikat train-test
    }


def fmt(m):
    return " | ".join(f"{k}={v:.3f}" for k, v in m.items())


# ---------- loss/PPL teacher-forcing (sama dgn val di train.py) ----------
@torch.no_grad()
def teacher_forced(model, tokenizer, device):
    ds = PantunDataset(TEST_TXT, tokenizer, max_seq_len=model.cfg.block_size, is_training=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=PantunCollator(tokenizer.pad_token_id))
    tot_loss, tot_tok = 0.0, 0
    for batch in dl:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits, _ = model(x)
        tot_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                    ignore_index=-100, reduction="sum").item()
        tot_tok += (y != -100).sum().item()
    loss = tot_loss / max(1, tot_tok)
    return loss, math.exp(min(loss, 20)), len(ds)


# ---------- generasi ----------
@torch.no_grad()
def generate_samples(model, tokenizer, items, device, gen_kwargs):
    gen_tokens = max(32, min(model.cfg.block_size - 16, 160))   # sama dgn train.py
    out = []
    for it in items:
        prefix = PantunDataset.build_prefix(it["rima_a"], it["rima_b"], "full")
        ids = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
        gen = model.generate(ids, max_new_tokens=gen_tokens, **gen_kwargs)
        out.append((cut_and_parse(tokenizer.decode(gen[0].tolist())), it["rima_a"], it["rima_b"]))
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(TOK_PATH):
        raise FileNotFoundError(f"{TOK_PATH} tidak ada. Pakai tokenizer yang dipakai training, jangan latih ulang di sini.")
    tokenizer = PantunCustomTokenizer()
    tokenizer.load(TOK_PATH)

    train_records = load_pantun_file(TRAIN_TXT, verbose=False)
    test_records = load_pantun_file(TEST_TXT)
    train_lines = {l for r in train_records for l in r["larik"]}
    train_words = {w for l in train_lines for w in extract_words(l)}
    items = test_records[:N_GEN]

    real = summarize([(r["larik"], r["rima_a"], r["rima_b"]) for r in items], train_lines, train_words)
    print("\n[REAL test pantun] = batas atas metrik (seen_in_train di sini = duplikat train/test)")
    print("  " + fmt(real))

    results = {"REAL": real}
    for name in CHECKPOINTS:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.exists(path):
            print(f"\n{name}: tidak ditemukan, dilewati")
            continue

        meta = torch.load(path, map_location="cpu")
        step, ckpt_val = meta.get("step"), meta.get("val_loss")
        del meta
        model = MiniTransformer.from_checkpoint(path, map_location=device).to(device).eval()
        assert model.cfg.vocab_size == tokenizer.vocab_size, (
            f"vocab model {model.cfg.vocab_size} != tokenizer {tokenizer.vocab_size}: tokenizer tidak cocok!")

        fp = sum(p.double().sum().item() for p in model.parameters())
        print(f"\n=== {name} | step={step} | val_loss di ckpt={ckpt_val} | fingerprint={fp:.6f}")

        loss, ppl, n = teacher_forced(model, tokenizer, device)
        print(f"  test loss={loss:.4f} | PPL={ppl:.2f} | n={n}")
        res = {"test_loss": loss, "test_ppl": ppl}

        for mode, kw in MODES.items():
            torch.manual_seed(SEED)
            samples = generate_samples(model, tokenizer, items, device, kw)
            res[mode] = summarize(samples, train_lines, train_words)
            print(f"  [{mode}] " + fmt(res[mode]))
            print("    contoh: " + " / ".join(samples[0][0]))
        results[name] = res

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nHasil disimpan ke eval_results.json")


if __name__ == "__main__":
    main()