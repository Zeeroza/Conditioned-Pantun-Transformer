import json
import math
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PantunCharTokenizer:
    """
    Tokenizer level karakter untuk dataset pantun.

    - Setiap karakter (huruf, spasi, newline, tanda baca) = 1 token.
    - Special token (<|bos|>, <|tema|>, dst.) tetap atomik: 1 token utuh, tidak dipecah per karakter.
    - Karakter yang tidak ada di vocab dipetakan ke <|unk|>.
    - Vocab dibangun dari korpus lalu disimpan ke JSON agar train / val / inference
      memakai mapping id yang identik.
    """
    SPECIAL_TOKENS = [
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|unk|>",
        "<|tema|>",
        "<|rima_a|>",
        "<|rima_b|>",
        "<|isi|>",
    ]

    def __init__(self, chars: Iterable[str], lowercase: bool = False):
        self.lowercase = lowercase
        self.chars: List[str] = sorted(set(chars))

        # Special token di id paling awal (pad = 0), lalu karakter.
        self.itos: List[str] = list(self.SPECIAL_TOKENS) + self.chars
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.special_tokens_map: Dict[str, int] = {t: self.stoi[t] for t in self.SPECIAL_TOKENS}

        self.pad_token_id = self.stoi["<|pad|>"]
        self.bos_token_id = self.stoi["<|bos|>"]
        self.eos_token_id = self.stoi["<|eos|>"]
        self.unk_token_id = self.stoi["<|unk|>"]
        self.isi_token_id = self.stoi["<|isi|>"]

        # Regex untuk memisahkan special token dari teks biasa (group capture -> token ikut tersimpan)
        self._special_re = re.compile(
            "(" + "|".join(re.escape(t) for t in self.SPECIAL_TOKENS) + ")"
        )

    # ------------------------------------------------------------------
    # Pembangunan vocab
    # ------------------------------------------------------------------
    @classmethod
    def build_from_files(
        cls,
        txt_paths: Union[str, Path, List[Union[str, Path]]],
        lowercase: bool = False,
    ) -> "PantunCharTokenizer":
        """Kumpulkan seluruh karakter unik dari field tema, rima, dan larik pada file dataset."""
        if isinstance(txt_paths, (str, Path)):
            txt_paths = [txt_paths]

        charset = set()
        for p in txt_paths:
            for rec in load_pantun_txt(p):
                parts = [rec["tema"], rec["rima_a"], rec["rima_b"], "\n".join(rec["larik"])]
                for part in parts:
                    charset.update(part.lower() if lowercase else part)

        tok = cls(charset, lowercase=lowercase)
        print(f"[Info] Vocab char-level terbentuk: {tok.vocab_size} token "
              f"({len(tok.SPECIAL_TOKENS)} special + {len(tok.chars)} karakter)")
        return tok

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars, "lowercase": self.lowercase}, f, ensure_ascii=False, indent=2)
        print(f"[Info] Vocab disimpan ke: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PantunCharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"], lowercase=data.get("lowercase", False))

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self.special_tokens_map:
                ids.append(self.special_tokens_map[piece])
            else:
                if self.lowercase:
                    piece = piece.lower()
                ids.extend(self.stoi.get(ch, self.unk_token_id) for ch in piece)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        special_ids = set(self.special_tokens_map.values())
        out = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            out.append(self.itos[tid])
        return "".join(out)


def load_pantun_txt(path: Union[str, Path]) -> List[Dict]:
    """
    Membaca dataset berformat TXT hasil splitting.py:

        <|start|> [TEMA: umum] [RIMA: ng, ar]
        larik 1
        larik 2
        larik 3
        larik 4
        <|end|>

    Blok yang tidak lengkap (header rusak atau bukan 4 larik) dilewati.
    """
    header_re = re.compile(
        r"\[TEMA:\s*(?P<tema>.*?)\]\s*\[RIMA:\s*(?P<rima_a>.*?)\s*,\s*(?P<rima_b>.*?)\]"
    )
    records = []
    current = None

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("<|start|>"):
                header = line[len("<|start|>"):].strip()
                match = header_re.search(header)
                if not match:
                    print(f"[Warning] Header tidak dikenali di baris {line_num}, blok dilewati: {header}")
                    current = None
                    continue
                current = {
                    "tema": match.group("tema").strip(),
                    "rima_a": match.group("rima_a").strip(),
                    "rima_b": match.group("rima_b").strip(),
                    "larik": [],
                    "_line": line_num,
                }
            elif line.startswith("<|end|>"):
                if current is not None:
                    if len(current["larik"]) == 4:
                        del current["_line"]
                        records.append(current)
                    else:
                        print(
                            f"[Warning] Blok mulai baris {current['_line']} memiliki "
                            f"{len(current['larik'])} larik (harus 4), dilewati."
                        )
                current = None
            elif current is not None:
                current["larik"].append(line)

    return records


class PantunDataset(Dataset):
    """
    Custom PyTorch Dataset dengan Dynamic Conditioning Dropout (70% Lengkap, 20% Tanpa Rima, 10% Polos).

    Hanya membaca dataset TXT hasil splitting.py (train.txt / val.txt).
    """
    def __init__(
        self,
        txt_path: Union[str, Path],
        tokenizer: PantunCharTokenizer,
        max_seq_len: int = 256,
        is_training: bool = True,
        dropout_probs: Tuple[float, float, float] = (0.70, 0.20, 0.10),
        mask_conditioning_loss: bool = False
    ):
        super().__init__()
        assert abs(sum(dropout_probs) - 1.0) < 1e-6, "Total probabilitas harus 1.0"
        assert Path(txt_path).suffix.lower() == ".txt", f"Dataset harus berformat .txt, bukan: {txt_path}"

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        self.dropout_probs = dropout_probs
        self.mask_conditioning_loss = mask_conditioning_loss
        self.data: List[Dict] = load_pantun_txt(txt_path)

    @staticmethod
    def build_prompt_string(item: Dict, mode: str = "full") -> str:
        """Helper statis untuk menyusun string sekuens berdasarkan mode conditioning."""
        tema = item["tema"].strip()
        rima_a = item["rima_a"].strip()
        rima_b = item["rima_b"].strip()
        larik_str = "\n".join(line.strip() for line in item["larik"])

        if mode == "full":
            prefix = f"<|bos|><|tema|>{tema}<|rima_a|>{rima_a}<|rima_b|>{rima_b}<|isi|>"
        elif mode == "no_rhyme":
            prefix = f"<|bos|><|tema|>{tema}<|isi|>"
        else:
            prefix = "<|bos|><|isi|>"

        return f"{prefix}{larik_str}<|eos|>"

    def _format_sequence(self, item: Dict) -> str:
        if not self.is_training:
            mode = "full"
        else:
            rand_val = random.random()
            p_full, p_no_rhyme, _ = self.dropout_probs
            if rand_val < p_full:
                mode = "full"
            elif rand_val < (p_full + p_no_rhyme):
                mode = "no_rhyme"
            else:
                mode = "unconditional"

        return self.build_prompt_string(item, mode=mode)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        formatted_text = self._format_sequence(item)
        token_ids = self.tokenizer.encode(formatted_text)

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        if self.mask_conditioning_loss:
            isi_positions = (input_ids == self.tokenizer.isi_token_id).nonzero(as_tuple=True)[0]
            if len(isi_positions) > 0:
                labels[:isi_positions[0].item()] = -100

        return {"input_ids": input_ids, "labels": labels}


class PantunCollator:
    def __init__(self, pad_token_id: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        padded_inputs = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_index)
        attention_mask = (padded_inputs != self.pad_token_id).long()

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_mask
        }


# =====================================================================
# Fungsi Profiler Distribusi Token & Penentu BLOCK_SIZE
# =====================================================================
def analyze_context_length(
    txt_path: Union[str, Path],
    tokenizer: PantunCharTokenizer,
    export_annotated_jsonl: Union[str, Path, None] = None
) -> Dict[str, Union[int, float]]:
    """
    Menganalisis distribusi panjang token (skenario terpanjang = Full Conditioning)
    untuk menentukan BLOCK_SIZE / CONTEXT_SIZE yang optimal pada Mini-Transformer.
    """
    lengths_full = []
    lengths_uncond = []
    annotated_records = []

    for obj in load_pantun_txt(txt_path):
        str_full = PantunDataset.build_prompt_string(obj, mode="full")
        str_uncond = PantunDataset.build_prompt_string(obj, mode="unconditional")

        len_full = len(tokenizer.encode(str_full))
        len_uncond = len(tokenizer.encode(str_uncond))

        lengths_full.append(len_full)
        lengths_uncond.append(len_uncond)

        if export_annotated_jsonl:
            obj["token_len_full"] = len_full
            obj["token_len_uncond"] = len_uncond
            annotated_records.append(obj)

    arr_full = np.array(lengths_full)
    arr_uncond = np.array(lengths_uncond)

    min_len = int(np.min(arr_full))
    mean_len = float(np.mean(arr_full))
    p50 = int(np.percentile(arr_full, 50))
    p95 = int(np.percentile(arr_full, 95))
    p99 = int(np.percentile(arr_full, 99))
    p99_9 = int(np.percentile(arr_full, 99.9))
    max_len = int(np.max(arr_full))

    # Rekomendasi BLOCK_SIZE (berbasis Max Length -> zero truncation):
    # 1. Kelipatan 8 terdekat di atas Max (ramah Tensor Cores FP16/BF16)
    rec_multiple_8 = int(math.ceil(max_len / 8.0) * 8)
    # 2. Pangkat 2 terdekat di atas Max (64, 128, 256, ...)
    rec_power_of_2 = 2 ** int(math.ceil(math.log2(max_len)))

    print("=" * 60)
    print("📊 LAPORAN DISTRIBUSI PANJANG TOKEN (CHAR-LEVEL)")
    print("=" * 60)
    print(f"Vocab Size                 : {tokenizer.vocab_size}")
    print(f"Total Sampel Pantun        : {len(arr_full):,} baris")
    print(f"Rata-rata Panjang (Polos)  : {np.mean(arr_uncond):.1f} token")
    print(f"Rata-rata Panjang (Lengkap): {mean_len:.1f} token")
    print("-" * 60)
    print("Distribusi Persentil (Full Conditioning):")
    print(f"  • Min (Terpendek)        : {min_len} token")
    print(f"  • P50 (Median)           : {p50} token")
    print(f"  • P95                    : {p95} token")
    print(f"  • P99                    : {p99} token")
    print(f"  • P99.9                  : {p99_9} token")
    print(f"  • Max (Terpanjang)       : {max_len} token")
    print("-" * 60)

    counts, bin_edges = np.histogram(arr_full, bins=6)
    print("Histogram Sebaran Panjang Token:")
    max_count = max(counts) if len(counts) > 0 else 1
    for i in range(len(counts)):
        bar_len = int((counts[i] / max_count) * 30)
        bar = "█" * bar_len
        print(f"  [{int(bin_edges[i]):3d} - {int(bin_edges[i+1]):3d} token] : {bar} ({counts[i]:,})")

    print("-" * 60)
    print("🎯 REKOMENDASI BLOCK_SIZE / CONTEXT_SIZE MINI-TRANSFORMER:")
    print(f"  • Opsi Strict (Zero-Truncation, Align-8) : {rec_multiple_8}")
    print(f"  • Opsi Standar (Power-of-2)              : {rec_power_of_2}")
    print("=" * 60)

    if export_annotated_jsonl:
        with open(export_annotated_jsonl, "w", encoding="utf-8") as out_f:
            for rec in annotated_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Info] File dataset dengan anotasi panjang token disimpan ke: {export_annotated_jsonl}\n")

    return {
        "max_len": max_len,
        "p99": p99,
        "recommended_align8": rec_multiple_8,
        "recommended_pow2": rec_power_of_2
    }


if __name__ == "__main__":
    DATA_DIR = Path("processed_data")
    TRAIN_PATH = DATA_DIR / "train.txt"
    VAL_PATH = DATA_DIR / "val.txt"
    VOCAB_PATH = DATA_DIR / "vocab_char.json"

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {TRAIN_PATH}. "
            "Jalankan 1.DATASET/preprocessing.py lalu 1.DATASET/splitting.py terlebih dahulu."
        )

    # 1. Bangun vocab dari train + val (hanya himpunan karakter, tidak ada kebocoran label),
    #    lalu simpan agar training, eval, dan generate memakai mapping yang sama.
    source_files = [TRAIN_PATH] + ([VAL_PATH] if VAL_PATH.exists() else [])
    tokenizer = PantunCharTokenizer.build_from_files(source_files, lowercase=False)
    tokenizer.save(VOCAB_PATH)

    # 2. Profiler untuk menentukan BLOCK_SIZE
    stats = analyze_context_length(
        txt_path=TRAIN_PATH,
        tokenizer=tokenizer,
        export_annotated_jsonl=None
    )

    # 3. Pakai rekomendasi ke dalam Dataset
    OPTIMAL_BLOCK_SIZE = stats["recommended_pow2"]

    train_dataset = PantunDataset(
        txt_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_seq_len=OPTIMAL_BLOCK_SIZE,
        is_training=True
    )

    # Sanity check: round-trip encode -> decode harus identik
    sample = PantunDataset.build_prompt_string(train_dataset.data[0], mode="full")
    ids = tokenizer.encode(sample)
    assert tokenizer.decode(ids) == sample, "Round-trip encode/decode tidak identik!"
    print(f"[OK] Round-trip lolos. Contoh ({len(ids)} token):\n{sample}")