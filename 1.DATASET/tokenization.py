import json
import math
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PantunBPETokenizer:
    """
    Wrapper untuk tiktoken BPE Tokenizer dengan penambahan custom special tokens
    khusus untuk kontrol struktural dan conditioning pantun.
    """
    def __init__(self, base_encoding: str = "cl100k_base"):
        base_enc = tiktoken.get_encoding(base_encoding)
        
        self.custom_special_tokens = [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|tema|>",
            "<|rima_a|>",
            "<|rima_b|>",
            "<|isi|>",
        ]
        
        base_vocab_size = base_enc.n_vocab
        print(f"[Info] Base encoding '{base_encoding}' memiliki vocab size: {base_vocab_size}")
        self.special_tokens_map: Dict[str, int] = {
            token: base_vocab_size + idx 
            for idx, token in enumerate(self.custom_special_tokens)
        }
        
        self.tokenizer = tiktoken.Encoding(
            name=f"{base_encoding}_pantun",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={**base_enc._special_tokens, **self.special_tokens_map}
        )
        
        self.pad_token_id = self.special_tokens_map["<|pad|>"]
        self.bos_token_id = self.special_tokens_map["<|bos|>"]
        self.eos_token_id = self.special_tokens_map["<|eos|>"]
        self.isi_token_id = self.special_tokens_map["<|isi|>"]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            special_ids = set(self.special_tokens_map.values())
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        return self.tokenizer.decode(token_ids)


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
        tokenizer: PantunBPETokenizer,
        max_seq_len: int = 128,
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
            try:
                isi_pos = (input_ids == self.tokenizer.isi_token_id).nonzero(as_tuple=True)[0].item()
                labels[:isi_pos] = -100
            except IndexError:
                pass

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
    tokenizer: PantunBPETokenizer,
    export_annotated_jsonl: Union[str, Path, None] = None
) -> Dict[str, Union[int, float]]:
    """
    Menganalisis distribusi panjang token dari dataset (menggunakan skenario terpanjang/Full Conditioning)
    untuk menentukan BLOCK_SIZE / CONTEXT_SIZE yang optimal pada Mini-Transformer.
    
    Args:
        txt_path: Path ke file dataset .txt hasil splitting.py.
        tokenizer: Instance PantunBPETokenizer.
        export_annotated_jsonl: (Opsional) Jika diisi path file baru, akan menyimpan dataset
                                yang sudah ditambahkan metadata 'token_length' untuk inspeksi.
    """
    lengths_full = []
    lengths_uncond = []
    annotated_records = []

    for obj in load_pantun_txt(txt_path):
        # Hitung panjang kondisi terpanjang (Full Conditioning) & terpendek (Polos)
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

    # Statistik Persentil
    min_len = int(np.min(arr_full))
    mean_len = float(np.mean(arr_full))
    p50 = int(np.percentile(arr_full, 50))
    p95 = int(np.percentile(arr_full, 95))
    p99 = int(np.percentile(arr_full, 99))
    p99_9 = int(np.percentile(arr_full, 99.9))
    max_len = int(np.max(arr_full))

    # Rekomendasi BLOCK_SIZE:
    # 1. Kelipatan 8 terdekat di atas P99.9 (Optimal untuk Tensor Cores / Mixed Precision FP16/BF16)
    rec_multiple_8 = int(math.ceil(max_len / 8.0) * 8)
    # 2. Pangkat 2 terdekat di atas Max Length (Standar arsitektur Transformer klasik: 64, 128, 256)
    rec_power_of_2 = 2 ** int(math.ceil(math.log2(max_len)))

    print("=" * 60)
    print("📊 LAPORAN DISTRIBUSI PANJANG TOKEN (BPE gpt2)")
    print("=" * 60)
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
    
    # Cetak Histogram ASCII sederhana
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
    DATASET_PATH = "1.DATASET/processed_data/train.txt"

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}. "
            "Jalankan 1.DATASET/preprocessing.py lalu 1.DATASET/splitting.py terlebih dahulu."
        )

    tokenizer = PantunBPETokenizer(base_encoding="gpt2")
    
    # 1. Jalankan profiler untuk mendapatkan angka BLOCK_SIZE
    stats = analyze_context_length(
        txt_path=DATASET_PATH,
        tokenizer=tokenizer,
        export_annotated_jsonl=None  # Isi "dataset_annotated.jsonl" jika ingin menyimpan panjang token ke file baru
    )

    # 2. Gunakan hasil rekomendasi secara otomatis ke dalam Dataset
    OPTIMAL_BLOCK_SIZE = stats["recommended_pow2"]
    
    train_dataset = PantunDataset(
        txt_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_seq_len=OPTIMAL_BLOCK_SIZE,
        is_training=True
    )