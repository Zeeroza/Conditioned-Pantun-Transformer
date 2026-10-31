import json
import math
import random
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Menggunakan library tokenizers dari HuggingFace untuk Custom BPE
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


class PantunCustomTokenizer:
    """
    Tokenizer BPE khusus bahasa Indonesia yang dilatih dari awal (from scratch)
    menggunakan korpus pantun agar suku kata dan rima tidak rusak.
    """
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.special_tokens = [
            "<|unk|>",
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|tema|>",
            "<|rima_a|>",
            "<|rima_b|>",
            "<|isi|>",
            "<|batas|>" # Token pemisah sampiran dan isi
        ]
        self.tokenizer = None
        self.pad_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.isi_token_id = None

    def train_from_txt(self, txt_path: Union[str, Path], save_path: str = "pantun_tokenizer.json"):
        """Melatih tokenizer HANYA dari file train.txt (mencegah data leakage)"""
        self.tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        def get_training_corpus():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            blocks = content.split('\n\n')
            for block in blocks:
                lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
                if len(lines) >= 6:
                    header = lines[0]
                    # Ekstrak tema dan rima agar tokenizer juga mengenali karakternya
                    tema_match = re.search(r'\[TEMA:\s*(.*?)\]', header)
                    rima_match = re.search(r'\[RIMA:\s*(.*?),\s*(.*?)\]', header)
                    
                    if tema_match: yield tema_match.group(1)
                    if rima_match: 
                        yield rima_match.group(1)
                        yield rima_match.group(2)
                        
                    # Yield teks murni pantun (4 baris)
                    yield "\n".join(lines[1:5])
                        
        print(f"[Info] Memulai proses training BPE Tokenizer murni dari {txt_path}...")
        self.tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        self.tokenizer.save(save_path)
        self._set_ids()
        print(f"[Info] Tokenizer berhasil dilatih dan disimpan ke '{save_path}'")

    def load(self, path: str):
        """Memuat tokenizer yang sudah dilatih sebelumnya"""
        self.tokenizer = Tokenizer.from_file(path)
        self._set_ids()
        
    def _set_ids(self):
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        self.bos_token_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|eos|>")
        self.isi_token_id = self.tokenizer.token_to_id("<|isi|>")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class PantunDataset(Dataset):
    """
    Custom PyTorch Dataset untuk membaca format TXT hasil splitting,
    dilengkapi Dynamic Conditioning Dropout dan Loss Masking Aktif.
    """
    def __init__(
        self,
        txt_path: Union[str, Path],
        tokenizer: PantunCustomTokenizer,
        max_seq_len: int = 128,
        is_training: bool = True,
        dropout_probs: Tuple[float, float, float] = (0.70, 0.20, 0.10),
        mask_conditioning_loss: bool = True
    ):
        super().__init__()
        assert abs(sum(dropout_probs) - 1.0) < 1e-6, "Total probabilitas harus 1.0"
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_training = is_training
        self.dropout_probs = dropout_probs
        self.mask_conditioning_loss = mask_conditioning_loss
        self.data: List[Dict] = self._load_txt(txt_path)

    def _load_txt(self, path: Union[str, Path]) -> List[Dict]:
        """Membaca train.txt/val.txt dan mengekstrak tag untuk dynamic dropout"""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            
            if len(lines) >= 6:
                header = lines[0]
                
                tema_match = re.search(r'\[TEMA:\s*(.*?)\]', header)
                rima_match = re.search(r'\[RIMA:\s*(.*?),\s*(.*?)\]', header)
                
                if tema_match and rima_match:
                    records.append({
                        "tema": tema_match.group(1),
                        "rima_a": rima_match.group(1),
                        "rima_b": rima_match.group(2),
                        "larik": lines[1:5]
                    })
        return records

    @staticmethod
    def build_prompt_string(item: Dict, mode: str = "full") -> str:
        tema = item["tema"].strip()
        rima_a = item["rima_a"].strip()
        rima_b = item["rima_b"].strip()
        
        sampiran = "\n".join(item["larik"][:2]).strip()
        isi = "\n".join(item["larik"][2:]).strip()
        
        larik_str = f"{sampiran}\n<|batas|>\n{isi}"

        if mode == "full":
            prefix = f"<|bos|><|tema|>{tema}<|rima_a|>{rima_a}<|rima_b|>{rima_b}<|isi|>"
        elif mode == "no_rhyme":
            prefix = f"<|bos|><|tema|>{tema}<|isi|>"
        else:
            prefix = "<|bos|><|isi|>"

        return f"{prefix}\n{larik_str}\n<|eos|>"

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


def analyze_context_length(
    txt_path: Union[str, Path],
    tokenizer: PantunCustomTokenizer
) -> Dict[str, Union[int, float]]:
    """Profiler sederhana untuk mendapatkan angka BLOCK_SIZE dari TXT"""
    lengths_full = []
    
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    blocks = content.split('\n\n')
    for block in blocks:
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) >= 6:
            header = lines[0]
            tema_match = re.search(r'\[TEMA:\s*(.*?)\]', header)
            rima_match = re.search(r'\[RIMA:\s*(.*?),\s*(.*?)\]', header)
            
            if tema_match and rima_match:
                obj = {
                    "tema": tema_match.group(1),
                    "rima_a": rima_match.group(1),
                    "rima_b": rima_match.group(2),
                    "larik": lines[1:5]
                }
                str_full = PantunDataset.build_prompt_string(obj, mode="full")
                len_full = len(tokenizer.encode(str_full))
                lengths_full.append(len_full)

    arr_full = np.array(lengths_full)
    max_len = int(np.max(arr_full))
    rec_power_of_2 = 2 ** int(math.ceil(math.log2(max_len)))

    print("=" * 60)
    print("📊 LAPORAN DISTRIBUSI PANJANG TOKEN (Custom BPE)")
    print("=" * 60)
    print(f"Total Sampel Pantun        : {len(arr_full):,} baris")
    print(f"Max (Terpanjang)           : {max_len} token")
    print(f"Rekomendasi BLOCK_SIZE     : {rec_power_of_2} token")
    print("=" * 60)

    return {"recommended_pow2": rec_power_of_2}


# =====================================================================
# BLOK EKSEKUSI UTAMA (Bebas dari Data Leakage)
# =====================================================================
if __name__ == "__main__":
    # Pastikan path ini sesuai dengan struktur direktori kamu
    TRAIN_TXT_PATH = "1.DATASET/processed_data/train.txt" 
    TOKENIZER_PATH = "pantun_tokenizer.json"
    
    # Inisialisasi Tokenizer dengan Vocab Size yang sudah dikecilkan menjadi 4000
    # (Diperkecil untuk meminimalisir underfitting dan output tidak koheren)
    tokenizer = PantunCustomTokenizer(vocab_size=4000)
    
    # 1. Melatih Tokenizer murni hanya dari TRAIN_TXT
    if not Path(TOKENIZER_PATH).exists():
        if Path(TRAIN_TXT_PATH).exists():
            tokenizer.train_from_txt(TRAIN_TXT_PATH, TOKENIZER_PATH)
        else:
            print(f"[Error] File {TRAIN_TXT_PATH} tidak ditemukan.")
            exit()
    else:
        print("[Info] Memuat tokenizer yang sudah ada...")
        tokenizer.load(TOKENIZER_PATH)
    
    # 2. Profiling untuk menentukan panjang konteks (BLOCK_SIZE)
    if Path(TRAIN_TXT_PATH).exists():
        stats = analyze_context_length(TRAIN_TXT_PATH, tokenizer)
        OPTIMAL_BLOCK_SIZE = stats["recommended_pow2"]
        
        # 3. Load Dataset dari train.txt
        train_dataset = PantunDataset(
            txt_path=TRAIN_TXT_PATH,
            tokenizer=tokenizer,
            max_seq_len=OPTIMAL_BLOCK_SIZE,
            is_training=True
        )
        
        collator = PantunCollator(pad_token_id=tokenizer.pad_token_id)
        dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collator)
        
        print(f"\n[Berhasil] Tokenizer dilatih tanpa kebocoran data.")
        print(f"DataLoader dimuat dari {TRAIN_TXT_PATH} dengan sampel: {len(train_dataset)}")