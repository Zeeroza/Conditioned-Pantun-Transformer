import os
import torch
import math
import sys
import re
import matplotlib.pyplot as plt

sys.path.append('../2.MODEL')
sys.path.append('../1.DATASET')

from model import MiniTransformer

# --- PATCHING ERROR CLASS DI LUAR FOLDER ---
import tokenization
if not (hasattr(tokenization.PantunCustomTokenizer, 'vocab_size') and isinstance(tokenization.PantunCustomTokenizer.vocab_size, property)):
    pass
else:
    original_property = tokenization.PantunCustomTokenizer.vocab_size
    tokenization.PantunCustomTokenizer.vocab_size = property(
        fget=original_property.fget,
        fset=lambda self, val: setattr(self, '_vocab_size', val)
    )
from tokenization import PantunCustomTokenizer
# --------------------------------------------------

def count_syllables(line: str) -> int:
    """Estimasi jumlah suku kata dalam bahasa Indonesia berdasarkan kelompok vokal."""
    line = line.lower().strip()
    # Mencari kelompok vokal (a, i, u, e, o)
    vowels = re.findall(r'[aiueoéèə]+', line)
    return len(vowels)

def check_rhyme(lines: list) -> bool:
    """Mengecek apakah rima akhir baris mengikuti pola a-b-a-b secara sederhana."""
    if len(lines) < 4:
        return False
    
    endings = [re.sub(r'[^a-z]', '', l.lower())[-2:] if len(re.sub(r'[^a-z]', '', l.lower())) >= 2 else "" for l in lines[:4]]
    
    match_a = endings[0] == endings[2] and endings[0] != ""
    match_b = endings[1] == endings[3] and endings[1] != ""
    
    return match_a and match_b

def evaluate_all_checkpoints():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}\n")
    
    # 1. Muat Tokenizer
    tokenizer_path = "pantun_tokenizer.json"
    train_txt_path = "../1.DATASET/processed_data/train.txt"
    
    try:
        tokenizer = PantunCustomTokenizer(vocab_size=4000)
    except TypeError:
        tokenizer = PantunCustomTokenizer()
    
    if not os.path.exists(tokenizer_path):
        if os.path.exists(train_txt_path):
            tokenizer.train_from_txt(train_txt_path, tokenizer_path)
        else:
            print(f"[Error] File training {train_txt_path} tidak ditemukan!")
            return
    else:
        tokenizer.load(tokenizer_path)

    # 2. Muat Test Set
    test_file = "../1.DATASET/processed_data/test.txt"
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = f.readlines()
    except FileNotFoundError:
        print(f"File {test_file} tidak ditemukan.")
        return
        
    loss_fn = torch.nn.CrossEntropyLoss()
    
    checkpoints = ["best13.pt", "best14.pt", "best15.pt"] 
    checkpoint_dir = "best-checkpoint"
    
    results_summary = {
        "checkpoint": [],
        "perplexity": [],
        "rhyme_accuracy": [],
        "valid_line_count": [],
        "avg_syllable_error": []
    }
    
    # 3. Iterasi setiap checkpoint
    for ckpt_name in checkpoints:
        print(f"\n=== Mengevaluasi Checkpoint: {ckpt_name} ===")
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"  -> Checkpoint {ckpt_path} tidak ditemukan, melewati...")
            continue

        try:
            model = MiniTransformer.from_checkpoint(ckpt_path, map_location=device).to(device)
        except Exception as e:
            print(f"  -> Gagal memuat model dari checkpoint {ckpt_name}: {e}")
            continue
            
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        # Variabel untuk metrik generasi
        correct_rhymes = 0
        valid_lines_count = 0
        total_syllable_errors = 0
        evaluated_samples = 0
        
        with torch.no_grad():
            for idx, pantun in enumerate(test_data):
                if not pantun.strip():
                    continue
                    
                token_ids = tokenizer.encode(pantun)
                if len(token_ids) < 2:
                    continue
                    
                # A. Hitung Loss & Perplexity (Teacher Forcing)
                input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
                targets = torch.tensor(token_ids[1:], dtype=torch.long).unsqueeze(0).to(device)

                outputs, _ = model(input_ids)
                loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                total_loss += loss.item() * targets.size(1)
                total_tokens += targets.size(1)
                
                # B. Evaluasi Kualitatif/Generasi (Ambil sampel subset agar cepat, misal 50 data pertama)
                if evaluated_samples < 50:
                    # Ambil beberapa token awal sebagai prompt (misal 10 token pertama)
                    prompt_len = min(10, len(token_ids) // 2)
                    prompt_tensor = torch.tensor([token_ids[:prompt_len]], dtype=torch.long).to(device)
                    
                    # Generate token baru
                    generated_ids = model.generate(prompt_tensor, max_new_tokens=40, temperature=0.8, top_k=50)
                    decoded_text = tokenizer.decode(generated_ids[0].tolist())
                    
                    # Pisahkan baris hasil generasi
                    lines = [l.strip() for l in decoded_text.split('\n') if l.strip()]
                    
                    # 1. Line Count Verification (Struktur Bait: idealnya 4 baris inti)
                    if len(lines) >= 4:
                        valid_lines_count += 1
                        
                    # 2. Rhyme Accuracy (Skor Rima)
                    if check_rhyme(lines):
                        correct_rhymes += 1
                        
                    # 3. Syllable Count Error (Target ideal 8-12 suku kata per baris)
                    line_errors = 0
                    checked_lines = lines[:4] if len(lines) >= 4 else lines
                    for l in checked_lines:
                        syl = count_syllables(l)
                        # Hitung deviasi dari rentang ideal 8-10 suku kata
                        if syl < 6 or syl > 14:
                            line_errors += 1
                    total_syllable_errors += line_errors
                    
                    eval_sample_increment = 1
                    evaluated_samples += eval_sample_increment

        # Kalkulasi Metrik Akhir
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
        
        rhyme_acc = (correct_rhymes / evaluated_samples) * 100 if evaluated_samples > 0 else 0
        line_struct_acc = (valid_lines_count / evaluated_samples) * 100 if evaluated_samples > 0 else 0
        avg_syl_err = (total_syllable_errors / (evaluated_samples * 4)) if evaluated_samples > 0 else 0
        
        print(f"  > Rata-rata Loss        : {avg_loss:.4f}")
        print(f"  > Perplexity            : {perplexity:.4f}")
        print(f"  > Rhyme Accuracy (A-B)  : {rhyme_acc:.2f}%")
        print(f"  > Line Count Validation : {line_struct_acc:.2f}%")
        print(f"  > Avg Syllable Error    : {avg_syl_err:.2f} deviasi/baris")
        
        # Simpan untuk plotting
        results_summary["checkpoint"].append(ckpt_name)
        results_summary["perplexity"].append(perplexity)
        results_summary["rhyme_accuracy"].append(rhyme_acc)
        results_summary["valid_line_count"].append(line_struct_acc)
        results_summary["avg_syllable_error"].append(avg_syl_err)

    # 4. Membuat Plot Visualisasi Grafik
    if len(results_summary["checkpoint"]) > 0:
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Perplexity (Semakin kecil semakin baik)
        plt.subplot(1, 3, 1)
        plt.bar(results_summary["checkpoint"], results_summary["perplexity"], color='skyblue')
        plt.title("Perplexity (Lower is Better)")
        plt.xlabel("Checkpoint")
        plt.ylabel("Perplexity")
        
        # Subplot 2: Rhyme & Line Accuracy (Semakin tinggi semakin baik)
        plt.subplot(1, 3, 2)
        x = range(len(results_summary["checkpoint"]))
        width = 0.35
        plt.bar([p - width/2 for p in x], results_summary["rhyme_accuracy"], width=width, label='Rhyme Acc (%)', color='orange')
        plt.bar([p + width/2 for p in x], results_summary["valid_line_count"], width=width, label='Line Struct Acc (%)', color='green')
        plt.title("Structural Accuracy (%)")
        plt.xlabel("Checkpoint")
        plt.xticks(x, results_summary["checkpoint"])
        plt.legend()
        
        # Subplot 3: Syllable Error (Semakin kecil semakin baik)
        plt.subplot(1, 3, 3)
        plt.bar(results_summary["checkpoint"], results_summary["avg_syllable_error"], color='salmon')
        plt.title("Syllable Deviation (Lower is Better)")
        plt.xlabel("Checkpoint")
        plt.ylabel("Avg Deviation")
        
        plt.tight_layout()
        plot_path = "evaluation_metrics_plot.png"
        plt.savefig(plot_path)
        print(f"\n[Info] Grafik visualisasi berhasil disimpan ke file '{plot_path}'!")
        
        plt.show()

if __name__ == "__main__":
    evaluate_all_checkpoints()