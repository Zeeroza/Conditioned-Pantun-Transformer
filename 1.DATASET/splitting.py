import json
import random
import os

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_pantun(item):
    """
    Mengonversi dictionary menjadi string berformat token eksplisit
    sesuai spesifikasi Tahap 4.
    """
    tema = item['tema']
    rima_a = item['rima_a']
    rima_b = item['rima_b']
    
    # Menggabungkan keempat baris pantun dengan enter (newline)
    teks_baris = '\n'.join(item['larik'])
    
    # Merakit struktur akhir
    # Format: <|start|> [TEMA: jenaka] [RIMA: an, ri]\nbaris1...\n<|end|>
    formatted_text = f"<|start|> [TEMA: {tema}] [RIMA: {rima_a}, {rima_b}]\n{teks_baris}\n<|end|>"
    return formatted_text

def split_and_save(data, output_dir):
    # Mengacak data dengan seed agar konsisten (reproducible)
    random.seed(42)
    random.shuffle(data)
    
    total_data = len(data)
    train_end = int(0.8 * total_data)
    val_end = int(0.9 * total_data)
    
    # Melakukan slicing list 80 - 10 - 10
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Distribusi Split:")
    print(f"- Train : {len(train_data)} bait (80%)")
    print(f"- Val   : {len(val_data)} bait (10%)")
    print(f"- Test  : {len(test_data)} bait (10%)")
    
    # Fungsi internal untuk menyimpan list ke TXT
    def save_to_txt(dataset, filename):
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # Tambahkan double newline antar pantun agar file mudah dibaca manusia
                f.write(format_pantun(item) + '\n\n')
    
    # Menyimpan ke masing-masing file
    save_to_txt(train_data, 'train.txt')
    save_to_txt(val_data, 'val.txt')
    save_to_txt(test_data, 'test.txt')
    
    print(f"\nBerhasil menyimpan file train.txt, val.txt, dan test.txt di folder '{output_dir}'")

# ==========================================
# BLOK EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # Path merujuk ke file hasil output preprocessing.py sebelumnya
    input_file = '1.DATASET/processed_data/master_data.jsonl'
    output_directory = '1.DATASET/processed_data/'
    
    # Pastikan file input tersedia sebelum diproses
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan. Jalankan preprocessing.py terlebih dahulu.")
    else:
        # 1. Load Data
        semua_pantun = load_jsonl(input_file)
        
        # 2. Split dan Format Data
        split_and_save(semua_pantun, output_directory)