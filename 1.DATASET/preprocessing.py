import pandas as pd
import string
import json

def ingest_and_standardize(file_txt, file_csv_2, file_csv_3):
    all_pantun = []
    
    # ---------------------------------------------------------
    # 1. Proses Dataset 1 (TXT)
    # Format: <BOS> baris1 <CLS> baris2 <CLS> baris3 <CLS> baris4 <EOS>
    # ---------------------------------------------------------
    try:
        with open(file_txt, 'r', encoding='utf-8') as f:
            for line in f:
                # Hilangkan tag BOS dan EOS
                clean_line = line.replace('<BOS>', '').replace('<EOS>', '').strip()
                # Pisahkan berdasarkan <CLS> dan bersihkan spasi
                larik = [l.strip() for l in clean_line.split('<CLS>') if l.strip()]
                
                # Pastikan jumlah baris tepat 4
                if len(larik) == 4:
                    all_pantun.append({
                        "tema": "umum", # Dataset 1 tidak punya tema, diset default
                        "larik": larik
                    })
    except Exception as e:
        print(f"Gagal memproses Dataset 1: {e}")

    # ---------------------------------------------------------
    # 2. Proses Dataset 2 (CSV)
    # Kolom relevan: text_pantun (dipisah dengan \n)
    # ---------------------------------------------------------
    try:
        df2 = pd.read_csv(file_csv_2)
        for text in df2['text_pantun'].dropna():
            # Pisahkan dengan \n dan abaikan baris yang kosong (menghindari error krn \n di akhir)
            text = text.replace('\\n', '\n') 
            larik = [l.strip() for l in text.split('\n') if l.strip()]
            
            if len(larik) == 4:
                all_pantun.append({
                    "tema": "umum", # Dataset 2 tidak punya tema eksplisit, diset default
                    "larik": larik
                })
    except Exception as e:
        print(f"Gagal memproses Dataset 2: {e}")

    # ---------------------------------------------------------
    # 3. Proses Dataset 3 (CSV)
    # Kolom relevan: teks (dipisah dengan \n) & tipe (sebagai tema)
    # ---------------------------------------------------------
    try:
        df3 = pd.read_csv(file_csv_3)
        for _, row in df3.iterrows():
            text = str(row['teks'])
            tipe = str(row['tipe'])
            
            # Pisahkan dengan \n dan abaikan baris yang kosong (menghindari error krn \n di akhir)
            text = text.replace('\\n', '\n')
            larik = [l.strip() for l in text.split('\n') if l.strip()]
            
            if len(larik) == 4:
                # Bersihkan tema (contoh: "Pantun Bijak" -> "bijak")
                clean_tema = tipe.lower().replace("pantun ", "").strip()
                all_pantun.append({
                    "tema": clean_tema,
                    "larik": larik
                })
    except Exception as e:
        print(f"Gagal memproses Dataset 3: {e}")

    return all_pantun

def get_suffix(line):
    # Buang tanda baca (koma, titik, dll) di akhir kalimat
    clean_line = line.translate(str.maketrans('', '', string.punctuation)).strip()
    # Ambil 2 huruf terakhir
    return clean_line[-2:].lower() if len(clean_line) >= 2 else clean_line.lower()

def filter_rhyme_and_save(all_pantun, output_path):
    valid_pantun = []
    
    for pantun in all_pantun:
        larik = pantun["larik"]
        
        # Ekstrak 2 huruf terakhir dari keempat baris
        suf1 = get_suffix(larik[0])
        suf2 = get_suffix(larik[1])
        suf3 = get_suffix(larik[2])
        suf4 = get_suffix(larik[3])
        
        # Cek apakah pola rima akhir menyilang a-b-a-b
        if suf1 == suf3 and suf2 == suf4:
            # Perekaman metadata Tahap 2
            pantun_bersih = {
                "tema": pantun["tema"],
                "rima_a": suf1,
                "rima_b": suf2,
                "larik": larik
            }
            valid_pantun.append(pantun_bersih)
            
    # Tulis hasil akhirnya menjadi format JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_pantun:
            f.write(json.dumps(item) + '\n')
            
    return valid_pantun

# ==========================================
# BLOK EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # Ganti nama file di bawah ini sesuai dengan nama file aslimu
    file_txt_1 = '1.DATASET/raw_data/dataset_1_Github.txt'
    file_csv_2 = '1.DATASET/raw_data/dataset_2_Mendeley.csv'
    file_csv_3 = '1.DATASET/raw_data/dataset_3_Kaggle.csv'
    
    # 1. Ingest Data (Milestone 1)
    master_list = ingest_and_standardize(file_txt_1, file_csv_2, file_csv_3)
    print(f"Total pantun mentah yang diekstrak: {len(master_list)} bait")
    
    # 2. Filter a-b-a-b dan Simpan ke JSONL (Milestone 2 & 3)
    output_jsonl = '1.DATASET/processed_data/master_data.jsonl' # Pastikan folder processed_data ada
    cleaned_list = filter_rhyme_and_save(master_list, output_jsonl)
    
    print(f"Total pantun valid (a-b-a-b) yang tersimpan: {len(cleaned_list)} bait")