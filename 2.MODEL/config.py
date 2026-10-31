# Model Architecture
VOCAB_SIZE = 50257  # atau len(chars) jika char-level
BLOCK_SIZE = 128    # panjang konteks maksimum
D_MODEL = 256       # dimensi embedding
N_HEAD = 4          # jumlah attention heads
N_LAYER = 4         # jumlah transformer blocks
DROPOUT = 0.1