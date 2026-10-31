"""
model.py - Mini-Transformer (Autoregressive Causal Decoder-Only)

Referensi: Vaswani et al., "Attention Is All You Need" (NIPS 2017)
Beda utama dari paper: encoder + cross-attention dibuang (decoder-only, ala GPT),
sisanya (sinusoidal PE, scaled dot-product, MHA, FFN, residual + LayerNorm,
weight tying, scaling embedding sqrt(d_model)) mengikuti paper.

Alur forward:
    idx (B, T)
      -> [1] TokenAndPositionalEmbedding         (B, T, D)
      -> N x TransformerBlock:
            [6] LayerNorm + Residual
            [4] MultiHeadAttention  <- [2] scaled dot-product Q,K,V  <- [3] causal mask
            [5] FeedForward
      -> LayerNorm akhir -> lm_head (weight tied) -> logits (B, T, V)
      -> (opsional) cross-entropy loss jika targets diberikan
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # --- sama persis dengan config.py ---
    vocab_size: int = 50257
    block_size: int = 128
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    # --- opsi tambahan (default aman, bisa di-override saat eksperimen) ---
    d_ff: Optional[int] = None      # None -> 4 * d_model (paper: 512 -> 2048)
    activation: str = "relu"        # "relu" (paper) atau "gelu"
    bias: bool = True
    norm_first: bool = True         # True: Pre-LN | False: Post-LN (persis paper)
    tie_weights: bool = True        # share embedding <-> lm_head (paper sec 3.4)
    use_flash: bool = False         # pakai F.scaled_dot_product_attention (lebih cepat)

    def __post_init__(self):
        assert self.d_model % self.n_head == 0, "d_model harus habis dibagi n_head"
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @classmethod
    def from_config_py(cls, module=None, **overrides) -> "ModelConfig":
        """
        Bangun ModelConfig dari config.py (konstanta UPPERCASE).
        Hyperparameter apa pun bisa di-override lewat kwargs:
            cfg = ModelConfig.from_config_py(n_layer=6, dropout=0.2)
        """
        if module is None:
            import config as module  # config.py di root project
        mapping = {
            "vocab_size": "VOCAB_SIZE",
            "block_size": "BLOCK_SIZE",
            "d_model": "D_MODEL",
            "n_head": "N_HEAD",
            "n_layer": "N_LAYER",
            "dropout": "DROPOUT",
        }
        kwargs = {k: getattr(module, v) for k, v in mapping.items() if hasattr(module, v)}
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# [1] Input Embedding + Positional Encoding
# ----------------------------------------------------------------------------
class TokenAndPositionalEmbedding(nn.Module):
    """
    x = Embedding(idx) * sqrt(d_model) + PE(pos)        (paper sec 3.4 & 3.5)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    PE disimpan sebagai buffer (bukan parameter): fixed, tidak ikut di-train,
    tapi ikut pindah device dan masuk state_dict.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer("pe", self._build_sinusoidal(cfg.block_size, cfg.d_model), persistent=False)

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)                   # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))  # (D/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (L, D)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) -> (B, T, D)
        T = idx.size(1)
        x = self.tok_emb(idx) * math.sqrt(self.d_model) + self.pe[:T]
        return self.drop(x)


# ----------------------------------------------------------------------------
# [3] Causal Masking (M)
# ----------------------------------------------------------------------------
def build_causal_mask(size: int, device=None) -> torch.Tensor:
    """
    Lower-triangular boolean mask, True = BOLEH di-attend.
    Posisi i hanya boleh melihat posisi j <= i (menjaga sifat auto-regressive).
    Bentuk (1, 1, size, size) agar broadcast ke (B, n_head, T, T).
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(1, 1, size, size)


# ----------------------------------------------------------------------------
# [2] Self-Attention: Q, K, V  ->  softmax(QK^T / sqrt(d_k) + M) V   (Eq. 1)
# ----------------------------------------------------------------------------
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k, v : (B, n_head, T, d_k)
    mask    : bool, broadcastable ke (B, n_head, T, T); True = boleh attend
    return  : (output (B, n_head, T, d_k), attention weights (B, n_head, T, T))
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, h, T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))            # posisi ilegal -> -inf -> softmax 0
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v), weights


# ----------------------------------------------------------------------------
# [4] Multi-Head Attention (MHA)
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    MultiHead(X) = Concat(head_1..head_h) W^O,  head_i = Attention(X Wq_i, X Wk_i, X Wv_i)

    Proyeksi Q, K, V untuk semua head digabung jadi satu Linear(D, 3D) (matematis identik
    dengan h proyeksi terpisah, tapi hanya satu matmul). d_k = d_model / n_head (paper: 512/8 = 64).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        self.dropout_p = cfg.dropout
        self.use_flash = cfg.use_flash

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)   # W^O
        self.out_proj._is_residual_proj = True   # dipakai untuk init skala kecil
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.register_buffer("causal_mask", build_causal_mask(cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape

        # (B, T, 3C) -> 3 x (B, T, C) -> (B, h, T, d_k)
        q, k, v = self.qkv_proj(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        weights = None
        if self.use_flash and not return_attn:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            mask = self.causal_mask[:, :, :T, :T]
            y, weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attn_drop)

        # concat heads: (B, h, T, d_k) -> (B, T, C), lalu proyeksi W^O
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return (y, weights) if return_attn else y


# ----------------------------------------------------------------------------
# [5] Feed-Forward Network (FFN / MLP)
# ----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """FFN(x) = act(x W1 + b1) W2 + b2  (Eq. 2). Dipakai per-posisi, parameter sama antar posisi."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.fc2._is_residual_proj = True
        self.act = nn.ReLU() if cfg.activation == "relu" else nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ----------------------------------------------------------------------------
# [6] Residual Connection & Layer Normalization  (satu Transformer block)
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Post-LN (persis paper, norm_first=False):
        x = LN(x + Attn(x));  x = LN(x + FFN(x))
    Pre-LN (default, norm_first=True):
        x = x + Attn(LN(x)); x = x + FFN(LN(x))
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_first = cfg.norm_first
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        weights = None
        if self.norm_first:
            a = self.attn(self.ln1(x), return_attn=return_attn)
            if return_attn:
                a, weights = a
            x = x + a
            x = x + self.ffn(self.ln2(x))
        else:
            a = self.attn(x, return_attn=return_attn)
            if return_attn:
                a, weights = a
            x = self.ln1(x + a)
            x = self.ln2(x + self.ffn(x))
        return (x, weights) if return_attn else x


# ----------------------------------------------------------------------------
# Model utuh
# ----------------------------------------------------------------------------
class MiniTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = TokenAndPositionalEmbedding(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        # Pre-LN butuh LayerNorm terakhir; Post-LN sudah ternormalisasi di tiap block.
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.norm_first else nn.Identity()
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.embed.tok_emb.weight   # paper sec 3.4

        self.apply(self._init_weights)
        # Skala init proyeksi residual ~ 1/sqrt(2 * n_layer) agar varians tidak membengkak
        # seiring kedalaman (praktik GPT-2).
        for m in self.modules():
            if getattr(m, "_is_residual_proj", False):
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- forward: output logits ATAU (logits, loss) ----
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        return_attn: bool = False,
    ):
        """
        idx     : (B, T) token ids, T <= block_size
        targets : (B, T) token ids target (idx yang digeser 1 ke kiri); None saat inference
        return  : logits (B, T, V), loss (scalar | None)  [+ list attention weights bila return_attn]
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Panjang input {T} > block_size {self.cfg.block_size}"

        x = self.embed(idx)
        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, w = block(x, return_attn=True)
                attn_maps.append(w)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                    # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_index
            )
        return (logits, loss, attn_maps) if return_attn else (logits, loss)

    # ---- generasi autoregressive (untuk evaluasi kualitatif) ----
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                 # crop ke konteks maksimum
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx

    # ---- helper untuk training ----
    def num_params(self, trainable_only: bool = True) -> int:
        # .parameters() otomatis tidak menghitung weight tied dua kali
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def configure_optimizers(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.98),   # paper: beta2 = 0.98
        eps: float = 1e-9,                           # paper: eps = 1e-9
    ) -> torch.optim.Optimizer:
        """AdamW; weight decay hanya untuk matriks (dim >= 2), bukan bias / LayerNorm."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)

    # ---- checkpoint sederhana: config ikut tersimpan ----
    def save_checkpoint(self, path: str, **extra):
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def from_checkpoint(cls, path: str, map_location="cpu") -> "MiniTransformer":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model


# ----------------------------------------------------------------------------
# Smoke test: python model.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    try:
        cfg = ModelConfig.from_config_py()
    except ImportError:
        cfg = ModelConfig()

    model = MiniTransformer(cfg)
    print(f"Params: {model.num_params() / 1e6:.2f}M | config: {cfg}")

    B, T = 2, 64
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(x, y)
    print("logits:", tuple(logits.shape), "| loss:", round(loss.item(), 4),
          "| ln(V) =", round(math.log(cfg.vocab_size), 4))

    # Sanity check causality: mengubah token masa depan tidak boleh mengubah logits masa lalu
    model.eval()
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    print("causal OK:", torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5))

    out = model.generate(x[:, :8], max_new_tokens=8, top_k=50)
    print("generate:", tuple(out.shape))