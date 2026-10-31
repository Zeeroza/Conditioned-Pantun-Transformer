"""
test_model.py - Sanity check untuk model.py (Mini-Transformer)

Jalankan salah satu:
    python test_model.py          # runner bawaan, tanpa dependensi tambahan
    pytest -v test_model.py       # kalau pytest terpasang

Semua test memakai config kecil supaya cepat (hitungan detik di CPU),
kecuali test_default_config_* yang memakai config.py asli kamu.
"""

import math
import os
import sys
import tempfile
import traceback

import torch

from model import MiniTransformer, ModelConfig


# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------
def tiny_cfg(**overrides) -> ModelConfig:
    kw = dict(vocab_size=64, block_size=32, d_model=64, n_head=4, n_layer=2, dropout=0.0)
    kw.update(overrides)
    return ModelConfig(**kw)


def make_batch(cfg: ModelConfig, B: int = 4, T: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y


# ----------------------------------------------------------------------------
# 1. Bentuk output & loss awal
# ----------------------------------------------------------------------------
def test_output_shapes_and_initial_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, y = make_batch(cfg)

    logits, loss = model(x, y)
    assert logits.shape == (4, 16, cfg.vocab_size)
    assert loss is not None and loss.dim() == 0

    # Loss awal harus dekat ln(V): kalau jauh lebih besar, init / scaling bermasalah
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.5, f"loss awal {loss.item():.3f} jauh dari ln(V)={expected:.3f}"


def test_no_targets_returns_none_loss():
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, _ = make_batch(cfg)
    logits, loss = model(x)
    assert loss is None
    assert logits.shape[:2] == x.shape


def test_input_longer_than_block_size_raises():
    cfg = tiny_cfg(block_size=16)
    model = MiniTransformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 17))
    try:
        model(x)
    except AssertionError:
        return
    raise AssertionError("Seharusnya AssertionError untuk T > block_size")


# ----------------------------------------------------------------------------
# 2. Causality (bug paling mahal di decoder-only)
# ----------------------------------------------------------------------------
def _check_causal(cfg: ModelConfig):
    torch.manual_seed(0)
    model = MiniTransformer(cfg).eval()
    x, _ = make_batch(cfg, B=2, T=16)

    t = 10  # ubah token di posisi t
    x2 = x.clone()
    x2[:, t] = (x2[:, t] + 1) % cfg.vocab_size

    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)

    # posisi < t tidak boleh berubah
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5), "KEBOCORAN: masa depan memengaruhi masa lalu"
    # posisi >= t harus berubah (memastikan test ini tidak lolos karena model mengabaikan input)
    assert not torch.allclose(a[:, t:], b[:, t:], atol=1e-5), "Perubahan token tidak berpengaruh sama sekali"


def test_causality_pre_ln():
    _check_causal(tiny_cfg(norm_first=True))


def test_causality_post_ln():
    _check_causal(tiny_cfg(norm_first=False))


def test_attention_weights_are_valid_causal_distributions():
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, _ = make_batch(cfg, B=2, T=12)

    with torch.no_grad():
        _, _, maps = model(x, return_attn=True)

    assert len(maps) == cfg.n_layer
    for w in maps:
        assert w.shape == (2, cfg.n_head, 12, 12)
        assert torch.allclose(w.sum(-1), torch.ones(2, cfg.n_head, 12), atol=1e-5), "baris softmax tidak berjumlah 1"
        assert torch.all(w.triu(diagonal=1) == 0), "ada bobot attention ke posisi masa depan"


# ----------------------------------------------------------------------------
# 3. Padding: right-padding aman + ignore_index bekerja
# ----------------------------------------------------------------------------
def test_right_padding_does_not_affect_valid_positions():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, _ = make_batch(cfg, B=2, T=10)
    x_padded = torch.cat([x, torch.zeros(2, 6, dtype=torch.long)], dim=1)  # pad kanan dengan id 0

    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x_padded)
    assert torch.allclose(a, b[:, :10], atol=1e-5)


def test_ignore_index_masks_loss():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, y = make_batch(cfg, B=2, T=12)

    y_masked = y.clone()
    y_masked[:, 6:] = -100  # separuh posisi diabaikan

    with torch.no_grad():
        logits, loss = model(x, y_masked)
        manual = torch.nn.functional.cross_entropy(
            logits[:, :6].reshape(-1, cfg.vocab_size), y[:, :6].reshape(-1)
        )
    assert torch.allclose(loss, manual, atol=1e-6)


# ----------------------------------------------------------------------------
# 4. Parameter, weight tying, gradien
# ----------------------------------------------------------------------------
def test_weight_tying():
    tied = MiniTransformer(tiny_cfg(tie_weights=True))
    untied = MiniTransformer(tiny_cfg(tie_weights=False))
    assert tied.lm_head.weight is tied.embed.tok_emb.weight
    assert untied.num_params() - tied.num_params() == untied.cfg.vocab_size * untied.cfg.d_model


def test_all_parameters_receive_finite_gradients():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).train()
    x, y = make_batch(cfg)

    _, loss = model(x, y)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} tidak menerima gradien"
        assert torch.isfinite(p.grad).all(), f"{name} punya gradien NaN/Inf"


def test_optimizer_param_groups_cover_all_params_once():
    model = MiniTransformer(tiny_cfg())
    opt = model.configure_optimizers(lr=1e-3, weight_decay=0.1)
    in_opt = [p for g in opt.param_groups for p in g["params"]]
    in_model = [p for p in model.parameters() if p.requires_grad]

    assert len(in_opt) == len(set(map(id, in_opt))), "ada parameter yang terdaftar dua kali"
    assert set(map(id, in_opt)) == set(map(id, in_model))
    assert opt.param_groups[0]["weight_decay"] == 0.1 and opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in opt.param_groups[0]["params"])
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])


# ----------------------------------------------------------------------------
# 5. Overfit satu batch (tes terpenting untuk jalur backward + optimizer)
# ----------------------------------------------------------------------------
def _overfit(cfg: ModelConfig, steps: int = 300, lr: float = 3e-3):
    torch.manual_seed(0)
    model = MiniTransformer(cfg).train()
    opt = model.configure_optimizers(lr=lr, weight_decay=0.0)
    x, y = make_batch(cfg, B=4, T=16)

    first = None
    for _ in range(steps):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return first, loss.item()


def test_overfit_single_batch_pre_ln():
    first, last = _overfit(tiny_cfg(norm_first=True))
    assert last < 0.5 and last < first * 0.2, f"loss tidak turun cukup: {first:.3f} -> {last:.3f}"


def test_overfit_single_batch_post_ln():
    first, last = _overfit(tiny_cfg(norm_first=False), lr=1e-3)
    assert last < 0.5 and last < first * 0.2, f"loss tidak turun cukup: {first:.3f} -> {last:.3f}"


# ----------------------------------------------------------------------------
# 6. Varian konfigurasi
# ----------------------------------------------------------------------------
def test_gelu_variant_runs():
    cfg = tiny_cfg(activation="gelu")
    model = MiniTransformer(cfg).eval()
    x, y = make_batch(cfg)
    _, loss = model(x, y)
    assert torch.isfinite(loss)


def test_flash_matches_manual_attention():
    torch.manual_seed(0)
    m1 = MiniTransformer(tiny_cfg(use_flash=False)).eval()
    m2 = MiniTransformer(tiny_cfg(use_flash=True)).eval()
    m2.load_state_dict(m1.state_dict())
    x, _ = make_batch(m1.cfg)

    with torch.no_grad():
        a, _ = m1(x)
        b, _ = m2(x)
    assert torch.allclose(a, b, atol=1e-4), f"selisih maksimum {(a - b).abs().max().item():.2e}"


def test_config_override_from_config_py():
    class FakeConfigModule:
        VOCAB_SIZE = 100
        BLOCK_SIZE = 16
        D_MODEL = 32
        N_HEAD = 2
        N_LAYER = 3
        DROPOUT = 0.2

    cfg = ModelConfig.from_config_py(FakeConfigModule, n_layer=5)
    assert (cfg.vocab_size, cfg.block_size, cfg.d_model, cfg.n_head, cfg.dropout) == (100, 16, 32, 2, 0.2)
    assert cfg.n_layer == 5, "override kwargs harus menang atas config.py"


def test_d_model_not_divisible_by_heads_raises():
    try:
        ModelConfig(d_model=250, n_head=4)
    except AssertionError:
        return
    raise AssertionError("Seharusnya AssertionError kalau d_model % n_head != 0")


# ----------------------------------------------------------------------------
# 7. Generate & checkpoint
# ----------------------------------------------------------------------------
def test_generate_shape_range_and_mode_restored():
    torch.manual_seed(0)
    cfg = tiny_cfg(block_size=16)
    model = MiniTransformer(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    # generate melewati block_size -> harus otomatis di-crop, tidak crash
    out = model.generate(x, max_new_tokens=20, top_k=10)
    assert out.shape == (2, 28)
    assert out.min() >= 0 and out.max() < cfg.vocab_size
    assert torch.equal(out[:, :8], x), "prompt tidak boleh berubah"
    assert model.training, "mode training harus dikembalikan setelah generate"


def test_greedy_generate_is_deterministic():
    cfg = tiny_cfg()
    model = MiniTransformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    a = model.generate(x, max_new_tokens=6, greedy=True)
    b = model.generate(x, max_new_tokens=6, greedy=True)
    assert torch.equal(a, b)


def test_checkpoint_roundtrip():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = MiniTransformer(cfg).eval()
    x, _ = make_batch(cfg)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        model.save_checkpoint(path, step=123)
        loaded = MiniTransformer.from_checkpoint(path).eval()

    assert loaded.cfg == cfg
    with torch.no_grad():
        a, _ = model(x)
        b, _ = loaded(x)
    assert torch.allclose(a, b, atol=1e-6)


# ----------------------------------------------------------------------------
# 8. Config asli dari config.py kamu
# ----------------------------------------------------------------------------
def test_default_config_param_count_and_loss():
    torch.manual_seed(0)
    cfg = ModelConfig()  # default = nilai di config.py
    model = MiniTransformer(cfg).eval()

    n = model.num_params()
    assert 15e6 < n < 17e6, f"jumlah parameter {n / 1e6:.2f}M di luar perkiraan (~16.03M)"

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    with torch.no_grad():
        logits, loss = model(x, y)
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


# ----------------------------------------------------------------------------
# Runner bawaan (tanpa pytest)
# ----------------------------------------------------------------------------
def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} test lolos")
    if failed:
        print("Gagal:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())