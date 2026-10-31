# Training Data Loading Fix

**Date:** 2026-10-31
**Scope:** `3.TRAINING/train.py`, `1.DATASET/tokenization.py`
**Status:** Verified with `--smoke_test`

## Problem

The training script was configured to open:

```text
1.DATASET/processed_data/train.jsonl
1.DATASET/processed_data/val.jsonl
```

Those files do not exist. When `train.jsonl` was missing, the script silently fell back to `master_data.jsonl` and used the **same file for both training and validation**:

```python
if not os.path.exists(train_path):
    train_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'master_data.jsonl')
    val_path = train_path
```

This caused two issues:

1. The real pre-computed splits (`train.txt`, `val.txt`) were never used.
2. Train and validation data were identical, so validation loss was not a valid held-out metric.

The actual split files produced by `1.DATASET/splitting.py` are:

| File | Records |
| --- | ---: |
| `train.txt` | 7,511 |
| `val.txt` | 939 |
| `test.txt` | 939 |

## Root Cause

`PantunDataset` only supported JSONL input. The generated `.txt` splits use a different, block-based format:

```text
<|start|> [TEMA: umum] [RIMA: ng, ar]
pergi ke lampung membeli tepung
kue talam dijual dipasar
nasib beruntung walau tak mengapung
sebelum tenggelam ditolong tim sar
<|end|>
```

The dataset had no parser for this format, so the training script could not simply point at the `.txt` files.

## Changes

### `3.TRAINING/train.py`

- Loads `train.txt` and `val.txt` directly:

```python
train_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'train.txt')
val_path = os.path.join(BASE_DIR, '1.DATASET', 'processed_data', 'val.txt')
```

- Removed the `master_data.jsonl` fallback entirely. The training pipeline no longer reads or probes any JSONL file.
- Added explicit existence checks with an actionable error message:

```text
File data train tidak ditemukan: .../train.txt
Jalankan 1.DATASET/preprocessing.py lalu 1.DATASET/splitting.py terlebih dahulu.
```

- Added empty-dataset checks after parsing.
- Added startup logs showing the resolved paths and record counts:

```text
[Init] Train samples: 7,511 (.../train.txt)
[Init] Val samples  : 939 (.../val.txt)
```

### `1.DATASET/tokenization.py`

- Added `load_pantun_txt(path)`, which parses the block format above into structured records:

```python
{
    "tema": "umum",
    "rima_a": "ng",
    "rima_b": "ar",
    "larik": ["...", "...", "...", "..."],
}
```

  The parser:
  - Matches `<|start|> [TEMA: ...] [RIMA: ..., ...]` headers with a regular expression.
  - Collects exactly four `larik` lines per block.
  - Skips malformed headers or incomplete blocks and prints a warning.

- Changed `PantunDataset` to be **TXT-only**:
  - Constructor parameter renamed from `jsonl_path` to `txt_path`.
  - Added `assert Path(txt_path).suffix.lower() == ".txt"`.
  - Removed `_load_jsonl()` and the extension-based JSONL dispatch.
  - All records are now loaded through `load_pantun_txt()`.

- Updated `analyze_context_length()` to read `.txt` records through `load_pantun_txt()` instead of parsing JSONL line by line.
- Updated the module `__main__` to use `1.DATASET/processed_data/train.txt` and raise a clear error instead of creating a dummy JSONL file.

## Data Flow After Fix

```text
preprocessing.py
      |
      v
processed_data/master_data.jsonl
      |
      v
splitting.py  ──> train.txt (80%)
             ──> val.txt   (10%)
             ──> test.txt  (10%)
      |
      v
PantunDataset (TXT only)
      |
      v
load_pantun_txt()
      |
      v
build_prompt_string()  ->  <|bos|><|tema|>...<|rima_a|>...<|rima_b|>...<|isi|>...<|eos|>
```

The conditioning/tokenization scheme is unchanged. TXT records are normalized into the model's existing special-token format, so `generate_sample()` and the model's conditioning dropout still work as before.

## Verification

Command:

```bash
.venv/bin/python 3.TRAINING/train.py --smoke_test
```

Result:

```text
[Init] Train samples: 7,511 (.../1.DATASET/processed_data/train.txt)
[Init] Val samples  : 939 (.../1.DATASET/processed_data/val.txt)
Step    0 | Loss: 10.8402 | LR: 6.00e-04 | Norm: 1.39
Step    1 | Loss: 10.4392 | LR: 6.00e-04 | Norm: 1.40
--- EVAL STEP 1 ---
Train Loss: 10.1448 | Val Loss: 10.1363 | Val PPL (BPE): 25242.76
...
✅ SMOKE TEST PASSED
```

Additional parser check:

```text
train records: 7511
val records  : 939
train 7511 torch.Size([56])
val 939 torch.Size([51])
```

Record counts match the expected 80/10/10 split, with no malformed-block warnings.

## Notes

- `master_data.jsonl` remains the output of `preprocessing.py` and the input to `splitting.py`; it is no longer read by the training script or by `PantunDataset`.
- Any code that previously called `PantunDataset(jsonl_path=...)` must switch to `PantunDataset(txt_path=...)`.
- The unrelated working-tree change in `generate_sample()` (`max_new_tokens` 32 → 80) was left untouched.

## Appendix: User Prompts

The following prompts were provided during this task, in chronological order.

### Prompt 1

> DO NOT CD OUTSIDE THIS DIRECTORY!
>
> i need you to fix this training python script, where it only reads properly from dataset/processed_data/master_data.jsonl
>
> where it's supposed to load the train.txt and val.txt

### Prompt 2

> wait don't do json as fallback, all in to the txt file, don't probe into the jsonl

### Prompt 3

> okay, after this document the changes as markdown file on TRAINING/report

### Prompt 4

> also add all of my prompt to you into those markdown documentation
