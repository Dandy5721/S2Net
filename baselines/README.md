# Baselines (paper Table comparisons)

## Baseline → script → model class

| Paper baseline | Base script | Base model (package) | Rhythm-gated ("skip") script | Rhythm model (package) |
|---|---|---|---|---|
| **FFSNN** (feed-forward SNN) | `main_bold-FFSNN.py` | `FFSNN` (`FFSNN/`) | `main_bold-skipFFSNN.py` | `FFSNN_mix` (`skip_FFSNN/`) |
| **SRNN** (recurrent SNN) | `main_bold-SRNN.py` | `SRNN` (`SRNN/`) | `main_bold-skipSRNN.py` | `SRNN_mix` (`skip_SRNN/`) |
| **ASRNN** (adaptive SRNN) | `main_bold-ASRNN.py` | `SRNN_ALIF` (`SRNN/`) | `main_bold-skipASRNN.py` | `SRNN_ALIF_mix_v3` (`skip_SRNN/`) |
| **LSNN** | `main_bold-LSNN.py` | `LSNN` (`LSNN/`) | `main_bold-skipLSNN.py` | `LSNN_mix` (`skip_LSNN/`) |
| **DH-SNN** (dendritic heterogeneity) | `main_bold-DHSRNN.py` | `DHSRNN` (`skipDHSRNN/`) | `main_bold-skipDHSRNN.py` | `rnn_test` (`skipDHSRNN/`) |

**Notes**
- The `skip*` scripts are the **Rhythm-SNN** (rhythmic-gating) variants of each base model; the paper's "Rhythm-SNN" baseline corresponds to these.
- `skipDHSRNN/` additionally contains `SNN_layers/` (vector-Kuramoto etc.) used by the dendritic/rhythm models.
- The ECG / general-temporal-task counterparts (`main_ecg-<method>.py`) reuse the **same** model packages; add them here in the same layout if the general-task baselines are also uploaded.

## Directory layout

```
baselines/
  README.md
  tools/            shared lib (set_seed, dump_json, count_parameters, dataset utils)
  FFSNN/  skip_FFSNN/
  SRNN/   skip_SRNN/          (SRNN, SRNN_ALIF, SRNN_mix, SRNN_ALIF_mix_v3)
  LSNN/   skip_LSNN/
  skipDHSRNN/                 (DHSRNN + rnn_test + SNN_layers/)
  main_bold-*.py              (10 training scripts, one per baseline/variant)
```

## Hyperparameter settings (paper Table 6)

Defaults in each package's `Hyperparameters.py` reproduce Table 6 of the paper.
All baselines use a **hidden dimension of 1024** (`--fc 1024`); only S²-Net uses 64.
Optimizer is Adam with `StepLR(step_size=100, gamma=0.75)` throughout.

| Model | Script(s) | LR | Batch | Epochs | Hidden |
|---|---|---|---|---|---|
| FFSNN | `main_bold-FFSNN.py` | 1e-2 | 256 | 250 | 1024 |
| SRNN  | `main_bold-SRNN.py`  | 1e-2 | 64  | 250 | 1024 |
| LSNN  | `main_bold-LSNN.py`  | 1e-2 | 256 | 250 | 1024 |
| ASRNN | `main_bold-ASRNN.py` | 1e-2 | 64  | 250 | 1024 |
| DH-SNN | `main_bold-DHSRNN.py` | 1e-2 | 16 | 500 | 1024 |
| Rhythm-SNN | `main_bold-skip*.py` (rhythm-gated variants) | 1e-2 | 16 | 500 | 1024 |

The hidden dimension is the recurrent width `--fc[0]`; the fixed 116→64 input
projection (`proj_dim=64`) is unchanged. These settings match the parameter
counts reported in Table 7 (e.g. SRNN/ASRNN ≈ 4.5 MB ≈ a 1024×1024 recurrent).

## Running

Each script trains on the BOLD dataset and reports Acc/Pre/F1. Defaults already
match Table 6, so no flags are needed; override only if you want to deviate:

```bash
cd baselines
python main_bold-SRNN.py --dataset PPMI          # uses Table 6 defaults
```

Seeds are set via `set_seed(...)` in `tools/lib.py`.
