# S²-Net: Spiking-by-Synchronization Neural Network

**From Cortical Synchronous Rhythm to Brain-Inspired Learning Architecture: An Oscillatory Spiking Neural Network with Time-Delayed Coordination**

Tingting Dan, Guorong Wu  
University of North Carolina at Chapel Hill

[[Paper (OpenReview)]](https://openreview.net/forum?id=aAslvH1QJK)

S²-Net is a brain-inspired architecture in which a graph-constrained Sakaguchi–Kuramoto oscillator layer rhythmically gates a population of dendritic spiking neurons through time-delayed phases.

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{dan2026s2net,
  title     = {From Cortical Synchronous Rhythm to Brain-Inspired Learning Architecture: An Oscillatory Spiking Neural Network with Time-Delayed Coordination},
  author    = {Dan, Tingting and Wu, Guorong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Model Architecture

Each entity $i$ (e.g., a brain region or an image pixel) is represented by an oscillatory gating unit with a $D$-dimensional phase $\theta_i \in \mathbb{R}^D$ and a dendritic spiking neuron. The entities interact through a connectivity graph $A$ (e.g., structural connectivity).

### 1. Oscillator layer (vector Sakaguchi–Kuramoto)
Code: `SNN/SNN_layers/vector_kuramoto.py`, class `graphVectorKuramoto`.
The input $X(t)$ is projected to a control signal $\gamma(t) \in \mathbb{R}^N$, and the phases evolve as

```math
\dot{\theta}_i^d = \omega_i^d + \kappa_i^d \sin(\gamma_i(t) - \theta_i^d) + \frac{K}{N} \sum_{j} A_{ij} \sin(\theta_j^d - \theta_i^d - \alpha_{ij}),
```

integrated with an Euler step of size `--dt`. $\omega$ and $\kappa$ are learnable.

### 2. Connectivity-derived phase lag
The lag is not a free parameter. Its magnitude comes from the transport cost $`C_{ij} = 1/(A_{ij} + \epsilon)`$, min–max normalized to $`\tilde{C}_{ij} \in [0,1]`$, and its sign from a learnable matrix $W$:

```math
\alpha_{ij} = \alpha_0 \tanh(W_{ij} - W_{ji})\, \tilde{C}_{ij}.
```

$A$ is symmetrized and $\tanh$ is odd, so $`\alpha_{ij} = -\alpha_{ji}`$ by construction.

### 3. Rhythmic gating with a time delay
Code: `models/s2net*.py`. With a delay of $\tau = 2$ steps, the gate is computed from the delayed phase averaged over its $D$ components:

```math
g_i(t) = \tfrac{1}{2}\left(1 + \sin\bar{\theta}_i(t-\tau)\right), \qquad \bar{\theta}_i = \tfrac{1}{D}\textstyle\sum_d \theta_i^d.
```

The spiking layer receives the gated phase features $`g_i(t)\,\sin\theta_i(t)`$ (projected from $D$ to 1) and a gate mask $`m(t) = \mathrm{sigmoid}(W_g\, g(t) + b_g)`$.

### 4. Rhythm-modulated dendritic spiking layer
Code: `SNN/SNN_layers/spike_rnn.py`, class `Rhy_spike_rnn_test_denri_wotanh_new`. Each neuron has $B$ dendritic branches with learnable decay rates; the branch currents are summed into $`I_i(t)`$. The membrane state uses a soft reset that does not depend on the gate, and the gate mask scales the emitted spikes:

```math
U_i(t) = \rho_i U_i(t-1) + (1-\rho_i) I_i(t) - \vartheta\, \delta_i(t-1), \qquad \delta_i(t) = \mathrm{H}(U_i(t) - \vartheta)\, m_i(t),
```

with $`\rho_i = \mathrm{sigmoid}(\tau_{m,i})`$, threshold $\vartheta = 0.5$, and a surrogate gradient for the Heaviside function $\mathrm{H}$. A spiking readout layer maps the spikes to task outputs.

## Repository Structure

The codebase is organized to separate core dynamical models from task-specific implementations:

```text
S2Net/
├── models/                  # [S²-Net Architectures] Task-specific model variants
│   ├── layers.py            # Wrappers for the oscillator layer and the spiking core
│   ├── s2net.py             # Standard S²-Net for Sequence Labeling (Eq. 2-6)
│   ├── s2net_cls.py         # S²-Net Classifier with Temporal Pooling (Subject Classification)
│   └── s2net_recon.py       # S²-Net Autoencoder (Self-Supervised Reconstruction)
│
├── SNN/                     # [Core Dynamics] Neural dynamics & Kuramoto layers
│   ├── SNN_layers/
│   │   ├── vector_kuramoto.py   # Vector Sakaguchi-Kuramoto with connectivity-derived phase lag (graphVectorKuramoto)
│   │   ├── spike_rnn.py         # Rhythm-modulated dendritic spiking layer
│   │   └── spike_neuron.py      # Neuron update and surrogate gradient
│   ├── spiking_model.py          # Spiking network definitions and readout layer
│   └── Hyperparameters.py        # Centralized configuration management
│
├── utils/                   # [Utilities] Data handling & Evaluation
│   ├── data_loader.py       # Data loaders for HCP, ADNI, UKB, NIFD datasets
│   ├── metrics.py           # Evaluation metrics (Accuracy, F1, MSE, Intra/Inter Sync)
│   └── lib.py               # Helper functions (Seed, Logging)
│
├── train_s2net_sim.py       # [Experiment 1] Proof-of-Concept Simulation (Synthetic Data)
├── train_s2net_recon.py     # [Experiment 2] Self-Supervised BOLD Reconstruction
├── train_s2net_cls.py       # [Experiment 3] Subject-Level Classification (HCPA/HCPYA/UKB/ADNI/PPMI/NIFD)
└── train_s2net_dyn.py       # [Experiment 4] Dynamic State Recognition (HCP-WM)
```

## Usage Instructions

## 1. Proof-of-Concept Simulation
Reproduce the "Binding-by-Synchrony" and "Segregation-by-Desynchrony" experiments on synthetic data (Disjoint Squares). This script validates the core physics without external data.

```bash
python train_s2net_sim.py --output_dir ./exp/simulation
```

## 2. Self-Supervised Reconstruction
Train S²-Net to reconstruct BOLD signals (Autoencoder mode), demonstrating its capability to capture intrinsic brain dynamics.

```bash
python train_s2net_recon.py \
  --dataset HCPYA \
  --data_root /path/to/HCP/BOLD/ \
  --sc_root /path/to/HCP/SC/ \
  --fallback_sc /path/to/fallback.mat
```

## 3. Subject Classification
Apply S²-Net for disease diagnosis or subject classification on datasets like ADNI or PPMI.
```bash
python train_s2net_cls.py \
  --dataset ADNI \
  --data_root /path/to/ADNI/BOLD/ \
  --sc_root /path/to/ADNI/SC/ \
  --label_file /path/to/ADNI/labels.csv
```

## 4. Dynamic State Recognition (Sequence Labeling)
Perform frame-wise cognitive state decoding (e.g., working memory tasks from HCP-WM) on task-fMRI data.

```bash
python train_s2net_dyn.py \
  --dataset HCPYA \
  --data_root /path/to/HCP/BOLD/ \
  --sc_root /path/to/HCP/SC/ \
  --label_csv_rl /path/to/labels_RL.csv \
  --label_csv_lr /path/to/labels_LR.csv
```


## Configuration
Key hyperparameters are managed in SNN/Hyperparameters.py. 
You can override them via command line arguments:

- --k: Global coupling strength $K$ (Eq. 2).
- --dt: Integration time step.
- --hidden: Latent dimension size.
- --low_n / --high_n: Spiking neuron adaptation parameters.
- --ablation: Choose ablation modes (no_coupling, no_mask, raw_feature, full).
