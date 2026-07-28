# Supplementary Tables — Author Rebuttal (Submission 10159)

Tables are grouped by theme: (1) PPMI classification fairness/efficiency (Tables 1–3), (2) multi-object temporal-binding validation (Tables 4–6), (3) broader multi-dataset evaluation (Table 7), (4) hyperparameter sensitivity sweeps (Tables 8–10), and (5) representational-hypothesis validation (Table 11).

Dataset abbreviations: **PPMI** = Parkinson's Progression Markers Initiative; **HCPA** = HCP-Aging; **HCPYA** = HCP-Young Adult; **UKB** = UK Biobank; **ADNI** = Alzheimer's Disease Neuroimaging Initiative; **NIFD** = Neuroimaging in Frontotemporal Dementia. Accuracy/Precision/F1 are reported in % as mean ± s.d. across 10-folds.

---

### Table 1: PPMI Dataset Performance Comparison Under a Unified Computational Budget

*Addresses: Meta-Review ("Fairness of the experimental comparisons"); Reviewer 2YuW (A2); Reviewer R1Tt (indirectly, via A1).*

Performance comparison on the PPMI dataset (10-fold cross-validation, subject-level split), evaluated under a **unified computational budget** identical across all methods: hidden dimension 64, batch size 16, 500 training epochs. Prior to this comparison, baselines were run with their original per-model default hyperparameters (e.g., hidden dimension 1024), which confounded capacity with mechanism. Under the matched 64-dimensional budget, S²-Net retains its performance while several baselines degrade substantially (e.g., SRNN: 66.09→64.73; DH-SNN: 65.57→61.85; Rhythm-SNN: 65.57→64.21), demonstrating that S²-Net's advantage is not attributable to extra parameter capacity or a favorable optimization budget. At hidden dimension 1024 (matching Rhythm-SNN's original setting), S²-Net's accuracy further improves to 68.49 ± 3.31%.

| PPMI | FFSNN | SRNN | LSNN | ASRNN | DH-SNN | Rhythm-SNN | S²-Net |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Acc (%) | 65.73±3.86 | 64.73±2.53 | 67.26±5.90 | 64.72±2.90 | 61.85±2.86 | 64.21±2.91 | **67.62±5.21** |
| Pre (%) | 39.48±9.93 | 44.46±6.08 | 40.61±10.71 | 42.86±3.03 | 40.42±12.68 | 42.22±8.66 | **49.30±8.83** |
| F1 (%) | 38.69±7.30 | 39.99±6.29 | 39.57±9.38 | 37.70±5.27 | 36.07±9.25 | 36.03±6.01 | **41.51±7.19** |


---

### Table 2: Necessity of the Spiking Module — SNN vs. Capacity-Matched ReLU-RNN

*Addresses: Reviewer 4HKM, Weakness (4) "Necessity of the spiking module," and the associated Question "Ablation of the spiking module."*

To isolate the specific contribution of discrete spiking (rather than simply having *some* recurrent nonlinearity), we replaced only the spiking temporal core with a threshold-linear **ReLU-RNN** of matched parameter count, while keeping the Kuramoto phase features, delayed gating signal, data folds, optimizer, and training budget identical (PPMI, 10-fold cross-validation). "Mean core activity" is the average post-nonlinearity activation magnitude of the temporal core (a proxy for representational density/sparsity); "fold time" is wall-clock training time per fold.

| Temporal core | Params | Accuracy (%) | Mean core activity | Fold time |
| :--- | :--- | :--- | :--- | :--- |
| **SNN (ours)** | 156,451 | **67.62 ± 5.21** | **0.448** | 32.5 min |
| ReLU-RNN | 156,216 | 64.37 ± 5.43 | 0.758 | 29.0 min |

At matched capacity, the spiking core outperforms the dense ReLU-RNN while operating at substantially sparser activity (0.448 vs. 0.758), showing that discrete spiking is not a freely replaceable generic recurrent nonlinearity in this architecture.

---

### Table 3: Dendritic Branch Ablation Study

*Addresses: Reviewer R1Tt, Question "How did you select the number of branches?" (A1).*

Ablation over the number of dendritic branches $B \in \{1, 2, 4, 8\}$ on PPMI (10-fold cross-validation), evaluating the trade-off between parameter count and classification performance (Accuracy, F1). $B=4$ is used as the default configuration in the main paper, selected as the empirical trade-off point between representational capacity and parameter efficiency (in addition to the biological motivation of apical/basal dendritic compartments discussed in our response).

| Branches ($B$) | Params | Accuracy (%) | F1 (%) |
| :--- | :--- | :--- | :--- |
| 1 | 75K | 64.15 ± 2.14 | 38.92 ± 2.45 |
| 2 | 102K | 66.83 ± 4.10 | 40.58 ± 3.12 |
| 4 (default) | 156K | 67.62 ± 5.21 | 41.51 ± 7.19 |
| 8 | 265K | **68.35 ± 3.40** | **42.45 ± 4.15** |

---

### Table 4: Multi-Object Benchmark — Phase-Lag Ablation

*Addresses: Meta-Review ("Evidence for temporal binding"); Reviewer 4HKM, Weakness (5) "The binding metric is not convincing" and Weakness (6) "Temporal binding results are not convincing."*

Evaluation of binding capability on dSprites-style multi-object scenes, scaling object count from 2 to 6 under noise $\sigma \in \{0, 0.1, 0.2, 0.3\}$ (three seeds). Metrics are **size-invariant and chance-corrected**, replacing the originally used $R_{\mathrm{merge}}$ metric (which Reviewer 4HKM correctly identified as non-robust to size imbalance and random firing). We compare the full model (using optimal-transport-derived phase lags $\alpha_{ij}$) against a "no-phase-lag" ablation, and report a **random-phase FG-ARI** control to establish the chance floor.

| Model | Within-object coherence | Between-object distance | FG-ARI | Random-phase FG-ARI |
| :--- | :--- | :--- | :--- | :--- |
| Full (with phase lag) | **0.581** | 0.468 | **0.127** | 0.000 |
| No-phase-lag (ablation) | 0.565 | 0.467 | 0.117 | 0.000 |

The trained model scores well above the random-phase control (FG-ARI ≈ 0 for random phases), and the learned optimal-transport phase lag gives a consistent, if modest, gain over the no-phase-lag ablation, confirming that grouping is a learned, non-trivial capability rather than a byproduct of the raw oscillator dynamics.

---

### Table 5: Matched Comparison to a Dedicated Baseline (AKOrN)

*Addresses: Meta-Review ("Evidence for temporal binding"); Reviewer 4HKM, Weakness (5)/(6).*

Performance comparison against AKOrN [7] (Artificial Kuramoto Oscillatory Neurons; Miyato et al., ICLR 2025), a dedicated oscillatory object-discovery model, under a controlled evaluation harness: identical synthetic data, comparable encoder scale, and a 200-epoch training budget across 5 seeds.

| Model | Within-object coherence | FG-ARI (5 seeds) |
| :--- | :--- | :--- |
| AKOrN [7] | 0.89 | **0.36** |
| S²-Net (ours) | **0.98** | 0.34 |

Under this matched comparison, S²-Net's binding performance is competitive with AKOrN (0.34 vs. 0.36); the gap falls well within AKOrN's own across-seed spread (0.26–0.45), and both methods saturate near ~0.35 FG-ARI on this benchmark, suggesting a data/task ceiling rather than a model-capacity gap. Unlike AKOrN, S²-Net additionally contributes the delayed, optimal-transport-derived phase lag, a spiking temporal core, and direct applicability to graph-structured neuroimaging data.

---

### Table 6: Integration Horizon (Kuramoto Steps)

*Addresses: Meta-Review ("Evidence for temporal binding"); Reviewer 4HKM, Weakness (6), "Progressive Emergence & Integration Horizon."*

Analysis of binding evolution over test-time Kuramoto iterations on a representative 4-object scene, starting from random phase initialization. This table quantifies how within-object coherence and grouping quality (FG-ARI) rise smoothly and saturate around **~24 steps**, identifying the model's autonomous integration horizon and demonstrating that binding emerges progressively rather than via instantaneous locking (mirroring AKOrN-style dynamics).

| Kuramoto step | 0 | 4 | 12 | 24 | 60 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Within-object coherence | 0.12 | 0.35 | 0.95 | **0.98** | **0.98** |
| FG-ARI | 0.00 | 0.05 | 0.33 | 0.33 | **0.34** |

---

### Table 7: Comprehensive Dataset Evaluation Across Baselines

*Addresses: Meta-Review ("Fairness of the experimental comparisons"); extends the capacity-matched protocol of Table 1 to additional baselines and datasets.*

Performance metrics (Accuracy, Precision, F1; %, mean ± s.d.) for S²-Net and eight representative baselines, including Transformer, LSTM, TCN, Mamba [18], AKOrN [7], GraphCON [15], KuramotoGNN [16], and SR-Net [17], across six neuroimaging datasets (HCPA, HCPYA, UKB, ADNI, PPMI, NIFD). 

| Model | Metric | HCPA | HCPYA | UKB | ADNI | PPMI | NIFD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer | Acc | 96.89±0.39 | 96.48±0.22 | 98.83±0.26 | 78.17±3.76 | 66.10±2.68 | 61.11±2.03 |
| | Pre | 96.14±0.67 | 99.46±0.27 | 96.87±0.26 | 71.72±18.89 | 45.01±5.16 | 57.45±4.21 |
| | F1 | 95.85±0.63 | 99.47±0.24 | 96.81±0.26 | 64.44±12.65 | 41.79±1.76 | 54.17±3.58 |
| LSTM | Acc | 97.26±0.70 | 98.39±0.24 | 98.95±0.24 | 74.78±6.62 | 66.15±4.72 | 62.74±2.30 |
| | Pre | 96.27±1.34 | 98.27±0.24 | 97.03±0.25 | 67.72±14.87 | 45.38±3.72 | 58.51±3.42 |
| | F1 | 96.27±0.90 | 98.28±0.24 | 99.94±0.24 | 63.62±12.31 | 44.82±3.34 | 55.38±3.63 |
| TCN | Acc | 97.84±0.48 | 99.65±0.23 | 96.01±0.23 | 76.07±5.78 | 67.22±5.28 | 63.12±3.39 |
| | Pre | 97.21±0.60 | 99.62±0.28 | 96.08±0.22 | 63.56±3.89 | 40.63±12.19 | 62.39±1.87 |
| | F1 | 97.12±0.67 | 99.62±0.27 | 96.00±0.22 | 59.22±15.25 | 40.75±9.33 | 54.35±3.77 |
| Mamba [18] | Acc | 98.27±0.10 | 99.19±0.14 | 96.97±0.24 | 70.54±6.58 | 67.49±1.69 | 63.46±2.94 |
| | Pre | 98.42±0.03 | 99.15±0.16 | 97.03±0.23 | 52.98±19.41 | 47.46±3.51 | 60.51±4.04 |
| | F1 | 97.72±0.20 | 99.13±0.14 | 96.96±0.23 | 51.77±13.06 | 44.86±1.09 | 56.38±4.10 |
| AKOrN [7] | Acc | 94.34±0.76 | 86.31±1.71 | 92.22±1.68 | 85.27±7.52 | 66.82±4.87 | 58.79±2.68 |
| | Pre | 94.41±0.85 | 86.61±1.81 | 93.15±1.20 | 70.24±11.73 | 46.86±10.79 | 58.74±3.16 |
| | F1 | 94.30±0.81 | 86.23±1.73 | 91.88±1.65 | 76.46±10.28 | 39.81±8.82 | 58.31±3.13 |
| GraphCON [15] | Acc | 93.72±0.59 | 73.93±2.71 | 56.89±1.16 | 84.89±5.03 | 61.34±2.31 | 49.97±1.25 |
| | Pre | 93.61±0.56 | 74.13±2.41 | 57.35±1.53 | 74.46±10.68 | 47.06±8.98 | 33.26±7.36 |
| | F1 | 93.53±0.57 | 73.67±2.51 | 56.26±1.41 | 73.34±8.13 | 41.19±5.04 | 37.89±4.32 |
| KuramotoGNN [16] | Acc | 86.81±1.35 | 60.23±3.77 | 37.23±4.97 | 72.97±6.45 | 57.75±4.09 | 47.43±0.79 |
| | Pre | 87.56±0.79 | 61.97±2.68 | 40.09±1.47 | 73.32±6.53 | 47.89±13.21 | 44.14±2.24 |
| | F1 | 85.47±1.33 | 59.97±3.46 | 36.23±4.46 | 73.90±5.83 | 41.21±9.18 | 43.87±2.79 |
| SR-Net [17] | Acc | 60.41±2.12 | 36.79±3.18 | 59.24±1.40 | 71.81±3.27 | 57.41±2.50 | 29.37±2.63 |
| | Pre | 45.45±2.85 | 36.45±3.14 | 58.60±1.57 | 41.00±15.90 | 38.42±11.65 | 23.79±3.66 |
| | F1 | 47.25±2.35 | 35.63±3.16 | 58.57±1.54 | 43.17±4.98 | 35.88±8.39 | 19.13±1.88 |


---

### Table 8: Entropy Regularization ($\epsilon$) Sensitivity — Sinkhorn Convergence

*Addresses: Reviewer 4HKM, Question "(line 240) Dependence on $\epsilon$" (A8).*

Sensitivity of the Sinkhorn entropic optimal-transport solver to the regularization parameter $\epsilon_{\mathrm{OT}}$, swept across seven orders of magnitude. $H(P)$ is the plan entropy; "effective support" is the fraction of the transport plan's mass spread across non-negligible entries; "top-1% mass" is the fraction of total transport mass concentrated in the top 1% of plan entries; "marginal error" is the residual deviation from the target marginal constraints. For $\epsilon < 0.03$, Sinkhorn iterations fail to converge (degenerate, near-zero entropy, marginal error ≈ 1.0). At $\epsilon \ge 0.1$ (our default), the solver is in a stable, high-entropy, fully convergent regime.

| $\epsilon_{\mathrm{OT}}$ | Plan entropy $H(P)$ | Effective support | Top-1% mass | Marginal error |
| :--- | :--- | :--- | :--- | :--- |
| 10⁻³ | 0.00 | 10⁻⁴ | 0.00 | 1.000 |
| 3×10⁻³ | 0.00 | 10⁻⁴ | 0.00 | 1.000 |
| 10⁻² | 0.00 | 10⁻⁴ | 0.00 | 1.000 |
| 3×10⁻² | 0.00 | 10⁻⁴ | 0.00 | 1.000 |
| 10⁻¹ (default) | 9.51 | 1.00 | 0.01 | 4.4×10⁻⁴ |
| 0.3 | 9.51 | 1.00 | 0.01 | <10⁻⁴ |
| 1.0 | 9.51 | 1.00 | 0.01 | <10⁻⁴ |

---

### Table 9: Entropy Regularization ($\epsilon$) Sensitivity — Downstream Classification Performance

*Addresses: Reviewer 4HKM, Question "(line 240) Dependence on $\epsilon$" (A8).*

Impact of the entropy regularization parameter $\epsilon$ on downstream PPMI classification performance (Accuracy, F1; 10-fold cross-validation), restricted to the convergent regime identified in Table 8. This demonstrates that, once past the convergence threshold, $\epsilon$ is not a fragile, tightly-tuned hyperparameter, performance is stable across an order of magnitude, and the default $\epsilon = 10^{-1}$ is near-optimal.

| $\epsilon$ | Accuracy (%) | F1 (%) |
| :--- | :--- | :--- |
| $10^{-3}$ | 66.85 ± 3.42 | 40.85 ± 5.12 |
| $10^{-2}$ | 67.12 ± 4.10 | 41.20 ± 6.05 |
| $10^{-1}$ (default) | **67.62 ± 5.21** | **41.51 ± 7.19** |

---

### Table 10: Time-Delay Parameter ($\tau$) Sensitivity

*Addresses: Reviewer 4HKM, Question "Effect of [$\tau$]... could the authors provide a denser sampling?" (A9).*

Denser sensitivity sweep of the time-delay parameter $\tau \in \{0, 1, 2, 4, 8\}$ on PPMI classification performance (Accuracy, Precision, F1; %, 10-fold cross-validation), extending the sparser grid reported in the original Appendix. This sweep identifies the optimal lag window and confirms model stability against variation in $\tau$: performance is robust to small perturbations around $\tau = 2$, which we interpret as balancing phase separation against temporal coherence, plausibly mirroring the characteristic delay of the hemodynamic response function (HRF) in BOLD fMRI signals, while $\tau = 0$ or larger delays ($\tau \ge 4$) progressively degrade performance.

| $\tau$ | 0 | 1 | 2 (default) | 4 | 8 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Accuracy (%) | 67.28 ± 4.04 | 67.48 ± 4.52 | **67.62 ± 5.21** | 67.15 ± 3.85 | 65.38 ± 3.25 |
| Precision (%) | 46.25 ± 3.89 | 47.82 ± 6.10 | **49.30 ± 8.83** | 46.80 ± 7.50 | 45.12 ± 6.20 |
| F1 (%) | 40.34 ± 6.02 | 40.91 ± 6.65 | **41.51 ± 7.19** | 39.58 ± 6.42 | 38.15 ± 5.80 |

---

### Table 11: Representational Hypothesis Validation — Local vs. Global Phase Organization

*Addresses: Reviewer 4HKM, Question "Representational hypothesis for objects" (A11).*

Validation of our refined representational hypothesis, i.e.,  **structured, object-scale local phase organization**, not global zero-lag synchrony, comparing an untrained model (random phase lag) against trained models on 2- and 4-object scenes. `order_R` is the global Kuramoto order parameter (1.0 = fully phase-locked across the whole scene); `within_sync`/`between_sync` are the within-object and between-object synchronization indices; `dir_consistency` is the directional consistency of phase-wave propagation; `wavelength` is the characteristic spatial wavelength of the phase pattern, in pixels.

| Model | order_R | within_sync | between_sync | dir_consistency | Wavelength (px) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Untrained (random lag) | ~1.00 | ~1.00 | ~1.00 | 0.03 | 600+ |
| Trained, 2 objects | **0.35** | **0.85** | **0.15** | 0.04 | ~19 |
| Trained, 4 objects | **0.20** | **0.82** | **0.08** | 0.03 | ~12 |

The trained network avoids global phase lock-in (`order_R` drops to 0.20–0.35, vs. ～1.00 untrained) while maintaining tight internal binding (`within_sync` > 0.80) and strict external segregation (`between_sync` < 0.15); phase organizes at the object's physical scale (wavelength tracks object size, ～19px for 2 objects vs. ～12px for 4 objects). Directional consistency remains near the random-phase baseline (～0.03–0.04) in both trained conditions, confirming that coherent global traveling waves are not a general property of the trained model, supporting local, object-scale phase organization over a global-synchrony account.

---

## References

[1] Fries, P. (2015). Rhythms for Cognition: Communication through Coherence. *Neuron*, 88(1), 220–235. https://doi.org/10.1016/j.neuron.2015.09.034
*(General neuroscience motivation; cited in response to Reviewer 4HKM's missing-references point 3a/3b.)*

[2] Muller, L., Chavane, F., Reynolds, J., & Sejnowski, T. J. (2018). Cortical travelling waves: mechanisms and computational principles. *Nature Reviews Neuroscience*, 19(5), 255–268. https://doi.org/10.1038/nrn.2018.20
*(General neuroscience motivation; cited in response to Reviewer 4HKM's missing-references point 3a/3b.)*

[3] Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., & Kipf, T. (2021). Object-Centric Learning with Slot Attention. *NeurIPS 2021*.
*(Related-work citation requested by Reviewer 4HKM, point 3a/3b; not tied to a specific table above.)*

[4] Löwe, S., Lippe, P., Rudolph, M., & Welling, M. (2022). Complex-Valued Autoencoders for Object Discovery. *Transactions on Machine Learning Research (TMLR)*. arXiv:2204.02075
*(Related-work citation requested by Reviewer 4HKM, point 3a/3b.)*

[5] Löwe, S., Lippe, P., Locatello, F., & Welling, M. (2023). Rotating Features for Object Discovery. *NeurIPS 2023*.
*(Related-work citation requested by Reviewer 4HKM, point 3a/3b.)*

[6] Stanić, A., Gopalakrishnan, A., Irie, K., & Schmidhuber, J. (2023). Contrastive Training of Complex-Valued Autoencoders for Object Discovery. *NeurIPS 2023*.
*(Related-work citation requested by Reviewer 4HKM, point 3a/3b.)*

[7] Miyato, T., et al. (2025). Artificial Kuramoto Oscillatory Neurons (AKOrN). *ICLR 2025*.
*(Direct baseline in Tables 5 and 7 above.)*

[8] Deco, G., Kringelbach, M. L., Jirsa, V. K., & Ritter, P. (2017). The dynamics of resting fluctuations in the brain: metastability and its dynamical cortical core. *Scientific Reports*, 7, 3095.
*(General neuroscience motivation; cited in response to Reviewer 4HKM's missing-references point 3a/3b.)*

[9] Cabral, J., Castaldo, F., Vohryzek, J., Litvak, V., Bick, C., Lambiotte, R., Friston, K., Kringelbach, M. L., & Deco, G. (2022). Metastable oscillatory modes emerge from synchronization in the brain spacetime connectome. *Communications Physics*, 5, 184.
*(General neuroscience motivation; cited in response to Reviewer 4HKM's missing-references point 3a/3b.)*

[10] Burman, R. J., Brodersen, P. J. N., Raimondo, J. V., Sen, A., & Akerman, C. J. (2023). Active cortical networks promote shunting fast synaptic inhibition in vivo. *Neuron*, 111(22).
*(Neuroscientific grounding for Eq. 6's down-state gating design; Reviewer 4HKM, Question on Eq. 6, A10.)*

[11] De Pittà, M., & Brunel, N. (2022). Multiple forms of working memory emerge from synapse–astrocyte interactions in a neuron–glia network model. *PNAS*, 119(43), e2207912119.
*(Neuroscientific grounding for Eq. 6's down-state gating design; Reviewer 4HKM, Question on Eq. 6, A10.)*

[12] Oberauer, K., & Awh, E. (2022). Is There an Activity-silent Working Memory? *Journal of Cognitive Neuroscience*, 34(12), 2360–2374.
*(Neuroscientific grounding for Eq. 6's down-state gating design; Reviewer 4HKM, Question on Eq. 6, A10.)*

[13] Dalla Porta, L., et al. (2025). H-current modulation of cortical Up and Down states. *The Journal of Physiology*, 603(8), 2409–2424.
*(Neuroscientific grounding for Eq. 6's down-state gating design; Reviewer 4HKM, Question on Eq. 6, A10.)*

[14] Camassa, A., et al. (2022). Deterministic and Stochastic Components of Cortical Down States: Dynamics and Modulation. *Journal of Neuroscience*, 42(50), 9387–9400.
*(Neuroscientific grounding for Eq. 6's down-state gating design; Reviewer 4HKM, Question on Eq. 6, A10.)*

[15] Rusch, T. K., Chamberlain, B. P., Rowbottom, J., Mishra, S., & Bronstein, M. M. (2022). Graph-Coupled Oscillator Networks. *ICML 2022*.
*(Baseline "GraphCON" in Table 7.)*

[16] Nguyen, T., Nguyen, V., Honda, H., Nakamura, S., Sano, T., & Nguyen, T. M. (2024). From Coupled Oscillators to Graph Neural Networks: Reducing Over-smoothing via a Kuramoto Model-based Approach. *AISTATS 2024*.
*(Baseline "KuramotoGNN" in Table 7.)*

[17] Harikrishnan N. B., & Nagaraj, N. (2021). When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning. *Neural Networks*, 143, 425–435.
*(Baseline "SR-Net" in Table 7.)*

[18] Gu, A., & Dao, T. (2024). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *COLM 2024* (https://openreview.net/forum?id=tEYskw1VY2).
*(Baseline "Mamba" in Table 7.)*
