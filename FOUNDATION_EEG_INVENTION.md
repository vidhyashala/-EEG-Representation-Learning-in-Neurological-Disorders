# NeuroVerse-Ω: A Neuro-Symbolic Foundation Model for Cross-Subject EEG Representation Learning and Zero-Shot Seizure Forecasting

## 1) Architecture Design

### 1.1 High-level concept
We propose **NeuroVerse-Ω**, a foundation model that treats EEG as a **structured neuro-physiological language** rather than raw time series. The architecture is built around four non-standard principles:

1. **Neuro-Symbolic Tensor Tokenization (NSTT)** — tokens encode local waveform, spectral morphology, electrode topology, and interpretable neuro-events simultaneously.
2. **Tri-Factor Latent Geometry (universal / disorder / subject)** — latent space explicitly decomposed into universal neural dynamics, disorder style, and subject identity.
3. **Causal Neuro-Transition World Model** — learns lawful transitions of latent brain microstates under counterfactual perturbations.
4. **Energy-based Pre-Ictal Divergence Head (label-free)** — zero-shot seizure forecasting via latent trajectory energy and manifold geodesic curvature deviations, without seizure labels.

---

### 1.2 Text diagram (module-level)

```text
Input EEG (multi-channel, multi-rate, multi-dataset)
   |
   |--> Adaptive Montage Harmonizer (AMH)
   |        - maps arbitrary electrode montages to canonical latent graph
   |
   |--> Neuro-Symbolic Tensor Tokenizer (NSTT)
   |        - Wavelet packet atoms
   |        - Cross-frequency phase-amplitude couplings
   |        - Graph Laplacian spatial motifs
   |        - Event symbols (spikes, bursts, micro-arousals, tremor-like rhythms)
   |
   |--> Hypergraph State Encoder (HSE)
   |        - temporal-hypergraph attention over (time, freq, electrode-clique, symbol)
   |
   |--> Latent Splitter (Tri-Factor)
   |        z_u: universal neurodynamics
   |        z_d: disorder-style code
   |        z_s: subject-style code
   |
   |--> Causal Neuro-Transition World Model (CNT-WM)
   |        predicts p(z_{t+1} | z_t, intervention)
   |
   |--> Physiological Constraint Engine (PCE)
   |        enforces biophysical plausibility and cross-view consistency
   |
   |--> Energy-Geodesic Sentinel (EGS)
            zero-shot seizure forecasting score
```

---

### 1.3 Module novelty

#### A) Adaptive Montage Harmonizer (AMH)
Different datasets have different channel sets (10–20 vs high-density caps). AMH learns a **graph transport operator** from observed montage graph \(G_m\) to a canonical latent neurograph \(G_*\):
\[
\mathbf{X}_* = T_{m\to *}(\mathbf{X}_m),\quad T_{m\to *}=\arg\min_T \|L_* - T L_m T^\top\|_F^2 + \lambda\|T\|_1
\]
where \(L_m, L_*\) are graph Laplacians. This is not simple interpolation; it is topology-aware domain transport.

#### B) Neuro-Symbolic Tensor Tokenizer (NSTT)
For each short window \(w_t\), create a **4-way token tensor**:
\[
\tau_t \in \mathbb{R}^{K_{time}\times K_{freq}\times K_{space}\times K_{symbol}}
\]
- **Time atoms:** transient morphology basis coefficients.
- **Freq atoms:** multiband and cross-frequency coupling coefficients.
- **Space atoms:** spectral graph wavelets over electrode graph cliques.
- **Symbol atoms:** probabilistic detectors of neurologically meaningful motifs (spike, spindle, tremor burst, K-complex-like events).

Tokens are then compressed with tensor sketching into sparse codewords. This is neuro-symbolic and far beyond patch tokenization.

#### C) Hypergraph State Encoder (HSE)
Instead of pairwise attention, HSE uses **hyperedge attention** connecting multi-way relations among \((ch_i, ch_j, f_a, f_b, t)\). Captures phenomena like “fronto-temporal high-gamma burst with delta phase lock”.

#### D) Tri-Factor Latent Splitter
The latent embedding is decomposed:
\[
z_t = [z_t^{u}, z_t^{d}, z_t^{s}],\quad z_t^{u}\in\mathbb{R}^{p},\ z_t^{d}\in\mathbb{R}^{q},\ z_t^{s}\in\mathbb{R}^{r}
\]
- \(z^u\): disorder-agnostic universal dynamics
- \(z^d\): disorder-specific style (epilepsy/PD/sleep)
- \(z^s\): subject signature

Orthogonality and adversarial independence constraints enforce disentanglement.

#### E) Causal Neuro-Transition World Model (CNT-WM)
Learns transition operator in latent ODE form:
\[
\frac{dz^u}{dt} = f_\theta(z^u, u_t) + \epsilon_t
\]
where \(u_t\) are synthetic interventions (phase jitter, channel dropout, physiological perturbation). Model predicts both next latent and uncertainty. This gives dynamics, not static representation.

#### F) Energy-Geodesic Sentinel (EGS) for zero-shot seizure prediction
Given latent trajectory \(\gamma(t)=z^u_t\), define:
1. **Energy deviation** from learned normal score network \(E(z)\)
2. **Geodesic curvature anomaly** on manifold metric \(M(z)\)
3. **Forward inconsistency** between predicted and realized latent

Combined pre-ictal risk score:
\[
\mathcal{R}_t = \alpha E(z_t^u) + \beta \kappa_g(\gamma,t) + \eta\|\hat z_{t+1}^u-z_{t+1}^u\|_2
\]
No seizure labels are used.

---

## 2) Mathematical Formulation

### 2.1 Data and domains
Let datasets \(\mathcal{D} = \{\mathcal{D}^{epi},\mathcal{D}^{pd},\mathcal{D}^{slp}\}\). Each sample has signal \(x\), subject \(s\), dataset \(m\), and optional disorder tag \(d\) (used only for alignment head, not seizure supervision).

### 2.2 Encoders
\[
\tau_t = \text{NSTT}(x_{t:t+W}),\quad h_t=\text{HSE}(\tau_{1:t}),\quad [z_t^u,z_t^d,z_t^s]=\Pi(h_t)
\]

### 2.3 Novel objective family: **Neuro-Physiological Law Consistency (NPLC)**
Total loss:
\[
\mathcal{L}=\lambda_1\mathcal{L}_{law}+\lambda_2\mathcal{L}_{tri}+\lambda_3\mathcal{L}_{counter}+\lambda_4\mathcal{L}_{subject\_inv}+\lambda_5\mathcal{L}_{manifold}
\]

#### (i) Law consistency loss \(\mathcal{L}_{law}\)
Predict transition statistics under two augmentations that should preserve physiology:
\[
\mathcal{L}_{law}=\sum_t D_{KL}(p_\theta(z_{t+1}^u|z_t^u,a_1)\|p_\theta(z_{t+1}^u|z_t^u,a_2))
\]
for augmentation pair \((a_1,a_2)\) preserving neural law.

#### (ii) Tri-factor disentanglement \(\mathcal{L}_{tri}\)
\[
\mathcal{L}_{tri}=\underbrace{\|{Z^u}^\top Z^d\|_F^2 + \|{Z^u}^\top Z^s\|_F^2 + \|{Z^d}^\top Z^s\|_F^2}_{\text{orthogonality}} + \text{HSIC}(Z^u,Z^d)+\text{HSIC}(Z^u,Z^s)
\]

#### (iii) Counterfactual transition coherence \(\mathcal{L}_{counter}\)
Given intervention \(\delta\), require local linearized effect consistency:
\[
\mathcal{L}_{counter}=\sum_t \left\|\big(\hat z_{t+1}^{u,\delta}-\hat z_{t+1}^{u}\big)-J_t\delta\right\|_2^2
\]
where \(J_t=\partial f_\theta/\partial u\).

#### (iv) Subject invariance adversary \(\mathcal{L}_{subject\_inv}\)
Train adversary \(g_\psi\) to predict subject from \(z^u\), encoder tries to fool:
\[
\min_\theta\max_\psi\ \mathbb{E}[\log p_\psi(s|z^u)]
\]
with gradient reversal.

#### (v) Manifold regularity \(\mathcal{L}_{manifold}\)
Encourage smooth geodesics for normal dynamics:
\[
\mathcal{L}_{manifold}=\sum_t \|\nabla_{\dot\gamma_t}\dot\gamma_t\|_{M(z_t)}^2
\]
reduces noisy jagged latent paths, making anomaly detection sharper.

---

## 3) Training Pipeline

### 3.1 Preprocessing
1. Bandpass + notch + robust re-referencing.
2. Montage graph build per dataset.
3. AMH transport to canonical graph.
4. Windowing with multi-scale durations (0.5s, 2s, 8s).
5. Construct symbolic events via weak detectors (unsupervised peak/rhythm finders).

### 3.2 Multi-dataset curriculum
- **Stage A (intra-dataset law pretraining):** learn lawful transitions in each dataset separately.
- **Stage B (cross-dataset fusion):** mix batches across disorders; activate tri-factor and subject invariance.
- **Stage C (hard shift training):** deliberately hold out one dataset/site and enforce transport robustness.
- **Stage D (long-horizon latent forecasting):** world model rollout consistency for 30–120s context.

### 3.3 Self-supervised tasks (non-contrastive, non-MAE)
1. **Law-consistency prediction** under physiology-preserving transforms.
2. **Brain microstate grammar modeling**: predict valid next symbol classes in NSTT symbolic axis.
3. **Counterfactual intervention prediction**: estimate latent response to synthetic perturbations.
4. **Hyperedge completion**: infer missing multi-way relations in hypergraph token structure.

---

## 4) Zero-Shot Seizure Prediction Mechanism

### 4.1 Build normative latent prior
Using only unlabeled/pre-ictal-unknown data, fit score model \(s_\phi(z)=\nabla_z\log p_{normal}(z)\) over \(z^u\).
Energy proxy:
\[
E(z)=\frac{1}{2}\|s_\phi(z)\|_2^2
\]

### 4.2 Online inference
For streaming windows:
1. Encode to \(z_t^u\)
2. Predict \(\hat z_{t+1}^u\) from CNT-WM
3. Compute risk \(\mathcal{R}_t\)
4. Aggregate via exponential hazard smoother:
\[
H_t = \rho H_{t-1} + (1-\rho)\mathcal{R}_t
\]

### 4.3 Decision boundary (label-free)
Threshold from unsupervised extreme value theory (EVT) on tail of \(H_t\):
\[
\text{Alarm if } H_t > u + \frac{\sigma}{\xi}\left((\frac{N}{N_u}\alpha)^{-\xi}-1\right)
\]
where \((u,\sigma,\xi)\) are GPD parameters fit on baseline sessions.

This yields seizure warning without seizure labels or fine-tuning.

---

## 5) Ablation Strategy

1. **Replace NSTT with time patches** → quantify drop in cross-dataset transfer.
2. **Remove tri-factor split** → test leakage of subject/disorder information into universal code.
3. **Disable law-consistency loss** → evaluate latent drift and zero-shot false alarm rate.
4. **Disable manifold term** → evaluate geodesic curvature separability.
5. **No AMH transport** → test montage robustness across datasets.
6. **No counterfactual training** → test forecasting calibration under perturbations.

Primary metrics:
- Cross-subject linear probe accuracy (diagnosis-related tasks)
- Cross-dataset retrieval consistency
- Zero-shot seizure prediction: sensitivity, FA/24h, mean warning horizon
- OOD robustness under synthetic montage/channel shifts

Novelty proof:
- Demonstrate that performance gain comes from **law-consistent dynamics + tri-factor geometry**, not model size.
- Show emergent universal motifs shared across epilepsy/PD/sleep in \(z^u\).

---

## 6) Full Pipeline Pseudocode

```python
# NeuroVerse-Ω
initialize AMH, NSTT, HSE, Splitter, CNT_WM, SubjectAdv, ScoreNet

for stage in [A, B, C, D]:
    for batch in loader(stage):
        x, subject_id, disorder_id, dataset_id = batch

        # 1) Montage harmonization
        x_star = AMH.transport(x, dataset_id)

        # 2) Multi-scale tokenization
        tau = NSTT(x_star)  # tensor tokens: time x freq x space x symbol

        # 3) Hypergraph encoding
        h = HSE(tau)

        # 4) Tri-factor latent split
        z_u, z_d, z_s = Splitter(h)

        # 5) Construct physiology-preserving paired views
        x1, x2 = physio_augment(x_star), physio_augment(x_star)
        z1_u = encode_universal(x1)
        z2_u = encode_universal(x2)

        # 6) Transition modeling + counterfactuals
        z_next_pred, dist1 = CNT_WM(z_u, intervention=a1)
        z_next_pred2, dist2 = CNT_WM(z_u, intervention=a2)
        z_cf_pred = CNT_WM(z_u, intervention=delta)

        # 7) Losses
        L_law = KL(dist1, dist2)
        L_tri = orthogonality_hsic(z_u, z_d, z_s)
        L_counter = counterfactual_jacobian_consistency(z_next_pred, z_cf_pred, delta)
        L_subj = adversarial_subject_invariance(z_u, subject_id, SubjectAdv)
        L_manifold = geodesic_smoothness(z_u)

        L = λ1*L_law + λ2*L_tri + λ3*L_counter + λ4*L_subj + λ5*L_manifold

        optimize(L, modules=[AMH, NSTT, HSE, Splitter, CNT_WM])
        optimize_subject_adversary(SubjectAdv)

# Fit unsupervised normal score model
for x in baseline_unlabeled_stream:
    z_u = encode_universal(x)
    train_score_matching(ScoreNet, z_u)

# Zero-shot online seizure forecasting
H = 0
for x_t in streaming_eeg:
    z_t = encode_universal(x_t)
    z_pred = CNT_WM.predict_next(z_t)
    E = 0.5 * norm(ScoreNet(z_t))**2
    kappa = geodesic_curvature(z_history + [z_t])
    pred_err = norm(z_pred - observe_next_latent())
    R = α*E + β*kappa + η*pred_err
    H = ρ*H + (1-ρ)*R
    if H > EVT_threshold:
        raise_alarm()
```

---

## 7) Paper-Style Contributions (NeurIPS/ICML-ready)

1. **Neuro-Symbolic Tensor Tokenization (NSTT):** a fundamentally new EEG tokenization framework that unifies waveform, frequency coupling, electrode topology, and interpretable neural event symbols in a single tensor-token space.
2. **Tri-Factor Cross-Disorder Latent Geometry:** explicit decomposition into universal, disorder-specific, and subject-specific factors with adversarial and geometric disentanglement, enabling robust cross-subject and cross-dataset transfer.
3. **Neuro-Physiological Law Consistency Learning:** a new self-supervised objective family that learns lawful latent transitions under physiological invariances and counterfactual perturbations, moving beyond contrastive and masked reconstruction paradigms.
4. **Zero-Shot Seizure Forecasting via Energy-Geodesic Sentinel:** a label-free prediction mechanism combining score-based energy, manifold geodesic curvature, and transition inconsistency to detect pre-ictal deviations without seizure annotations.
5. **Unified Foundation Model Across Epilepsy, Parkinson’s, and Sleep Disorders:** first framework to jointly model multiple neurological disorder domains while preserving universal neurodynamics and delivering zero-shot clinical forecasting.

---

## 8) EXTRA Novelty Boost (new idea beyond current literature)

### Neural Grammar Field (NGF) — EEG as a Generative Neuro-Language
We introduce a **continuous neural grammar field** where symbolic EEG motifs are treated like grammar tokens with physiological production rules.

- Define nonterminal states \(\mathcal{N}\): latent neurostates (e.g., stable wake rhythm, transitional instability).
- Terminal symbols \(\Sigma\): NSTT event symbols.
- Production probabilities conditioned on \(z^u\):
\[
P(A \to BC\mid z^u_t),\quad P(A\to a\mid z^u_t)
\]
- Grammar violation score indicates pathological transitions.

This gives interpretable, disease-agnostic trajectories:
- normal: high-probability grammatical transitions
- pre-ictal: rising grammar perplexity and forbidden-rule activation

NGF is novel because it fuses symbolic formal language structure with continuous latent neurodynamics for EEG foundation modeling.

---

## Why this is publication-worthy
- It reframes EEG SSL from static representation learning to **lawful neurodynamic world modeling**.
- It introduces a principled route to **zero-shot clinical forecasting** using geometry + energy + grammar violations.
- It explicitly solves subject and dataset variability with decomposition and topology transport.
- It offers mechanistic interpretability that clinicians can inspect (symbols, grammar, transition laws).
