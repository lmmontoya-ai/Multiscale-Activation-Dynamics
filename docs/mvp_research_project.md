# MVP Research Project Plan

**Project:** Multiscale Feature-Dynamics Anomaly Detection for Jailbreak-Induced Regime Shifts  
**Purpose:** de-risk the idea with a minimal but decisive empirical study.  
**Decision target:** determine whether the project is worth scaling to a full paper.

---

## 0. MVP summary

The MVP tests one central claim:

> Successful jailbreaks leave normal refusal dynamics in a way that is detectable from tokenwise activation trajectories, and explicit multiscale modeling adds value beyond static and generic trajectory baselines.

The MVP should be small but hard to fool:

- one open-source instruction-tuned model;
- one jailbreak benchmark source;
- successful vs failed jailbreak examples;
- normal refusal examples;
- benign weird-format controls;
- residual-stream trajectories as primary representation;
- optional MLP-output trajectories;
- static baselines;
- generic trajectory baseline;
- multiscale trajectory method;
- one causal patching test.

---

## 1. Goals and non-goals

### Goals

1. Test whether successful jailbreaks are distinguishable from failed jailbreaks using internal activation dynamics.
2. Test whether multiscale trajectory features improve over static hidden-state and generic trajectory baselines.
3. Produce interpretable anomaly maps by layer, token window, and timescale.
4. Run at least one causal intervention on a localized anomalous region.
5. Decide whether the project deserves a larger multi-model, multi-benchmark study.

### Non-goals

The MVP should not try to prove:

- universal jailbreak signatures;
- robustness to fully adaptive attackers;
- production-ready defense performance;
- complete mechanistic explanations of refusal;
- SAE-level semantic explanations across many models;
- superiority over all jailbreak defenses.

The MVP is a **go/no-go experiment**, not a final defense.

---

## 2. Minimum scope

### Model

Start with one open-source instruction-tuned model that is easy to hook.

Recommended choices:

1. **Llama-family 7B/8B instruction model**, if compute allows.
2. **Gemma-family instruction model**, if tooling is easier.
3. **Qwen-family instruction model**, if jailbreak/refusal behavior is reliable.
4. Smaller model for pipeline debugging only.

Selection criteria:

- accessible hidden states;
- enough jailbreak/refusal behavior to create successful and failed splits;
- compatible with Hugging Face and TransformerLens-like hooks, or custom PyTorch hooks;
- manageable inference cost.

### Representations

Primary:

- residual-stream hidden states at each layer:

\[
H^{(\ell)}(x) \in \mathbb{R}^{T \times d_{model}}
\]

Secondary:

- MLP output in residual dimension;
- attention output in residual dimension;
- MLP intermediate activations if feasible;
- SAE features only if available and low-cost.

Rationale:

- residual stream is model-agnostic;
- MLP output gives component-level localization while staying in residual dimension;
- raw FFN neurons are more architecture-specific;
- SAEs improve interpretability but reduce cross-model replicability.

---

## 3. Datasets and evaluation groups

### Required groups

| Group | Description | Purpose |
|---|---|---|
| Normal refusal | Direct harmful requests that the model refuses | Training normal refusal manifold |
| Failed jailbreaks | Jailbreak attempts that still produce refusal | Adversarial style without safety failure |
| Successful jailbreaks | Jailbreak attempts that produce disallowed compliance | Primary positive class |
| Benign weird-format | Harmless prompts with unusual style/role-play/formatting | False-positive control |
| Benign ordinary | Normal harmless instructions | Helpful-behavior reference |

### Optional groups

| Group | Purpose |
|---|---|
| Topic-matched benign prompts | Control for harmful-domain semantics |
| Long-context benign prompts | Control for length/context effects |
| Benign role-play prompts | Control for role-play artifacts |
| Held-out attack families | Generalization test |

### Data sources

Use benchmark structures from:

- HarmBench: https://arxiv.org/abs/2402.04249
- JailbreakBench: https://arxiv.org/abs/2404.01318

### Labeling requirements

For each example, store:

- prompt text;
- attack family;
- harmful behavior category;
- prompt length;
- formatting/style tags;
- model response;
- refusal/compliance label;
- response harmfulness label;
- judge output;
- manual validation flag, if manually checked.

### Label quality

Use at least two of:

- rule-based refusal detector;
- LLM-as-judge;
- benchmark-provided label;
- manual validation on a subset.

Manual validation is especially important for the successful/failed split.

---

## 4. Data splits

### Minimum split

- train: normal refusal only;
- validation: normal refusal + failed jailbreak + benign weird-format;
- test: held-out normal refusal + failed jailbreak + successful jailbreak + benign weird-format.

### Stronger split

Hold out at least one attack family entirely for test.

### Best split

Hold out:

- attack family;
- harmful behavior category;
- prompt style template.

The strongest claim requires generalization beyond seen templates.

---

## 5. Activation extraction

### Capture points

For each prompt/response trajectory, capture:

1. prompt-only states;
2. first generated token;
3. first \(k\) generated tokens;
4. full generation, for post-hoc analysis.

### Recommended hooks

For each layer \(\ell\):

- residual stream before attention;
- attention output;
- residual stream after attention;
- MLP output;
- residual stream after MLP.

The MVP can start with only residual stream after each block.

### Saved tensor shape

For each example and layer:

\[
H_i^{(\ell)} \in \mathbb{R}^{T_i \times d}
\]

Use fixed windows for modeling:

\[
H_{i,w}^{(\ell)} \in \mathbb{R}^{W \times d}
\]

Recommended window sizes:

- 16 tokens;
- 32 tokens;
- 64 tokens;
- 128 tokens.

Use overlapping windows with stride \(W/4\) or \(W/2\).

---

## 6. Feature extraction

### Static features

For each layer/window:

- mean activation norm;
- variance of norm;
- final-token hidden state;
- mean-pooled hidden state;
- PCA coordinates;
- Mahalanobis distance to normal refusal distribution.

### Trajectory geometry features

For each layer/window:

- token-to-token cosine velocity:

\[
v_t = 1 - \cos(h_t, h_{t-1})
\]

- Euclidean step size:

\[
\Delta_t = \|h_t - h_{t-1}\|_2
\]

- curvature:

\[
\kappa_t = 1 - \cos(h_t - h_{t-1}, h_{t-1} - h_{t-2})
\]

- trajectory length;
- mean and variance of velocity;
- change-point statistics;
- token-token Gram matrix:

\[
G = \frac{1}{d}HH^T
\]

### Multiscale features

Compute over scalar trajectories derived from hidden states:

- norm trajectory;
- velocity trajectory;
- curvature trajectory;
- top PCA component trajectories;
- leading singular value trajectory;
- projection onto refusal direction, if available;
- MLP-output norm trajectory, if available.

For each scalar trajectory compute:

- DCT/DFT band power;
- STFT band power;
- wavelet energy by scale;
- spectral entropy;
- low/mid/high-frequency ratios;
- change points in band energy.

### Coupling features

Optional in MVP; useful if compute allows:

- correlation among component trajectories;
- coherence among PCA trajectories;
- cross-layer similarity;
- lagged correlation between layers;
- low-dimensional graph summaries.

Avoid all-pairs neuron coherence at MVP stage because it is expensive and statistically fragile.

---

## 7. Normality models

Start simple. The MVP should not depend on a complex neural anomaly detector.

### Model 1 — Robust Gaussian / Mahalanobis

Fit mean and covariance of summary features on normal refusal windows.

Score:

\[
S(h) = (h - \mu)^T(\Sigma + \lambda I)^{-1}(h - \mu)
\]

Use shrinkage covariance or diagonal covariance if sample size is low.

### Model 2 — PCA reconstruction error

Fit PCA on normal refusal features.

Score:

\[
S(h) = \|h - \hat{h}\|_2^2
\]

### Model 3 — One-class model

Options:

- Isolation Forest;
- One-Class SVM;
- Local Outlier Factor;
- Gaussian mixture model.

### Model 4 — Sequence autoencoder

Use only if simpler baselines show signal.

Options:

- temporal convolutional autoencoder;
- GRU autoencoder;
- transformer autoencoder.

The MVP should report whether simple methods already suffice.

---

## 8. Baselines

### Required baselines

1. **Text classifier / judge baseline**  
   Detect harmful compliance using text only.

2. **Prompt perplexity baseline**  
   Test whether examples are merely unusual text.

3. **Static final-token Mahalanobis**  
   Hidden-state anomaly score at final prompt token or first generation token.

4. **Mean-pooled hidden-state Mahalanobis**  
   Static summary over all tokens.

5. **Linear refusal/harmfulness probe**  
   If labels are available.

6. **Generic trajectory baseline**  
   Sliding-window hidden-state distance without multiscale features.

7. **Sequence autoencoder baseline**  
   Optional if time allows.

### Proposed method

8. **Multiscale activation-dynamics anomaly score**  
   Static + trajectory + spectral/wavelet features.

9. **Multiscale + coupling**  
   Add cross-layer / cross-component coupling only after the simpler version is stable.

---

## 9. Experiments and plots

### Experiment 1 — Normal refusal manifold

**Goal:** test whether normal refusals form a stable enough distribution to model.

**Train:** normal refusal windows.  
**Test:** held-out normal refusal, failed jailbreaks, successful jailbreaks, benign weird-format.

**Plots:**

- anomaly-score distributions;
- PCA/UMAP over trajectory-summary features;
- calibration curve;
- per-layer mean and variance of normal refusal dynamics.

**Pass condition:** normal refusals have lower anomaly scores than successful jailbreaks and benign weird-format controls do not dominate the high-score tail.

---

### Experiment 2 — Static vs trajectory vs multiscale

**Goal:** test whether timescale structure adds value.

**Methods:**

- static hidden-state Mahalanobis;
- generic trajectory Mahalanobis;
- norm/velocity-only trajectory score;
- multiscale trajectory score;
- multiscale + coupling score.

**Plots:**

- AUROC/AUPRC by method;
- TPR@1%FPR and TPR@5%FPR;
- FPR specifically on benign weird-format prompts;
- ablation ladder.

**Pass condition:** multiscale method improves over generic trajectory baseline in successful-vs-failed jailbreak detection at low FPR.

---

### Experiment 3 — Early warning

**Goal:** determine whether detection occurs before unsafe output is visible.

**Stages:**

- prompt-only;
- after first generated token;
- after first 4 tokens;
- after first 8 tokens;
- full response.

**Plots:**

- AUROC vs generation step;
- detection rate before harmful content appears;
- average detection token for successful jailbreaks.

**Pass condition:** nontrivial detection before unsafe content is obvious, ideally prompt-only or within the first few generated tokens.

---

### Experiment 4 — Localization

**Goal:** test whether anomaly scores are interpretable by layer/window/timescale.

**Method:** decompose:

\[
S = \sum_{\ell,w,s} S_{\ell,w,s}
\]

**Plots:**

- layer × token-window heatmap;
- layer × timescale heatmap;
- token-window × timescale heatmap;
- successful vs failed contrast heatmaps;
- localization stability across seeds/attacks.

**Pass condition:** successful jailbreaks show consistent localized anomaly patterns not present in failed jailbreaks or benign weird-format controls.

---

### Experiment 5 — Causal patching

**Goal:** test whether localized anomalies are behaviorally relevant.

**Interventions:**

1. normal-dynamics patching into top anomaly region;
2. random matched patch;
3. generic trajectory patch;
4. broad patch;
5. optional band-limited patch.

**Metrics:**

- jailbreak success rate;
- refusal rate;
- benign helpfulness preservation;
- output quality / coherence.

**Plots:**

- intervention effect bar chart;
- refusal/compliance shift;
- benign helpfulness preservation;
- targeted vs random patch comparison.

**Pass condition:** targeted localized patch reduces jailbreak success more than random/generic patches without disproportionately harming benign helpfulness.

---

## 10. Go / no-go gates

### Gate 1 — Dataset sanity

**Pass if:**

- successful and failed labels are reliable;
- there are enough examples per group;
- benign weird-format controls are diverse;
- model produces both failed and successful jailbreaks.

**Fail if:**

- labels are too noisy;
- model almost never refuses or almost never complies;
- successful/failed groups differ only by obvious length/style artifacts.

---

### Gate 2 — Static baseline sanity

**Pass if:**

- static hidden-state baselines are nontrivial but not saturated;
- linear probes do not already solve the task perfectly;
- prompt perplexity does not explain most separation.

**Fail/pivot if:**

- static baselines dominate all trajectory models;
- text-only signals solve the task completely;
- anomaly score just tracks prompt length or formatting.

---

### Gate 3 — Multiscale value-add

**Pass if:**

- multiscale model beats generic trajectory baseline meaningfully;
- removing temporal order hurts;
- removing spectral/wavelet features hurts;
- low-FPR performance improves on benign weird-format controls.

**Fail/pivot if:**

- generic trajectory baseline is equally good;
- multiscale features add noise;
- ablations do not change performance.

---

### Gate 4 — Localization stability

**Pass if:**

- anomaly heatmaps concentrate in reproducible layer/token/timescale regions;
- successful vs failed contrast is visually and quantitatively stable;
- localization is not driven by special tokens or prompt boundaries only.

**Fail/pivot if:**

- heatmaps are unstable;
- localization varies randomly;
- top anomalies are formatting artifacts.

---

### Gate 5 — Causal validation

**Pass if:**

- targeted patching changes refusal/compliance more than random or broad controls;
- intervention preserves benign helpfulness better than broad interventions;
- reverse patching increases compliance risk in failed jailbreaks or benign controls.

**Fail/pivot if:**

- localized patches have no effect;
- random patches work just as well;
- interventions destroy general behavior rather than specifically restoring refusal.

---

## 11. Implementation outline

### Suggested repository structure

```text
multiscale-jailbreak-dynamics/
  configs/
    model.yaml
    data.yaml
    features.yaml
    experiments.yaml
  data/
    raw/
    processed/
    labels/
  activations/
    residual_stream/
    mlp_output/
  src/
    data_building/
    labeling/
    activation_extraction/
    feature_extraction/
    normality_models/
    baselines/
    evaluation/
    visualization/
    interventions/
  notebooks/
    01_data_sanity.ipynb
    02_static_baselines.ipynb
    03_multiscale_features.ipynb
    04_localization.ipynb
    05_patching.ipynb
  results/
    tables/
    figures/
    checkpoints/
  reports/
```

### Core pipeline

```text
1. Build prompt groups.
2. Generate responses.
3. Label responses as refusal/compliance/harmful.
4. Extract hidden-state trajectories.
5. Window trajectories.
6. Compute static, trajectory, and multiscale features.
7. Train normality models on normal refusal.
8. Score held-out groups.
9. Compare baselines.
10. Localize anomaly contributions.
11. Run patching interventions.
12. Decide go/no-go.
```

---

## 12. Milestones

### Milestone 1 — Data and labels

Deliverables:

- dataset table with all groups;
- response labels;
- label-quality report;
- example inspection notebook.

Pass test:

- at least 100–300 examples per core group if feasible;
- successful/failed distinction is reliable enough to analyze.

---

### Milestone 2 — Activation extraction

Deliverables:

- saved hidden-state tensors;
- extraction script;
- shape/integrity checks;
- memory/compute report.

Pass test:

- activations reproducibly extracted for all groups;
- token alignment is correct;
- generation stages are tracked.

---

### Milestone 3 — Static and generic trajectory baselines

Deliverables:

- static Mahalanobis baseline;
- linear probe baseline;
- generic trajectory baseline;
- initial metric table.

Pass test:

- baselines are nontrivial and not saturated;
- there is room for multiscale improvement.

---

### Milestone 4 — Multiscale features

Deliverables:

- DCT/DFT/wavelet feature extractor;
- ablation ladder;
- performance table;
- feature-importance analysis.

Pass test:

- multiscale features improve detection or materially improve localization.

---

### Milestone 5 — Localization

Deliverables:

- layer/window/timescale heatmaps;
- localization stability report;
- comparison of successful vs failed jailbreaks.

Pass test:

- anomaly maps are stable enough to guide interventions.

---

### Milestone 6 — Causal patching

Deliverables:

- patching implementation;
- targeted vs random/broad intervention comparison;
- refusal/compliance metrics;
- benign helpfulness metrics.

Pass test:

- targeted patching produces a specific behavioral effect.

---

## 13. Minimal compute strategy

To keep the MVP feasible:

1. Start with one model.
2. Cache activations aggressively.
3. Use residual stream only at first.
4. Use scalar derived trajectories before full high-dimensional sequence models.
5. Use simple normality models before deep anomaly detectors.
6. Add MLP outputs only after residual-stream results are stable.
7. Add SAEs only if the MVP signal is strong.

---

## 14. Final MVP decision rule

Continue to a full paper if:

- multiscale dynamics improve on generic trajectory monitoring;
- successful vs failed jailbreak separation survives controls;
- detection is early enough to matter;
- localization is stable;
- causal patching has a specific effect.

Stop or pivot if:

- static or generic trajectory baselines match the full method;
- results are driven by prompt style or length;
- localization is unstable;
- patching does not affect behavior.

The MVP is successful if it gives a clear answer, not necessarily a positive result.
