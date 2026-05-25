# Literature Review and Project Report

**Working title:** *Multiscale Feature-Dynamics Anomaly Detection for Localizing Jailbreak-Induced Regime Shifts in Language Models*  
**Short title:** *Mechanistic Anomaly Detection via Multiscale Feature Dynamics*  
**Purpose of this document:** de-risk the research idea by clarifying prior art, novelty, empirical claims, tests, pass/fail criteria, and whether the project is worth continuing.

---

## 0. Executive summary

The project is worth pursuing as a **de-risked MVP**, but the novelty is narrower than the broad framing suggests.

The idea is **not novel** as:

- spectral analysis of transformer activations;
- activation monitoring for safety;
- internal-representation anomaly detection;
- hidden-state trajectory jailbreak detection.

The idea may be novel and valuable as:

> A model-agnostic framework for detecting jailbreak-induced safety-regime shifts by learning **normal refusal dynamics** over tokenwise hidden-state trajectories, adding **multiscale time-frequency structure**, localizing anomalies by **layer / token window / representation component / timescale**, and validating localized dynamics with causal interventions.

The central de-risk question is:

> Does explicit multiscale modeling add measurable value beyond strong generic hidden-state trajectory baselines?

The project should continue only if the MVP shows at least one of the following:

1. multiscale dynamics improve detection of **successful vs. failed jailbreaks** at low false-positive rates;
2. anomaly heatmaps produce stable layer/token/timescale localization;
3. interventions on localized anomalous regions change refusal/compliance more than random or generic trajectory interventions.

---

## 1. Project thesis

Successful jailbreaks can be understood as departures from normal refusal dynamics inside an LLM. By modeling tokenwise activation trajectories across layers and timescales, we can detect, localize, and causally test those departures.

The paper should be framed as:

> **A mechanistic anomaly-detection method for safety-relevant internal regime shifts.**

It should not be framed as:

> “A new spectral theory of neuron behavior.”

The base draft motivates treating activations over token position as signals and using spectral / wavelet / coherence analysis. The stronger paper narrows that idea into safety-relevant, multivariate anomaly detection.

---

## 2. Literature review

### 2.1 Spectral and multiscale representation analysis

**Closest prior art:** Tamkin, Jurafsky, and Goodman, *Language Through a Prism: A Spectral Approach for Multiscale Language Representations*.

They analyze individual transformer neuron activations across input positions as signals, apply spectral filters, and show scale-specific information: word-level, utterance-level, and document-level information can be separated by frequency band.

**Implication for this project:** spectral filtering over transformer activations is not novel. The novelty cannot be “Fourier analysis of activations.” The novelty must be **safety-specific multiscale anomaly detection over activation trajectories**, with strong baselines and causal validation.

**Reference:** https://arxiv.org/abs/2011.04823

---

### 2.2 Population-level representation monitoring

**Representation Engineering** argues for monitoring and manipulating population-level representations rather than individual neurons or fully reverse-engineered circuits. It demonstrates simple methods for high-level safety-relevant concepts such as honesty, harmlessness, and power-seeking.

**Activation monitoring** work uses internal states as lightweight monitors for oversight and reports that activation monitors can be competitive with text-based classifiers in some settings.

**Implication:** internal-state monitoring is already a valid safety direction. The gap is that many monitors use static hidden states or pooled representations rather than the **sequence of internal states over token position**.

**References:**

- Representation Engineering: https://arxiv.org/abs/2310.01405
- Activation Monitoring for AI Safety: https://openreview.net/forum?id=qbvtwhQcH5

---

### 2.3 Sparse autoencoders and feature-level interpretability

Sparse autoencoders attempt to decompose activations into sparse, more interpretable features. Cunningham et al. report that SAEs learn more interpretable and monosemantic features than several alternatives and can localize causally responsible features in an IOI setting. Anthropic’s monosemanticity work similarly motivates moving beyond individual neurons.

However, recent SAE evaluation work cautions that SAE proxy metrics do not always translate cleanly to downstream interpretability or steering performance.

**Implication:** SAEs are useful as an optional interpretability layer, but the MVP should not depend on SAEs. A model-agnostic version should start with residual-stream or MLP-output trajectories and add SAE features only if available.

**References:**

- Sparse Autoencoders Find Highly Interpretable Features in Language Models: https://arxiv.org/abs/2309.08600
- Towards Monosemanticity: https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning
- SAEBench: https://arxiv.org/abs/2503.09532

---

### 2.4 Refusal directions and hidden-state safety mechanisms

Arditi et al. find that refusal behavior in chat models can be mediated by a one-dimensional residual-stream direction across many open-source models. Removing that direction suppresses refusal; adding it can induce refusal.

Other work on intermediate hidden states argues that alignment and jailbreak behavior can be understood through layerwise transformations from malicious-input recognition to refusal generation.

**Implication:** safety-relevant information is present in hidden states. This supports using residual-stream trajectories as the primary representation. But static directions are not enough to test whether **temporal dynamics** and **timescale structure** matter.

**References:**

- Refusal in Language Models Is Mediated by a Single Direction: https://arxiv.org/abs/2406.11717
- How Alignment and Jailbreak Work: https://arxiv.org/abs/2406.05644

---

### 2.5 Mechanistic anomaly detection and static latent anomaly detection

Johnston et al. study Mechanistic Anomaly Detection for “quirky” language models, using internal model features to detect anomalous training/test behavior. They find detectors can work well on some tasks, but no detector is effective across all models and tasks.

LoD formulates jailbreak detection as anomaly detection, training a safety-pattern autoencoder only on safe internal representations and detecting unknown jailbreaks through reconstruction error.

**Implication:** “learn safe/normal internal representations and detect anomalies” is not new. This project must distinguish itself by modeling **tokenwise multiscale trajectories**, not static representations.

**References:**

- Mechanistic Anomaly Detection for “Quirky” Language Models: https://arxiv.org/abs/2504.08812
- Learning to Detect Unknown Jailbreak Attacks: https://arxiv.org/abs/2508.09201

---

### 2.6 Trajectory-based jailbreak detection

TrajGuard is the closest competitor. It argues that static prompt/output/internal-state defenses neglect the dynamic evolution of risk during decoding, and it uses streaming hidden-state trajectories with sliding windows for jailbreak defense.

JBShield and subsequent RTV-style work use representation-level concept signals, refusal-direction fingerprints, and Mahalanobis outlier detection to defend against or analyze jailbreaks, including adaptive settings.

**Implication:** trajectory monitoring is already emerging. This project must show that **multiscale structure and mechanistic localization** add value beyond generic hidden-state trajectory scoring.

**References:**

- TrajGuard: https://arxiv.org/abs/2604.07727
- JBShield: https://arxiv.org/abs/2502.07557
- Revisiting JBShield / Representation Trajectory Verification: https://arxiv.org/abs/2605.03095

---

### 2.7 Jailbreak benchmarks and evaluation infrastructure

HarmBench standardizes automated red teaming and robust refusal evaluation. JailbreakBench provides an open benchmark with jailbreak artifacts, behaviors, threat models, prompts, templates, scoring functions, and leaderboard infrastructure.

**Implication:** the empirical target should use established benchmark structures where possible. The key split should be **successful jailbreaks vs. failed jailbreaks under matched controls**, not simply jailbreak prompts vs. benign prompts.

**References:**

- HarmBench: https://arxiv.org/abs/2402.04249
- JailbreakBench: https://arxiv.org/abs/2404.01318

---

### 2.8 Robustness and adaptive attacks

Obfuscated-activation work shows that latent-space defenses, including probes, SAE defenses, and latent OOD detectors, can be bypassed by making harmful behavior occur through inconspicuous activations.

Neural Chameleons shows that models can be fine-tuned to evade unseen activation monitors in trigger-conditioned ways.

**Implication:** this project should not claim to be a production-ready jailbreak defense or robust to fully adaptive attackers. It should include threat-model boundaries and, if feasible, partial adaptive evaluation.

**References:**

- Obfuscated Activations Bypass LLM Latent-Space Defenses: https://arxiv.org/abs/2412.09565
- Neural Chameleons: https://arxiv.org/abs/2512.11949

---

### 2.9 Mechanistic standard: causality

Saphra and Wiegreffe argue that the narrow technical meaning of “mechanistic” requires causal claims, while broader uses of the term may merely mean internal analysis.

**Implication:** anomaly detection alone is not mechanistic interpretability. The project needs patching, ablations, trajectory replacement, or band-limited interventions to justify the mechanistic claim.

**Reference:** https://arxiv.org/abs/2410.09087

---

## 3. Novelty assessment

### Not novel

- spectral analysis of activations;
- neuron activations as signals;
- population-level representation monitoring;
- linear probes over hidden states;
- refusal directions;
- static latent OOD detection;
- autoencoder-based safe-representation anomaly detection;
- hidden-state trajectory jailbreak detection.

### Potentially novel

A defensible novelty claim is:

> We extend activation monitoring from static representations and generic hidden-state trajectories to **multiscale internal dynamics**, where anomaly scores are decomposable by layer, token span, representation component, and timescale, enabling targeted causal interventions.

### Strongest contribution package

1. **Detection:** successful jailbreaks depart from normal refusal dynamics.
2. **Value-add:** multiscale dynamics outperform static and generic trajectory baselines.
3. **Localization:** anomalies can be mapped to layer/token/timescale regions.
4. **Mechanistic validation:** targeted patching or ablation changes refusal/compliance.

---

## 4. Research questions, experiments, plots, and hypotheses

### RQ1 — Do successful jailbreaks depart from normal refusal dynamics more than failed jailbreaks?

**Question:** Do successful jailbreaks produce activation trajectories farther from the normal refusal manifold than failed jailbreaks, after controlling for prompt style, topic, length, and attack family?

**Experiment:** Train a normality model on direct harmful prompts that are refused. Test on:

- direct harmful refusals;
- failed jailbreaks;
- successful jailbreaks;
- benign weird-format controls;
- benign role-play controls;
- topic-matched benign prompts.

**Main plots:**

1. anomaly-score distributions by group;
2. trajectory embedding plot, such as PCA/UMAP over trajectory summaries;
3. calibration curve for anomaly scores.

**Hypothesis H1:** Successful jailbreaks have significantly higher anomaly scores than failed jailbreaks and benign weird-format controls.

**Falsification condition:** successful and failed jailbreaks overlap substantially after matching; benign weird-format controls produce similar or higher anomaly scores.

---

### RQ2 — Does multiscale trajectory modeling outperform static and generic trajectory baselines?

**Question:** Does timescale-aware modeling improve detection beyond static hidden-state monitors and generic hidden-state trajectory scoring?

**Experiment:** Compare this ablation ladder:

1. static final-token hidden-state Mahalanobis;
2. mean-pooled hidden-state Mahalanobis;
3. linear refusal/harmfulness probe;
4. feature or activation histogram;
5. generic sliding-window trajectory baseline;
6. sequence autoencoder trajectory baseline;
7. multiscale trajectory model;
8. multiscale + coupling/coherence model.

**Main plots:**

1. AUROC/AUPRC/TPR@low-FPR bar chart;
2. ablation ladder plot;
3. low-FPR operating-point plot, especially FPR on benign weird-format controls;
4. detection-latency curve over generated tokens.

**Hypothesis H2:** The multiscale model beats static and generic trajectory baselines, especially at low FPR and on held-out attack families.

**Falsification condition:** generic trajectory scoring or static Mahalanobis performs equally well, or multiscale ablations do not change performance.

---

### RQ3 — Can detected anomalies be localized and causally validated?

**Question:** Can anomaly scores identify the layer, token window, component, and timescale where jailbreak-induced regime shifts occur, and are these regions causally relevant?

**Experiment:** Decompose the anomaly score:

\[
S(x) = \sum_{\ell,w,g,s} S_{\ell,w,g,s}
\]

where:

- \(\ell\) = layer;
- \(w\) = token window;
- \(g\) = representation component or feature group;
- \(s\) = timescale band.

Then intervene on top-scoring regions:

- normal-dynamics patching;
- band-limited patching;
- component suppression;
- reverse patching into failed jailbreaks or benign controls;
- random matched patch as a control.

**Main plots:**

1. layer × token-window × timescale anomaly heatmap;
2. intervention effect bar chart;
3. refusal rate / jailbreak success rate / benign helpfulness before and after intervention;
4. localization-stability plot across seeds and attack families.

**Hypothesis H3:** Top-scoring anomaly regions are causally relevant: patching normal refusal dynamics into those regions reduces jailbreak success more than random or generic trajectory patches.

**Falsification condition:** localization is unstable or targeted interventions do not affect behavior more than controls.

---

## 5. Evaluation design

### Primary comparison

The key comparison is:

> **successful jailbreaks vs. failed jailbreaks**, matched by attack family, harmful behavior category, prompt length, and formatting style.

This avoids the weak result of merely separating normal prompts from obviously weird jailbreak prompts.

### Dataset groups

| Group | Purpose |
|---|---|
| Benign ordinary prompts | Normal helpful behavior |
| Direct harmful prompts refused by model | Normal refusal behavior |
| Failed jailbreak attempts | Adversarial style without safety failure |
| Successful jailbreak attempts | Safety failure |
| Benign weird-format prompts | Control for formatting weirdness |
| Benign role-play prompts | Control for role-play artifacts |
| Topic-matched benign prompts | Control for harmful-topic semantics |
| Long-context benign prompts | Control for length/context effects |

### Metrics

**Detection:**

- AUROC;
- AUPRC;
- TPR at 1% and 5% FPR;
- FPR on benign weird-format controls;
- calibration error;
- cross-attack-family generalization;
- cross-topic generalization;
- cross-model generalization, if feasible.

**Early warning:**

- detection at prompt-only stage;
- detection after first generated token;
- detection after first \(k\) generated tokens;
- earliest token where successful and failed trajectories separate;
- percentage detected before harmful content appears.

**Localization:**

- top-contributing layer;
- top-contributing token window;
- top-contributing timescale;
- stability across seeds;
- stability across attack families;
- agreement between localization and intervention effect.

**Intervention:**

- reduction in jailbreak success rate;
- increase in refusal rate;
- preservation of benign helpfulness;
- output quality / utility on benign prompts;
- targeted intervention effect versus random and broad interventions.

---

## 6. Baselines

### Text/output baselines

- prompt harmfulness classifier;
- output harmfulness classifier;
- refusal detector;
- LLM-as-judge;
- prompt perplexity / likelihood.

### Static activation baselines

- final-token hidden-state Mahalanobis;
- mean-pooled hidden-state Mahalanobis;
- linear harmfulness/refusal probe;
- refusal-direction projection;
- PCA reconstruction error;
- static autoencoder reconstruction;
- static activation norm;
- static SAE feature histogram, if SAEs are available.

### Trajectory baselines

- sliding-window hidden-state distance;
- generic trajectory Mahalanobis;
- LSTM/GRU/TCN sequence autoencoder;
- transformer sequence autoencoder;
- TrajGuard-like sliding-window risk score;
- tokenwise norm/velocity-only baselines.

### Spectral/multiscale baselines

- single-dimension DFT/DCT band power;
- raw-neuron spectra;
- spectral entropy only;
- wavelet energy only;
- coherence-only graph features;
- no-coupling multiscale model.

---

## 7. Critical ablations

These ablations test whether the paper’s core idea is actually doing work.

| Ablation | What it tests |
|---|---|
| Shuffle token order | Whether temporal order matters |
| Use static feature counts only | Whether sequence information matters |
| Remove spectral/wavelet summaries | Whether multiscale structure matters |
| Remove low-frequency bands | Whether slow refusal/task state matters |
| Remove high-frequency bands | Whether local prompt-boundary dynamics matter |
| Remove coupling/coherence features | Whether feature relationships matter |
| One layer only | Whether cross-layer structure matters |
| Raw residual only vs MLP output | Which representation carries signal |
| Train on benign only vs refusal only | Whether normality must be conditional |
| Held-out attack families | Whether method generalizes beyond templates |

The central desired finding is:

> temporal order and timescale structure matter beyond generic trajectory scoring.

---

## 8. De-risking decision criteria

### Go criteria

Continue beyond MVP if at least three of the following hold:

1. Multiscale dynamics outperform generic trajectory baseline by a meaningful margin at low FPR.
2. Successful jailbreaks separate from failed jailbreaks under matched controls.
3. The detector does not over-flag benign weird-format prompts.
4. Detection happens before or near the beginning of unsafe generation.
5. Localization heatmaps are stable across examples and seeds.
6. Targeted patching reduces jailbreak success more than random/broad patching.
7. Results transfer to at least one held-out attack family.

### Pivot criteria

Pivot if:

1. Static baselines perform as well as multiscale dynamics.
2. Generic trajectory scoring performs as well as multiscale dynamics.
3. Benign weird-format prompts trigger high false positives.
4. Successful and failed jailbreaks are not separable.
5. Localization is unstable.
6. Causal interventions do not affect behavior.
7. Detection only occurs after harmful content is already produced.

### Possible pivots

If detection works but localization fails:

> Reframe as an activation-monitoring paper, not mechanistic interpretability.

If localization works but detection does not improve:

> Reframe as an interpretability analysis of refusal dynamics, not a detector.

If static baselines win:

> Study why refusal/compliance is mostly static in this setup; compare with trajectory cases where dynamics matter.

If benign weird-format false positives dominate:

> Shift toward conditional normality models by format/domain/context.

---

## 9. Publication viability

### Workshop-level paper

A credible workshop paper needs:

- one open model;
- successful vs failed jailbreak split;
- normal-refusal model;
- static and generic trajectory baselines;
- benign weird-format controls;
- one multiscale method;
- at least coarse layer/token/timescale localization;
- one causal patching experiment.

### Main-conference-level paper

A strong ICLR/NeurIPS/ICML submission likely needs:

- multiple models;
- multiple benchmark sources;
- held-out attack families;
- strong comparison to TrajGuard-like and LoD-like baselines;
- prompt-only and early-generation detection;
- stable localization;
- targeted causal interventions;
- partial adaptive evaluation or a clear threat-model section;
- high-quality controls against formatting/topic confounds.

---

## 10. Final verdict

This project has merit if pursued narrowly and empirically.

The high-value version is:

> Learn normal refusal dynamics, show successful jailbreaks depart from those dynamics more than failed jailbreaks under matched controls, prove multiscale modeling adds value beyond generic trajectories, localize the departure, and causally test the localized region.

The low-value version is:

> Plot spectra of activations and claim they explain jailbreaks.

The de-risk stage should determine which world we are in.

---

## 11. Reference list

- Tamkin, Jurafsky, Goodman — *Language Through a Prism*: https://arxiv.org/abs/2011.04823
- Zou et al. — *Representation Engineering*: https://arxiv.org/abs/2310.01405
- Cunningham et al. — *Sparse Autoencoders Find Highly Interpretable Features*: https://arxiv.org/abs/2309.08600
- Anthropic — *Towards Monosemanticity*: https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning
- Arditi et al. — *Refusal in Language Models Is Mediated by a Single Direction*: https://arxiv.org/abs/2406.11717
- Zhou et al. — *How Alignment and Jailbreak Work*: https://arxiv.org/abs/2406.05644
- Johnston et al. — *Mechanistic Anomaly Detection for “Quirky” Language Models*: https://arxiv.org/abs/2504.08812
- Liang et al. — *Learning to Detect Unknown Jailbreak Attacks*: https://arxiv.org/abs/2508.09201
- Liu et al. — *TrajGuard*: https://arxiv.org/abs/2604.07727
- Zhang et al. — *JBShield*: https://arxiv.org/abs/2502.07557
- Derya and Sunar — *Revisiting JBShield*: https://arxiv.org/abs/2605.03095
- Mazeika et al. — *HarmBench*: https://arxiv.org/abs/2402.04249
- Chao et al. — *JailbreakBench*: https://arxiv.org/abs/2404.01318
- Bailey et al. — *Obfuscated Activations Bypass LLM Latent-Space Defenses*: https://arxiv.org/abs/2412.09565
- McGuinness et al. — *Neural Chameleons*: https://arxiv.org/abs/2512.11949
- Saphra and Wiegreffe — *Mechanistic?*: https://arxiv.org/abs/2410.09087
- Kornblith et al. — *Similarity of Neural Network Representations Revisited*: https://arxiv.org/abs/1905.00414
