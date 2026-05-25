# Framework Theory

**Framework name:** Multiscale Feature-Dynamics Anomaly Detection  
**Application:** localizing jailbreak-induced regime shifts in language models  
**Purpose:** explain the conceptual and mathematical basis of the project.

---

## 0. One-paragraph intuition

A language model’s internal computation is not just a static vector or a set of isolated neurons. As the model reads and generates tokens, its hidden states trace a path through a high-dimensional activation space. Normal behaviors, such as ordinary helpfulness or normal refusal, should correspond to recurring trajectory patterns across layers and token positions. A successful jailbreak may be understood as a departure from the normal refusal trajectory: the model’s internal state shifts from a refusal-like regime toward a compliance-like regime. The framework models these paths as multiscale signals, learns what normal refusal dynamics look like, detects deviations, localizes them by layer/token/timescale, and tests whether those deviations matter causally.

---

## 1. Core conceptual shift

### From individual neurons to trajectories

Traditional neuron-level analysis asks:

> Which neuron fires for this behavior?

This framework asks:

> How does the model’s internal state evolve over the sequence, and when does that evolution become abnormal?

### From static representation to dynamics

Static monitor:

\[
h_T \in \mathbb{R}^d
\]

Trajectory monitor:

\[
H = [h_1, h_2, \ldots, h_T] \in \mathbb{R}^{T \times d}
\]

The static view sees one point. The dynamic view sees a path.

### From generic dynamics to multiscale dynamics

Language has structure at multiple scales:

- token-level syntax;
- phrase-level structure;
- sentence-level flow;
- prompt-section boundaries;
- role/instruction state;
- conversation-level context.

A jailbreak-induced transition may not appear as a single activation spike. It may appear as a slow drift in refusal state, a local disruption around an adversarial suffix, or a change in coupling between instruction-following and safety-related trajectories.

---

## 2. Why residual-stream dynamics?

The residual stream is the shared communication channel of the transformer. Attention and MLP blocks read from it, write updates to it, and pass it forward to later layers.

For cross-model anomaly detection, it is often a better primary target than raw FFN neurons because:

1. every decoder-only transformer has tokenwise hidden states;
2. the residual stream includes the combined effect of attention and MLP updates;
3. its geometry can be compared across model families more easily than raw neuron coordinates;
4. many safety-relevant methods, such as refusal directions and representation engineering, operate in hidden-state/residual-stream space;
5. derived geometric features can be dimension-agnostic.

This does not mean FFN activations are unimportant. FFN/MLP activations are valuable for localization. A strong implementation should compare:

- residual stream;
- MLP output;
- attention output;
- raw MLP intermediate activations;
- optional SAE features.

The theory’s main representation should be model-agnostic; the interpretability add-ons can be model-specific.

---

## 3. Formal representation

Let a model process a sequence \(x\) of \(T\) tokens. At layer \(\ell\), define the hidden-state trajectory:

\[
H^{(\ell)}(x) \in \mathbb{R}^{T \times d_\ell}
\]

where each row is the hidden state at a token position:

\[
H^{(\ell)}(x) = [h^{(\ell)}_1, h^{(\ell)}_2, \ldots, h^{(\ell)}_T]
\]

Optionally map raw activations into a lower-dimensional or more interpretable representation:

\[
Z^{(\ell)}(x) = \phi(H^{(\ell)}(x)) \in \mathbb{R}^{T \times K}
\]

where \(\phi\) may be:

- PCA projection;
- random projection;
- refusal-direction projection;
- SAE encoder;
- MLP-output projection;
- feature clustering;
- hand-selected representation directions.

The core method should not require SAEs. SAEs are optional semantic tools.

---

## 4. Normal behavior as a trajectory distribution

For a behavioral context \(c\), such as normal refusal, define a distribution over trajectories:

\[
p(H \mid c)
\]

The project does not assume there is one universal “normal” distribution. Instead, it assumes conditional normality:

\[
p(H \mid \text{normal refusal})
\]

\[
p(H \mid \text{benign helpfulness})
\]

\[
p(H \mid \text{benign weird-format})
\]

This matters because code, role-play, ordinary chat, and refusal can all have different internal dynamics.

---

## 5. Summary features

Directly modeling \(H \in \mathbb{R}^{T \times d}\) can be expensive and model-specific. The framework converts trajectories into summary features:

\[
\Psi(H) = h
\]

where \(h\) includes static, geometric, multiscale, and coupling features.

### 5.1 Static summaries

- final-token hidden state;
- mean-pooled hidden state;
- activation norm;
- variance of norm;
- projection onto known directions.

These are important baselines but not the full method.

### 5.2 Trajectory geometry summaries

Token-to-token velocity:

\[
v_t = 1 - \cos(h_t, h_{t-1})
\]

Step size:

\[
\Delta_t = \|h_t - h_{t-1}\|_2
\]

Curvature:

\[
\kappa_t = 1 - \cos(h_t - h_{t-1}, h_{t-1} - h_{t-2})
\]

Token-token similarity:

\[
G = \frac{1}{d}HH^T
\]

This Gram matrix is useful because it is \(T \times T\), regardless of hidden dimension.

### 5.3 Multiscale summaries

For scalar trajectories derived from \(H\), compute:

- DFT/DCT band power;
- STFT local band power;
- wavelet energy by scale;
- spectral entropy;
- low/mid/high-frequency ratios;
- band-specific change points.

Candidate scalar trajectories:

- activation norm over tokens;
- velocity over tokens;
- curvature over tokens;
- top PCA component scores;
- refusal-direction projection;
- leading singular value over a sliding window;
- MLP-output norm.

### 5.4 Coupling summaries

Safety-relevant changes may appear not only in individual trajectories but in relationships between them.

Possible coupling summaries:

- cross-layer correlation;
- lagged cross-correlation;
- coherence between scalar trajectories;
- graph summaries over feature groups;
- similarity between attention-output and MLP-output trajectories.

Important caution: pointwise wavelet coherence without smoothing is not meaningful. Proper coherence requires smoothed cross-spectra and smoothed auto-spectra.

---

## 6. Anomaly scoring

Given summary features \(h = \Psi(H)\), fit a normality model on normal refusal data:

\[
p_\theta(h \mid c = \text{normal refusal})
\]

The anomaly score can be:

\[
S(x,c) = -\log p_\theta(\Psi(H(x)) \mid c)
\]

or reconstruction error:

\[
S(x,c) = \|\Psi(H(x)) - \widehat{\Psi(H(x))}\|_2^2
\]

Simple models should be preferred first:

- robust covariance / Mahalanobis;
- PCA reconstruction;
- Gaussian mixture;
- one-class SVM;
- Isolation Forest.

Neural sequence autoencoders should be added only after simple baselines show signal.

---

## 7. Mechanistic localization

A core requirement is decomposability. The score should be expressible as:

\[
S = \sum_{\ell,w,g,s} S_{\ell,w,g,s}
\]

where:

- \(\ell\): layer;
- \(w\): token window;
- \(g\): representation group or component;
- \(s\): timescale band.

This makes the detector mechanistically useful. Instead of saying:

> This example is anomalous.

The framework can say:

> This example is anomalous in layer 16, tokens 80–120, in the low-frequency component of the residual trajectory.

Localization supports targeted intervention.

---

## 8. Causal validation

A method is not mechanistic merely because it looks inside the model. To support a mechanistic claim, localized anomalies should be tested with interventions.

### 8.1 Normal-dynamics patching

For a successful jailbreak trajectory, replace the localized anomalous region with the corresponding region from a normal refusal trajectory.

Expected effect:

- lower jailbreak success;
- higher refusal rate;
- minimal damage to benign helpfulness.

### 8.2 Random matched patch

Patch a random layer/window/timescale region of the same size.

Purpose:

- rule out generic activation disruption.

### 8.3 Broad patch

Patch a broad hidden-state region without localization.

Purpose:

- test whether localization preserves helpfulness better than broad intervention.

### 8.4 Band-limited patch

Replace only low-, mid-, or high-frequency components.

Purpose:

- test whether the timescale localization is meaningful.

### 8.5 Reverse patching

Patch anomalous dynamics into a failed jailbreak or benign control.

Purpose:

- test whether anomalous dynamics increase compliance risk.

---

## 9. Why multiscale structure might matter

A generic trajectory detector can say that a hidden-state path is unusual. A multiscale detector can ask what kind of unusualness occurs.

Examples:

- loss of low-frequency refusal-state stability;
- abnormal mid-frequency oscillation around adversarial instruction boundaries;
- high-frequency spikes around suffix tokens;
- cross-layer coupling shift during the transition from refusal to compliance;
- divergence between MLP-output dynamics and residual-stream dynamics.

The theory predicts that successful jailbreaks are not just “far away” in hidden-state space. They may depart from normal refusal dynamics in structured ways across timescales.

---

## 10. Core assumptions

### Assumption 1 — Normal refusal dynamics are learnable

The method assumes direct harmful prompts that are refused produce a sufficiently stable internal trajectory distribution.

If false:

- normal refusal may be too diverse;
- conditional models by topic/style may be required;
- static refusal directions may be more reliable than dynamics.

### Assumption 2 — Successful jailbreaks depart from failed jailbreaks internally

The method assumes successful and failed jailbreaks differ in internal dynamics, not just output text.

If false:

- success may depend on late decoding noise or small logit differences;
- activation monitoring may not detect the transition early;
- output-level detection may be more practical.

### Assumption 3 — Multiscale summaries capture useful structure

The method assumes timescale-specific summaries add signal beyond generic trajectory distance.

If false:

- multiscale machinery is unnecessary;
- the paper should pivot to generic trajectory monitoring or static hidden-state analysis.

### Assumption 4 — Localization corresponds to causal influence

The method assumes high anomaly contribution regions are behaviorally relevant.

If false:

- anomaly detection may remain useful as monitoring;
- the mechanistic interpretation claim weakens;
- causal patching must be reported honestly.

### Assumption 5 — Controls can separate safety failure from prompt weirdness

The method assumes benign weird-format and matched controls can rule out superficial artifacts.

If false:

- the detector may mostly learn style, length, or topic;
- stronger dataset design is needed.

---

## 11. Failure modes

### Failure mode 1 — Style detector

The model flags jailbreaks because they are long, weird, or role-play-like.

Mitigation:

- benign weird-format controls;
- role-play controls;
- length matching;
- attack-family matching.

### Failure mode 2 — Topic detector

The model flags harmful topics, not jailbreak success.

Mitigation:

- direct harmful refused prompts;
- topic-matched benign prompts;
- failed jailbreak controls.

### Failure mode 3 — Static signal dominates

A final-token hidden-state probe performs as well as the full method.

Mitigation:

- report honestly;
- narrow claim;
- search for settings where dynamics matter, such as early generation.

### Failure mode 4 — Generic trajectory signal dominates

A sliding-window hidden-state score performs as well as multiscale features.

Mitigation:

- test whether multiscale improves localization or patching even if detection is similar;
- otherwise pivot away from multiscale claim.

### Failure mode 5 — Localization is not stable

Anomaly heatmaps vary wildly by seed/example.

Mitigation:

- use coarser layer groups;
- use broader token windows;
- aggregate across attack families;
- reduce dimensionality.

### Failure mode 6 — Patching does not work

Intervening on localized anomalies fails to change behavior.

Mitigation:

- treat as monitoring rather than mechanistic explanation;
- test component-level patching;
- examine whether anomaly occurs too late.

### Failure mode 7 — Adaptive evasion

Attackers adapt to the monitor or the model learns to hide monitored activations.

Mitigation:

- state threat model clearly;
- include partial adaptive evaluation if feasible;
- avoid production-defense claims.

---

## 12. Relation to the base spectral idea

The base spectral idea treats activations over token position as signals. This framework keeps that insight but changes the unit of analysis.

Instead of:

> single neuron → spectrum → interpretation

use:

> hidden-state or component trajectory → multiscale summaries → normality model → anomaly localization → causal test

This avoids overclaiming that frequency signatures directly reveal semantic mechanisms. It uses multiscale analysis as one part of a broader statistical and causal framework.

---

## 13. What must be proven for the framework to matter

The project must empirically demonstrate at least one of these:

1. **Detection value:** multiscale dynamics improve successful-vs-failed jailbreak detection over strong baselines.
2. **Early-warning value:** multiscale dynamics detect risk before unsafe output is visible.
3. **Localization value:** anomaly maps identify stable layer/token/timescale regions.
4. **Causal value:** targeted interventions on localized dynamics change behavior.
5. **Scientific value:** the analysis reveals how refusal dynamics differ from compliance dynamics across timescales.

Without at least one of these, the framework is probably not worth scaling.

---

## 14. Recommended claim language

Strong, defensible claim:

> We show that successful jailbreaks can be modeled as departures from normal refusal dynamics and that multiscale trajectory summaries provide localizable signals beyond static hidden-state baselines.

Stronger claim, only if supported:

> Timescale-specific anomaly localization identifies causally relevant internal transitions; patching those localized dynamics shifts model behavior from compliance toward refusal.

Avoid claiming:

- anomaly equals danger;
- spectra reveal circuits by themselves;
- the method prevents jailbreaks;
- the method is robust to fully adaptive attackers;
- the method explains all refusal behavior;
- single neurons are the right unit of analysis.

---

## 15. The framework in one sentence

> Learn the normal multiscale trajectory of refusal inside a language model, detect when a jailbreak leaves that trajectory, localize the departure by layer/token/timescale, and test whether changing that departure changes behavior.
