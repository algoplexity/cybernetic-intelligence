# CI Unified — Closed-Form Synthesis

> **Canonical documentation file for the Cybernetic Intelligence (CI) research programme.**
> This document merges `CIv7` through `CIv16` into a single, formula-preserving synthesis anchored
> on the observer-compression closure framework. The original source files are preserved verbatim in
> [`docs/archive/`](archive/).

---

## Table of Contents

1. [The Closure Framework](#1-the-closure-framework)
   1. [Primitive Observer](#11-primitive-observer)
   2. [Core Compression Quantity](#12-core-compression-quantity)
   3. [Mesoscopic Structure](#13-mesoscopic-structure)
   4. [Dynamics — the Intelligence Signal](#14-dynamics--the-intelligence-signal)
   5. [Geometry (Derived, Not Primitive)](#15-geometry-derived-not-primitive)
   6. [Observer Duality](#16-observer-duality)
   7. [Causality and Perturbation](#17-causality-and-perturbation)
   8. [Learning via Observer Update](#18-learning-via-observer-update)
   9. [Autopoiesis](#19-autopoiesis)
   10. [Substrate Tower](#110-substrate-tower)
   11. [Mesoscope (Instrument)](#111-mesoscope-instrument)
   12. [Final Definition of Intelligence](#112-final-definition-of-intelligence)
2. [CI Version Mapping](#2-ci-version-mapping)
3. [CI Version Summaries](#3-ci-version-summaries)
   1. [CIv7-ECA](#31-civ7-eca)
   2. [CIv7-LLM](#32-civ7-llm)
   3. [CIv7 Grand Story](#33-civ7-grand-story)
   4. [CIv7 Grand Story Poem](#34-civ7-grand-story-poem)
   5. [CIv7 Dependency Graph](#35-civ7-dependency-graph)
   6. [CIv10-ECA](#36-civ10-eca)
   7. [CIv10-LLM](#37-civ10-llm)
   8. [CIv11 — Entropic Geometry](#38-civ11--entropic-geometry)
   9. [CIv15 — Autopoietic Planner](#39-civ15--autopoietic-planner)
   10. [CIv16 — The Tower Hypothesis](#310-civ16--the-tower-hypothesis)
4. [Consolidated References](#4-consolidated-references)
   1. [External References with Direct URLs](#41-external-references-with-direct-urls)
   2. [Ambiguous or Cryptic References with Source-File Context](#42-ambiguous-or-cryptic-references-with-source-file-context)
   3. [CI Source-File Index](#43-ci-source-file-index)
5. [Closure](#5-closure)

---

## 1. The Closure Framework

All constructs in the CI programme reduce to four primitives: observer $\mathcal{O}$, compression
$K_{\mathcal{O}}$, change $\Delta K_{\mathcal{O}}$, and non-additivity $E_{\mathcal{O}}$. No
additional primitives are required. The sections below define each component and map it to the
versioned CI material.

---

### 1.1 Primitive Observer

$$\mathcal{O} = (\phi,\, M,\, B)$$

| Component | Role |
|-----------|------|
| $\phi$ | Representation map — encodes raw experience into a compressible form |
| $M$ | Model / compressor class — the family of models used to compress $\phi(X)$ |
| $B$ | Resource bounds — computational, memory, and time constraints on $M$ |

All quantities in the CI framework are defined *relative to* $\mathcal{O}$. Different observers
inhabit the same world but measure fundamentally different descriptions of it.

---

### 1.2 Core Compression Quantity

$$K_{\mathcal{O}}(X) := L_M(\phi(X))$$

$K_{\mathcal{O}}(X)$ is the description length of $X$ under observer $\mathcal{O}$. It is the
single unified quantity from which all CI measures are derived:

* **MDL, BDM, CTM** are approximations of or estimators for $K_{\mathcal{O}}$.
* Compression gain corresponds to a decrease in $K_{\mathcal{O}}$.
* Compression failure corresponds to an increase or discontinuity in $K_{\mathcal{O}}$.

---

### 1.3 Mesoscopic Structure

$$E_{\mathcal{O}}(X) = K_{\mathcal{O}}(X) - \sum_i K_{\mathcal{O}}(X_i)$$

$E_{\mathcal{O}}$ measures *non-additivity* — the shared, coordinated structure that is not
captured by summing the parts:

* $E_{\mathcal{O}} = 0$ implies the parts are statistically independent under $\mathcal{O}$.
* $E_{\mathcal{O}} > 0$ implies emergent, coordinated structure.
* Mesoscopic variables in CIv9–CIv15 are instances of $E_{\mathcal{O}}$.

---

### 1.4 Dynamics — the Intelligence Signal

$$\Delta K_{\mathcal{O}}(t) = K_{\mathcal{O}}(X_{t+1}) - K_{\mathcal{O}}(X_t)$$

All intelligence signals reduce to $\Delta K_{\mathcal{O}}$:

| Signal type | Manifestation |
|-------------|---------------|
| Structural break | Large $|\Delta K_{\mathcal{O}}|$ |
| Entropy spike | Positive $\Delta K_{\mathcal{O}}$ |
| Compression gain / learning | Negative $\Delta K_{\mathcal{O}}$ |
| Stable regime | $\Delta K_{\mathcal{O}} \approx 0$ |

This quantity is the core detection signal in CIv7-ECA structural-break detection, the utility
signal in CIv15 autopoietic planning, and the self-evolution constraint in CIv15–CIv16.

---

### 1.5 Geometry (Derived, Not Primitive)

Define the gradient and Hessian of $K_{\mathcal{O}}$ over the space of inputs:

$$\nabla K_{\mathcal{O}}, \qquad \nabla^2 K_{\mathcal{O}}$$

| Geometric object | Interpretation |
|-----------------|----------------|
| $\nabla K_{\mathcal{O}}$ | Direction of increasing compressibility |
| $\nabla^2 K_{\mathcal{O}}$ | Curvature — regime stability or boundary |
| Discontinuity in $\nabla K_{\mathcal{O}}$ | Torsion / folding (Walch) |
| Path minimising $\int K_{\mathcal{O}}$ | Geodesic through information space |
| Constrained evolution of $K_{\mathcal{O}}$ | Contact flow (CIv10-LLM, GCF) |

Geometry is thus *derived from* compression, not imposed as a separate ontology. The entropic
geometry of CIv11 (curvature, harmonic forms, metriplectic flow) is a coordinate representation
of $\nabla^2 K_{\mathcal{O}}$.

---

### 1.6 Observer Duality

Different observers:

$$\mathcal{O}_1 \neq \mathcal{O}_2$$

induce different compressions:

$$K_{\mathcal{O}_1}(X) \neq K_{\mathcal{O}_2}(X)$$

The divergence between two observers:

$$D(X) = K_{\mathcal{O}_1}(X) - K_{\mathcal{O}_2}(X)$$

* measures representational misalignment;
* drives learning — an agent learns by detecting divergence and updating $\phi$ or $M$;
* captures symbolic-vs-latent duality (CIv7–CIv10): symbolic and latent substrates are two
  different $\phi$, not competing ontologies.

---

### 1.7 Causality and Perturbation

The causal importance of an element $e$ within $X$:

$$I_{\mathcal{O}}(X, e) = K_{\mathcal{O}}(X) - K_{\mathcal{O}}(X \setminus e)$$

* $I_{\mathcal{O}} > 0$: $e$ is a structural element — removing it increases description length.
* $I_{\mathcal{O}} < 0$: $e$ is noise — removing it improves compression.
* $I_{\mathcal{O}} = 0$: $e$ is irrelevant to the observer's model.

Conditional causality: $K_{\mathcal{O}}(Y \mid X) < K_{\mathcal{O}}(Y)$ captures the mutual
compression gain when $X$ is available as context.

This maps onto CIv7's BDM-based structural-break detection and CIv15's causal perturbation testing.

---

### 1.8 Learning via Observer Update

Learning is triggered by:

* large $|\Delta K_{\mathcal{O}}|$ — structural surprise;
* large divergence $D(X)$ between two observers — representational misalignment;
* increasing residual — compression capacity exceeded.

The update:

$$\phi \rightarrow \phi', \qquad M \rightarrow M'$$

Goal: minimise $K_{\mathcal{O}}(X)$ subject to bounds $B$.

This encompasses: SEAL-style self-editing (CIv16), curriculum evolution in CIv10/CIv15, contact
geometry steering in CIv10-LLM, and the symbolic motif refactoring of CIv10-ECA.

---

### 1.9 Autopoiesis

The observer modifies itself over time:

$$\mathcal{O}_t \rightarrow \mathcal{O}_{t+1}$$

subject to the autopoietic viability constraint:

$$\Delta K_{\mathcal{O}} \le 0$$

Utility:

$$U = -\Delta K_{\mathcal{O}} + \text{task reward}$$

CIv15 operationalises this as: accept a self-edit if and only if it improves ΔMDL (overall
compression) and forecast skill. CIv16 extends this to full substrate-level observer refinement.

---

### 1.10 Substrate Tower

The hierarchy in CIv16 is *observer refinement*, not separate ontology:

$$\mathcal{O}^{(1)} \preceq \mathcal{O}^{(2)} \preceq \cdots \preceq \mathcal{O}^{(n)}$$

Each step enriches the observer by:

* richer $\phi$ (expanded representation: set → algebraic → topological → geometric → analytic → meta-structural);
* larger $M$ (more expressive model class);
* relaxed $B$ (greater computational budget).

Each step increases compressibility and representational power. The tower is the ordered set of
observer refinements, driven purely by $\Delta K_{\mathcal{O}} \le 0$.

---

### 1.11 Mesoscope (Instrument)

A mesoscope is a collection of observers:

$$\{\mathcal{O}_1,\, \mathcal{O}_2,\, \ldots,\, \mathcal{O}_n\}$$

Measuring:

* $K_{\mathcal{O}_i}(X)$ — compression under each observer;
* $E_{\mathcal{O}_i}(X)$ — emergent structure under each observer;
* $D_{ij}(X) = K_{\mathcal{O}_i}(X) - K_{\mathcal{O}_j}(X)$ — divergence between observers.

**Regime** = region of $X$-space where all observers agree ($D_{ij} \approx 0$).  
**Break / phase transition** = region where observers diverge ($|D_{ij}|$ large).

The mesoscope is the unified diagnostic instrument connecting CIv7 structural-break detection,
CIv10 multiscale alignment, CIv11 entropic geometry, and CIv15/CIv16 self-evolution.

---

### 1.12 Final Definition of Intelligence

$$\text{Intelligence} = \text{the process that selects and transforms } \mathcal{O}$$
$$\text{to minimise } K_{\mathcal{O}}(X)$$
$$\text{while preserving predictive and causal structure.}$$

This definition subsumes all prior CI versions:

* It selects $\phi$ and $M$ (CIv7: symbolic/latent substrate choice).
* It updates them under compression feedback (CIv10: evolution of $\phi$).
* It operates over a curved information geometry (CIv11: geometry of $K$).
* It maintains autopoietic viability (CIv15: dynamics of $\mathcal{O}$).
* It climbs a hierarchy of observer refinements (CIv16: hierarchy of $\mathcal{O}$).

---

## 2. CI Version Mapping

| CI Version | Core Contribution | Closure Mapping |
|------------|------------------|-----------------|
| **CIv7** | Structural break detection via symbolic (ECA) and latent (LLM) substrates | $\Delta K_{\mathcal{O}}$, $I_{\mathcal{O}}(X,e)$ |
| **CIv10** | Byte-level symbolic emergence; contact-geometric latent control | Evolution of $\phi$: $\phi \rightarrow \phi'$ |
| **CIv11** | Entropic geometry; metriplectic computation as unified substrate | Geometry of $K$: $\nabla K_{\mathcal{O}}$, $\nabla^2 K_{\mathcal{O}}$ |
| **CIv15** | Autopoietic planner; compression-aligned self-evolution | Dynamics of $\mathcal{O}$: $\mathcal{O}_t \rightarrow \mathcal{O}_{t+1}$, $\Delta K \le 0$ |
| **CIv16** | Tower hypothesis; substrate hierarchy as observer enrichment | Hierarchy of $\mathcal{O}$: $\mathcal{O}^{(1)} \preceq \cdots \preceq \mathcal{O}^{(n)}$ |

Versions CIv8–CIv9 are transitional refinements between CIv7 and CIv10; they are preserved in
the full retrospective documents in `docs/` but not individually summarised here.

---

## 3. CI Version Summaries

### 3.1 CIv7-ECA

**Source:** [`docs/archive/CIv7-ECA-hypothesis.md`](archive/CIv7-ECA-hypothesis.md)

**Core hypothesis:** Structural breaks in univariate time series can be robustly detected by
encoding the input as symbolic sequences (e.g. via permutation entropy, delta-sign encoding),
evolving these through Elementary Cellular Automata (ECA), and analysing the resulting 2D symbolic
evolution as an algorithmic and topological substrate.

**Closure mapping:** The symbolic substrate is a bounded observer with $\phi = \text{ECA evolution}$
and $M = \text{BDM/CTM compressor}$. A structural break is a large $|\Delta K_{\mathcal{O}}|$. The
causal importance of an element corresponds to $I_{\mathcal{O}}(X, e)$.

**Key signals:**
* Algorithmic compressibility shift via BDM/CTM ($\Delta K_{\mathcal{O}}$).
* Joint compression failure between adjacent symbolic segments (Sutskever).
* Torsional bifurcation — topological instability under semantic perturbation (Walch).
* Negative complexity from motif degeneracy/symmetry folding (Grosse et al.).
* Edge-of-chaos attractor instabilities via Class IV ECA dynamics (Zhang et al.).

**Distinguished application:** Real-valued time series → delta-sign or permutation encoding →
ECA evolution → BDM/torsion analysis → structural break localisation.

---

### 3.2 CIv7-LLM

**Source:** [`docs/archive/CIv7-LLM-hypothesis.md`](archive/CIv7-LLM-hypothesis.md)

**Core hypothesis:** Structural failures in language model behaviour — hallucination, steering
unreliability, generalisation collapse, semantic drift — can be detected and interpreted by
analysing latent representations as a compressive and topological substrate. The latent substrate
exhibits discontinuities that serve as failure surfaces.

**Closure mapping:** The latent substrate is an observer with $\phi = \text{residual stream / attention flow}$
and $M = \text{LLM compressor}$. Failures manifest as $|\Delta K_{\mathcal{O}}|$ spikes or
divergence $D(X)$ between symbolic and latent observers.

**Key signals:**
* Steering vector unreliability — failure to consistently adjust output distribution (Braun et al.).
* Joint compression failure between latent summaries and originating corpus segments.
* Semantic attractor collapse — multiple interpretations degenerate to a single summary.
* Torsion and cohomology loss in attention-induced manifold flows (Walch, Hodge, Langlands).
* Latent leakage paths — information in non-salient neurons invertible across prompts (Jha et al.).
* Conceptual over-compression — thematic diversity sacrificed for KL minimisation (Shani et al.).

**Distinguished application:** Thematic analysis of large corpora — detecting topic drift, loss of
conceptual granularity, and semantic incoherence via latent fault geometry.

---

### 3.3 CIv7 Grand Story

**Source:** [`docs/archive/CIv7-grand-story.md`](archive/CIv7-grand-story.md)

**Summary:** A narrative bridge for CIv7. The story of the "Cartographers of the Unknown" maps
each CIv7 component instrument to a character in an expedition:

| Instrument | CIv7 Component |
|------------|----------------|
| Crystal lens (Geometric MDL Core) | CIv7-GMC — overfit / contradiction detector |
| Whispering notebook (Semantic Tracker) | CIv7-SAT — meaning-shift tracker |
| Wind chime (Entropy Monitor) | CIv7-EFM — entropy-flux detector |
| Tuning fork (Compression Analyser) | CIv7-JCA — joint compression analyser |
| Glyph-mapper (Motif Tracker) | CIv7-MRT — symbolic motif evolution tracker |
| Topological Map Spinner | CIv7-TGM — holes, loops, torsion |
| Mirror (Core Updater) | CIv7-ACU — self-rewiring / autopoietic update |

The three journey phases (stability in compression → geometric collapse → diagnosing blindness)
correspond to increasing $|\Delta K_{\mathcal{O}}|$ and increasing observer adaptation.

---

### 3.4 CIv7 Grand Story Poem

**Source:** [`docs/archive/CIv7-grand-story-poem.md`](archive/CIv7-grand-story-poem.md)

**Summary:** *"The Fable of the Fractured Compass"* — a seven-scene poetic rendering of the CIv7
expedition through five landscapes (Vale of Vanishing Patterns, Forest of Divergent Echoes, Hollow
Archive, Motif Spiral, Torsion Bridge, Oracle Pool) to the Chorus of Collapse. Each scene
corresponds to a failure mode in $K_{\mathcal{O}}$: from curvature quivers (GMC) through motif
drift (MRT) and torsion surges (TGM) to the final summit where the compass turns inward.

---

### 3.5 CIv7 Dependency Graph

**Source:** [`docs/archive/CI7-dependency-graph.md`](archive/CI7-dependency-graph.md)

**Summary:** Master reference graph tracing theoretical and empirical dependencies within CIv7.
Three functional pillars:

1. **Latent Representation Compression** — Sutskever (compression as prediction), Grosse et al.
   (geometric Occam / FIM degeneracy), Walch & Hodge (torsion topology as harmonic lattice breakdown).
2. **Attribution Drift Analysis** — Sakabe et al. (token-level path divergence), Anthropic Circuit
   Tracer (internal drift as geometric path inconsistency), Jha et al. / vec2vec (cross-model drift).
3. **Three implementation phases:** latent compression stability → geometric breakdown detection →
   model-agnostic diagnostics.

**Reference-linked CIv7 documents (GitHub Pages):**
* [CIv7-ECA Hypothesis](https://algoplexity.github.io/cybernetic-intelligence/CIv7-ECA-hypothesis)
* [CIv7-LLM Hypothesis](https://algoplexity.github.io/cybernetic-intelligence/CIv7-LLM-hypothesis)
* [CIv7 Unified Framework](https://algoplexity.github.io/cybernetic-intelligence/CIv7-ECA-LLM-unified-framework-hypothesis)
* [CIv7-ECA Solution Proposal](https://algoplexity.github.io/cybernetic-intelligence/CIv7-ECA-solution-proposal)
* [CIv7-LLM Solution Proposal](https://algoplexity.github.io/cybernetic-intelligence/CIv7-LLM-solution-proposal)
* [CIv7 Unified Framework Solution Proposal](https://algoplexity.github.io/cybernetic-intelligence/CIv7-ECA-LLM-unified-framework-solution-proposal)

---

### 3.6 CIv10-ECA

**Source:** [`docs/archive/CIv10-ECA-hypothesis.md`](archive/CIv10-ECA-hypothesis.md)

**Core hypothesis:** Intelligence involves a symbolic substrate that emerges from hierarchical
compression patterns within raw byte sequences. The substrate encodes causal skeletons by
identifying minimal self-organising motifs — formed not by predefined token units but through
dynamic split hierarchies learned from data (e.g. AU-Net).

**Closure mapping:** The representation map $\phi$ now evolves from fixed ECA rules (CIv7) to a
*learned, multiscale byte-hierarchy* $\phi = \text{AU-Net layers}$. Symbolic faults are
$$|\Delta C_i| > \varepsilon \quad \text{or} \quad |T(M_i[t]) - T(M_i[t-1])| > \delta$$
i.e. large $|\Delta K_{\mathcal{O}}|$ or large torsion change — the same closure conditions as CIv7,
now at learnable, language-agnostic scales.

**CIv10-specific extensions:**
* Symbolic structures form across compression-aligned scales, not token boundaries.
* Byte-to-Concept emergence driven by latent pooling dynamics and entropy-aware segmentation.
* Enables cross-lingual and morphologically rich reasoning without tokeniser retraining.
* SEAL-style self-editing and entropy-regularised sampling govern curriculum mutation.

---

### 3.7 CIv10-LLM

**Source:** [`docs/archive/CIv10-LLM-hypothesis.md`](archive/CIv10-LLM-hypothesis.md)

**Core hypothesis:** Intelligence in LLMs emerges from the *controlled evolution of latent
dynamics* governed by contact geometry. The latent substrate is a $(2d+1)$-dimensional contact
manifold, where each conceptual trajectory is shaped by a contact Hamiltonian — encoding
stability, dissipation, and meaning-preserving structure.

**Closure mapping:** The contact manifold is the geometric representation of $\nabla K_{\mathcal{O}}$;
the contact Hamiltonian $H(z)$ drives evolution along paths of decreasing $K_{\mathcal{O}}$.
Fault surface:
$$F = \{z \mid |\Delta C(z)| > \varepsilon \text{ or } |\nabla T(z)| > \delta\}$$
Reflexive repair: a contactomorphism $\varphi$ that warps latent space to restore compression
regularity — an instance of $\phi \rightarrow \phi'$ in the learning update.

**Key mechanisms:**
* Uncertainty-aware geodesic control: ensemble uncertainty reshapes the metric to bend inference
  away from under-trained / volatile embeddings.
* Contactomorphism injection: symbolic fault description conditions a latent patch $\varphi$.
* Compression as control signal: $\Delta C$ and $\nabla T$ are the feedback gradients.

---

### 3.8 CIv11 — Entropic Geometry

**Source:** [`docs/archive/CIv11-unified-story.md`](archive/CIv11-unified-story.md)

**Core hypothesis:** Intelligence is an emergent phenomenon of spatiotemporal geometric
computation — where entropy, curvature, and energy are the fundamental primitives. Transformers
are computational gravity engines enacting a *metriplectic flow*: a fusion of conservative
(Hamiltonian) and dissipative (entropy-maximising) dynamics.

**Closure mapping:** CIv11 *collapses the substrate dualism* of CIv7–CIv10 into a single entropic
manifold. Every component is a coordinate representation of $K_{\mathcal{O}}$:

| CIv11 Component | Closure equivalent |
|----------------|--------------------|
| Attention (thermodynamic lensing) | $\nabla K_{\mathcal{O}}$ flow |
| Softmax (Gibbs entropy) | Normalisation of $K_{\mathcal{O}}$ |
| Curvature / torsion collapse | $\nabla^2 K_{\mathcal{O}}$ discontinuity |
| Geodesic traversal | Path minimising $\int K_{\mathcal{O}}$ |
| LoRA / skip connection patches | Local $\phi' \rightarrow \phi$ update |

CIv11 predicts that intelligence failures are *topological singularities*: discontinuities in
$\nabla K_{\mathcal{O}}$, geodesic predictability collapse, and phase distortion between harmonic forms.

---

### 3.9 CIv15 — Autopoietic Planner

**Source:** [`docs/archive/CIv15-hypothesis.md`](archive/CIv15-hypothesis.md)

**Core hypothesis:** CIv15 operationalises autopoietic planning: minimal generative programs are
autonomously edited, future outcomes are simulated via decompression, and actions are selected to
maximise compressibility and downstream utility. The system maintains its own viability by keeping
$\Delta\text{MDL} \le 0$ over a rolling window.

**Closure mapping:**

| CIv15 component | Closure equivalent |
|----------------|--------------------|
| Program library $L_t$ | Current observer $\mathcal{O}_t$ |
| ΔMDL ≤ 0 over rolling window | Autopoietic constraint $\Delta K_{\mathcal{O}} \le 0$ |
| Utility $U = f(\Delta\text{BDM}, \text{forecast}, \text{reward})$ | $U = -\Delta K_{\mathcal{O}} + \text{task reward}$ |
| Self-edit acceptance | Learning update $\phi \rightarrow \phi'$ |
| Causal perturbation testing | $I_{\mathcal{O}}(X, e)$ |

**Substrate variants:** Symbolic substrate (program library), latent substrate (BDM/MDL
signals), and unified substrate (decision-making over decompression forecasts) are all
interpretations of the same observer $\mathcal{O}$ under different $\phi$ and $M$.

---

### 3.10 CIv16 — The Tower Hypothesis

**Source:** [`docs/archive/CIv16-hypothesis.md`](archive/CIv16-hypothesis.md)

**Core hypothesis:** Cybernetic intelligence emerges through a hierarchical enrichment of
substrates, where each layer arises via symmetry breaking and structural constraint applied to the
previous. This mirrors Azari's *Conceptual Tower of Mathematical Structures* (2025), situating
symbolic, probabilistic, and latent forms of cognition as nested stages — not competing paradigms.

**Closure mapping:** Each substrate tier is an observer refinement:

| Substrate tier | $\phi$ | $M$ |
|---------------|--------|-----|
| Set-like (percepts) | raw signal identity | counting / frequency |
| Algebraic (symbolic rules) | compositional grammar | MDL / BDM |
| Topological / order (network flows) | graph embedding | topological complexity |
| Geometric (latent embeddings) | neural encoder | contact geometry |
| Manifold (localised regimes) | multiscale encoder | curvature-aware compressor |
| Analytic / probabilistic | Bayesian / variational | MDL + uncertainty |
| Meta-structural (reflexive loops) | self-editing functor | category-theoretic compressor |

Each step up the tower is justified by $\Delta K_{\mathcal{O}} \le 0$: the richer observer
achieves better compression, satisfying the autopoietic constraint.

**Implementation roadmap phases:**
1. Symbolic–Latent Mapping (dual-stream encoders).
2. Structural Break Dynamics (topological order → manifold substrates).
3. Probabilistic Enrichment (Bayesian / variational modules).
4. Meta-Structural Reflexivity (SEAL-style self-rewriting substrate transitions).
5. Tower Integration (full substrate ladder in ECA dynamics, financial alphas, language modelling).

---

## 4. Consolidated References

### 4.1 External References with Direct URLs

| Citation | URL(s) |
|----------|--------|
| Anderson, P. W. — *More Is Different* (1972) | https://www.science.org/doi/10.1126/science.177.4047.393 |
| Ashby, W. R. — *Design for a Brain* (1956) | https://archive.org/details/designforbrain00ashb |
| Beer, S. — *Brain of the Firm* (1972) | https://www.wiley.com/en-us/Brain+of+the+Firm%2C+2nd+Edition-p-9780471921224 |
| Goldenfeld, N. — *Lectures on Phase Transitions and the Renormalization Group* | https://www.routledge.com/Lectures-on-Phase-Transitions-and-the-Renormalization-Group/Goldenfeld/p/book/9780367096143 |
| Maturana, H. & Varela, F. — *Autopoiesis and Cognition* (1980) | https://mitpress.mit.edu/9780262630543/autopoiesis-and-cognition/ |
| Wu, Y. et al. — *SEAL: Self-Evolving Agents via Loop Editing* (2025) | https://arxiv.org/abs/2504.18116 |
| Zhang, Y. et al. — *Intelligence at the Edge of Chaos* (2023) | https://arxiv.org/abs/2301.03266 |
| Zenil, H. et al. — Algorithmic Information Dynamics | https://www.algorithmicdynamics.net/ · https://arxiv.org/abs/1609.00110 |
| Grünwald, P. & Roos, T. — MDL resources | https://arxiv.org/abs/1703.01417 |
| Crutchfield, J. & Young, K. — ε-machines / statistical complexity | https://arxiv.org/abs/cond-mat/9506022 |
| Anthropic — Circuit Tracing / interpretability hub | https://transformer-circuits.pub/ |

---

### 4.2 Ambiguous or Cryptic References with Source-File Context

For the references below, no canonical URL was reliably identifiable from the CI docs. Each
entry cites the source file(s) it was extracted from to provide context for later search.

| Citation | Source context |
|----------|---------------|
| AU-Net — *From Bytes to Ideas*; multiscale byte-level symbolic emergence | `docs/archive/CIv10-ECA-hypothesis.md` |
| Azari, E. — *A Conceptual Tower of Mathematical Structures* (2025), arXiv:2507.xxxxx | `docs/archive/CIv16-hypothesis.md` |
| Bianconi, G. — entropy / curvature / gravity framing | `docs/archive/CIv11-unified-story.md` |
| Braun et al. — steering vector failure / latent separability | `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| BrightStar Labs — emergent models / CA-like substrates | `docs/archive/CIv7-ECA-hypothesis.md` |
| Burtsev et al. (2023) — learning rules at the edge of chaos | `docs/archive/CIv15-hypothesis.md` |
| Chen et al. — SASR: Adaptive Integration of SFT and RL | `docs/archive/CIv7-ECA-hypothesis.md` |
| Chen et al. — GFSE: Universal Graph Structural Encoder | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` |
| Darwin Gödel Machine — autonomous open-ended self-improvement | `docs/archive/CIv7-ECA-hypothesis.md` |
| DeepMind — AlphaEvolve: LLM-supervised evolutionary algorithm discovery | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Dijkstra et al. (2025) — neural basis of reality monitoring | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` |
| GCF (2025) — contact geometry for uncertainty-aware latent evolution | `docs/archive/CIv10-LLM-hypothesis.md` |
| Grosse et al. — geometric Occam's Razor / negative complexity | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Grundy et al. (2025) — forecast error-based online changepoint detection | `docs/archive/CIv7-ECA-hypothesis.md` |
| Ha, D. & Schmidhuber, J. — World Models | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Hernández-Espinosa et al. (2024) — SuperARC | `docs/archive/CIv15-hypothesis.md` |
| Hodge et al. (2024) — algebraic / topological inductive bias in transformers | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Jha et al. — vec2vec: universal geometry of embeddings | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Langlands duality / transformer algebraic-topological priors | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Li et al. — RD-Agent(Q): multi-agent symbolic financial factor-model co-design | `docs/archive/CIv7-ECA-hypothesis.md` |
| LLM Geometry (2025) — human-aligned concept manifolds in low dimensions | `docs/archive/CIv10-LLM-hypothesis.md` |
| OpenThoughts (2024) — structured symbolic reasoning / recipes for compositional reasoning | `docs/archive/CIv7-ECA-hypothesis.md` |
| Riedel, P. & Zenil, H. (2025) — ECA rule primality and causal decomposition | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv15-hypothesis.md` |
| Sakabe et al. — attribution drift from symbolic perturbations | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| Schmidhuber, J. (1997) — compression as cognition | `docs/archive/CIv10-ECA-hypothesis.md` · `docs/archive/CIv10-LLM-hypothesis.md` |
| SEAL (2024) — symbolic self-editing / curriculum evolution | `docs/archive/CIv10-ECA-hypothesis.md` · `docs/archive/CIv16-hypothesis.md` |
| Shani et al. (2024) — compression-meaning divergence in LLM vs. human concepts | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` |
| Sutskever, I. (2023) — compression as prediction / unsupervised learning | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CIv10-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |
| T2L (2025) — symbolic descriptions conditioning parameter injections | `docs/archive/CIv10-LLM-hypothesis.md` |
| Vargas, M. et al. (2022) — cybernetic systems at the meso scale | `docs/archive/CIv16-hypothesis.md` |
| Vivier-Ardisson et al. (2025) — symbolic MCMC / reversible inference layers | `docs/archive/CIv7-ECA-hypothesis.md` |
| Walch, M. (2024) — torsion as topological signal in symbolic dynamics | `docs/archive/CIv7-ECA-hypothesis.md` · `docs/archive/CIv7-LLM-hypothesis.md` · `docs/archive/CIv10-LLM-hypothesis.md` · `docs/archive/CI7-dependency-graph.md` |

---

### 4.3 CI Source-File Index

The original source files for this document are archived at:

| File | Description |
|------|-------------|
| [`docs/archive/CIv7-ECA-hypothesis.md`](archive/CIv7-ECA-hypothesis.md) | Structural break detection via ECA symbolic substrates |
| [`docs/archive/CIv7-LLM-hypothesis.md`](archive/CIv7-LLM-hypothesis.md) | Latent fault geometry in language models |
| [`docs/archive/CIv7-grand-story.md`](archive/CIv7-grand-story.md) | Narrative exposition of CIv7 — "The Cartographers of the Unknown" |
| [`docs/archive/CIv7-grand-story-poem.md`](archive/CIv7-grand-story-poem.md) | Poetic rendering — "The Fable of the Fractured Compass" |
| [`docs/archive/CI7-dependency-graph.md`](archive/CI7-dependency-graph.md) | Master reference graph for CIv7 theoretical dependencies |
| [`docs/archive/CIv10-ECA-hypothesis.md`](archive/CIv10-ECA-hypothesis.md) | Byte-level symbolic substrate emergence via AU-Net |
| [`docs/archive/CIv10-LLM-hypothesis.md`](archive/CIv10-LLM-hypothesis.md) | Contact-geometric latent control and reflexive repair |
| [`docs/archive/CIv11-unified-story.md`](archive/CIv11-unified-story.md) | Entropic geometry and metriplectic computation |
| [`docs/archive/CIv15-hypothesis.md`](archive/CIv15-hypothesis.md) | Autopoietic planner with compression-aligned self-evolution |
| [`docs/archive/CIv16-hypothesis.md`](archive/CIv16-hypothesis.md) | Tower hypothesis — substrate hierarchy as observer enrichment |

---

## 5. Closure

All constructs in this document reduce to four quantities:

| Quantity | Role |
|----------|------|
| $\mathcal{O} = (\phi, M, B)$ | The primitive observer |
| $K_{\mathcal{O}}(X) := L_M(\phi(X))$ | Description length — the universal compression quantity |
| $\Delta K_{\mathcal{O}}(t)$ | Change in compression — the intelligence signal |
| $E_{\mathcal{O}}(X) = K_{\mathcal{O}}(X) - \sum_i K_{\mathcal{O}}(X_i)$ | Non-additivity — the mesoscopic / emergent-structure quantity |

No additional primitives are required.

---

*This document is the canonical entry point for the CI research programme.  
For the full verbatim text of any source document, see [`docs/archive/`](archive/).*
