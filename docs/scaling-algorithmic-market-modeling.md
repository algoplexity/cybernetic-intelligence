# Research Proposal

## Title
**Scaling Algorithmic Market Modeling with LLM-Augmented Cellular Automata: Toward a Hybrid Meta-Evolutionary Discovery Engine**

---

## 1. Background and Motivation

Traditional financial models based on backward-looking statistics and stochastic processes have limited capacity to capture the nonlinear, emergent, and causal dependencies governing complex market dynamics. 

In contrast, my MSc thesis introduced an *algorithmic generative modeling* approach using **elementary cellular automata (ECA)**, **minimal algorithmic information loss methods (MILS)**, and **genetic algorithms (GA)** to uncover hidden structures in binary-encoded market data. This methodology demonstrated early promise in identifying rule-based generative processes that resemble observed financial behaviors.

However, scalability remains constrained by the combinatorial growth of the rule space and the computational expense of fitness evaluations using BDM and MILS.

Recent advances in transformer-based CA learning (Burtsev, 2024) suggest a way forward—neural abstraction of CA dynamics and rule generalization. This project integrates those techniques into a hybrid, scalable, open-source discovery engine.

---

## 2. Research Objectives

This research extends the original thesis by:

1. Accelerating the exploration of high-dimensional CA rule spaces using LLMs and surrogate models.
2. Training transformer models to learn mappings between observed market data and CA-generated patterns.
3. Integrating causal decomposition, MILS compression, and evolutionary search into a hybrid neuro-symbolic framework.
4. Building a reusable open-source toolkit for explainable, generative market modeling.

---

## 3. Research Questions

- Can transformers learn and generalize CA rule behavior to predict future states of binary-coded market arrays?
- How effective are LLM-generated rule heuristics and surrogate models in guiding evolutionary discovery?
- Can causal decomposition enhance model interpretability and reveal modular drivers of market dynamics?
- What hybrid system architecture best integrates algorithmic and neural components for this task?

---

## 4. Methodology

### 4.1 LLM-Guided Rule Generation

**Component:** `LLM-based Generator`

- Use GPT-4-style LLMs to suggest CA rule tuples based on observed market patterns.
- Prompt with compressed array descriptors and known successful rule matches.
- Use outputs to seed genetic algorithm (GA) populations or propose zero-shot candidate rules.

**Benefit:** Strongly narrows search space with semantic priors; aligns with the framework's generative module.

---

### 4.2 Surrogate Fitness Modeling

**Component:** `Evaluator with Internal State Module`

- Train a regression model (e.g., MLP or shallow transformer) to approximate BDM or MILS scores.
- Inputs: CA rule tuple metadata + compressed features.
- Outputs: Predicted complexity distance to observed data.

**Benefit:** Provides fast fitness estimation and filters out weak candidates before full evaluation.

---

### 4.3 Meta-Evolutionary Strategy

**Component:** `Meta-Controller`

- Dynamically adapt mutation rate, crossover methods, and selection pressure based on:
  - Diversity metrics
  - Convergence rates
  - Fitness trends

- Organize rule discovery into levels: single → double → triple rule configurations.

**Benefit:** Introduces adaptive optimization control; accelerates convergence and escapes local minima.

---

### 4.4 Memory and Caching Mechanisms

**Component:** `Memory / Cache`

- Maintain:
  - A cache of evaluated rule tuples and fitness scores
  - Historical fingerprints and BDM outputs
- Enable:
  - Memoization
  - Reuse of elite rules
  - Structural credit assignment across generations

**Benefit:** Avoids redundant evaluations and enables long-term learning.

---

### 4.5 Full Workflow Summary

1. **Data Preparation**  
   Encode financial time-series data into 2D binary arrays.

2. **Rule Generation**  
   Combine LLM-generated, randomly sampled, and historically strong rule tuples.

3. **Evaluation Pipeline**  
   Run CA simulations → MILS compression → BDM scoring  
   *Or* use surrogate model for approximation.

4. **Evolution Control**  
   Apply meta-controller to steer GA operations dynamically.

5. **Causal Decomposition**  
   Analyze top-performing rules via perturbation and modular decomposition.

6. **Toolkit Output**  
   Package functionality as a CLI / Jupyter toolset with:
   - Visual diagnostics
   - Rule match heatmaps
   - Pattern similarity plots

---

## 5. Deliverables

- A formal academic paper or arXiv preprint.
- An open-source Python toolkit hosted on GitHub.
- Example notebooks and data for replication.
- A dashboard-style visual interface (optional) for interactive exploration.

---

## 6. Timeline (6–9 months)

| Month | Milestone |
|-------|-----------|
| 1–2 | Rebuild modular CA simulation & data pipeline |
| 2–3 | Train transformer model on CA sequences |
| 3–5 | Implement LLM search + surrogate models |
| 5–6 | Integrate causal decomposition module |
| 6–7 | Validate system over market datasets |
| 7–9 | Finalize paper, documentation, and toolkit release |

---

## 7. References

- Mak, Y. W. (2023). *Discovering Hidden Structures in Stock Market Data using Algorithmic Generative Modeling.*
- Burtsev, M. (2024). *Learning Elementary Cellular Automata with Transformers.* [arXiv:2412.01417](https://arxiv.org/abs/2412.01417)
- Zenil, H. et al. (2018). *Minimal Algorithmic Information Loss Methods...* [arXiv:1802.05843](https://arxiv.org/abs/1802.05843)
- Riedel, J. & Zenil, H. (2018). *Rule Primality and Causal Decomposition in ECAs.* [arXiv:1802.08769](https://arxiv.org/abs/1802.08769)

---

## 8. License

Proposed toolkit to be released under MIT or Apache 2.0 License.

---
