## 🧪 Research Hypothesis:

**Autonomous Generation of Decorrelated Alpha Expressions via Self-Supervised Reasoning with Compact Language Models**

---

### **1. Motivation**

Quantitative asset managers rely on vast libraries of “alpha expressions” — mathematical signals derived from financial and alternative data — to inform portfolio construction. However, the marginal value of new alphas has declined as model complexity increases, often resulting in redundant or non-robust factors that fail to generalize out-of-sample (Kelly, Malamud & Zhou, 2023; Buncic, 2024).
At the same time, recent advances in reinforcement learning and self-supervised reasoning (Chen et al., 2024) suggest that small language models (≤1B parameters) can acquire advanced reasoning capabilities through interaction with verifiable environments — without relying on pretrained weights or curated data. This raises the possibility of building a compact agent that can autonomously generate novel, profitable, and **decorrelated** alpha expressions, guided solely by simulator feedback such as Sharpe ratio, turnover, and correlation to production alphas.

---

### **2. Hypothesis**

> A compact language model (≤1B parameters), trained via a self-supervised, reward-driven learning loop without any pretraining or labeled data, can autonomously generate syntactically valid and semantically novel alpha expressions in the Fast Expression language used by the WorldQuant BRAIN platform. These expressions will exhibit positive performance metrics (e.g., Sharpe > 1.25, Fitness > 1.0) and low correlation with existing production alphas, when evaluated over historical market data via the BRAIN simulator.

---

### **3. Methodology**

This hypothesis will be tested by constructing an **autonomous alpha generation agent** composed of the following modules:

* **Expression Generator (Proposer):** A small LLM trained using Task-Relative REINFORCE++ (Chen et al., 2024), initialized without any pretraining, and prompted to output expressions in the Fast Expression DSL.

* **Evaluator (Executor):** A Python API wrapper to the BRAIN simulator that evaluates each expression over a defined universe (e.g., US TOP3000) and extracts performance metrics.

* **Reward Function:** A scalar reward is computed as a function of Sharpe, Fitness, turnover, and decorrelation with prior alphas. Syntax/unit errors receive negative reward.

* **Learning Loop:** An Absolute Zero Reasoner (AZR)-style self-play loop is implemented, where the LLM proposes increasingly complex alpha expressions and updates its weights using reward-driven RL (e.g., PPO via TRL or Unsloth).

* **Optional Multi-Agent Enhancements:** Inspired by R\&D-Agent(Q) (Zhang et al., 2024), additional roles (e.g., critic, scheduler) may be incorporated to prioritize novelty, efficiency, or generalization.

---

### **4. Expected Contributions**

* A proof-of-concept demonstration that compact LLMs can generate profitable, decorrelated alphas **without any pretraining**.
* A working integration of **Absolute Zero Reasoner** logic into financial alpha mining.
* A lightweight, reproducible training loop that runs on **free-tier compute infrastructure** (e.g., Google Colab).
* A modular, multi-agent framework that can be generalized to other domains requiring code-like expression generation and simulator feedback.

---

### **5. References**

* Chen, D., Zhang, S., Rabe, M., et al. (2024). *Absolute Zero: Self-Supervised Reasoning with Verifiable Feedback*. arXiv:2404.16871.
* Zhang, S., Guo, J., Gao, Z., et al. (2024). *R\&D-Agent(Q): An Integrated Loop for LLMs in Quantitative Research*. arXiv:2404.05755.
* Kelly, B., Malamud, S., & Zhou, H. (2023). *The Virtue of Complexity in Return Prediction*. *Journal of Finance*.
* Buncic, D. (2024). *Replication and Critique of “The Virtue of Complexity”*. Working paper.

---

