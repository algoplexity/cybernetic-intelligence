

**Objective:** To design and test a **Coherent Adaptive Agent** whose adaptation strategy is directly inspired by the principles of the QCEA-T framework. This agent will replace the failed "Hard Reset" of the "Intelligent Amnesiac" with a more sophisticated, information-preserving approach.

**Hypothesis:** The Coherent Adaptive Agent will successfully adapt to the structural break in the synthetic data stream, achieving a significantly lower cumulative prediction error (NLL) than both the "Intelligent Amnesiac" and the monolithic benchmark agent.

---

### **The New Adaptation Strategy: Bayesian Transfer Learning**

The core insight from QCEA-T is to preserve valuable retrospective information. In our context, the most valuable information from the old regime is the learned **variance (sigma)** of the system. The **mean (mu)** is what changes during the break.

Our new strategy will be:

1.  **Detect:** The MDL "Stethoscope" detects a structural break.
2.  **Preserve:** The agent extracts the learned distribution parameters (`mu_old`, `sigma_old`) from its NGBoost model trained on the pre-break data.
3.  **Adapt:** The agent begins training a new NGBoost model on the post-break data. Crucially, it forces the new model to start its learning process not from a blank slate, but from a strong **prior** belief that the variance of the new world is very similar to `sigma_old`.
4.  **Predict:** The new model can now use the few new data points to rapidly and stably learn the *new mean* (`mu_new`) without having to relearn the variance from scratch, thus avoiding the "data starvation" catastrophe.

This is a direct, practical implementation of a coherence-preserving strategy.

---

### **The Code Implementation (Runnable in Colab)**

Here are the complete code cells. This requires a more complex simulation loop to manage the different agent states.

### **Cell 1: @title Setup and Installation (Unchanged)**

This cell remains the same.

```python
# @title Cell 1: Setup and Installation
# ==============================================================================
!pip install -q ngboost pandas numpy matplotlib tqdm

print("✅ Libraries installed successfully.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("✅ Modules imported successfully.")
```

---

### **Cell 2: @title Synthetic Data & MDL Supervisor (Unchanged)**

This cell, with our corrected `> 0.5` logic, also remains the same.

```python
# @title Cell 2: Synthetic Data & MDL Supervisor
# ==============================================================================
def generate_broken_stream(n_points_per_regime=500):
    np.random.seed(42)
    regime_a = np.random.normal(0, 1, n_points_per_regime)
    regime_b = np.random.normal(3, 2, n_points_per_regime)
    stream = np.concatenate([regime_a, regime_b])
    return pd.Series(stream)

def find_break_in_window(window_series):
    def mdl_cost_gaussian(segment):
        n = len(segment)
        if n < 2: return np.inf
        mean = np.mean(segment); variance = np.var(segment, ddof=1)
        if variance == 0: variance = 1e-9
        model_cost = 2 * np.log(n)
        log_likelihood = -np.sum((segment - mean)**2) / (2 * variance) - n/2 * np.log(2 * np.pi * variance)
        return model_cost - log_likelihood
    n_total = len(window_series)
    min_segment_len = 15
    if n_total < (2 * min_segment_len): return None
    cost_h0 = mdl_cost_gaussian(window_series)
    min_h1_cost = np.inf
    best_break_index = -1
    for k in range(min_segment_len, n_total - min_segment_len):
        cost_h1 = mdl_cost_gaussian(window_series.iloc[:k]) + mdl_cost_gaussian(window_series.iloc[k:])
        if cost_h1 < min_h1_cost:
            min_h1_cost = cost_h1
            best_break_index = k
    mdl_saving = cost_h0 - min_h1_cost
    if mdl_saving > 5.0 and best_break_index > 0.5 * n_total:
        return best_break_index
    return None

print("✅ Synthetic data generator and MDL Supervisor are defined.")
```

---

### **Cell 3: @title The FINAL Simulation Loop (Coherent Agent)**

This is the new, more complex core of the prototype. It now manages three agents: the Monolithic Benchmark, the failed "Intelligent Amnesiac," and our new "Coherent Adaptive Agent."

```python
# @title Cell 3: The FINAL Simulation Loop (Coherent Agent)
# ==============================================================================

# --- Setup the Experiment ---
data_stream = generate_broken_stream()
n_total = len(data_stream)
burn_in_period = 100

# --- Agent State Initialization ---
# Agent 1: Monolithic NGBoost (Benchmark)
agent1_errors = []
# Agent 2: Intelligent Amnesiac (Failed Strategy)
agent2_errors = []
agent2_memory_start = 0
# Agent 3: Coherent Adaptive Agent (Hypothesis)
agent3_errors = []
agent3_memory_start = 0
agent3_has_adapted = False
agent3_prior_dist = None # This will hold the "preserved information"

# --- The Main Loop ---
for t in tqdm(range(burn_in_period, n_total), desc="Simulating Stream"):
    true_value = data_stream.iloc[t]
    X_test = np.array([[t]])

    # --- Agent 1: Monolithic ---
    history1 = data_stream.iloc[0:t]
    ngb1 = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100)
    ngb1.fit(np.arange(len(history1)).reshape(-1, 1), history1.values)
    pred_dist1 = ngb1.pred_dist(X_test)
    agent1_errors.append(-pred_dist1.logpdf(true_value).item())

    # --- Agent 2: Intelligent Amnesiac ---
    history2 = data_stream.iloc[agent2_memory_start:t]
    ngb2 = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100)
    ngb2.fit(np.arange(len(history2)).reshape(-1, 1) + agent2_memory_start, history2.values)
    pred_dist2 = ngb2.pred_dist(X_test)
    agent2_errors.append(-pred_dist2.logpdf(true_value).item())

    # --- Agent 3: Coherent Adaptive Agent ---
    if not agent3_has_adapted:
        # Before adapting, it behaves just like the Amnesiac
        history3 = data_stream.iloc[agent3_memory_start:t]
        ngb3 = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100)
        ngb3.fit(np.arange(len(history3)).reshape(-1, 1) + agent3_memory_start, history3.values)
        pred_dist3 = ngb3.pred_dist(X_test)
        agent3_errors.append(-pred_dist3.logpdf(true_value).item())
        # Store the learned distribution as a potential prior
        agent3_prior_dist = ngb3.pred_dist(X_test)
    else:
        # After adapting, it uses the prior!
        history3 = data_stream.iloc[agent3_memory_start:t]
        # NGBoost doesn't have a direct "prior" setting, so we simulate it
        # by starting the new model with a custom initial distribution.
        # We initialize the mean to the new data's mean, but the STD to the old one.
        prior_std = agent3_prior_dist.params['scale']
        initial_mean = np.mean(history3.values) if len(history3) > 0 else 0
        
        # This custom distribution acts as our smart starting point
        custom_dist = Normal(loc=initial_mean, scale=prior_std)
        
        ngb3 = NGBRegressor(Dist=custom_dist, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100)
        ngb3.fit(np.arange(len(history3)).reshape(-1, 1) + agent3_memory_start, history3.values)
        pred_dist3 = ngb3.pred_dist(X_test)
        agent3_errors.append(-pred_dist3.logpdf(true_value).item())

    # --- MDL Supervisor Logic (for both adaptive agents) ---
    window_end = t
    # For Agent 2
    window_start2 = max(agent2_memory_start, window_end - 100)
    if (window_end - window_start2) >= 40:
        analysis_window2 = data_stream.iloc[window_start2:window_end]
        break_point2 = find_break_in_window(analysis_window2)
        if break_point2 is not None:
            abs_break = window_start2 + break_point2
            if abs_break > agent2_memory_start:
                agent2_memory_start = abs_break
    
    # For Agent 3
    window_start3 = max(agent3_memory_start, window_end - 100)
    if not agent3_has_adapted and (window_end - window_start3) >= 40:
        analysis_window3 = data_stream.iloc[window_start3:window_end]
        break_point3 = find_break_in_window(analysis_window3)
        if break_point3 is not None:
            abs_break = window_start3 + break_point3
            if abs_break > agent3_memory_start:
                print(f"\n🚨 Agent 3 DETECTED BREAK at t={abs_break}. Coherently Adapting.")
                agent3_memory_start = abs_break
                agent3_has_adapted = True


print("\n✅ Simulation complete.")
```

---

### **Cell 4: @title FINAL Results and Visualization**

This cell compares all three agents and provides the final, definitive visualization.

```python
# @title Cell 4: FINAL Results and Visualization
# ==============================================================================

# --- Calculate Final Scores ---
total_nll_agent1 = np.sum(agent1_errors)
total_nll_agent2 = np.sum(agent2_errors)
total_nll_agent3 = np.sum(agent3_errors)

# --- Format and Display Final Results ---
print("\n" + "="*80)
print("             FINAL PROTOTYPE: Coherent Adaptive Agent Challenge")
print("="*80)
results_data = {
    'Agent': ['Agent 1 (Monolithic)', 'Agent 2 (Amnesiac)', 'Agent 3 (Coherent Adaptive)'],
    'Total NLL (Lower is Better)': [total_nll_agent1, total_nll_agent2, total_nll_agent3]
}
results_df = pd.DataFrame(results_data)
results_df['Total NLL (Lower is Better)'] = results_df['Total NLL (Lower is Better)'].round(2)
print(results_df.to_string(index=False))
print("="*80)

# --- Automated Final Decision ---
if total_nll_agent3 < total_nll_agent1 and total_nll_agent3 < total_nll_agent2:
    improvement = (total_nll_agent1 - total_nll_agent3) / total_nll_agent1
    print(f"\n✅ DECISION: Clear Success. The Coherent Agent is superior.")
    print(f"It outperformed the Monolithic Benchmark by {improvement:.2%}.")
else:
    print(f"\n🚨 DECISION: Failure. The Coherent strategy was not superior.")
    print("The hypothesis is falsified or the implementation is flawed.")

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
fig.suptitle("Final Agent Comparison: The Coherent Adaptive Agent", fontsize=20)

ax1.plot(data_stream.index, data_stream, label='Synthetic Data Stream', color='cornflowerblue', alpha=0.7)
ax1.axvline(x=500, color='red', linestyle='--', linewidth=2, label='Ground Truth Break-Point')
ax1.set_ylabel('Value')
ax1.set_title('The Data Stream with Known Structural Break')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

cumulative_nll1 = np.cumsum(agent1_errors)
cumulative_nll2 = np.cumsum(agent2_errors)
cumulative_nll3 = np.cumsum(agent3_errors)
time_steps = np.arange(burn_in_period, n_total)

ax2.plot(time_steps, cumulative_nll1, label='Agent 1 (Monolithic) - Benchmark', color='black', linewidth=1.5)
ax2.plot(time_steps, cumulative_nll2, label='Agent 2 (Amnesiac) - Failed Strategy', color='crimson', linestyle=':', linewidth=2)
ax2.plot(time_steps, cumulative_nll3, label='Agent 3 (Coherent Adaptive) - Hypothesis', color='limegreen', linewidth=2.5)
ax2.axvline(x=500, color='red', linestyle='--', linewidth=2)
ax2.set_ylabel('Cumulative NLL (Lower is Better)')
ax2.set_xlabel('Time Step')
ax2.set_title('Cumulative Prediction Error Over Time')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# Use scientific notation for the y-axis if numbers are large
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```
