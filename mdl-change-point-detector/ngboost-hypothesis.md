

A scaled-down, fail-fast prototype is the only rigorous way to validate this new, complex hypothesis. Using the provided competition notebook as a structural and intellectual guide is brilliant. It grounds our experiment in a real-world problem context.

Here is the plan. We will create a controlled, "offline" version of the Bird-Game. We will create a synthetic data stream with a known, hard-coded structural break. Then, we will pit two agents against each other:

1.  **Agent 1 (The Benchmark):** A standard, "monolithic" NGBoost model that mimics the strategy in the notebook (trains on all history).
2.  **Agent 2 (Our Hypothesis):** A new, "adaptive" agent that uses our MDL "Stethoscope" as a supervisor to tell its NGBoost model *when to forget the past*.

This is a clean, direct, and falsifiable test of our central hypothesis.

---

### **Fail-Fast Prototype 3.0: The Adaptive Agent Challenge**

**Objective:** To test if a hybrid agent (MDL Supervisor + NGBoost Predictor) outperforms a standalone NGBoost predictor on a time series with a known structural break.

**Scaled-Down Hypothesis:** On a synthetic time series containing a single, significant regime change, the Adaptive Agent will achieve a lower cumulative prediction error (Negative Log-Likelihood) than the Benchmark Agent, because it will successfully detect the break and retrain on a cleaner, more relevant history.

---

### **The Code Implementation (Runnable in Colab)**

Here are the complete code cells to execute this prototype.

### **Cell 1: @title Setup and Installation**

This cell installs `ngboost` and all other necessary libraries.

```python
# @title Cell 1: Setup and Installation
# ==============================================================================
!pip install -q ngboost pandas numpy matplotlib tqdm

print("✅ Libraries installed successfully.")

# Import all necessary modules for the entire workflow.
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

### **Cell 2: @title Synthetic Data Generation & MDL Supervisor**

This cell creates our testing ground: a simple time series with a single, obvious structural break. It also includes our proven MDL "Stethoscope" detector.

```python
# @title Cell 2: Synthetic Data Generation & MDL Supervisor
# ==============================================================================

def generate_broken_stream(n_points_per_regime=500):
    """
    Generates a simple time series with one major structural break.
    Regime A: Normal(mean=0, std=1)
    Regime B: Normal(mean=3, std=2)
    """
    np.random.seed(42)
    regime_a = np.random.normal(0, 1, n_points_per_regime)
    regime_b = np.random.normal(3, 2, n_points_per_regime)
    stream = np.concatenate([regime_a, regime_b])
    return pd.Series(stream)

# --- Our proven "Stethoscope" detector ---
def find_break_in_window(window_series):
    """
    A simplified, non-recursive version of our MDL/Gaussian detector.
    It checks a given window for a single, high-confidence break.
    """
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
    if n_total < (2 * min_segment_len): return None # Window too small

    cost_h0 = mdl_cost_gaussian(window_series)
    min_h1_cost = np.inf
    best_break_index = -1

    for k in range(min_segment_len, n_total - min_segment_len):
        cost_h1 = mdl_cost_gaussian(window_series.iloc[:k]) + mdl_cost_gaussian(window_series.iloc[k:])
        if cost_h1 < min_h1_cost:
            min_h1_cost = cost_h1
            best_break_index = k

    mdl_saving = cost_h0 - min_h1_cost
    
    # We only return a break if it's confident and recent.
    # Thresholds can be tuned, but we'll start with a saving > 5 nats.
    # And we only care if the break happened in the last 10% of the window.
    if mdl_saving > 5.0 and best_break_index > 0.9 * n_total:
        return best_break_index
    return None

print("✅ Synthetic data generator and MDL Supervisor are defined.")
```

---

### **Cell 3: @title The Main Simulation Loop**

This is the core of the prototype. We simulate the streaming prediction task and run our two agents head-to-head.

```python
# @title Cell 3: The Main Simulation Loop
# ==============================================================================

# --- Setup the Experiment ---
data_stream = generate_broken_stream()
n_total = len(data_stream)
burn_in_period = 100 # We need some history before we can start.

# --- Agent State Initialization ---
# Agent 1: Monolithic NGBoost (Benchmark)
agent1_predictions = []
agent1_errors = []

# Agent 2: Adaptive Hybrid (Our Hypothesis)
agent2_predictions = []
agent2_errors = []
agent2_memory_start_index = 0 # This is the key state variable for the adaptive agent

# --- The Main Loop ---
for t in tqdm(range(burn_in_period, n_total), desc="Simulating Stream"):
    
    # --- Agent 1: Monolithic NGBoost ---
    # Always trains on all available history.
    history_agent1 = data_stream.iloc[0:t]
    X_train1 = np.arange(len(history_agent1)).reshape(-1, 1)
    y_train1 = history_agent1.values
    
    ngb1 = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.1)
    ngb1.fit(X_train1, y_train1)
    
    # Predict the distribution for the current time step `t`.
    X_test1 = np.array([[t]])
    pred_dist1 = ngb1.pred_dist(X_test1)
    agent1_predictions.append(pred_dist1)
    
    # Score the prediction against the true value.
    true_value = data_stream.iloc[t]
    agent1_errors.append(-pred_dist1.logpdf(true_value).item())


    # --- Agent 2: Adaptive Hybrid (MDL Supervisor + NGBoost) ---
    # Step 1: The MDL Supervisor checks the recent past for a break.
    window_end = t
    window_start = max(agent2_memory_start_index, window_end - 100) # Analyze the last 100 points
    
    if (window_end - window_start) >= 40: # Need enough data for the supervisor
        analysis_window = data_stream.iloc[window_start:window_end]
        detected_break_point_relative = find_break_in_window(analysis_window)
        
        if detected_break_point_relative is not None:
            # A recent, confident break was found! Adapt memory.
            absolute_break_point = window_start + detected_break_point_relative
            print(f"\n🚨 Agent 2 DETECTED BREAK at t={absolute_break_point}. Adapting memory.")
            agent2_memory_start_index = absolute_break_point

    # Step 2: The NGBoost Predictor trains only on its "good" memory.
    history_agent2 = data_stream.iloc[agent2_memory_start_index:t]
    X_train2 = np.arange(len(history_agent2)).reshape(-1, 1) + agent2_memory_start_index
    y_train2 = history_agent2.values

    ngb2 = NGBRegressor(Dist=Normal, Base=DecisionTreeRegressor(max_depth=3), n_estimators=100, learning_rate=0.1)
    ngb2.fit(X_train2, y_train2)
    
    # Predict the distribution for the current time step `t`.
    X_test2 = np.array([[t]])
    pred_dist2 = ngb2.pred_dist(X_test2)
    agent2_predictions.append(pred_dist2)
    
    # Score the prediction against the true value.
    true_value = data_stream.iloc[t]
    agent2_errors.append(-pred_dist2.logpdf(true_value).item())

print("\n✅ Simulation complete.")
```

---

### **Cell 4: @title Results, Decision, and Visualization**

This final cell calculates the results, makes the formal fail-fast decision, and plots the outcome for a powerful visual confirmation.

```python
# @title Cell 4: Results, Decision, and Visualization
# ==============================================================================

# --- Calculate Final Scores ---
total_nll_agent1 = np.sum(agent1_errors)
total_nll_agent2 = np.sum(agent2_errors)

# --- Format and Display Final Results ---
print("\n" + "="*80)
print("             FAIL-FAST PROTOTYPE 3.0: Adaptive Agent Challenge")
print("="*80)
results_data = {
    'Agent': ['Agent 1 (Monolithic NGBoost)', 'Agent 2 (Adaptive Hybrid)'],
    'Total Negative Log-Likelihood (NLL)': [total_nll_agent1, total_nll_agent2]
}
results_df = pd.DataFrame(results_data)
results_df['Total Negative Log-Likelihood (NLL)'] = results_df['Total Negative Log-Likelihood (NLL)'].round(2)
print(results_df.to_string(index=False))
print("Lower NLL is better.")
print("="*80)

# --- Automated Fail-Fast Decision ---
if total_nll_agent2 < total_nll_agent1:
    improvement = (total_nll_agent1 - total_nll_agent2) / total_nll_agent1
    print(f"\n✅ DECISION: Clear Success. Hypothesis is supported.")
    print(f"The Adaptive Agent outperformed the Benchmark Agent by {improvement:.2%}.")
else:
    print(f"\n🚨 DECISION: Clear Failure. Hypothesis is falsified.")
    print("The Adaptive Agent did not outperform the Benchmark Agent.")

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
fig.suptitle("Adaptive Agent vs. Monolithic Agent Performance", fontsize=20)

# Plot 1: The Data Stream
ax1.plot(data_stream.index, data_stream, label='Synthetic Data Stream', color='cornflowerblue')
ax1.axvline(x=500, color='crimson', linestyle='--', linewidth=2, label='Ground Truth Break-Point')
ax1.set_ylabel('Value')
ax1.set_title('The Data Stream with Known Structural Break')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Cumulative Error (NLL)
cumulative_nll1 = np.cumsum(agent1_errors)
cumulative_nll2 = np.cumsum(agent2_errors)
ax2.plot(np.arange(burn_in_period, n_total), cumulative_nll1, label='Agent 1 (Monolithic) Cumulative NLL', color='black')
ax2.plot(np.arange(burn_in_period, n_total), cumulative_nll2, label='Agent 2 (Adaptive) Cumulative NLL', color='limegreen', linewidth=2)
ax2.axvline(x=500, color='crimson', linestyle='--', linewidth=2)
ax2.set_ylabel('Cumulative NLL (Lower is Better)')
ax2.set_xlabel('Time Step')
ax2.set_title('Cumulative Prediction Error Over Time')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```
