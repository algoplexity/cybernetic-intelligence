

We have defined the components responsible for processing real-world data (`PermutationSymbolizer` and `SeriesProcessor`). Now, we must define their counterpart: the component responsible for generating the **synthetic pre-training data**.

This completes the "Data" portion of our `Core Services Library`. The `ECADataGenerator` is the factory that will produce the high-quality, curated "gym" where our model will first learn the fundamental principles of dynamics, as dictated by our unified hypothesis.

Here is the detailed class design for `ECADataGenerator`.

---

### **Detailed Class Design: `ECADataGenerator`**

| Component | `ECADataGenerator` |
| :--- | :--- |
| **Module** | `core_library/data_processing.py` |
| **Role & Responsibilities** | **Synthetic Data Factory for Pre-training.**<br>- Its primary role is to create a high-quality, synthetic dataset for the MDL pre-training phase.<br>- It is responsible for simulating Elementary Cellular Automata based on a *curated* list of rules, not just all 256.<br>- The curation is guided by our research: focusing on complex rules (Zhang's "edge of chaos") and composite rules (Riedel & Zenil's primality findings).<br>- It processes the 2D spacetime output of the simulations into `(sequence, label)` pairs, where the `sequence` is a window of the ECA's evolution and the `label` is the ID of the rule that generated it.<br>- It must handle both simple (e.g., Rule 110) and multi-step composite rules (e.g., `Rule 51 -> Rule 118`) by assigning them unique labels. |
| **Key Collaborators**| *(None - Foundational Data Source)*. It uses standard libraries like `numpy` for its internal calculations but does not depend on other custom classes in our architecture. |

#### **Code Skeleton**

This skeleton defines the public interface and the necessary configuration.

```python
# In core_library/data_processing.py

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple

# ... existing classes (PermutationSymbolizer, SeriesProcessor) ...

class ECADataGenerator:
    """
    Generates a curated dataset of Elementary Cellular Automata evolutions
    for pre-training the dynamical autoencoder.
    """
    def __init__(self, config: Dict):
        """
        Initializes the generator with a detailed configuration.

        Args:
            config (Dict): A dictionary containing parameters like:
                - 'rules_to_use': List of integer rule IDs (e.g., [30, 54, 90, 110]).
                - 'composite_rules': List of lists for composite rules (e.g., [[51, 118]]).
                - 'width': The spatial width of the cellular automaton.
                - 'steps': The number of time steps to simulate.
                - 'sequence_length': The length of the sequences to extract for training.
                - 'warmup_steps': Number of initial steps to discard to allow chaos to develop.
                - 'seed': An integer for reproducible random initial conditions.
        """
        self.config = config
        self.rng = np.random.default_rng(self.config['seed'])
        self._rule_map = self._create_rule_map()

    def _create_rule_map(self) -> Dict[int, str]:
        """Creates a mapping from internal integer labels to human-readable rule names."""
        # This allows us to handle simple and composite rules with unique integer labels.
        # e.g., {0: "Rule 30", 1: "Rule 54", ..., 10: "Rule 51->118"}
        pass

    def _simulate_single_rule(self, rule_id: int, initial_state: np.ndarray) -> np.ndarray:
        """Simulates a single ECA rule for the configured number of steps."""
        # Core ECA simulation logic goes here.
        pass

    def _slice_simulation(self, simulation: np.ndarray, label: int) -> Tuple[List[np.ndarray], List[int]]:
        """Slices a 2D simulation into multiple (sequence, label) pairs."""
        # Takes the [steps, width] numpy array and creates a list of
        # [sequence_length, width] arrays.
        pass

    def generate_and_save(self, output_path: str) -> None:
        """
        The main public method. Orchestrates the generation of the full dataset
        and saves it to a file (e.g., .npz or .pt).
        """
        all_sequences = []
        all_labels = []

        # Iterate through simple and composite rules defined in the config
        for label, rule_def in self.config['rules_to_use'].items():
            # 1. Create a random initial state
            initial_state = self.rng.integers(0, 2, size=self.config['width'])
            
            # 2. Simulate the rule(s)
            simulation_result = self._simulate_single_rule(rule_def, initial_state) # Or a composite version
            
            # 3. Slice the result into training samples
            sequences, labels = self._slice_simulation(simulation_result, label)
            
            all_sequences.extend(sequences)
            all_labels.extend(labels)

        # 4. Save the compiled dataset
        # e.g., torch.save({'sequences': torch.tensor(all_sequences), 'labels': torch.tensor(all_labels)}, output_path)
        print(f"Generated and saved {len(all_sequences)} sequences to {output_path}")

```

### **How This Fits Into the Plan**

1.  **Completes the Data Layer:** With `ECADataGenerator`, we have now fully defined the "data processing" capabilities of our `Core Services Library`. We can create data for both pre-training (synthetic) and inference (real-world).
2.  **Enables Pre-training:** This component is the direct dependency for our `MDLPreTrainer`. The trainer will instantiate this generator to create the dataset on which our `HierarchicalDynamicalAutoencoder` will be trained.
3.  **Feeds the MDL Loss:** The `(sequence, label)` pairs produced by this generator are precisely what our dual-loss function needs: the `sequence` for the **Reconstruction Loss** and the `label` for the **Classification Loss**.
