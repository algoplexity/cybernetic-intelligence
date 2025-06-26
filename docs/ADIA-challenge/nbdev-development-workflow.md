

**One Controller, Many Source Notebooks.**

This is the standard and most scalable pattern for managing an `nbdev` project.

---

### **The "One Controller, Many Sources" Pattern**

Let's visualize how your project will evolve:

**Your Google Drive Folder Structure (Your "Source of Truth"):**
```
/MyDrive/Colab Notebooks/CrunchDAO/
├── ADIA_Structural_Break_Challenge.ipynb  <-- This is your "Controller"
└── adia_library_source/
    ├── 01_core_data_processing.ipynb      <-- Source for data processing code
    ├── 02_core_model_architecture.ipynb   <-- Source for model classes
    ├── 03_pipelines_training.ipynb        <-- Source for the training orchestrator
    └── 04_pipelines_inference.ipynb     <-- Source for the inference orchestrator
```

**Your Colab "Controller" Notebook (`ADIA_Structural_Break_Challenge.ipynb`):**
*   Its job is to run the **session setup** (install, clone, auth, mount).
*   Then, it will have a series of cells that **orchestrate the `nbdev` build process**. It will copy *all* the source notebooks from your Drive into the `nbs/` directory of the cloned project, run `nbdev_prepare`, run `nbdev_test`, and then commit and push everything.

**Your GitHub Repository Structure (The "Final Product"):**
```
/adia-structural-break-challenge/
├── nbs/                                <-- Your WORKSPACE, a mirror of your `adia_library_source`
│   ├── 01_core_data_processing.ipynb
│   ├── 02_core_model_architecture.ipynb
│   ├── 03_pipelines_training.ipynb
│   └── 04_pipelines_inference.ipynb
│
├── adia_structural_break_challenge/    <-- Your auto-generated LIBRARY
│   ├── __init__.py
│   ├── core_data_processing.py         <-- Generated from 01_...
│   ├── core_model_architecture.py      <-- Generated from 02_...
│   ├── pipelines_training.py           <-- Generated from 03_...
│   └── pipelines_inference.py          <-- Generated from 04_...
│
├── settings.ini                        <-- Project configuration
├── .gitignore
└── ... (other config files)
```

### **Why This Pattern is Powerful**

1.  **Separation of Concerns:** Your `Controller` notebook handles the messy, session-specific details of the Colab environment. Your `Source` notebooks are clean, portable, and contain only the code, docs, and tests relevant to their specific module.
2.  **Scalability:** As your project grows, you just add new `Source` notebooks (e.g., `05_utils.ipynb`). You only need to add one line to your `Controller` notebook's copy script. The rest of the workflow (`nbdev_prepare`, `nbdev_test`) automatically picks up the new files.
3.  **Single Point of Control:** To build and deploy your entire project, you only need to open and run one notebook: the `Controller`. It acts as your master build script.
4.  **Clarity:** It's immediately clear to anyone looking at your project where the source code lives (`nbs/` on GitHub, `adia_library_source/` on your Drive) and how the project is built (by looking at the `Controller` notebook).

So, to summarize, you will have **one main "Controller" notebook** that you use in Colab to manage the entire project lifecycle, and a **growing collection of focused "Source" notebooks** on your Google Drive that contain the actual logic of your library. This is the path to a clean, organized, and professional `nbdev` project.
---

This is another fantastic question that connects our design (C4) directly to our implementation (`nbdev`).

The answer is: **Each `.ipynb` file corresponds to a Component in our C4 architecture.**

Let's break down this mapping.

### **The C4-to-`nbdev` Mapping**

| C4 Level | C4 Concept | `nbdev` Implementation |
| :--- | :--- | :--- |
| **Level 2** | **Container** | A **logical grouping** of Python modules. In our case, `Core Services Library` is a container. This corresponds to a **subdirectory** in the final Python package (e.g., `adia_structural_break_challenge/core_library/`). |
| **Level 3** | **Component** | A single `.py` file (a Python module). This is the key mapping. A component like `PermutationSymbolizer` or `SeriesProcessor` lives inside a specific module. |
| **Level 4** | **Code** | The classes and functions within a `.py` file. |

**How `nbdev` Fits In:**

*   An `.ipynb` notebook is the **source code for a Level 3 Component**.
*   The `nbdev` directive `#| default_exp core_data_processing` tells the system: "When you export this notebook, create a Python module named `core_data_processing.py`."
*   The classes defined within that notebook (`PermutationSymbolizer`, `SeriesProcessor`) become the **Level 4 Code** inside that module.

### **Applying this to Our Project**

Based on our C4 design, here is the direct mapping for our `Core Services Library` container:

| C4 Component Name | `nbdev` Source Notebook Name | Generated Python Module |
| :--- | :--- | :--- |
| **Data Processing Components** | `01_core_data_processing.ipynb` | `adia_structural_break_challenge/core_data_processing.py` |
| **Model Architecture Components** | `02_core_model_architecture.ipynb`| `adia_structural_break_challenge/core_model_architecture.py` |

And for our `Pipeline` containers:

| C4 Component Name | `nbdev` Source Notebook Name | Generated Python Module |
| :--- | :--- | :--- |
| **Training Pipeline Components** | `03_pipelines_training.ipynb` | `adia_structural_break_challenge/pipelines_training.py` |
| **Inference Pipeline Components**| `04_pipelines_inference.ipynb` | `adia_structural_break_challenge/pipelines_inference.py` |

**What about the Container level?**

In a very large project, you might structure your library with subdirectories. For example:
*   `adia_structural_break_challenge/core/data_processing.py`
*   `adia_structural_break_challenge/core/model_architecture.py`

You can achieve this in `nbdev` by using dot notation in the `#| default_exp` directive:
```python
#| default_exp core.data_processing
```
This would tell `nbdev` to create a `core` subdirectory and place the `data_processing.py` file inside it.

For our current project size, keeping the modules flat within the main library (`adia_structural_break_challenge/`) is perfectly fine and standard. The logical "container" concept is maintained by our naming convention (`core_*`, `pipelines_*`).

**Conclusion:**

You are correct to think in these terms. The mapping is clean and direct:

*   **1 C4 Component = 1 `.py` module = 1 `.ipynb` source notebook.**

This ensures that our final project structure on GitHub perfectly mirrors the logical components we designed in our C4 architecture diagrams, which is the hallmark of a well-engineered system.

