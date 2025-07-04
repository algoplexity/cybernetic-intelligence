You are absolutely right. The two workflow documents you linked are excellent starting points, but they are indeed outdated and describe two different pieces of the puzzle. It's time to synthesize them into a single, modern, and complete workflow that aligns with `nbdev` v2, Quarto, and our specific project architecture.

Here is the revised, unified development workflow. This will be our definitive guide.

---

### **The Modernized `nbdev` Development Workflow for the ADIA Challenge**

This document replaces the outdated workflows and provides a single source of truth for our development process.

#### **Core Principles**

1.  **Notebook-First:** The `.ipynb` notebooks are the single source of truth for all code, tests, and documentation.
2.  **Separation of Concerns:** We make a clean distinction between two types of notebooks:
    *   **Library Notebooks (`01_*`, `02_*`, etc.):** These build the reusable "Lego bricks" of our project (`SeriesProcessor`, `MDL_AU_Net_Autoencoder`, `MDLPreTrainer`). They are exported into an installable Python library (`adia_cybernet`).
    *   **Controller Notebook (`00_main_controller.ipynb`):** This is the high-level application script that *uses* the Lego bricks to build the final "spaceship." It contains the `train()` and `infer()` logic specific to the ADIA challenge and is **not** part of the installable library.

---

### **Phase 1: Library Development (Building the Lego Bricks)**

This is the cycle for creating and modifying the core components of our solution in notebooks like `01_core_data_processing.ipynb`, `02_model_architecture.ipynb`, and `03_training_pipeline.ipynb`.

**Workflow Steps (in a new Colab session):**

1.  **Setup the Environment (The "Header Cell"):** At the top of your library notebook, run this standard setup cell.
    ```python
    # 1. Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # 2. Navigate to the project root directory
    %cd /content/drive/MyDrive/ADIA-Structural-Break-Challenge

    # 3. Install dependencies for this specific notebook
    !pip install -q cellpylib numpy pandas # Add other deps as needed

    # 4. Perform an "editable install" of our own library
    # This is CRUCIAL. It makes code from 01_* available to 02_*, etc.
    !pip install -e '.[dev]'
    ```

2.  **Develop and Test:**
    *   Write your Python classes and functions in code cells.
    *   Use `#| export` at the top of a cell to mark it for inclusion in the final Python library.
    *   Write your explanations and documentation in markdown cells.
    *   Write your unit tests in separate code cells. Use `#| include: false` at the top of these test cells to ensure they are executed but not shown in the final documentation website.

3.  **Export to the Library:**
    *   Once your code is working and your tests pass, you must **export the notebook to the Python library**. This updates the `.py` files in the `adia_cybernet/` directory.
    *   Run this command in a new cell:
        ```python
        from nbdev.export import nb_export
        # Replace with the name of the notebook you are working on
        nb_export('01_core_data_processing.ipynb', lib_path='adia_structural_break_challenge')
        ```
        *Alternatively, from the terminal: `!nbdev_export`*

4.  **Commit Your Work:**
    *   The `nbdev` philosophy is to commit **both** the notebook (`.ipynb`) and the exported Python module (`.py`). This keeps everything in sync.
    *   Run the following git commands:
        ```python
        # Configure git (only needed once per session)
        !git config --global user.name "Your Name"
        !git config --global user.email "your.email@example.com"
        
        # Add ALL changes (both .ipynb and .py files)
        !git add .
        
        # Commit with a clear message
        !git commit -m "feat: Implement and test PermutationSymbolizer"
        
        # Push to GitHub
        !git push origin main
        ```

---

### **Phase 2: Application Orchestration (Building the Spaceship)**

This is the cycle for working on the `00_main_controller.ipynb`. This notebook uses the library you built in Phase 1 to run the end-to-end experiment.

**Workflow Steps:**

1.  **Setup the Environment:**
    *   Use the exact same "Header Cell" as in Phase 1. The `pip install -e '.[dev]'` step is critical here, as it's what allows this notebook to `import` your library.

2.  **Develop the `train()` and `infer()` Logic:**
    *   **Important:** This notebook should **NOT** have a `#| default_exp` or any `#| export` directives. It is a user of the library, not a part of it.
    *   Import your components: `from adia_cybernet.core_data_processing import SeriesProcessor`, etc.
    *   Write the high-level orchestration logic:
        *   Define experiment configurations.
        *   Instantiate your classes (`MDLPreTrainer`, `BreakClassifierFinetuner`).
        *   Call the `.pretrain()` and `.finetune()` methods.
        *   Implement the `train()` and `infer()` functions as required by the ADIA challenge structure.
    *   Run the cells to execute and debug the full end-to-end pipeline.

3.  **Commit Your Work:**
    *   Since this notebook doesn't export any `.py` files, the commit is simpler.
    *   Run the `git` commands:
        ```python
        !git add 00_main_controller.ipynb
        !git commit -m "refactor: Update main training loop in controller"
        !git push origin main
        ```

---

### **Workflow Cheatsheet**

| Notebook Type | Purpose | Key Directives | Git Artifacts to Commit |
| :--- | :--- | :--- | :--- |
| **Library Notebooks**<br>(`01_*`, `02_*`, `03_*`) | Build reusable components. | `#| export` (for code)<br>`#| include: false` (for tests) | The `.ipynb` notebook **and** the exported `.py` file. |
| **Controller Notebook**<br>(`00_main_controller.ipynb`) | Run the end-to-end experiment. | **None.** This notebook is a script, not a library module. | Only the `.ipynb` notebook itself. |

This modernized workflow provides a clear, robust, and scalable process for developing our solution, leveraging the full power of `nbdev` while maintaining a clean separation between our core library and the main application script.
