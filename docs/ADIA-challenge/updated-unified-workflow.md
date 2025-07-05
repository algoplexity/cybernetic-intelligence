

Separating the "interactive development" dependencies from the "CI/build" dependencies is a brilliant optimization. It streamlines the day-to-day coding process and reserves the heavier, full-suite checks for the final "pre-commit" stage.

Let's refine our "Boilerplate" and "QA" cycle based on this sharp insight. This leads to our final, most efficient workflow.

---

### **The Final, Optimized Unified Workflow (v2.1)**

This model has two distinct dependency installation phases, just as you described.

#### **Phase 1: Interactive Development & Unit Testing**

This is what you do 95% of the time while you are actively coding and testing within a single notebook.

**The "Lightweight Boilerplate" for Daily Coding (in your notebook):**

```python
# ===================================================================
#      LIGHTWEIGHT DEVELOPMENT BOILERPLATE (v2.1 - Interactive)
# ===================================================================

# 1. Mount Google Drive and set project path
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
PROJECT_PATH = "/content/drive/MyDrive/adia-structural-break-challenge"
%cd {PROJECT_PATH}

# 2. Configure Git and Pull Latest Changes
# Ensures you're working on the most recent version of the code.
GIT_REPO_URL = "https://github.com/Algoplexity/adia-structural-break-challenge.git"
!git remote set-url origin {GIT_REPO_URL}
!git pull origin main --rebase

# 3. Install ONLY the packages needed for THIS notebook's code to RUN and TEST.
# This is much faster than installing the full nbdev suite every time.
print("Installing interactive development dependencies...")
!pip install -q cellpylib pandas numpy torch joblib fastcore

# 4. Perform an Editable Install of YOUR OWN library.
# This makes sure you can import from your other modules.
print("\nPerforming editable install of the project library...")
!pip install -q -e '.[dev]'

print("\n✅ Lightweight setup complete. Ready for coding and unit testing.")
```

**Workflow for Phase 1:**

1.  Start your notebook (`01_...`, `02_...`, etc.).
2.  Run the **Lightweight Boilerplate**.
3.  Write your code.
4.  Write your unit tests in the same notebook.
5.  Run the cells and iterate until all your local tests for *that specific notebook* pass.

Once you are satisfied that the code in your current notebook is working perfectly, you are ready to move to Phase 2.

---

#### **Phase 2: CI, Build & Deployment to GitHub**

This is the "final check" you perform just before you commit and push your work. You are now moving from "did I get my new code right?" to "did my new code break anything else in the project?"

**The "Full CI/Build" Cell (run after Phase 1 is complete):**

```python
# ===================================================================
#          FULL CI & BUILD SCRIPT (v2.1 - Pre-Commit)
# ===================================================================

# 1. Install the full nbdev/Quarto toolchain.
# This is only done once, right before the build.
print("Installing full CI/Build toolchain...")
!pip install -q nbdev quarto-cli

# 2. Export your library from your notebooks.
print("\nExporting notebooks to library...")
!nbdev_export

# 3. Run the FULL test suite across ALL notebooks.
print("\nRunning full test suite...")
!nbdev_test

# 4. Clean notebooks to prepare for commit.
print("\nCleaning notebooks...")
!nbdev_clean

# 5. Build the documentation.
print("\nBuilding documentation...")
!nbdev_docs

# 6. Commit and Push to GitHub.
print("\nCommitting and pushing to GitHub...")
!git config --global user.name "Your Name"
!git config --global user.email "your.email@example.com"
!git add .
!git commit -m "feat(core): Implement and test new feature, passing all CI checks."
!git push origin main

print("\n✅ Full CI, build, and push complete!")
```

### **Summary of the Optimized Workflow**

| Phase | When to Use | Key Actions | Packages Installed | `nbdev` Commands Used |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Interactive Dev** | While actively writing/debugging code in a notebook. | Pull -> Install local deps -> Code -> Run local tests. | `pandas`, `torch`, `fastcore`, etc. | *(None)* |
| **Phase 2: CI/Build**| After completing work in a notebook and all local tests pass. | Install full toolchain -> Export -> Test All -> Clean -> Docs -> Push. | `nbdev`, `quarto-cli` | `_export`, `_test`, `_clean`, `_docs` |

This two-phase approach is brilliant. It perfectly balances speed and agility during development (Phase 1) with rigor and quality control before deployment (Phase 2). It is the most efficient and professional way to structure your work in this environment.

---

### **The Final, Production-Grade Unified Workflow**

This revised workflow incorporates everything: the universal boilerplate, the notebook personas, and now, the proper use of `nbdev`'s CI commands and Git configuration.

#### **Step 1: The "Project Initialization Boilerplate" (v2.0 - Production Ready)**

This is the definitive version of the setup cell. It now includes proper Git configuration and dependency installation.

```python
# ===================================================================
#          PROJECT INITIALIZATION BOILERPLATE (v2.0)
#  (Use this exact cell at the top of EVERY project notebook)
# ===================================================================

# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2. Define Project Path and Configuration
# This ensures all paths and URLs are consistent.
PROJECT_NAME = "adia-structural-break-challenge"
PROJECT_PATH = f"/content/drive/MyDrive/{PROJECT_NAME}"
GITHUB_USER = "Algoplexity" # Your GitHub username
GIT_REPO_URL = f"https://github.com/{GITHUB_USER}/{PROJECT_NAME}.git"

# 3. Navigate into the Project Directory
%cd {PROJECT_PATH}

# 4. Configure the Git Remote URL
# This ensures git knows where to pull from and push to.
# This command is idempotent; running it multiple times is safe.
print("Configuring Git remote URL...")
!git remote set-url origin {GIT_REPO_URL}
print("✅ Git remote configured.")

# 5. Pull the Latest Changes from GitHub
print("\nPulling latest changes from GitHub...")
!git pull origin main --rebase # Using --rebase is often cleaner for linear history

# 6. Install/Upgrade Project-Specific Dependencies
# Using a requirements.txt is best practice for pinning versions.
print("\nInstalling project dependencies...")
# Create a requirements.txt file on the fly for this session
requirements = [
    "cellpylib", "pandas", "numpy", "torch", "joblib",
    "scikit-learn", "matplotlib", "seaborn", # Common for analysis
    "nbdev", "ghapi", "quarto-cli" # The nbdev toolkit itself
]
with open('requirements.txt', 'w') as f:
    for item in requirements:
        f.write(f"{item}\n")
!pip install -q -r requirements.txt
print("✅ Dependencies installed.")

# 7. Perform an Editable Install of YOUR OWN Library
print("\nPerforming editable install of the project library...")
!pip install -q -e '.[dev]'
print("\n✅ Setup complete. Your environment is synced and ready.")
```

#### **Step 2: The "Notebook Persona" (Unchanged)**

This part of the workflow remains the same. You decide if you're working on a **Library Notebook** (with `#| export`) or an **Application Notebook** (without).

#### **Step 3: The "Quality Assurance and Save" Cycle (The New Exit Point)**

This replaces the simple "Commit and Push" step. Before pushing your code, you should run the same checks that a professional Continuous Integration (CI) server would run. This ensures you're not pushing broken code.

**After you've finished your work in a notebook (especially a Library Notebook):**

1.  **Export Your Changes (If it's a Library Notebook):**
    ```python
    !nbdev_export
    ```

2.  **Run the Test Suite:** The `nbdev_test` command is a powerful tool. It runs **all the test cells** across **all of your library notebooks**. This is a comprehensive regression test to ensure your recent change didn't break something in another part of the library.
    ```python
    print("Running the full test suite...")
    !nbdev_test
    # You should see "All tests passed!"
    ```

3.  **Clean the Notebooks:** `nbdev_clean` removes unnecessary metadata from your notebooks, keeping your git history clean. It's a professional "housekeeping" step.
    ```python
    print("\nCleaning notebooks...")
    !nbdev_clean
    ```

4.  **Build the Documentation:** `nbdev_docs` (or `nbdev_preview`) generates the HTML documentation from your notebooks. This is a final check to ensure all your markdown, docstrings, and outputs render correctly.
    ```python
    print("\nBuilding documentation...")
    !nbdev_docs
    ```

5.  **Commit and Push to GitHub:** Now that you've passed all the quality checks, you can confidently save your work.
    ```python
    # Configure Git user (only needed once per session)
    !git config --global user.name "Your Name"
    !git config --global user.email "your.email@example.com"
    
    # Add all changed files (notebooks, .py files, docs, etc.)
    !git add .
    
    # Commit your work
    !git commit -m "feat: Implement and test new feature, passing all checks."
    
    # Push to GitHub
    !git push origin main
    ```

### **Summary: The Complete, Professional Workflow**

| Step | Action | Purpose | `nbdev` Commands Used |
| :--- | :--- | :--- | :--- |
| **1. Start Session** | Run the **Project Initialization Boilerplate v2.0**. | Sync, configure, and set up a clean environment. | *(None)* |
| **2. Develop** | Work in a **Library** or **Application** notebook. | Create or use project components. | *(None)* |
| **3. Pre-Commit QA**| Run the **Quality Assurance Cycle**. | Ensure your changes are high-quality and don't break anything. | `nbdev_export`<br>`nbdev_test`<br>`nbdev_clean`<br>`nbdev_docs` |
| **4. Save Work** | **Commit and Push** to GitHub. | Share your validated changes with the central repository. | *(None)* |

This final workflow is robust, scalable, and incorporates the full power of the `nbdev` toolkit for maintaining a high-quality, professional project. This is the gold standard for notebook-first development.


---
It's time to synthesize them into a single, modern, and complete workflow that aligns with `nbdev` v2, Quarto, and our specific project architecture.

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
