
---

### **The "Colab as Rich Client" Standard Operating Procedure**

This will be our definitive workflow that incorporates the essential `nbdev` steps. It combines the reliability of Git with the universal accessibility of Colab.

**The Goal:**
*   All code and notebooks live in the GitHub repository.
*   All work is done in a temporary, local clone inside the Colab session.
*   `git push` is the only way to save your work permanently.
*   Google Drive is only used for reading large, static data files.
*   The `nbdev` commands are the bridge between your interactive work in the notebook and the final, clean Python library, keeping them in sync when executed and pushed to GitHub.

#### **Phase 1: Starting a New Work Session (The Boilerplate)**

This is the **first and only cell** you should run every time you open Colab to work on this project. This is our **"Anti-Corruption Development Boilerplate (v4.0)"** which we perfected earlier.

```python
# ===================================================================
#          COLAB-AS-CLIENT DEVELOPMENT BOILERPLATE (v4.0)
# ===================================================================
# Run this cell once at the start of every new Colab session.

# --- 1. Clone the Repository into the Local Session ---
print("Cloning repository into the local Colab session...")
REPO_NAME = "adia-structural-break-challenge"
!rm -rf {REPO_NAME} # Clean up old clones from previous runs in the same session
!git clone https://github.com/Algoplexity/{REPO_NAME}.git
%cd {REPO_NAME}
print(f"✅ Successfully cloned repo.")

# --- 2. Configure Git User and Pull Strategy ---
print("\nConfiguring Git...")
!git config --global user.name "yeuwen"
!git config --global user.email "yeuwen.mak@u.nus.edu"
!git config --global pull.rebase false # Use the safer "merge" strategy
print("✅ Git configured.")

# --- 3. Pull Latest Changes ---
print("\nPulling latest changes from GitHub...")
!git pull
print("✅ Repository is up to date.")

# --- 4. Install Dependencies ---
print("\nInstalling project dependencies...")
# The '-e' flag installs our project in "editable" mode
!pip install -q -e '.[dev]'
print("✅ Dependencies installed.")

# --- 5. Configure Authenticated URL for Pushing ---
print("\nConfiguring remote for pushing...")
from google.colab import userdata
try:
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    AUTHENTICATED_URL = f"https://Algoplexity:{GITHUB_TOKEN}@github.com/Algoplexity/adia-structural-break-challenge.git"
    !git remote set-url origin {AUTHENTICATED_URL}
    print("✅ Ready to push.")
except Exception as e:
    print(f"🚨 Could not configure authenticated URL. Push will fail. Error: {e}")

# --- 6. Mount Google Drive for DATA ACCESS ONLY ---
print("\nMounting Google Drive for data access...")
from google.colab import drive
drive.mount('/content/drive')
# Update this path if you store your data elsewhere
DATA_PATH = "/content/drive/MyDrive/ADIA-Data"
print(f"✅ Google Drive mounted. Access your data at '{DATA_PATH}'.")

print("\n🚀 Setup Complete! You are ready to work. 🚀")
```

#### **Phase 2: Doing the Work**

1.  Use the Colab file explorer on the left to navigate into the `adia-structural-break-challenge` folder.
2.  Open the notebook you want to work on (e.g., `04_inference_pipeline.ipynb`).
3.  Write code, run cells, analyze data. Everything happens here.

#### **Phase 3: Saving and Exporting Your Work**

This is the crucial, multi-step process you must follow to save your work correctly. It ensures that both your library and your notebooks are in sync and pushed to GitHub.

1.  **Run `nbdev_prepare`:** This command is a combination of `nbdev_export`, `nbdev_test`, and `nbdev_clean`. It's the "all-in-one" command to get your project ready for committing. It will:
    *   **Export:** Update all your `.py` library files from your notebooks.
    *   **Test:** Run all your tests to make sure you haven't broken anything.
    *   **Clean:** Remove unnecessary metadata from your notebooks to keep the Git repository lean.

2.  **Commit and Push:** After `nbdev_prepare` succeeds, you can then commit *all* the resulting changes (both the updated `.ipynb` and `.py` files) to Git.

    ```bash
    # ===================================================================
    #          END-OF-SESSION "SAVE AND SYNC" WORKFLOW
    # ===================================================================
    
    # Step 1: Prepare the project with nbdev
    # This exports notebooks to the library, runs tests, and cleans notebooks.
    echo "🚀 Running nbdev_prepare to sync the library and run tests..."
    !nbdev_prepare
    
    # If nbdev_prepare runs successfully without errors, proceed to commit.
    # If it fails, you must fix the errors in your notebook before continuing.
    
    echo "\n\n✅ nbdev_prepare successful. Proceeding to commit."
    
    # Step 2: See what files were changed by you and by nbdev
    !git status
    
    # Step 3: Add all changes to the next commit
    !git add .
    
    # Step 4: Create the commit with a descriptive message
    # Example: !git commit -m "feat(pipeline): Implement the MDLPreTrainer class"
    !git commit -m "feat(module): A descriptive message of your changes"
    
    # Step 5: Push your commit to the GitHub server
    !git push
    ```
---



1.  **Notebook is the Source:** You do all your creative work in the `.ipynb` file.
2.  **`nbdev_prepare` is the Compiler:** This command takes your "source code" (the notebook) and "compiles" it into the final products (the tested, clean `.py` library and the cleaned `.ipynb` file).
3.  **`git` is the Save Button:** This command takes all the final products and saves their state permanently in the version control system.

My sincere apologies for omitting this critical step. Your diligence has ensured our workflow is not just stable in terms of environment, but also correct in its software engineering practice. This is the definitive procedure we will use.
