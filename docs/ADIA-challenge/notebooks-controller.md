
This is the notebook you will run in your Google Colab session. Its only job is to manage the environment and orchestrate the build process using the clean "Source Notebooks" stored on your Google Drive.

---

### **Instructions:**

1.  **Open a new or existing notebook in Google Colab.** This will be your Controller.
2.  **Add your GitHub Token** to Colab's Secrets Manager. Name the secret `GITHUB_TOKEN`.
3.  **Create the source folder on your Google Drive:** `My Drive/adia_project_source/`.
4.  **Place your clean source notebooks** (e.g., `00_index.ipynb`, `01_core_data_processing.ipynb`) inside that folder.
5.  **Copy and paste the two cells below** into your Controller notebook and run them in order.

---

### **Cell 1: Environment Setup & GitHub Integration**

This cell installs all necessary tools, clones your repository using your GitHub token, and mounts your Google Drive. Run this cell once at the beginning of each session.

```python
# @title Cell 1: Environment Setup & GitHub Integration
import os
import shutil
from google.colab import userdata, drive
import getpass

# --- 1. User-configurable variables ---
# Your GitHub details
GIT_USER_NAME = "algoplexity"
GIT_USER_EMAIL = getpass.getpass("Enter your GitHub email: ") # More secure than hardcoding
# Project details
REPO_NAME = "adia-structural-break-challenge"
# Name of the GitHub token in Colab's Secrets Manager
GITHUB_TOKEN_SECRET_NAME = "GITHUB_TOKEN"
# Name of the folder on your Google Drive containing your source notebooks
DRIVE_PROJECT_FOLDER = "adia_project_source"

# --- 2. Install necessary tools ---
print("--- Installing nbdev and dependencies... ---")
!pip install -Uqq nbdev ghapi fastcore
print("✅ Tools installed.")

# --- 3. Get GitHub token from Colab Secrets ---
print("--- Accessing GitHub token... ---")
try:
    GITHUB_TOKEN = userdata.get(GITHUB_TOKEN_SECRET_NAME)
except userdata.SecretNotFoundError:
    raise RuntimeError(f"Secret '{GITHUB_TOKEN_SECRET_NAME}' not found. Please add it to your Colab Secrets.")
print("✅ GitHub token accessed.")

# --- 4. Clone the repository ---
print(f"--- Cloning repository: {REPO_NAME}... ---")
# Remove the repo if it already exists to ensure a clean start
if os.path.exists(REPO_NAME):
    shutil.rmtree(REPO_NAME)
# Construct the remote URL with the token for authentication
remote_url_with_token = f"https://{GITHUB_TOKEN}@github.com/{GIT_USER_NAME}/{REPO_NAME}.git"
!git clone {remote_url_with_token}
# Change the current working directory to the repository root
os.chdir(REPO_NAME)
print(f"✅ Repository cloned. Current directory: {os.getcwd()}")

# --- 5. Configure Git for commits ---
print("--- Configuring Git user... ---")
!git config user.name "{GIT_USER_NAME}"
!git config user.email "{GIT_USER_EMAIL}"
print("✅ Git user configured.")

# --- 6. Mount Google Drive and verify source folder ---
print("--- Mounting Google Drive... ---")
drive.mount('/content/drive', force_remount=True)
drive_source_path = f"/content/drive/MyDrive/{DRIVE_PROJECT_FOLDER}"
if not os.path.exists(drive_source_path):
    raise FileNotFoundError(f"The source folder '{DRIVE_PROJECT_FOLDER}' was not found in your Google Drive's 'My Drive'. Please create it.")
print(f"✅ Google Drive mounted and source folder verified: {drive_source_path}")

print("\n🚀 Environment setup complete! Ready to build.")
```

---

### **Cell 2: Build, Test, and Deploy from Source**

This cell executes the core workflow. It copies your clean source notebooks from Google Drive into the project, runs the `nbdev` toolchain to build the library and run tests, and then commits and pushes the final, clean state back to your GitHub repository.

```python
# @title Cell 2: Build, Test, and Deploy from Source
import os
import json

# --- 1. Define paths and source notebooks ---
# This list is the "table of contents" for your project.
# To add a new module, simply add a new line here.
DRIVE_SOURCE_PATH = f"/content/drive/MyDrive/{DRIVE_PROJECT_FOLDER}"
SOURCE_NOTEBOOKS = [
    # (Source Filename on Drive, Destination Path in Repo)
    ("00_index.ipynb", "nbs/00_index.ipynb"),
    ("01_core_data_processing.ipynb", "nbs/01_core_data_processing.ipynb"),
    # Add your next notebook here when it's ready:
    # ("02_core_model_architecture.ipynb", "nbs/02_core_model_architecture.ipynb"),
]
COMMIT_MESSAGE = "build: Update library from source notebooks"

# --- 2. Copy source notebooks from Drive into the project ---
print("--- [Step 1/4] Copying source notebooks into project... ---")
for src_name, dest_path in SOURCE_NOTEBOOKS:
    full_src_path = os.path.join(DRIVE_SOURCE_PATH, src_name)
    if not os.path.exists(full_src_path):
        print(f"⚠️ WARNING: Source notebook '{src_name}' not found in Drive. Skipping.")
        continue
    !cp -p "{full_src_path}" "{dest_path}"
    print(f"  - Copied '{src_name}' to '{dest_path}'")
print("✅ Source notebooks copied.")

# --- 3. Run the nbdev toolchain ---
print("\n--- [Step 2/4] Running nbdev_prepare to build library... ---")
!nbdev_prepare

print("\n--- [Step 3/4] Running nbdev_test to validate code... ---")
!nbdev_test

# --- 4. Commit and push to GitHub ---
print("\n--- [Step 4/4] Committing and pushing to GitHub... ---")
print("  - Staging all changes...")
!git add .
print(f"  - Committing with message: '{COMMIT_MESSAGE}'")
!git commit -m "{COMMIT_MESSAGE}"
print("  - Pushing to remote 'origin'...")
# The remote was cloned with the token, so push will be authenticated
!git push origin main
print("\n✅ Workflow complete! Project updated on GitHub.")
```
---

**You will use the Controller notebook almost *every single time* you work on the project.**

It's a common misconception to think of it as a one-time initialization script. Instead, you should think of it as your **"Session Starter"** or **"Build & Deploy Script"**.

### Why the Controller is Used Every Time

The key reason is the nature of Google Colab: **Colab environments are ephemeral (temporary).**

1.  **Clean Slate:** When you start a new Colab session (or if your old one times out), you are given a fresh, blank virtual machine.
    *   Your cloned GitHub repository is **gone**.
    *   The tools you installed (`nbdev`, etc.) are **gone**.
    *   The `nbs/` directory with your notebooks is **gone**.

2.  **The Controller's Job:** The Controller notebook's primary purpose is to **recreate your entire development environment from scratch** in a perfectly clean, reproducible way, every single time.

### The Workflow for a Subsequent Push (Day 2, Day 3, etc.)

Let's say you've finished for the day and successfully pushed your `01_core_data_processing.ipynb`. The next day, you want to add the model architecture.

Here is your new workflow:

**Step 1: Do Your Development Work on Google Drive (The "Source")**

*   You do **NOT** open the Controller notebook first.
*   You go to your Google Drive folder (`adia_project_source/`).
*   You create a new notebook there named `02_core_model_architecture.ipynb`.
*   You write your `TransformerEncoder` class, its tests, and documentation inside this new notebook. You save it.

**Step 2: Open and Run the Controller Notebook in Colab (The "Build Script")**

*   Now that your new source code is ready, you open your **Controller notebook** in Colab.
*   **Update the Controller:** You make one tiny change to Cell 2's `SOURCE_NOTEBOOKS` list to make it aware of your new file:
    ```python
    SOURCE_NOTEBOOKS = [
        ("00_index.ipynb", "nbs/00_index.ipynb"),
        ("01_core_data_processing.ipynb", "nbs/01_core_data_processing.ipynb"),
        # You add this new line:
        ("02_core_model_architecture.ipynb", "nbs/02_core_model_architecture.ipynb"),
    ]
    COMMIT_MESSAGE = "feat(core): Add model architecture components"
    ```
*   **Run Cell 1 (Environment Setup):** This will re-clone your repo from GitHub (with yesterday's changes), reinstall `nbdev`, and mount your Drive.
*   **Run Cell 2 (Build, Test, Deploy):** This will now:
    1.  Copy *both* `01_...` and your new `02_...` notebooks from Drive into the fresh `nbs/` directory.
    2.  Run `nbdev_prepare`, which will now generate **two** Python files: `core_data_processing.py` and `core_model_architecture.py`.
    3.  Run `nbdev_test` on **both** notebooks.
    4.  Commit and push the new state of the project to GitHub.

### Summary: The Role of Each Notebook Type

| Notebook Type | Role | When to Edit It | When to Run It in Colab |
| :--- | :--- | :--- | :--- |
| **Controller Notebook** | **Session Manager & Build Script.** Creates the environment and orchestrates the build. | Rarely. Only to add a new source file to the list or change the commit message. | **Every time** you start a new Colab session to build and push your project. |
| **Source Notebooks** (e.g., `01_...ipynb`) | **Code, Docs, & Tests.** This is where you write your library. | **Constantly.** This is your primary workspace on Google Drive. | You **never** run this notebook in the sense of building the project. You run its cells individually on Drive to test your logic as you develop. |

This workflow might seem repetitive at first, but it is incredibly robust. It guarantees that every single build is clean and reproducible, which is a cornerstone of professional software development.


