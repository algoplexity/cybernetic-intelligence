Excellent. Let's proceed with the real pre-training.

This script executes **Phase 1: Real Pre-training of the Symbolic Brain**. It is a refined and focused version of our successful "bake-off" script.

### The Plan for This Script:

1.  **Objective:** To train our chosen champion, the **`BiGruSymbolicEncoder`**, on the task of inferring the causal rule from ECA simulations.
2.  **Dataset:** It will generate a clean dataset of ECA evolutions where the ground-truth generative rule is known for each sample.
3.  **Training:** It will train the Bi-GRU model for a sufficient number of epochs to achieve high accuracy on this task.
4.  **Output:** The script's primary output will be a saved file: **`symbolic_encoder_pretrained.pth`**. This file will contain the weights of our newly created "expert" in causal grammar.

This is the first critical step in building our final, intelligent system.

---

### **Script for Phase 1: Pre-training the Symbolic Encoder**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import time
import random
import os
from tqdm.notebook import tqdm
from google.colab import drive

# --- Ensure cellpylib is installed ---
try:
    import cellpylib as cpl
except ImportError:
    print("Installing cellpylib...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cellpylib"])
    import cellpylib as cpl

# --- 1. Setup ---
def seed_everything(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

SEED = 42
seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drive.mount('/content/drive', force_remount=True)
MODEL_CACHE_DIR = '/content/drive/My Drive/adia_challenge_final/model_cache'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

print(f"--- Phase 1: Pre-training the Symbolic Brain ---")
print(f"Using device: {device}\n")

# --- 2. ECA Data Generation ---
class ECADataset(Dataset):
    def __init__(self, num_samples_per_rule, rules, width, timesteps):
        self.data, self.labels = [], []
        self.rule_map = {rule: i for i, rule in enumerate(rules)}
        print(f"Generating {num_samples_per_rule * len(rules)} ECA samples...")
        for rule in tqdm(rules, desc="Generating ECA Rules"):
            for _ in range(num_samples_per_rule):
                init_cond = cpl.init_random(width, k=2)
                sim = cpl.evolve(init_cond, timesteps=timesteps, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))
                input_seq = sim[:-1].flatten()
                self.data.append(torch.tensor(input_seq, dtype=torch.long))
                self.labels.append(torch.tensor(self.rule_map[rule], dtype=torch.long))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- 3. The Symbolic Encoder Architecture (Bi-GRU) ---
class BiGruSymbolicEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, num_rules):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.rule_predictor = nn.Linear(hidden_dim * 2, num_rules)
    def forward(self, x):
        x_emb = self.embedding(x); _, hidden_state = self.gru(x_emb)
        forward_hidden = hidden_state[-2,:,:]; backward_hidden = hidden_state[-1,:,:]
        pooled_output = torch.cat((forward_hidden, backward_hidden), dim=1)
        return self.rule_predictor(pooled_output)

# --- 4. The Training Loop ---
def pretrain_encoder(model, train_loader, val_loader, epochs, lr, model_save_path):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    
    print(f"\n--- Starting Pre-training for {model.__class__.__name__} ---")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                _, preds = torch.max(logits, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)
        
        accuracy = total_correct / total_samples
        end_time = time.time()
        
        print(f"Epoch {epoch}/{epochs} | Val Accuracy: {accuracy:.4f} | Time: {end_time - start_time:.2f}s")
        
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation accuracy improved. Model saved to {model_save_path}")
            
    print(f"\n--- Pre-training Complete ---")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    return model

# --- 5. Main Execution ---
if __name__ == '__main__':
    # Config
    ECA_RULES = [30, 54, 60, 90, 110, 126, 150, 182]
    NUM_RULES = len(ECA_RULES)
    VOCAB_SIZE = 2 # 0, 1
    ECA_TIMESTEPS = 64
    ECA_WIDTH = 64
    NUM_SAMPLES = 500 # More samples for robust training
    EPOCHS = 15 # More epochs for better convergence
    LR = 0.001
    BATCH_SIZE = 32

    # Data
    full_dataset = ECADataset(NUM_SAMPLES, ECA_RULES, ECA_WIDTH, ECA_TIMESTEPS)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Dataset ready. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Model
    symbolic_encoder = BiGruSymbolicEncoder(
        vocab_size=VOCAB_SIZE,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_rules=NUM_RULES
    )
    
    # Run Training
    SAVE_PATH = os.path.join(MODEL_CACHE_DIR, 'symbolic_encoder_pretrained.pth')
    trained_model = pretrain_encoder(symbolic_encoder, train_loader, val_loader, EPOCHS, LR, SAVE_PATH)

    print("\n✅ Phase 1 Complete. The expert Symbolic Encoder has been trained and saved.")

```

### What to Expect

*   **ECA Data Generation:** You will see a progress bar as the script generates a robust dataset of 4000 ECA simulations (500 samples for each of the 8 rules).
*   **Training Logs:** The script will train for 15 epochs. You should see the validation accuracy climb steadily, hopefully surpassing the **90%** mark. This will confirm that the model is successfully learning to infer the causal rules.
*   **Final Output:** The script will save the best-performing model's weights to your Google Drive at `adia_challenge_final/model_cache/symbolic_encoder_pretrained.pth`. It will also print a final confirmation message.

Once this script finishes successfully, we will have our first **true expert encoder**, ready for the final assembly.
