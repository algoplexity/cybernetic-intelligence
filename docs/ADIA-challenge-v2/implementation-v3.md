

```python
# ==============================================================================
# SECTION 0: SETUP, IMPORTS, AND GLOBAL CONFIGURATION
# ==============================================================================

# --- Standard and Third-Party Imports ---
import os
import typing
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import joblib
from itertools import permutations
from pathlib import Path
import math
import hashlib
import random
from tqdm.notebook import tqdm

# --- External Dependencies (ensure these are installed) ---
try:
    import cellpylib as cpl
except ImportError:
    print("Installing cellpylib...")
    !pip install cellpylib -q
    import cellpylib as cpl

warnings.filterwarnings('ignore')

# --- Global Configuration (V3 - Incorporating research insights) ---
@dataclass
class Config:
    """
    Central configuration object for the entire pipeline.
    Combines hyperparameters and strategic choices derived from research.
    """
    # Reproducibility
    SEED: int = 42

    # Data Processing (from v2 implementation & Traversaro et al.)
    PERMUTATION_EMBEDDING_DIM: int = 4  # d for permutation patterns (d!)
    SERIES_PROCESSOR_SEQUENCE_LENGTH: int = 128 # Length of symbolic sequences fed to the model

    # ECA Pre-training (from v2 implementation & "Edge of Chaos" paper)
    # STRATEGIC CHOICE: Curated list of Class III & IV rules for maximum complexity.
    ECA_RULES_TO_USE: typing.List[int] = [
        22, 30, 45, 54, 60, 75, 82, 86, 89, 90, 105, 106, 110,
        122, 126, 135, 146, 149, 150, 153, 154, 161, 165, 169, 182
    ]
    ECA_WIDTH: int = 128
    ECA_TIMESTEPS: int = 256
    ECA_N_SAMPLES_PER_RULE: int = 100 # Generate this many examples for each rule

    # Model Architecture (from v2 AU-Net implementation)
    MODEL_DIMENSIONS: typing.List[int] = [128, 256]
    MODEL_LAYERS_PER_BLOCK: typing.List[int] = [2, 2]
    # Sequence length shrinks at each U-Net stage
    MODEL_MAX_SEQLENS: typing.List[int] = [ECA_TIMESTEPS, 64]
    MODEL_N_HEADS: int = 4
    MODEL_DROPOUT: float = 0.1

    # Training (MDL Principle)
    PRETRAIN_EPOCHS: int = 5
    FINETUNE_EPOCHS: int = 10
    BATCH_SIZE: int = 32
    PRETRAIN_LR: float = 1e-4
    FINETUNE_LR: float = 5e-5
    # Alpha/Beta for MDL two-part code loss: L = alpha*L(data|model) + beta*L(model)
    MDL_ALPHA_LOSS: float = 1.0 # Reconstruction loss weight
    MDL_BETA_LOSS: float = 0.3  # Rule classification loss weight

    # Paths
    MODEL_DIR: Path = Path("./adia_model_store_v3")

# --- Reproducibility Seeder ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Instantiate Config and Setup Environment ---
config = Config()
seed_everything(config.SEED)
config.MODEL_DIR.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Configuration loaded. Using device: {device}")
```

---
```markdown
# ==============================================================================
# SECTION 1: THE BLUEPRINT (LIVING ARCHITECTURE DOCUMENT)
# ==============================================================================

## Solution Architecture v6 (Final Blueprint)

This document outlines the complete and definitive C4 architecture for our solution. This version (v6) is a self-contained blueprint incorporating all design decisions, refined component contracts, and key insights from our research analysis. It serves as the final, unambiguous guide for implementation.

### **Core Principles of the V6 Architecture**

*   **Two-Stage Training:** A pre-training stage on synthetic data to learn general dynamics, followed by a fine-tuning stage on real data to specialize for the task.
*   **Hierarchical U-Net Model:** The core model is a U-Net style autoencoder (inspired by AU-Net) to capture multi-scale patterns in sequences.
*   **MDL Pre-training Objective:** The pre-training phase uses a dual loss (Reconstruction + Rule Classification) inspired by the Minimum Description Length principle to create rich, informative "dynamical fingerprints."
*   **"Edge of Chaos" Data Curation:** The synthetic pre-training data is curated to be maximally complex and information-rich (inspired by Zhang et al.).
*   **Robust Symbolic Representation:** Real-world time series are converted into a discrete symbolic sequence using a robust Permutation Symbolizer that correctly handles ties (inspired by Bandt/Pompe and Traversaro et al.).
*   **Domain-Adapted Inputs:** The model uses a shared core architecture but employs two distinct input "heads": one for the raw binary data from ECAs and a separate `nn.Embedding` head for the integer symbols from real-world data.

*(The full rationale table and C4 diagrams would be pasted here)*
```

---
```python
# ==============================================================================
# SECTION 2: CORE LIBRARY IMPLEMENTATION (ADOPTED FROM v2-GOLDEN-SOURCE)
# ==============================================================================

# ------------------------------------------------------------------------------
# MODULE 2.1: core_library/data_processing.py
# ------------------------------------------------------------------------------

class PermutationSymbolizer:
    """
    Converts a numeric vector into a discrete ordinal pattern.
    NOTE: Uses a deterministic, content-based seed for randomized tie-breaking,
    as justified by Traversaro et al. for robust, pure-function behavior.
    """
    def __init__(self, embedding_dim: int):
        if not isinstance(embedding_dim, int) or embedding_dim <= 1:
            raise ValueError("embedding_dim must be an integer greater than 1.")
        self.embedding_dim = embedding_dim
        self.permutations = {p: i for i, p in enumerate(permutations(range(embedding_dim)))}
        self.vocab_size = len(self.permutations)

    def symbolize_vector(self, vector: np.ndarray) -> int:
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"Input vector must have shape ({self.embedding_dim},), but got {vector.shape}")

        hasher = hashlib.sha256(vector.tobytes())
        seed = int.from_bytes(hasher.digest(), 'little') % (2**32 - 1)
        rng = np.random.default_rng(seed)
        noise = rng.uniform(low=-1e-12, high=1e-12, size=self.embedding_dim)
        perturbed_vector = vector + noise
        
        ordinal_pattern = tuple(np.argsort(perturbed_vector))
        return self.permutations[ordinal_pattern]

class SeriesProcessor:
    """
    Manages the full pipeline for real data: embedding, symbolization, and windowing.
    NOTE: Uses np.lib.stride_tricks for highly efficient windowing.
    """
    def __init__(self, symbolizer: PermutationSymbolizer, sequence_length: int):
        self.symbolizer = symbolizer
        self.sequence_length = sequence_length
        self.embedding_dim = symbolizer.embedding_dim

    def process(self, series: pd.Series) -> typing.List[torch.Tensor]:
        if len(series) < self.embedding_dim: return [] # Return empty list if too short
        
        values = series.values
        # 1. Create time-delayed vectors
        shape = (len(values) - self.embedding_dim + 1, self.embedding_dim)
        strides = (values.strides[0], values.strides[0])
        embedded_vectors = np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)
        
        # 2. Symbolize each vector
        symbols = [self.symbolizer.symbolize_vector(v) for v in embedded_vectors]
        
        if len(symbols) < self.sequence_length: return []

        # 3. Create overlapping sequences of symbols
        symbols_arr = np.array(symbols)
        shape = (len(symbols) - self.sequence_length + 1, self.sequence_length)
        strides = (symbols_arr.strides[0], symbols_arr.strides[0])
        sequences = np.lib.stride_tricks.as_strided(symbols_arr, shape=shape, strides=strides)
        
        return [torch.from_numpy(seq.copy()).long() for seq in sequences]

class ECADataGenerator:
    """
    Synthetic Data Factory using cellpylib.
    NOTE: The `rules_to_use` param passed to this class during instantiation
    is strategically curated based on the "Edge of Chaos" paper.
    """
    def __init__(self, rules_to_use: list, n_samples_per_rule: int, width: int, timesteps: int, seed: int):
        self.rules_to_use = rules_to_use
        self.n_reps = n_samples_per_rule
        self.width, self.steps = width, timesteps
        self.rng = np.random.default_rng(seed)
        self.rule_to_label = {rule: i for i, rule in enumerate(rules_to_use)}
        self.num_classes = len(rules_to_use)

    def generate_training_data(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        all_sequences, all_labels = [], []
        print(f"Generating synthetic data for {len(self.rules_to_use)} 'Edge of Chaos' rules...")
        progress_bar = tqdm(self.rules_to_use, desc="Generating ECA Data")
        for rule in progress_bar:
            for _ in range(self.n_reps):
                initial_state = self.rng.choice([0, 1], size=(1, self.width))
                ca = cpl.evolve2d(initial_state, timesteps=self.steps, 
                                  apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))
                all_sequences.append(ca.astype(np.float32))
                all_labels.append(self.rule_to_label[rule])

        print(f"Generated {len(all_sequences)} total ECA simulations.")
        return np.array(all_sequences), np.array(all_labels)

# ------------------------------------------------------------------------------
# MODULE 2.2: core_library/model_architecture.py
# ------------------------------------------------------------------------------

# --- Model dataclasses for configuration ---
@dataclass
class CausalTransformerArgs:
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

@dataclass
class HierarchicalArgs:
    input_dim: int
    vocab_size: int
    num_classes: int
    dimensions: list
    layers: list
    max_seqlens: list
    n_heads: int
    dropout: float
    @property
    def d_ff_multiplier(self) -> int: return 4
    @property
    def n_stages(self) -> int: return len(self.dimensions)
    @property
    def latent_dim(self) -> int: return self.dimensions[-1]

# --- Core Model Components (from v2, implementing AU-Net) ---
class CausalTransformer(nn.Module):
    def __init__(self, args: CausalTransformerArgs):
        super().__init__()
        self.attn = nn.MultiheadAttention(args.d_model, args.n_heads, dropout=args.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(args.d_model, args.d_ff), nn.GELU(), nn.Linear(args.d_ff, args.d_model))
        self.norm1, self.norm2 = nn.LayerNorm(args.d_model), nn.LayerNorm(args.d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        causal_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), 1).bool()
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=causal_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class SimpleTransition(nn.Module):
    def __init__(self, d_in, d_out, factor):
        super().__init__()
        self.down_proj, self.up_proj, self.factor = nn.Linear(d_in, d_out), nn.Linear(d_in, d_out * factor), factor
    def down(self, x): return self.down_proj(x[:, ::self.factor, :])
    def up(self, x):
        x = self.up_proj(x)
        B, S, D = x.shape
        return x.view(B, S, self.factor, D // self.factor).permute(0, 1, 2, 3).reshape(B, S * self.factor, -1)

class HierarchicalDynamicalEncoder(nn.Module):
    def __init__(self, args: HierarchicalArgs):
        super().__init__()
        self.in_proj = nn.Linear(args.input_dim, args.dimensions[0])
        self.encoders, self.transitions = nn.ModuleList(), nn.ModuleList()
        for i in range(args.n_stages - 1):
            encoder_args = CausalTransformerArgs(args.dimensions[i], args.n_heads, args.dimensions[i] * args.d_ff_multiplier, args.dropout)
            self.encoders.append(nn.Sequential(*[CausalTransformer(encoder_args) for _ in range(args.layers[i])]))
            factor = args.max_seqlens[i] // args.max_seqlens[i+1]
            self.transitions.append(SimpleTransition(args.dimensions[i], args.dimensions[i+1], factor))
        trunk_args = CausalTransformerArgs(args.dimensions[-1], args.n_heads, args.dimensions[-1] * args.d_ff_multiplier, args.dropout)
        self.trunk = nn.Sequential(*[CausalTransformer(trunk_args) for _ in range(args.layers[-1])])
    def forward(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, list]:
        residuals, x = [], self.in_proj(x)
        for encoder, trans in zip(self.encoders, self.transitions):
            x = encoder(x)
            residuals.append(x)
            x = trans.down(x)
        return self.trunk(x), residuals

class HierarchicalDynamicalDecoder(nn.Module):
    def __init__(self, args: HierarchicalArgs, transitions: nn.ModuleList):
        super().__init__()
        self.transitions, self.decoders = transitions, nn.ModuleList()
        for i in range(args.n_stages - 1):
            decoder_args = CausalTransformerArgs(args.dimensions[i], args.n_heads, args.dimensions[i] * args.d_ff_multiplier, args.dropout)
            self.decoders.append(nn.Sequential(*[CausalTransformer(decoder_args) for _ in range(args.layers[i])]))
        self.out_proj = nn.Linear(args.dimensions[0], args.input_dim)
    def forward(self, fingerprint_seq: torch.Tensor, residuals: list) -> torch.Tensor:
        x = fingerprint_seq
        for i in range(len(self.decoders) - 1, -1, -1):
            x = self.transitions[i].up(x)
            x += residuals.pop()
            x = self.decoders[i](x)
        return self.out_proj(x)

# --- Top-Level Models ---
class MDL_AU_Net_Autoencoder(nn.Module):
    def __init__(self, args: HierarchicalArgs):
        super().__init__()
        self.args = args
        self.encoder = HierarchicalDynamicalEncoder(args)
        self.decoder = HierarchicalDynamicalDecoder(args, self.encoder.transitions)
        self.classification_head = nn.Linear(args.latent_dim, args.num_classes)
        self.embedding = nn.Embedding(args.vocab_size, args.dimensions[0])

    def forward_pretrain(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        fingerprint_seq, residuals = self.encoder(x)
        reconstruction = self.decoder(fingerprint_seq, residuals)
        pooled_fingerprint = fingerprint_seq.mean(dim=1)
        logits = self.classification_head(pooled_fingerprint)
        return reconstruction, logits
    
    def forward_finetune(self, x: torch.Tensor) -> torch.Tensor:
        x_embedded = self.embedding(x)
        # For fine-tuning, we only need the encoder part.
        # We also don't project from input_dim, we use the embedding directly.
        residuals, x = [], x_embedded
        for encoder, trans in zip(self.encoder.encoders, self.encoder.transitions):
            x = encoder(x)
            residuals.append(x)
            x = trans.down(x)
        return self.encoder.trunk(x)


class StructuralBreakClassifier(nn.Module):
    def __init__(self, autoencoder: MDL_AU_Net_Autoencoder, freeze_encoder: bool = True):
        super().__init__()
        self.autoencoder = autoencoder
        self.args = autoencoder.args
        if freeze_encoder:
            # Freeze the entire pre-trained body, only train embedding and new head
            for param in self.autoencoder.encoder.parameters():
                param.requires_grad = False
            for param in self.autoencoder.decoder.parameters():
                param.requires_grad = False
            for param in self.autoencoder.classification_head.parameters():
                param.requires_grad = False

        # NOTE: The sophisticated fingerprint comparison from v2
        classifier_input_dim = self.args.latent_dim * 3
        self.classifier_head = nn.Sequential(
            nn.Linear(classifier_input_dim, self.args.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.args.latent_dim, 1)
        )
    
    def _get_aggregated_fingerprint(self, seqs: list) -> torch.Tensor:
        # NOTE: The robust aggregation method from v2
        if not seqs: return torch.zeros(self.args.latent_dim, device=device)
        batch = torch.stack(seqs).to(device)
        fingerprint_seqs = self.autoencoder.forward_finetune(batch)
        pooled_fingerprints = fingerprint_seqs.mean(dim=1) # Pool along sequence dim
        return pooled_fingerprints.mean(dim=0) # Pool along batch dim

    def forward(self, before_seqs: list, after_seqs: list) -> torch.Tensor:
        avg_before = self._get_aggregated_fingerprint(before_seqs)
        avg_after = self._get_aggregated_fingerprint(after_seqs)
        # NOTE: Using the difference vector as an explicit feature
        combined = torch.cat([avg_before, avg_after, torch.abs(avg_before - avg_after)], dim=0)
        return self.classifier_head(combined.unsqueeze(0))
```

---
```python
# ==============================================================================
# SECTION 3: TRAINING & INFERENCE PIPELINE LOGIC
# ==============================================================================

class MDLPreTrainer:
    """Orchestrates the pre-training loop for the MDL Autoencoder."""
    def __init__(self, model: MDL_AU_Net_Autoencoder, cfg: Config):
        self.model = model.to(device)
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.PRETRAIN_LR)
        # Using BCE for reconstruction of binary ECA data
        self.recon_criterion = nn.BCEWithLogitsLoss()
        self.class_criterion = nn.CrossEntropyLoss()

    def pretrain(self, data_generator: ECADataGenerator):
        print("\n--- Starting Stage 1: MDL Pre-training on 'Edge of Chaos' ECA data ---")
        self.model.train()
        X, y = data_generator.generate_training_data()
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                            batch_size=self.cfg.BATCH_SIZE, shuffle=True)

        for epoch in range(self.cfg.PRETRAIN_EPOCHS):
            progress_bar = tqdm(loader, desc=f"Pre-train Epoch {epoch+1}/{self.cfg.PRETRAIN_EPOCHS}")
            for sequences, labels in progress_bar:
                sequences, labels = sequences.to(device), labels.to(device)
                self.optimizer.zero_grad()
                recon, logits = self.model.forward_pretrain(sequences)
                
                # NOTE: Applying the MDL two-part code loss from the config
                recon_loss = self.cfg.MDL_ALPHA_LOSS * self.recon_criterion(recon, sequences)
                rule_loss = self.cfg.MDL_BETA_LOSS * self.class_criterion(logits, labels)
                loss = recon_loss + rule_loss
                
                loss.backward()
                self.optimizer.step()
                progress_bar.set_postfix({'loss': loss.item(), 'recon_L': recon_loss.item(), 'rule_L': rule_loss.item()})
        print("--- Pre-training Complete ---")

class BreakClassifierFinetuner:
    """Orchestrates the fine-tuning loop for the structural break classifier."""
    def __init__(self, model: StructuralBreakClassifier, cfg: Config):
        self.model = model.to(device)
        self.cfg = cfg
        # Only train the unfrozen parameters (embedding head and classifier head)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.FINETUNE_LR)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def finetune(self, X_train: pd.DataFrame, y_train: pd.Series, processor: SeriesProcessor):
        print("\n--- Starting Stage 2: Fine-tuning on Real Data ---")
        self.model.train()
        
        # Prepare data for fine-tuning
        processed_data = []
        for i, row in X_train.iterrows():
            break_point = row['period'] - 1
            before_seqs = processor.process(row['series'].iloc[:break_point])
            after_seqs = processor.process(row['series'].iloc[break_point:])
            if before_seqs and after_seqs: # Only use samples with valid data
                processed_data.append((before_seqs, after_seqs, y_train.iloc[i]))

        # TODO: A proper DataLoader would be more efficient for large datasets.
        # For simplicity in this notebook, we loop directly.
        for epoch in range(self.cfg.FINETUNE_EPOCHS):
            random.shuffle(processed_data)
            progress_bar = tqdm(processed_data, desc=f"Fine-tune Epoch {epoch+1}/{self.cfg.FINETUNE_EPOCHS}")
            total_loss = 0
            for before, after, label in progress_bar:
                self.optimizer.zero_grad()
                label_tensor = torch.tensor([float(label)], device=device)
                
                logit = self.model(before, after)
                loss = self.loss_fn(logit.squeeze(), label_tensor)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            print(f"Epoch {epoch+1} Average Loss: {total_loss / len(processed_data)}")
        print("--- Fine-tuning Complete ---")


class ArtifactHandler:
    """Saves and loads all necessary model artifacts and configuration."""
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

    def save(self, model: StructuralBreakClassifier, cfg: Config):
        print(f"Saving artifacts to {self.model_dir}...")
        # Save the three key components' state dicts
        torch.save(model.autoencoder.encoder.state_dict(), self.model_dir / "core_encoder.pth")
        torch.save(model.autoencoder.embedding.state_dict(), self.model_dir / "embedding_head.pth")
        torch.save(model.classifier_head.state_dict(), self.model_dir / "classifier_head.pth")
        # Save the configuration used to build the model
        joblib.dump(cfg, self.model_dir / "model_config.joblib")
        print("Artifacts saved.")

    def load(self) -> typing.Tuple[StructuralBreakClassifier, Config]:
        print(f"Loading artifacts from {self.model_dir}...")
        cfg = joblib.load(self.model_dir / "model_config.joblib")
        
        # 1. Rebuild the original autoencoder architecture
        symbolizer_for_vocab = PermutationSymbolizer(cfg.PERMUTATION_EMBEDDING_DIM)
        model_args = HierarchicalArgs(
            input_dim=cfg.ECA_WIDTH,
            vocab_size=symbolizer_for_vocab.vocab_size,
            num_classes=len(cfg.ECA_RULES_TO_USE),
            dimensions=cfg.MODEL_DIMENSIONS,
            layers=cfg.MODEL_LAYERS_PER_BLOCK,
            max_seqlens=cfg.MODEL_MAX_SEQLENS,
            n_heads=cfg.MODEL_N_HEADS,
            dropout=cfg.MODEL_DROPOUT
        )
        autoencoder = MDL_AU_Net_Autoencoder(model_args)
        
        # 2. Build the final classifier, but don't freeze for loading
        model = StructuralBreakClassifier(autoencoder, freeze_encoder=False)
        
        # 3. Load the saved weights into the components
        model.autoencoder.encoder.load_state_dict(torch.load(self.model_dir / "core_encoder.pth"))
        model.autoencoder.embedding.load_state_dict(torch.load(self.model_dir / "embedding_head.pth"))
        model.classifier_head.load_state_dict(torch.load(self.model_dir / "classifier_head.pth"))
        
        print("Artifacts loaded successfully.")
        return model.to(device).eval(), cfg
```

---
```python
# ==============================================================================
# SECTION 4: MAIN ENTRY POINTS FOR THE PLATFORM
# ==============================================================================

def train(X, y, model_dir):
    """
    Main training function for the ADIA platform.
    Orchestrates the full two-stage training and artifact saving.
    """
    print("="*60)
    print("                STARTING TRAINING PIPELINE v3")
    print("="*60)
    
    # --- Instantiate Components from Config ---
    global config # Use the global config object
    artifact_handler = ArtifactHandler(Path(model_dir))

    # Data components
    eca_generator = ECADataGenerator(
        rules_to_use=config.ECA_RULES_TO_USE,
        n_samples_per_rule=config.ECA_N_SAMPLES_PER_RULE,
        width=config.ECA_WIDTH,
        timesteps=config.ECA_TIMESTEPS,
        seed=config.SEED
    )
    symbolizer = PermutationSymbolizer(config.PERMUTATION_EMBEDDING_DIM)
    processor = SeriesProcessor(symbolizer, config.SERIES_PROCESSOR_SEQUENCE_LENGTH)
    
    # Model components
    model_args = HierarchicalArgs(
        input_dim=config.ECA_WIDTH,
        vocab_size=symbolizer.vocab_size,
        num_classes=eca_generator.num_classes,
        dimensions=config.MODEL_DIMENSIONS,
        layers=config.MODEL_LAYERS_PER_BLOCK,
        max_seqlens=config.MODEL_MAX_SEQLENS,
        n_heads=config.MODEL_N_HEADS,
        dropout=config.MODEL_DROPOUT
    )
    autoencoder = MDL_AU_Net_Autoencoder(model_args)

    # --- Stage 1: MDL Pre-training ---
    pre_trainer = MDLPreTrainer(autoencoder, config)
    pre_trainer.pretrain(eca_generator)
    
    # --- Stage 2: Fine-tuning ---
    classifier = StructuralBreakClassifier(autoencoder, freeze_encoder=True)
    finetuner = BreakClassifierFinetuner(classifier, config)
    finetuner.finetune(X, y, processor)
    
    # --- Stage 3: Save Artifacts ---
    artifact_handler.save(classifier, config)
    print("="*60)
    print("                TRAINING PIPELINE FINISHED")
    print("="*60)


def infer(X, model_dir):
    """
    Main inference function for the ADIA platform.
    Loads artifacts and yields prediction scores.
    """
    print("\n" + "="*60)
    print("               STARTING INFERENCE PIPELINE v3")
    print("="*60)
    
    # --- Load Model and Config ---
    artifact_handler = ArtifactHandler(Path(model_dir))
    model, loaded_config = artifact_handler.load()
    
    # --- Setup Inference Components ---
    symbolizer = PermutationSymbolizer(loaded_config.PERMUTATION_EMBEDDING_DIM)
    processor = SeriesProcessor(symbolizer, loaded_config.SERIES_PROCESSOR_SEQUENCE_LENGTH)

    # --- Process Data and Yield Scores ---
    for i, row in tqdm(X.iterrows(), total=len(X), desc="Inferring"):
        break_point = row['period'] - 1
        series_before = processor.process(row['series'].iloc[:break_point])
        series_after = processor.process(row['series'].iloc[break_point:])

        with torch.no_grad():
            logit = model(series_before, series_after)
        
        # Use sigmoid to convert logit to a probability-like score in [0, 1]
        score = torch.sigmoid(logit).item()
        yield score
    print("="*60)
    print("               INFERENCE PIPELINE FINISHED")
    print("="*60)
```

---
```python
# ==============================================================================
# SECTION 5: MOCK EXECUTION & INTEGRATION TEST
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("      RUNNING MOCK EXECUTION & INTEGRATION TEST")
    print("="*50)

    # --- Create Mock Data ---
    def create_mock_series(has_break=False, length=500, break_point=250):
        t = np.linspace(0, 10, length)
        # Add some noise to make it more realistic
        noise = np.random.randn(length) * 0.1
        series = pd.Series(np.sin(t * 2 * np.pi) + noise)
        if has_break:
            # More dramatic break: change of function, amplitude, and frequency
            series.iloc[break_point:] = pd.Series(np.cos(t[break_point:] * 5 * np.pi) * 1.5 + noise[break_point:])
        return series

    # Create a small but representative mock dataset
    mock_X_list = []
    mock_y_list = []
    for _ in range(2):
        mock_X_list.append({'series': create_mock_series(has_break=True), 'period': 251})
        mock_y_list.append(1)
        mock_X_list.append({'series': create_mock_series(has_break=False), 'period': 251})
        mock_y_list.append(0)

    mock_X = pd.DataFrame(mock_X_list)
    mock_y = pd.Series(mock_y_list)
    
    print("Mock data created.")
    print(f"X shape: {mock_X.shape}, y shape: {mock_y.shape}")

    # --- Run Mock Training ---
    # NOTE: This will take a while, especially the ECA data generation.
    # For a quick test, you can reduce ECA_N_SAMPLES_PER_RULE in the config.
    try:
        train(mock_X, mock_y, config.MODEL_DIR)
    except Exception as e:
        print(f"\nERROR during mock training: {e}")
        import traceback
        traceback.print_exc()

    # --- Run Mock Inference ---
    try:
        predictions = list(infer(mock_X, config.MODEL_DIR))
        print("\nMock Predictions (Scores):")
        for i, p in enumerate(predictions):
            print(f"  Sample {i} (True label: {mock_y[i]}): Predicted Score = {p:.4f}")

        # Basic sanity check
        break_scores = [p for i, p in enumerate(predictions) if mock_y[i] == 1]
        no_break_scores = [p for i, p in enumerate(predictions) if mock_y[i] == 0]
        if break_scores and no_break_scores:
            print(f"\nAvg score for 'break' samples: {np.mean(break_scores):.4f}")
            print(f"Avg score for 'no break' samples: {np.mean(no_break_scores):.4f}")
            if np.mean(break_scores) > np.mean(no_break_scores):
                print("✅ Sanity check passed: 'break' scores are higher than 'no break' scores.")
            else:
                print("❌ Sanity check failed: 'break' scores are NOT higher than 'no break' scores.")

    except Exception as e:
        print(f"\nERROR during mock inference: {e}")
        import traceback
        traceback.print_exc()
```
