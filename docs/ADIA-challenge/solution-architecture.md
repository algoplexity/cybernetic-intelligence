
---

## **Solution Architecture: MDL Structural Break Detector**

### **1. System Purpose & Vision**

The **MDL Structural Break Detector** is a software system designed to win the ADIA Lab Structural Break Challenge. It operates on a unified hypothesis that structural breaks in a time series can be detected by identifying fundamental changes in the series' underlying causal dynamics.

The system's core strategy is to:
1.  **Pre-train** a sophisticated Transformer-based autoencoder on a curated dataset of complex, rule-based systems (Elementary Cellular Automata) to make it an expert at recognizing and compressing abstract dynamics.
2.  **Formalize** this training through the lens of the Minimum Description Length (MDL) principle, where the model learns to find the most compact "dynamical fingerprint" for any given sequence.
3.  **Deploy** this trained model during inference to analyze segments of a real-world time series, transform them into a comparable symbolic format, and calculate a "break score" based on the distance between their respective dynamical fingerprints.

---

### **2. C4 View: System Context (Level 1)**

*Shows how our system fits into its environment.*

**Diagram:**
```
     +-----------------+
     |  Quantitative   |
     |   Researcher    |
     |     (Person)    |
     +-----------------+
            |
            | Initiates Training & Inference
            v
+------------------------------------------------+
|                                                |
|      MDL Structural Break Detector             |
|                (Our Software System)           |
|                                                |
|   Analyzes time series using a pre-trained     |
|   dynamical model to predict the probability   |
|   of a structural break.                       |
|                                                |
+------------------------------------------------+
     ^      |
     |      |
Provides  |      | Submits
Train/Test|      | Predictions
Data      |      |
     |      v
+-----------------------------+
|                             |
|    ADIA Challenge Platform  |
|      (External System)      |
|                             |
+-----------------------------+
```

---

### **3. C4 View: Container Diagram (Level 2)**

*Shows the major, high-level, independently runnable parts of our system.*

**Diagram:**
```
+-----------------------------------------------------------------------------------+
|                                                                                   |
| System Boundary: MDL Structural Break Detector                                    |
|                                                                                   |
|    +-----------------------------+       +------------------------------------+   |
|    |   Pre-Training Pipeline     |------>|           Model Store              |   |
|    |      (Python Script)        |       | (File System: .pth, .joblib files) |<--+
|    |      (Implements train())   |       +------------------------------------+   |
|    +-------------^---------------+             ^                                  |
|                  |                             | Reads Trained Encoder            |
|       Writes     |                             |                                  |
|       /Reads     v                             |                                  |
|    +-----------------------------+       +-----+-----------------------+          |
|    |  Synthetic ECA Dataset      |       |      Inference Pipeline     |--------->|
|    | (File System: .pt file)     |       |       (Implements infer())  |          |
|    +-----------------------------+       +-------------^---------------+          |
|                                                        | Reads Test Data          |
|                                                        |                          |
+--------------------------------------------------------+--------------------------+
                                                         |
                                                         v
                                               +-------------------+
                                               | ADIA Data Store   |
                                               | (data/*.parquet)  |
                                               +-------------------+
```

---

### **4. C4 View: Component Diagram (Level 3)**

*Zooms into each container to show its logical modules and their responsibilities.*

#### **4.1. Pre-Training Pipeline Components**

**Diagram:**
```
+-------------------------------------------------------------------------+
|                                                                         |
| Container: Pre-Training Pipeline                                        |
|                                                                         |
|    +---------------------+    Requests Data    +----------------------+ |
|    |                     |-------------------->|                      | |
|    |     MDLTrainer      |<--------------------|    ECADataGenerator  | |
|    | (Manages training   |    Provides Batch   |  (Creates synthetic  | |
|    |  loop, dual loss)   |                     |     ECA data)        | |
|    +----------+----------+                     +----------------------+ |
|               |                                                         |
|    Passes Data, | Computes Loss                                           |
|    Returns      v                                                         |
|    Logits/Recons|                                                         |
|    +--------------------------+  Provides   +--------------------------+ |
|    |   DynamicalAutoencoder   |---Trained-->|       EncoderSaver       | |
|    | (Encoder, Decoder,       |   Model     | (Extracts and saves just | |
|    |  Classification Head)    |             |    the encoder state)    | |
|    +--------------------------+             +-------------+------------+ |
|                                                           |             |
|                                                           | Writes .pth |
|                                                           v             |
+-------------------------------------------------------------------------+
                                                            |
                                                 To: Model Store (External)
```

#### **4.2. Inference Pipeline Components**

**Diagram:**
```
+---------------------------------------------------------------------------------+
|                                                                                 |
| Container: Inference Pipeline                                                   |
|                                                                                 |
|    +--------------------+    Loads State    +--------------------------------+  |
|    |   EncoderLoader    |<-----------------|     Model Store (External)       |  |
|    |  (Loads .pth file  |                  +--------------------------------+  |
|    |   into nn.Module)  |                                                    |  |
|    +---------+----------+                                                    |  |
|              |                                                               |  |
|   Provides   | Trained Encoder                                               |  |
|   Loaded     v                                                               |  |
|   Model  +----------------------+  Feeds Symbol Seq  +----------------------+  |
|          |                      |------------------->|                      |  |
|          |     Fingerprinter    |                    |    SeriesProcessor   |  |
|          |  (Generates stable   |                    | (Applies full data   |  |
|          |    fingerprint for a |<-------------------| transformation pipe) |  |
|          |     time series)     |  Returns Symbol Seq+---------^-------------+  |
|          +-----------+----------+                    |         |               |
|                      |                               | Reads Raw | Series Data     |
|   Returns            | Fingerprint                   |         v               |
|   Before/After       v                               | +-------------------+ |
|          +----------------------+                    | | ADIA Data Store   | |
|          |   BreakScoreCalculator |                    |  (data/*.parquet) | |
|          |   (Computes distance   |                    +-------------------+ |
|          |  between fingerprints) |                                          |
|          +-----------+----------+                                          |
|                      |                                                     |
|                      | Final [0,1] Score                                   |
|                      v                                                     |
+---------------------------------------------------------------------------------+
```

---

### **5. C4 View: Code (Level 4)**

*Defines the key classes and their public interfaces (the "contracts") for implementation and testing.*

#### **File: `model_architecture.py`**
```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, latent_dim: int): ...
    def forward(self, sequence_batch: torch.Tensor) -> torch.Tensor: ...

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim: int, model_dim: int, num_heads: int, num_layers: int, output_dim: int): ...
    def forward(self, fingerprint_batch: torch.Tensor, seq_len: int) -> torch.Tensor: ...

class DynamicalAutoencoder(nn.Module):
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, num_classes: int, latent_dim: int): ...
    def forward(self, sequence_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def encode(self, sequence_batch: torch.Tensor) -> torch.Tensor: ...
```

#### **File: `data_processing.py`**
```python
import numpy as np
import pandas as pd

class ECADataGenerator:
    def __init__(self, config: dict): ...
    def generate_training_data(self) -> tuple[np.ndarray, np.ndarray]: ...

class PermutationSymbolizer:
    def __init__(self, embedding_dim: int): ...
    def transform_series(self, series: pd.Series) -> np.ndarray: ...

class SeriesProcessor:
    def __init__(self, symbolizer: PermutationSymbolizer, sequence_length: int): ...
    def process(self, series: pd.Series) -> torch.Tensor | None: ...
```

#### **File: `pipelines.py`**
```python
import torch
import pandas as pd
from model_architecture import DynamicalAutoencoder, TransformerEncoder
from data_processing import ECADataGenerator, SeriesProcessor

class MDLTrainer:
    def __init__(self, model: DynamicalAutoencoder, data_generator: ECADataGenerator, config: dict): ...
    def train(self) -> nn.Module: ...
    def save_encoder(self, encoder: nn.Module, path: str): ...

class InferencePipeline:
    def __init__(self, encoder: TransformerEncoder, series_processor: SeriesProcessor): ...
    def calculate_break_score(self, series_before: pd.Series, series_after: pd.Series) -> float: ...
```

This complete document provides a top-to-bottom, coherent view of the entire system. It establishes a clear plan, defines responsibilities, and sets up explicit contracts between all components, laying a solid foundation for successful implementation.
