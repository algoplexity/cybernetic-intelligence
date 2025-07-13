## Solution Architecture v3 (Final Blueprint)

This document outlines the complete and definitive C4 architecture for our solution. This version (v3) is a self-contained blueprint incorporating all design decisions, refined component contracts, and key insights from our research analysis. It serves as the final, unambiguous guide for implementation.

### **Core Principles of the V6 Architecture**

*   **Two-Stage Training:** A pre-training stage on synthetic data to learn general dynamics, followed by a fine-tuning stage on real data to specialize for the task.
*   **Hierarchical U-Net Model:** The core model is a U-Net style autoencoder (inspired by AU-Net) to capture multi-scale patterns in sequences.
*   **MDL Pre-training Objective:** The pre-training phase uses a dual loss (Reconstruction + Rule Classification) inspired by the Minimum Description Length principle to create rich, informative "dynamical fingerprints."
*   **"Edge of Chaos" Data Curation:** The synthetic pre-training data is curated to be maximally complex and information-rich (inspired by Zhang et al.).
*   **Robust Symbolic Representation:** Real-world time series are converted into a discrete symbolic sequence using a robust Permutation Symbolizer that correctly handles ties (inspired by Bandt/Pompe and Traversaro et al.).
*   **Domain-Adapted Inputs:** The model uses a shared core architecture but employs two distinct input "heads": one for the raw binary data from ECAs and a separate `nn.Embedding` head for the integer symbols from real-world data.

---
### **Design Rationale & Traceability**

This section maps the core research findings from our source papers to the specific architectural decisions made in this V6 blueprint.

| Research Finding & Source Paper | Corresponding Architectural Design Decision | Rationale |
| :--- | :--- | :--- |
| **"Intelligence emerges at the 'edge of chaos'."**<br>*(Zhang et al., "Intelligence at the Edge of Chaos")* | The **`ECADataGenerator`** will be configured to create a curated dataset heavily biased towards Wolfram Class IV (e.g., Rule 110) and complex Class III rules. | To build a model that learns generalizable, robust features of dynamics, we must pre-train it on a dataset that is maximally complex and information-rich, rather than a simple or purely random one. |
| **"Explicitly encouraging the model to infer the generating rule can enhance its ability to make longer-term predictions."**<br>*(Burtsev, "Learning ECA with Transformers")* | The **`MDL_Hierarchical_Autoencoder`** is trained with a **dual-loss objective** (reconstruction + rule classification), inspired by the Minimum Description Length (MDL) principle. | This forces the model's internal "fingerprint" to be not just descriptive of the data's appearance, but predictive of its underlying causal mechanism. This creates a richer, more abstract internal representation. |
| **A hierarchical, multi-scale architecture is highly effective for processing long sequences and learning semantics.**<br>*(AU-Net Paper, "From Bytes to Ideas")* | The core model architecture (`HierarchicalDynamicalEncoder` and `Decoder`) is a **U-Net**, not a standard "flat" Transformer. | The U-Net's contracting path naturally creates a multi-scale representation, forcing the bottleneck (the fingerprint) to capture long-range, high-level dynamics while allowing shallower layers to handle local patterns. This is ideal for identifying a shift in the overall "regime." |
| **Tied values in real-world time series should be treated as observational ambiguity, not new states.**<br>*(Traversaro et al., "Comparing methods for computing PE")* | The **`PermutationSymbolizer`** uses **randomized, deterministic tie-breaking**. | This prevents the model from learning "fictitious states" from data quantization artifacts and provides a more principled way to handle the conversion from continuous real-world data to a discrete symbolic representation. |
| **Cellular automata evolution is a discrete, rule-based process.**<br>*(All papers)* | The pre-training stage uses **two separate input "heads"**. The core model sees raw binary/float data from `ECADataGenerator`, while a separate `nn.Embedding` head is used to translate the integer symbols from `SeriesProcessor` into a compatible vector space for fine-tuning. | This solves the "domain gap." It ensures the model learns the fundamental physics of ECAs on data that looks exactly like an ECA. The embedding head then acts as a learned "translator" for the language of real-world market dynamics, allowing the core pre-trained knowledge to be applied effectively. |

---

### Level 1: System Context

This view shows our system in relation to the user and the external platform.

```mermaid
C4Context
    title System Context Diagram for ADIA Challenge

    System_Ext(adia_platform, "ADIA Platform Runner", "The environment that calls our train() and infer() functions.")
    
    System(our_system, "Our Solution (V6)", "A two-stage, hierarchical deep learning pipeline to detect structural breaks in time series.")

    Rel(adia_platform, our_system, "Calls train() and infer(), providing data.")
```

---

### Level 2: Container Diagram

This view breaks down our solution into its major structural blocks.

```mermaid
C4Container
    title System Container Diagram for ADIA Challenge

    System_Ext(adia_platform, "ADIA Platform Runner")
    
    Container_Boundary(our_system, "Our Solution") {
        Container(core_lib, "Core Services Library", "Python Module", "Contains all foundational, reusable code for data processing and model definitions.")
        Container(training_pipeline, "Training Pipeline Logic", "Python Module", "Orchestrates the two-stage training process.")
        Container(inference_pipeline, "Inference Pipeline Logic", "Python Module", "Orchestrates the prediction process using the trained model.")
        ContainerDb(model_store, "Model Store", "File System Directory", "Stores the final, trained model artifact and its configuration.")
    }

    Rel(adia_platform, training_pipeline, "Calls train()")
    Rel(adia_platform, inference_pipeline, "Calls infer()")
    
    Rel(training_pipeline, core_lib, "Uses")
    Rel(training_pipeline, model_store, "Writes to")
    Rel(inference_pipeline, core_lib, "Uses")
    Rel(inference_pipeline, model_store, "Reads from")
```

---

### Level 3: Component Diagrams

This level shows the components inside each container.

#### **Container 1: Core Services Library**
Contains all foundational, reusable building blocks.

```mermaid
C4Component
    title Component Diagram for Core Services Library

    Container_Boundary(core_container, "Core Services Library") {
        Component(perm_sym, "PermutationSymbolizer", "Converts a vector to a symbolic permutation integer.")
        Component(series_proc, "SeriesProcessor", "Transforms a raw time series into a list of integer symbol sequences.")
        Component(eca_gen, "ECADataGenerator", "Creates synthetic ECA data as 2D float tensors.")

        Component(dyn_ae, "MDL_Hierarchical_Autoencoder", "Composite U-Net model with two input heads and a dual-loss objective.")
        Component(break_class, "StructuralBreakClassifier", "Fine-tuning model that uses the pre-trained encoder.")
    }
```

#### **Container 2: Training Pipeline Logic**
Orchestrates the `train()` function.

```mermaid
C4Component
    title Component Diagram for Training Pipeline

    Container_Boundary(core_container, "Core Services Library") {
        Component(eca_gen, "ECADataGenerator")
        Component(dyn_ae, "MDL_Hierarchical_Autoencoder")
        Component(series_proc, "SeriesProcessor")
        Component(break_class, "StructuralBreakClassifier")
    }

    Container_Boundary(training_container, "Training Pipeline Logic") {
        Component(pre_trainer, "MDLPreTrainer", "Manages the pre-training loop.")
        Component(fine_tuner, "BreakClassifierFinetuner", "Manages the fine-tuning loop.")
        Component(saver, "ArtifactSaver", "Saves the final model artifacts.")
    }

    System_Ext(model_store, "Model Store")
    
    Rel(pre_trainer, eca_gen, "Uses")
    Rel(pre_trainer, dyn_ae, "Trains")
    Rel(fine_tuner, series_proc, "Uses")
    Rel(fine_tuner, break_class, "Fine-tunes")
    Rel_D(break_class, saver, "Provides Final Model Artifacts to")
    Rel_R(saver, model_store, "Writes artifacts to")
```

#### **Container 3: Inference Pipeline Logic**
Orchestrates the `infer()` function.

```mermaid
C4Component
    title Component Diagram for Inference Pipeline

    System_Ext(model_store, "Model Store")
    System_Ext(adia_platform, "ADIA Platform Runner")

    Container_Boundary(inference_container, "Inference Pipeline Logic") {
        Component(loader, "ArtifactLoader", "Loads all necessary model artifacts from the Model Store.")
        Component(encoder, "HierarchicalDynamicalEncoder", "The loaded, pre-trained core encoder.")
        Component(embedding_head, "EmbeddingHead", "The loaded, fine-tuned symbol embedding layer.")
        Component(series_proc, "SeriesProcessor", "Transforms raw test data into integer symbol sequences.")
        Component(fingerprinter, "Fingerprinter", "Generates a stable fingerprint for a data segment.")
        Component(scorer, "BreakScoreCalculator", "Computes the final distance score.")
    }

    Rel_R(loader, model_store, "Reads from")
    Rel_D(loader, encoder, "Instantiates")
    Rel_D(loader, embedding_head, "Instantiates")
    Rel_D(fingerprinter, series_proc, "Uses")
    Rel_D(fingerprinter, embedding_head, "Uses")
    Rel_D(fingerprinter, encoder, "Uses")
    Rel_D(scorer, fingerprinter, "Gets fingerprints from")
    Rel_R(scorer, adia_platform, "Yields Prediction to")
```

#### **Container 4: Model Store**
Represents the persisted artifacts.

```mermaid
C4Component
    title Component Diagram for Model Store

    Container_Boundary(model_store_container, "Model Store (File System Directory)") {
        Component(encoder_weights, "core_encoder.pth", "PyTorch State Dictionary", "Weights of the core U-Net encoder.")
        Component(embedding_weights, "embedding_head.pth", "PyTorch State Dictionary", "Weights of the trained symbol embedding layer.")
        Component(model_config, "model_config.joblib", "Configuration File", "Hyperparameters needed to build the model architectures.")
    }
```

---

### Level 4: Code View (The Definitive Blueprint for Implementation)

This level details the primary classes and their corrected "code contracts."

#### **Module 1: `core_library/data_processing.py`**

| Class Name | Role & Responsibilities | Key Public Methods | Collaborators |
| :--- | :--- | :--- | :--- |
| **`PermutationSymbolizer`** | Converts a single numeric vector into a single integer symbol using randomized, deterministic tie-breaking. | `__init__(embedding_dim)`<br>`symbolize_vector(vector)` | *(None)* |
| **`SeriesProcessor`** | Transforms a raw `pd.Series` into a list of integer symbol sequences (`List[torch.Tensor]`). | `__init__(symbolizer, sequence_length)`<br>`process(series)` | `PermutationSymbolizer` |
| **`ECADataGenerator`** | Creates synthetic ECA data as a batch of 2D float tensors `(batch, timesteps, width)` and corresponding integer rule labels. | `__init__(config)`<br>`generate_training_data()` | *(Uses `cellpylib`)* |

#### **Module 2: `core_library/model_architecture.py`**

| Class Name | Role & Responsibilities | Key Public Methods | Collaborators |
| :--- | :--- | :--- | :--- |
| **`MDL_Hierarchical_Autoencoder`** | **The main pre-training model.**<br>- Contains the core `HierarchicalDynamicalEncoder` and `Decoder`.<br>- Contains a separate `nn.Embedding` layer for symbol processing.<br>- Has a `forward_pretrain` method for ECA data (floats).<br>- Has a `forward_finetune_encode` method for real data (integers). | `__init__(args)`<br>`forward_pretrain(float_tensor)`<br>`forward_finetune_encode(int_tensor)` | *(Internal)* |
| **`StructuralBreakClassifier`** | **The fine-tuning model.**<br>- Wraps a pre-trained `MDL_Hierarchical_Autoencoder`.<br>- Its `forward` pass takes lists of integer symbol sequences `(before_seqs, after_seqs)` and correctly calls the `forward_finetune_encode` method of its collaborator. | `__init__(autoencoder, ...)`<br>`forward(before_seqs, after_seqs)` | `MDL_Hierarchical_Autoencoder` |

#### **Module 3: `training_pipeline.py`**

| Class Name | Role & Responsibilities | Key Public Methods | Collaborators |
| :--- | :--- | :--- | :--- |
| **`MDLPreTrainer`** | Orchestrates the pre-training loop for the `MDL_Hierarchical_Autoencoder` using float tensor data from `ECADataGenerator`. | `__init__(model, config)`<br>`pretrain(data_generator)` | `MDL_Hierarchical_Autoencoder`, `ECADataGenerator` |
| **`BreakClassifierFinetuner`**| Orchestrates the fine-tuning loop for the `StructuralBreakClassifier` using lists of integer symbol sequences from `SeriesProcessor`. | `__init__(model, config)`<br>`finetune(X_train, y_train, processor)` | `StructuralBreakClassifier`, `SeriesProcessor` |
| **`ArtifactSaver`** | Saves the core encoder weights, the embedding head weights, and the model configuration. | `save(model, config, path)` | `MDL_Hierarchical_Autoencoder` |

#### **Module 4: `inference_pipeline.py`**

| Class Name | Role & Responsibilities | Key Public Methods | Collaborators |
| :--- | :--- | :--- | :--- |
| **`ArtifactLoader`** | Reads the config, core encoder weights, and embedding head weights from the `Model Store`. | `load(path)` | *(None)* |
| **`Fingerprinter`** | Generates a stable fingerprint for a raw `pd.Series`. It orchestrates the `SeriesProcessor`, `EmbeddingHead`, and `Encoder` to produce a single vector. | `__init__(encoder, embedding_head, processor)`<br>`generate(series)` | `SeriesProcessor` |
| **`BreakScoreCalculator`** | Takes two fingerprint vectors and computes their cosine distance. | `calculate(fp_before, fp_after)` | *(None)* |

---
--- END OF FILE ---
