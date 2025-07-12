

## Solution Architecture v4

This document outlines the complete C4 architecture for our solution to the ADIA challenge. This version (v4) is a self-contained blueprint incorporating all design decisions and corrections, providing a definitive guide for implementation.

### Level 1: System Context

This view shows our system in relation to the user and the external platform. Our solution is a self-contained library that is called by the ADIA Platform Runner, which expects `train()` and `infer()` entrypoints.

```mermaid
C4Context
    title System Context Diagram for ADIA Challenge

    System_Ext(adia_platform, "ADIA Platform Runner", "The environment that calls our train() and infer() functions.")
    
    System(our_system, "Our Solution", "A two-stage deep learning pipeline to detect structural breaks in time series.")

    Rel(adia_platform, our_system, "Calls train() and infer(), providing data.")
```

### Level 2: Container Diagram

This view breaks down our solution into its major, high-level structural blocks or "containers."

```mermaid
C4Container
    title System Container Diagram for ADIA Challenge

    System_Ext(adia_platform, "ADIA Platform Runner", "The environment that calls our train() and infer() functions.")
    
    Container_Boundary(our_system, "Our Solution") {
        Container(core_lib, "Core Services Library", "Python Module", "Contains all foundational, reusable code for data processing and model definitions.")
        Container(training_pipeline, "Training Pipeline Logic", "Python Module", "Orchestrates the two-stage training process (pre-training and fine-tuning).")
        Container(inference_pipeline, "Inference Pipeline Logic", "Python Module", "Orchestrates the prediction process using the trained model.")
        ContainerDb(model_store, "Model Store", "File System Directory", "Stores the final, trained model artifact and its configuration.")
    }

    Rel(adia_platform, training_pipeline, "Calls train()")
    Rel(adia_platform, inference_pipeline, "Calls infer()")
    
    Rel(training_pipeline, core_lib, "Uses data processing & model components from")
    Rel(training_pipeline, model_store, "Writes final model artifact to")

    Rel(inference_pipeline, core_lib, "Uses data processing components from")
    Rel(inference_pipeline, model_store, "Reads final model artifact from")
```

---

### Level 3: Component Diagrams

This level shows the components inside each container.

#### Container 1: Core Services Library

This container holds all the foundational, reusable building blocks of our system.

```mermaid
C4Component
    title Component Diagram for Core Services Library

    Container_Boundary(core_container, "Core Services Library") {
        Component(perm_sym, "PermutationSymbolizer", "Converts a vector to a symbolic permutation.")
        Component(series_proc, "SeriesProcessor", "Transforms a full time series into symbolic sequences.")
        Component(eca_gen, "ECADataGenerator", "Creates synthetic ECA data for pre-training.")

        Component(trans_enc, "HierarchicalDynamicalEncoder", "Model primitive for encoding sequences.")
        Component(trans_dec, "HierarchicalDynamicalDecoder", "Model primitive for decoding sequences.")
        Component(dyn_ae, "MDL_AU_Net_Autoencoder", "Composite model for pre-training.")
        Component(break_class, "StructuralBreakClassifier", "Composite model for fine-tuning.")
    }

    Rel(series_proc, perm_sym, "Uses")
    Rel(dyn_ae, trans_enc, "Is composed of")
    Rel(dyn_ae, trans_dec, "Is composed of")
    Rel(break_class, trans_enc, "Is composed of")
```

#### Container 2: Training Pipeline Logic

This container's components are pure **orchestrators** that manage the two-stage training flow.

```mermaid
C4Component
    title Component Diagram for Training Pipeline

    Container_Boundary(core_container, "Core Services Library") {
        Component(eca_gen, "ECADataGenerator")
        Component(dyn_ae, "MDL_AU_Net_Autoencoder")
        Component(series_proc, "SeriesProcessor")
        Component(break_class, "StructuralBreakClassifier")
    }

    Container_Boundary(training_container, "Training Pipeline Logic") {
        Component(pre_trainer, "MDLPreTrainer", "Manages the pre-training loop.")
        Component(fine_tuner, "BreakClassifierFinetuner", "Manages the fine-tuning loop.")
        Component(saver, "EncoderSaver", "Saves the final model artifact.")
    }

    System_Ext(model_store, "Model Store")
    
    Rel(pre_trainer, eca_gen, "Uses")
    Rel(pre_trainer, dyn_ae, "Trains")

    Rel(fine_tuner, series_proc, "Uses")
    Rel(fine_tuner, break_class, "Fine-tunes")
    
    Rel_D(break_class, saver, "Provides Final Encoder to")
    Rel_R(saver, model_store, "Writes artifact to")
```

#### Container 3: Inference Pipeline Logic

This container's components load the final model and use core services to generate predictions.

```mermaid
C4Component
    title Component Diagram for Inference Pipeline

    System_Ext(model_store, "Model Store")
    System_Ext(adia_platform, "ADIA Platform Runner")

    Container_Boundary(inference_container, "Inference Pipeline Logic") {
        Component(loader, "EncoderLoader", "Loads model from the Model Store.")
        Component(encoder, "HierarchicalDynamicalEncoder", "The loaded, fine-tuned model artifact.")
        Component(series_proc, "SeriesProcessor", "Transforms raw test data into symbolic sequences.")
        Component(fingerprinter, "Fingerprinter", "Generates a stable fingerprint for a data segment.")
        Component(scorer, "BreakScoreCalculator", "Computes the final distance score.")
    }

    Rel_R(loader, model_store, "Reads artifact from")
    Rel_D(loader, encoder, "Instantiates")
    
    Rel_D(fingerprinter, series_proc, "Uses")
    Rel_D(fingerprinter, encoder, "Uses")
    
    Rel_D(scorer, fingerprinter, "Gets 'before' and 'after' fingerprints from")
    
    Rel_R(scorer, adia_platform, "Yields Prediction to")
```

#### Container 4: Model Store

This container represents the persistence layer (`model_directory_path`).

```mermaid
C4Component
    title Component Diagram for Model Store

    Container_Boundary(model_store_container, "Model Store (File System Directory)") {
        Component(model_weights, "final_encoder.pth", "PyTorch State Dictionary", "The learned numerical weights of the final encoder.")
        Component(model_config, "model_config.joblib", "Configuration File", "Hyperparameters needed to build the model architecture before loading weights.")
    }
```

---

### Level 4: Code View (The Blueprint for Implementation)

This level details the primary classes and their corrected "code contracts."

#### Module 1: `core_library/data_processing.py`

| Class Name | Role & Responsibilities | Key Public Methods | Key Collaborators |
| :--- | :--- | :--- | :--- |
| **`PermutationSymbolizer`** | **Symbolic Converter.**<br>- Converts a single numeric vector into a discrete ordinal pattern symbol.<br>- Uses randomized tie-breaking for robustness. | `__init__(embedding_dim, seed)`<br>`symbolize_vector(vector)` | *(None - Foundational)* |
| **`SeriesProcessor`** | **Real Data Transformer.**<br>- Manages the full pipeline: time-delay embedding, symbolization, and windowing into sequences.<br>- Handles edge cases like series being too short. | `__init__(symbolizer, sequence_length)`<br>`process(series)` | `PermutationSymbolizer` |
| **`ECADataGenerator`** | **Synthetic Data Factory.**<br>- Simulates Elementary Cellular Automata to create a labeled dataset.<br>- Handles composite rules and ensures reproducibility. | `__init__(config)`<br>`generate_training_data()` | *(None - Uses `cellpylib` externally)* |

#### Module 2: `core_library/model_architecture.py`

| Class Name | Role & Responsibilities | Key Public Methods | Key Collaborators |
| :--- | :--- | :--- | :--- |
| **`HierarchicalDynamicalEncoder`** | **Sequence Encoder (Contracting Path).**<br>- A `nn.Module` that compresses a sequence into a final "fingerprint" sequence.<br> - Its `forward` pass MUST return a tuple: `(fingerprint_sequence, residuals_list)`. | `__init__(args)`<br>`forward(sequence_batch)` | *(None - Primitive)* |
| **`HierarchicalDynamicalDecoder`** | **Sequence Decoder (Expanding Path).**<br>- A `nn.Module` that reconstructs the original sequence.<br>- Its `forward` pass MUST accept two arguments: `(fingerprint_seq, residuals)`. | `__init__(args, transitions)`<br>`forward(fingerprint_seq, residuals)`| *(None - Primitive)* |
| **`MDL_AU_Net_Autoencoder`** | **Pre-training Model.**<br>- A composite `nn.Module` that combines the Encoder, Decoder, and a classification head.<br>- Its internal logic correctly handles the tuple returned by the encoder. | `__init__(args)`<br>`forward(sequence_batch)`<br>`encode(sequence_batch)` | `HierarchicalDynamicalEncoder`, `HierarchicalDynamicalDecoder` |
| **`StructuralBreakClassifier`** | **Fine-tuning Model.**<br>- A composite `nn.Module` that predicts a break from processed `before` and `after` periods.<br>- Its `forward` pass MUST accept two lists of tensors: `(before_seqs, after_seqs)`.<br>- Its internal logic MUST correctly unpack the `(fingerprint, _)` tuple when calling its encoder. | `__init__(encoder, latent_dim, ...)`<br>`forward(before_seqs, after_seqs)` | `HierarchicalDynamicalEncoder` |

#### Module 3: `training_pipeline.py`

| Class Name | Role & Responsibilities | Key Public Methods | Key Collaborators |
| :--- | :--- | :--- | :--- |
| **`MDLPreTrainer`** | **Pre-training Orchestrator.**<br>- Manages the training loop for the `MDL_AU_Net_Autoencoder`. | `__init__(model, config)`<br>`pretrain(data_generator)` | `MDL_AU_Net_Autoencoder`, `ECADataGenerator` |
| **`BreakClassifierFinetuner`** | **Fine-tuning Orchestrator.**<br>- Manages the training loop for the `StructuralBreakClassifier`.<br>- Its implementation must pass lists of tensors to the classifier. | `__init__(model, config)`<br>`finetune(X_train, y_train, processor)`| `StructuralBreakClassifier`, `SeriesProcessor` |
| **`EncoderSaver`** | **Artifact Manager.**<br>- Saves the final fine-tuned encoder and its configuration. | `save(model, config, path)` | `StructuralBreakClassifier` |

#### Module 4: `inference_pipeline.py`

| Class Name | Role & Responsibilities | Key Public Methods | Key Collaborators |
| :--- | :--- | :--- | :--- |
| **`EncoderLoader`** | **Artifact Loader.**<br>- Reads the config and weights from the `Model Store`. | `load(path)` | `HierarchicalDynamicalEncoder` |
| **`Fingerprinter`** | **Vector Generator.**<br>- Orchestrates producing a single, stable fingerprint for a time series segment.<br>- Its implementation must handle lists of sequences and average the resulting fingerprints. | `__init__(encoder, processor)`<br>`generate(series)` | `SeriesProcessor`, `HierarchicalDynamicalEncoder` |
| **`BreakScoreCalculator`** | **Prediction Calculator.**<br>- Takes two fingerprint vectors and computes their cosine distance. | `calculate(fp_before, fp_after)` | *(None - simple math)* |

---
