# 📄 AS-IS SYSTEM WORKFLOW DOCUMENTATION

This document captures the current architecture of the long-document classification and information extraction system, as extracted from `code_summary.txt`. It includes:

* An overview of the as-is pipeline
* A detailed text-based component diagram
* A Component–Responsibility–Collaborator (CRC) table
* Notes for managing risk-aware incremental upgrades
* A discussion on handling extremely long documents with modern architectures

---

## 🧹 OVERVIEW OF CURRENT WORKFLOW

The system processes legal and compliance documents (often PDFs) to classify them, extract relevant named entities, and detect the presence of digital or visual signatures. It is built in modular components using both traditional ML and deep learning, relying on caching, multiprocessing, and format-specific processing pipelines.

---

## 🔗 TEXT-BASED WORKFLOW DIAGRAM

```
                           ┌────────────────────────────┐
                           │    Input PDF Files     │
                           └─────────────────────────┘
                                        │
                                [1] PDF LOADING
                                        │
                ┌───────────────────────────────────────┐
                │                                                │
     ┌────────▼───────────┐                         ┌───────────────────────┐
     │ OCR Pipeline       │                         │ Digital Signature Check │
     │ (tesserocr +       │                         │ (PDF bytes + regex)     │
     │  pdf2texts)        │                         └─────────────────────┘
     └──────────────────────┘                                      │
                │                                                ▼
       ┌─────▼────────────┐                         ┌───────────────────────┐
       │ Chunking        │                         │ Visual Signature Detector │
       │ (semchunk)      │                         │ (signitractor ONNX)      │
       └────────────┘                         └─────────────────────┘
              │                                                  ▼
       ┌────▼───────────────┐                         ┌────────────────────┐
       │ Embedding Module │                         │  Signature Classifier  │
       │ (RoBERTa, TF-IDF)│                         │  (signitector LGBM)    │
       └────────────────────┘                         └─────────────────┘
              │                                                                
       ┌────▼────────────────────┐                                            
       │ Chunk Classification     │                                            
       │ (RoBERTa + LSTM fusion)  │                                            
       └────────────────────┘                                            
              │                                                                
       ┌────▼────────────┐                                                   
       │ Entity Extraction │                                                   
       │ (TokenClassifier) │                                                   
       └─────────────┘                                                   
                │                                                              
           ┌───▼───┐                                                         
           │  Cache  │                                                         
           │ + Persist│                                                        
           └───────┘                                                         
```

---

## 📋 COMPONENT–RESPONSIBILITY–COLLABORATOR (CRC) TABLE

| Component                | Responsibilities                                                          | Collaborators                      |
| ------------------------ | ------------------------------------------------------------------------- | ---------------------------------- |
| **PDF Loader**           | Load PDF files, manage byte-level access                                  | Signature Checker                  |
| **OCR Pipeline**         | Convert scanned or image-based PDFs into text using Tesseract             | Chunker, Visual Signature Detector |
| **Digital Sig. Check**   | Check for cryptographic digital signatures in PDF structure               | PDF Loader                         |
| **Visual Sig. Detect**   | Detect presence of visual signature bounding boxes using ONNX model       | OCR, Signature Classifier          |
| **Signature Classifier** | Classify detected visual regions as true/false signatures                 | Visual Detector                    |
| **Chunker**              | Split full text into semantically coherent chunks (e.g. \~512 tokens)     | OCR, Embedding Module              |
| **Embedding Module**     | Generate chunk-level representations using RoBERTa, TF-IDF                | Chunker, Classifier                |
| **Chunk Classifier**     | Classify document using embeddings + LSTM fusion                          | Embedding, Cache                   |
| **Entity Extractor**     | Named entity recognition using RoBERTa-style token classification         | Chunker                            |
| **Cache Manager**        | Cache all intermediate outputs on disk for efficiency and fault tolerance | All major components               |

---

## 📌 RISK-MANAGED UPGRADE GUIDANCE (by Component)

| Upgrade Target       | Risk Level | Recommendation                                                   |
| -------------------- | ---------- | ---------------------------------------------------------------- |
| PDF Loader + OCR     | Medium     | Replace with unified MLLM-based model (e.g., Donut, MMDocReader) |
| Signature Detection  | High       | Maintain for now, refactor to MLLM only after testing            |
| Chunking + Embedding | Medium     | Replace with long-context model inference over entire doc        |
| Chunk Classifier     | Low        | Easy to refactor into FusionNet-style or MoE classifier          |
| Entity Extractor     | Low        | Migrate to InstructNER or few-shot NER                           |
| Caching Framework    | Low        | Can retain as-is; decouples stability from model upgrades        |

---

## ♻️ HANDLING EXTREMELY LONG DOCUMENTS (MODERN SOLUTIONS)

To address long (potentially infinite) document lengths that exceed standard transformer context windows, modern 2025-era models offer viable alternatives:

### ✅ Long-Context Transformers

Use foundation models like:

* **Claude 3 Opus**, **Command R+**, **GPT-4.5**, **Mistral-7B-Long**, or **Phi-3-Long**
* Native context lengths from 32K to 256K+ tokens

**Benefits:**

* Eliminate manual chunking
* Preserve cross-section dependencies in classification
* Allow direct classification, extraction, or summarization across the full document

### ✅ Recurrent Memory-Augmented Models

If full sequence input isn't feasible:

* Use **RetNet**, **RWKV**, or **Memorizing Transformers** with chunkwise recurrence

### ✅ Compression-Aware Planning

Hybrid techniques:

* Use semantic hashing or symbolic sketching to reduce context
* Augment with **retrieval-based augmentation** over the entire doc corpus

### ✅ Agentic Refactor with Memory

The existing agentic orchestrator can manage:

* Long-document streaming (via a memory-augmented plan)
* Caching intermediate judgments
* Chunk-based voting with justification replay

These strategies allow you to shift from token-limited classification to full-document, multi-pass comprehension and labeling.

---

## ✅ Summary

This modular architecture is highly swappable, meaning a **gradual migration** to newer models can be achieved without full system disruption. The **first candidates for upgrade** include:

* Long-context embedding + classifier to replace chunk+LSTM
* Replacing OCR + chunker with layout-aware transformer
* Swapping entity extraction with instruction-tuned few-shot LLM-based NER

By combining modern long-context architectures with your agentic wrapping approach, the system can evolve into a robust, streaming-friendly platform for industrial-scale document intelligence.


