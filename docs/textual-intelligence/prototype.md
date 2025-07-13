
# Pipeline 1: Analysis (Discover & Validate)
Purpose: To process raw text submissions, use an LLM to discover potential underlying themes (motifs), and then use the Minimum Description Length (MDL) principle to validate which themes genuinely explain the data by measuring data compression.

Input: A JSON file containing raw, verbatim text responses.

Output: A JSON file containing the raw, validated themes and their associated MDL metrics.

# @title Cell 1.1: Analysis - Configuration & Imports
# ==============================================================================
# STAGE 1: ANALYSIS PIPELINE (DISCOVER & VALIDATE)
# ==============================================================================
# This pipeline ingests raw text and produces validated, machine-readable themes.
#
# PURPOSE:
#   - Stage 1 (Discover): Use an LLM to extract potential semantic themes (motifs)
#     from raw text responses.
#   - Stage 3 (Validate): Use the Minimum Description Length (MDL) principle to
#     validate that these themes genuinely explain the data.
#
# INPUT: JSON file with verbatim responses.
# OUTPUT: A "raw analysis" JSON file containing validated motifs and MDL scores.
# ==============================================================================

import os
import json
import re
import time
import logging
import traceback
from typing import List, Dict, Set, Tuple
from collections import Counter
import hashlib

# --- Try to import heavy libraries, with user-friendly errors ---
try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    from pybdm import BDM
except ImportError as e:
    print(f"ERROR: A required library is missing: {e}")
    print("Please install the necessary packages by running:")
    print("!pip install torch transformers bitsandbytes accelerate pybdm")
    raise

# --- Project Configuration ---
BASE_PROJECT_DIR = '/content/drive/MyDrive/Colab Notebooks/Legal/'
# Ensure the base directory exists
os.makedirs(BASE_PROJECT_DIR, exist_ok=True)

# --- Input Files ---
# The single source file containing all responses grouped by respondent
P1_VERBATIM_RESPONSES_FILE = os.path.join(BASE_PROJECT_DIR, 'Phase1_PDF_Extraction_Outputs', 'phase1_pdf_analysis_by_response.json')

# --- Output Files ---
# The final output of this analysis pipeline
MDL_RAW_ANALYSIS_OUTPUT_FILE = os.path.join(BASE_PROJECT_DIR, f"mdl_raw_analysis_results_{time.strftime('%Y%m%d')}.json")
# A debug log for troubleshooting LLM interactions
LLM_DEBUG_LOG_FILE = os.path.join(BASE_PROJECT_DIR, f"llm_analysis_debug_log_{time.strftime('%Y%m%d')}.txt")

# --- Processing Configuration ---
# Specify which Question IDs (QIDs) to analyze.
# Set to an empty list [] to process all QIDs found in the input file.
P3_QIDS_TO_PROCESS_THEMATICALLY = ["Q4"]

# --- LLM Configuration ---
LOCAL_LLM_MODEL_ID = 'google/gemma-2b-it'
USE_QUANTIZATION_FOR_LOCAL_LLM = True  # Set to True to use 4-bit quantization (requires GPU)
LLM_BATCH_SIZE_RESPONSES = 5  # Number of text responses to group into one chunk for the LLM
LLM_RETRY_ATTEMPTS = 2 # Number of times to retry if the LLM fails to produce valid JSON
MAX_TEXT_CHARS_PER_LLM_PROMPT_CHUNK = 7000  # Max characters of corpus text to feed into one prompt
LLM_MAX_NEW_TOKENS_MOTIF_EXTRACTION = 1000 # Max tokens the LLM can generate for its response
MAX_MOTIFS_PER_CHUNK = 3  # Max themes to extract from each chunk of text

# --- MDL (Minimum Description Length) & BDM (Block Decomposition Method) Configuration ---
MATRIX_SIZE_GLOBAL = (8, 8)  # Dimensions for converting text hash to a binary matrix for BDM
BDM_SEGMENT_LENGTH = 2000    # Length of text segments for calculating corpus BDM complexity

# --- MDL Hypothesis Cost L(H) Configuration ---
# These values determine the "cost" of storing the themes themselves.
MOTIF_SYMBOLIC_LABEL_COST = 0.5         # Cost for a theme's symbolic label (e.g., [DATA_PRIVACY])
MOTIF_DESCRIPTION_TEXT_BASE_COST = 0.5  # Base cost for having a description
MOTIF_DESCRIPTION_TOKEN_COST = 0.1      # Cost per token in the description
MOTIF_SURFACE_FORMS_LIST_BASE_COST = 0.0  # Base cost for the list of surface forms
MOTIF_SURFACE_FORM_TOKEN_COST_IN_LH = 0.0 # Cost for surface form tokens (set to 0 to focus on explanatory power)

# --- Theme Filtering Configuration ---
MIN_SF_FREQUENCY_IN_FULL_CORPUS = 2  # A theme's phrase must appear at least this many times in the entire set of responses for a question
MAX_SF_TOKEN_LENGTH_FOR_FINAL_MOTIF = 6 # Max words allowed in a surface form phrase

# --- Logger Setup ---
logger = logging.getLogger("ThematicAnalysis")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("Pipeline 1 (Analysis) Configuration Loaded.")

# @title Cell 1.2: Analysis - Helper Functions
# This cell contains all the functions needed for the analysis pipeline.

# --- 1. Data Loading & Text Utilities ---

def load_original_verbatim_responses(filepath: str) -> Dict[str, List[str]]:
    """Loads verbatim responses, grouping all text passages by their QID."""
    responses_by_qid = {}
    if not os.path.exists(filepath):
        logger.error(f"Input file not found: {filepath}"); return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        respondent_data = data.get("pdf_analysis_by_response", {})
        for resp_id, answers in respondent_data.items():
            for qid, q_data in answers.items():
                passages = q_data.get("extracted_passages")
                if isinstance(passages, list) and passages:
                    full_answer = " ".join(p.strip() for p in passages if isinstance(p, str) and p.strip())
                    if full_answer:
                        if qid not in responses_by_qid: responses_by_qid[qid] = []
                        responses_by_qid[qid].append(full_answer)
        logger.info(f"Loaded {sum(len(v) for v in responses_by_qid.values())} responses for {len(responses_by_qid)} QIDs.")
        return responses_by_qid
    except Exception as e:
        logger.error(f"Failed to load or parse {filepath}: {e}"); return {}

def tokenize_phrase(phrase_text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    if not isinstance(phrase_text, str): return []
    return phrase_text.lower().split()

def preprocess_corpus(text: str) -> str:
    """Basic text cleaning for LLM processing."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def count_sf_occurrences(corpus_text: str, surface_form: str) -> int:
    """Case-insensitive count of a phrase in a text body."""
    if not all(isinstance(arg, str) for arg in [corpus_text, surface_form]) or not surface_form.strip(): return 0
    try:
        return len(re.findall(re.escape(surface_form.lower()), corpus_text.lower()))
    except re.error:
        return 0

# --- 2. LLM Interaction ---

def initialize_llm_pipeline(model_id: str, use_quantization: bool):
    """Initializes and returns a Hugging Face pipeline for text generation."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing LLM '{model_id}' on device '{device}'. Quantization: {use_quantization}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"device_map": "auto"}
        if use_quantization and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        logger.info("LLM pipeline initialized successfully.")
        return llm_pipeline, tokenizer
    except Exception as e:
        logger.critical(f"LLM pipeline initialization failed: {e}")
        traceback.print_exc()
        return None, None

def create_motif_prompt(text_chunk: str, max_motifs: int) -> str:
    """Creates the detailed prompt for the LLM to extract themes."""
    # This prompt is engineered to force the LLM into a specific JSON output format.
    return f"""You are a precise thematic analysis assistant. Your task is to extract key recurring themes from the provided text.

STRICT OUTPUT REQUIREMENTS:
1. Your ENTIRE response MUST be a single, valid JSON list.
2. Each element in the list MUST be a JSON object with exactly three keys: "label", "description", and "surface_forms".
3. "label": A string in ALL_CAPITAL_SNAKE_CASE, enclosed in square brackets. Example: "[DATA_SECURITY_POLICY]".
4. "description": A single, concise sentence (string).
5. "surface_forms": A JSON list of 2 to 3 short (2-6 words) VERBATIM phrases extracted DIRECTLY from the text that exemplify the theme.
6. Identify up to {max_motifs} themes. Do NOT include any text or markdown (like ```json) before or after the JSON list.

Text to analyze:
\"\"\"
{text_chunk[:MAX_TEXT_CHARS_PER_LLM_PROMPT_CHUNK]}
\"\"\"

Your valid JSON response (ONLY the JSON list):
"""

def call_llm(prompt: str, llm_pipeline, tokenizer) -> str:
    """Calls the LLM and returns the generated text."""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm_pipeline(
            formatted_prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS_MOTIF_EXTRACTION,
            do_sample=False, # Use deterministic generation for theme extraction
            pad_token_id=tokenizer.pad_token_id
        )
        if outputs and outputs[0].get('generated_text'):
            return outputs[0]['generated_text'].strip()
        return ""
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

def parse_llm_response(raw_text: str) -> List[Dict]:
    """Parses and validates the JSON output from the LLM."""
    # Find the JSON list within the raw response
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if not match: return []
    json_str = match.group(0)

    try:
        data = json.loads(json_str)
        if not isinstance(data, list): return []

        # Schema validation
        valid_motifs = []
        for item in data:
            if (isinstance(item, dict) and
                all(k in item for k in ["label", "description", "surface_forms"]) and
                isinstance(item["label"], str) and re.fullmatch(r"\[[A-Z0-9_]+\]", item["label"]) and
                isinstance(item["description"], str) and
                isinstance(item["surface_forms"], list)):
                valid_motifs.append(item)
        return valid_motifs
    except json.JSONDecodeError:
        return []

# --- 3. Motif Processing & Filtering ---

def get_motifs_for_qid(texts: List[str], llm_pipeline, tokenizer, qid: str) -> List[Dict]:
    """Orchestrates motif extraction for all responses to a single question."""
    all_motifs = []
    # Batch responses into chunks
    for i in range(0, len(texts), LLM_BATCH_SIZE_RESPONSES):
        chunk_texts = texts[i:i + LLM_BATCH_SIZE_RESPONSES]
        corpus_chunk = preprocess_corpus("\n\n".join(chunk_texts))
        if len(corpus_chunk) < 100: continue

        logger.info(f"  Analyzing chunk {i//LLM_BATCH_SIZE_RESPONSES + 1} for QID {qid}...")

        prompt = create_motif_prompt(corpus_chunk, MAX_MOTIFS_PER_CHUNK)
        motifs_from_chunk = []
        for attempt in range(LLM_RETRY_ATTEMPTS):
            raw_response = call_llm(prompt, llm_pipeline, tokenizer)
            parsed = parse_llm_response(raw_response)
            if parsed:
                motifs_from_chunk = parsed
                break
            logger.warning(f"    Attempt {attempt+1} failed to get valid motifs. Retrying...")
            time.sleep(1)

        if motifs_from_chunk:
            logger.info(f"    Extracted {len(motifs_from_chunk)} motifs from chunk.")
            all_motifs.extend(motifs_from_chunk)
        else:
            with open(LLM_DEBUG_LOG_FILE, "a") as f:
                f.write(f"--- FAILED CHUNK: QID {qid} ---\nPROMPT:\n{prompt}\nRAW_RESPONSE:\n{raw_response}\n---\n")

    return all_motifs

def consolidate_motifs(raw_motifs: List[Dict]) -> List[Dict]:
    """Merges motifs with the same label, combining their surface forms."""
    consolidated = {}
    for motif in raw_motifs:
        label = motif["label"]
        if label not in consolidated:
            consolidated[label] = {
                "label": label,
                "description": motif["description"],
                "surface_forms": set()
            }
        # Add surface forms, converting to lowercase for uniqueness
        sfs = {sf.lower().strip() for sf in motif.get("surface_forms", []) if sf.strip()}
        consolidated[label]["surface_forms"].update(sfs)

    # Convert sets back to sorted lists
    return [
        {**m, "surface_forms": sorted(list(m["surface_forms"]))}
        for m in consolidated.values()
    ]

def filter_motifs(motifs: List[Dict], corpus: str) -> List[Dict]:
    """Filters motifs and their surface forms based on frequency and length criteria."""
    final_motifs = []
    for motif in motifs:
        valid_sfs = []
        for sf in motif.get("surface_forms", []):
            count = count_sf_occurrences(corpus, sf)
            num_tokens = len(tokenize_phrase(sf))
            if count >= MIN_SF_FREQUENCY_IN_FULL_CORPUS and 1 < num_tokens <= MAX_SF_TOKEN_LENGTH_FOR_FINAL_MOTIF:
                valid_sfs.append(sf)

        if valid_sfs:
            final_motifs.append({**motif, "surface_forms": sorted(list(set(valid_sfs)))})
    return final_motifs

# --- 4. MDL / BDM Calculations ---

def initialize_bdm():
    """Initializes a 2D BDM object."""
    try:
        bdm = BDM(ndim=2)
        logger.info("BDM instance initialized.")
        return bdm
    except Exception as e:
        logger.critical(f"BDM initialization failed: {e}"); return None

def text_to_binary_matrix(text: str, size: Tuple[int, int]) -> np.ndarray:
    """Converts text to a binary matrix via SHA256 hashing."""
    if not text.strip(): return np.zeros(size, dtype=int)
    h = hashlib.sha256(text.encode('utf-8')).hexdigest()
    binary_str = bin(int(h, 16))[2:].zfill(256)
    matrix_size = size[0] * size[1]
    padded_binary_str = binary_str.ljust(matrix_size, '0')[:matrix_size]
    return np.array(list(map(int, padded_binary_str))).reshape(size)

def get_bdm(text: str, bdm_instance: BDM, size: Tuple[int, int]) -> float:
    """Calculates BDM for a single text segment."""
    if not text.strip(): return 0.0
    matrix = text_to_binary_matrix(text, size)
    try:
        return bdm_instance.bdm(matrix)
    except Exception:
        return -1.0 # Error indicator

def get_corpus_bdm(corpus: str, bdm_instance: BDM) -> float:
    """Calculates total BDM for a large corpus by segmenting it."""
    total_bdm = 0.0
    for i in range(0, len(corpus), BDM_SEGMENT_LENGTH):
        segment = corpus[i:i+BDM_SEGMENT_LENGTH]
        segment_bdm = get_bdm(segment, bdm_instance, MATRIX_SIZE_GLOBAL)
        if segment_bdm < 0: return -1.0 # Propagate error
        total_bdm += segment_bdm
    return total_bdm

def get_L_H(motifs: List[Dict]) -> float:
    """Calculates L(H), the cost of the hypothesis (the themes themselves)."""
    cost = 0.0
    for motif in motifs:
        cost += MOTIF_SYMBOLIC_LABEL_COST
        desc_tokens = tokenize_phrase(motif.get("description", ""))
        cost += MOTIF_DESCRIPTION_TEXT_BASE_COST + (len(desc_tokens) * MOTIF_DESCRIPTION_TOKEN_COST)
        sfs = motif.get("surface_forms", [])
        if sfs:
            cost += MOTIF_SURFACE_FORMS_LIST_BASE_COST
            for sf in sfs:
                sf_tokens = tokenize_phrase(sf)
                cost += len(sf_tokens) * MOTIF_SURFACE_FORM_TOKEN_COST_IN_LH
    return cost

def compress_text_with_motifs(text: str, motifs: List[Dict]) -> str:
    """Replaces surface forms in text with short placeholders."""
    compressed_text = text.lower()
    for i, motif in enumerate(motifs):
        placeholder = f"@@M{i:03d}@@"
        # Sort by length, longest first, to avoid partial matches
        sorted_sfs = sorted(motif.get("surface_forms", []), key=len, reverse=True)
        for sf in sorted_sfs:
            # Use word boundaries (\b) to match whole phrases
            compressed_text = re.sub(r'\b' + re.escape(sf.lower()) + r'\b', placeholder, compressed_text)
    return compressed_text

def get_mdl_cost(corpus: str, motifs: List[Dict], bdm_instance: BDM) -> Tuple[float, float, float]:
    """Calculates the total MDL cost: L(H) + L(D|H)."""
    l_h = get_L_H(motifs)
    compressed_corpus = compress_text_with_motifs(corpus, motifs)
    l_d_h = get_corpus_bdm(compressed_corpus, bdm_instance)
    if l_d_h < 0: return l_h, -1.0, -1.0 # Propagate error
    return l_h, l_d_h, l_h + l_d_h

logger.info("Pipeline 1 (Analysis) Helper Functions Defined.")

# @title Cell 1.3: Analysis - Orchestration
# This cell runs the main analysis pipeline.

def main_analysis():
    """The main function to orchestrate the entire analysis pipeline."""
    script_version = "Thematic Intelligence Engine - Analysis v1.0"
    logger.info(f"--- Running: {script_version} ---")

    # --- 1. Initialization ---
    llm_pipeline, tokenizer = initialize_llm_pipeline(LOCAL_LLM_MODEL_ID, USE_QUANTIZATION_FOR_LOCAL_LLM)
    if not llm_pipeline: return

    bdm_instance = initialize_bdm()
    if not bdm_instance: return

    # --- 2. Load Data ---
    responses_by_qid = load_original_verbatim_responses(P1_VERBATIM_RESPONSES_FILE)
    if not responses_by_qid: return

    qids_to_process = P3_QIDS_TO_PROCESS_THEMATICALLY or list(responses_by_qid.keys())

    all_qid_results = []
    # --- 3. Per-QID Analysis Loop ---
    for qid in qids_to_process:
        if qid not in responses_by_qid:
            logger.warning(f"QID '{qid}' specified but not found in data. Skipping.")
            continue

        logger.info(f"--- Analyzing QID: {qid} ---")
        response_texts = responses_by_qid[qid]
        full_corpus = "\n\n<RSP_SEP>\n\n".join(response_texts)

        # --- 4. Baseline MDL Calculation ---
        baseline_mdl = get_corpus_bdm(full_corpus, bdm_instance)
        if baseline_mdl < 0:
            logger.error(f"  Baseline BDM calculation failed for QID {qid}. Skipping.")
            continue
        logger.info(f"  Baseline MDL (L(D)) for QID {qid}: {baseline_mdl:.2f}")

        # --- 5. Motif Discovery & Filtering ---
        raw_motifs = get_motifs_for_qid(response_texts, llm_pipeline, tokenizer, qid)
        logger.info(f"  Extracted {len(raw_motifs)} raw motifs.")
        consolidated = consolidate_motifs(raw_motifs)
        logger.info(f"  Consolidated into {len(consolidated)} unique motifs.")
        final_motifs = filter_motifs(consolidated, full_corpus)
        logger.info(f"  Filtered to {len(final_motifs)} final motifs after validation.")

        if not final_motifs:
            logger.warning(f"  No valid motifs found for QID {qid} after filtering.")
            # Still save the baseline result
            result = {
                "qid": qid, "num_responses": len(response_texts),
                "mdl_metrics": {"baseline_mdl": baseline_mdl, "compression_achieved": 0},
                "final_motifs": []
            }
            all_qid_results.append(result)
            continue

        # --- 6. Final MDL Calculation with Hypothesis ---
        l_h, l_d_h, total_mdl = get_mdl_cost(full_corpus, final_motifs, bdm_instance)
        if total_mdl < 0:
            logger.error(f"  Final MDL calculation failed for QID {qid}. Skipping.")
            continue

        compression = baseline_mdl - total_mdl
        logger.info(f"  L(H): {l_h:.2f}, L(D|H): {l_d_h:.2f}, Total MDL: {total_mdl:.2f}")
        logger.info(f"  Compression achieved: {compression:.2f}")

        # --- 7. Assemble & Store Result ---
        result = {
            "qid": qid,
            "num_responses": len(response_texts),
            "mdl_metrics": {
                "baseline_mdl": baseline_mdl,
                "l_h": l_h,
                "l_d_h": l_d_h,
                "final_mdl": total_mdl,
                "compression_achieved": compression
            },
            "final_motifs": final_motifs # These are the validated but un-enriched motifs
        }
        all_qid_results.append(result)

    # --- 8. Final Output ---
    final_output = {
        "metadata": {
            "script_version": script_version,
            "analysis_timestamp": time.asctime(),
            "source_file": os.path.basename(P1_VERBATIM_RESPONSES_FILE),
            "llm_model_id": LOCAL_LLM_MODEL_ID,
        },
        "results_by_qid": all_qid_results
    }

    with open(MDL_RAW_ANALYSIS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ Analysis complete. Raw validated themes saved to: {MDL_RAW_ANALYSIS_OUTPUT_FILE}")


# --- Execute the Pipeline ---
if __name__ == "__main__":
    main_analysis()

# Pipeline 2: Enrichment (Translate & Report)
Purpose: To take the raw, validated themes from the Analysis Pipeline and enrich them with human-centric context. This makes the themes understandable and ready for reporting.

Input: The "raw analysis" JSON from Pipeline 1 and the original verbatim responses JSON.

Output: An "enriched analysis" JSON file containing themes with frequency metrics, exemplary quotes, and human-readable descriptions.

# @title Cell 2.1: Enrichment - Configuration & Imports
# ==============================================================================
# STAGE 2: ENRICHMENT PIPELINE (TRANSLATE & REPORT)
# ==============================================================================
# This pipeline takes raw themes and makes them human-readable.
#
# PURPOSE:
#   - Stage 2 (Translate): Enriches raw, validated themes with human-centric context:
#     - Frequency metrics (% of respondents, mention counts).
#     - Exemplary quotations from the original text.
#     - (Optional) LLM-refined theme names and descriptions for clarity.
#
# INPUT:
#   1. Raw analysis JSON from Pipeline 1.
#   2. Original verbatim responses JSON.
#
# OUTPUT: A final, "enriched" JSON file ready for reporting and synthesis.
# ==============================================================================

import os
import json
import re
import time
import logging
import traceback
from typing import List, Dict, Tuple

# --- Try to import heavy libraries, with user-friendly errors ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
except ImportError as e:
    print(f"ERROR: A required library is missing: {e}")
    print("Please install the necessary packages by running:")
    print("!pip install torch transformers bitsandbytes accelerate")
    raise

# --- Project Configuration ---
# Assuming BASE_PROJECT_DIR is set from the previous pipeline
if 'BASE_PROJECT_DIR' not in locals():
    BASE_PROJECT_DIR = '/content/drive/MyDrive/Colab Notebooks/Legal/'
    os.makedirs(BASE_PROJECT_DIR, exist_ok=True)

# --- Input Files ---
# IMPORTANT: Update this filename to match the output of your Analysis Pipeline run.
# You can use os.listdir() to find the latest file if needed.
RAW_ANALYSIS_FILE = os.path.join(BASE_PROJECT_DIR, f"mdl_raw_analysis_results_{time.strftime('%Y%m%d')}.json")
VERBATIM_RESPONSES_FILE = os.path.join(BASE_PROJECT_DIR, 'Phase1_PDF_Extraction_Outputs', 'phase1_pdf_analysis_by_response.json')

# --- Output File ---
ENRICHED_OUTPUT_FILE = os.path.join(BASE_PROJECT_DIR, f"enriched_thematic_analysis_{time.strftime('%Y%m%d')}.json")

# --- Enrichment Configuration ---
# Set to True to use an LLM to rewrite theme descriptions for better clarity.
# This requires a GPU and may significantly increase processing time.
REFINE_DESCRIPTIONS_WITH_LLM = False
QUOTATIONS_TO_EXTRACT = 3 # Number of example quotes to find for each theme

# --- LLM Configuration (only used if REFINE_DESCRIPTIONS_WITH_LLM is True) ---
LLM_MODEL_ID_ENRICH = 'google/gemma-2b-it'
USE_QUANTIZATION_ENRICH = True

# --- Logger Setup ---
logger_enrich = logging.getLogger("ThematicEnrichment")
if not logger_enrich.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_enrich.addHandler(handler)
    logger_enrich.setLevel(logging.INFO)

logger_enrich.info("Pipeline 2 (Enrichment) Configuration Loaded.")

# @title Cell 2.2: Enrichment - Helper Functions
# This cell contains all functions for the enrichment pipeline.
# Note: Some functions like load_original_verbatim_responses are duplicated
# from the analysis pipeline for modularity, allowing this pipeline to be run independently.

# --- 1. Data Loading & Text Utilities ---

def load_original_verbatim_responses(filepath: str) -> Dict[str, List[str]]:
    """Loads verbatim responses, grouping all text passages by their QID."""
    # (This function is identical to the one in the Analysis pipeline)
    responses_by_qid = {}
    if not os.path.exists(filepath):
        logger_enrich.error(f"Input file not found: {filepath}"); return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        respondent_data = data.get("pdf_analysis_by_response", {})
        for resp_id, answers in respondent_data.items():
            for qid, q_data in answers.items():
                passages = q_data.get("extracted_passages")
                if isinstance(passages, list) and passages:
                    full_answer = " ".join(p.strip() for p in passages if isinstance(p, str) and p.strip())
                    if full_answer:
                        if qid not in responses_by_qid: responses_by_qid[qid] = []
                        responses_by_qid[qid].append(full_answer)
        logger_enrich.info(f"Loaded {sum(len(v) for v in responses_by_qid.values())} responses for {len(responses_by_qid)} QIDs.")
        return responses_by_qid
    except Exception as e:
        logger_enrich.error(f"Failed to load or parse {filepath}: {e}"); return {}

def create_question_map(filepath: str) -> Dict[str, str]:
    """Creates a map from QID to the full question text."""
    qid_map = {}
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        for resp in data.get("pdf_analysis_by_response", {}).values():
            for qid, q_data in resp.items():
                if qid not in qid_map and "question_text" in q_data:
                    qid_map[qid] = q_data["question_text"]
        return qid_map
    except Exception: return {}

def count_sf_occurrences_case_insensitive(corpus: str, sf: str) -> int:
    """Case-insensitive count of a surface form in a text body."""
    if not all(isinstance(arg, str) for arg in [corpus, sf]) or not sf.strip(): return 0
    return len(re.findall(re.escape(sf.lower()), corpus.lower()))

# --- 2. Enrichment Calculation Functions ---

def calculate_frequency_metrics(corpus: str, individual_responses: List[str], motif: Dict) -> Dict:
    """Calculates frequency and respondent penetration for a theme."""
    total_occurrences = 0
    hit_respondent_indices = set()

    for sf in motif.get("surface_forms", []):
        total_occurrences += count_sf_occurrences_case_insensitive(corpus, sf)
        for i, response_text in enumerate(individual_responses):
            if count_sf_occurrences_case_insensitive(response_text, sf) > 0:
                hit_respondent_indices.add(i)

    num_responses_hit = len(hit_respondent_indices)
    percent_hit = (num_responses_hit / len(individual_responses)) * 100 if individual_responses else 0

    return {
        "total_sf_occurrences": total_occurrences,
        "unique_responses_hit": num_responses_hit,
        "percent_responses_hit": round(percent_hit, 2)
    }

def extract_exemplary_quotations(corpus: str, motif: Dict, num_quotes: int) -> List[str]:
    """Extracts high-quality, diverse quotations illustrating a theme."""
    quotes = []
    used_spans = []
    # Sort surface forms by length (desc) to find more substantive quotes first
    sorted_sfs = sorted(motif.get("surface_forms", []), key=len, reverse=True)

    for sf in sorted_sfs:
        if len(quotes) >= num_quotes: break
        try:
            for match in re.finditer(re.escape(sf.lower()), corpus.lower()):
                start, end = match.span()
                # Check for overlap with already extracted quotes
                if any(max(start, s) < min(end, e) for s, e in used_spans):
                    continue

                # Expand context around the match
                context_start = corpus.rfind('.', 0, start) + 1
                context_end = corpus.find('.', end)
                if context_end == -1: context_end = len(corpus)

                snippet = corpus[context_start:context_end].strip()
                # Highlight the found surface form
                highlighted_snippet = re.sub(f"({re.escape(sf)})", r"**\1**", snippet, flags=re.IGNORECASE)

                quotes.append(highlighted_snippet)
                used_spans.append((context_start, context_end))
                if len(quotes) >= num_quotes: break
        except re.error:
            continue
    return quotes

# --- 3. LLM-based Enrichment (Optional) ---

def initialize_enrichment_llm():
    """Initializes LLM pipeline specifically for enrichment tasks."""
    # This function can be the same as the analysis one, but is separated for clarity
    if not REFINE_DESCRIPTIONS_WITH_LLM: return None, None
    try:
        from .Cell_1_2__Analysis_-_Helper_Functions import initialize_llm_pipeline
        return initialize_llm_pipeline(LLM_MODEL_ID_ENRICH, USE_QUANTIZATION_ENRICH)
    except (NameError, ImportError):
        # Fallback if running standalone
        logger_enrich.warning("Could not import `initialize_llm_pipeline`. You may need to copy it to this cell.")
        return None, None


def refine_description_with_llm(motif: Dict, question: str, quotes: List[str], llm_pipeline, tokenizer) -> str:
    """Uses an LLM to generate a more fluent, human-readable description for a theme."""
    if not all([llm_pipeline, tokenizer, quotes]):
        return motif.get("description", "")

    quote_str = "\n".join(f'- "{q.replace("**", "")}"' for q in quotes)
    prompt = f"""You are an expert qualitative analyst. Your task is to synthesize a clear, concise theme description.

Background:
- Original Question: "{question}"
- Theme Label: "{motif['label']}"
- Original AI-generated Description: "{motif['description']}"
- Example respondent statements for this theme:
{quote_str}

Based on all the information above, write a new, improved, single-sentence description for this theme. The description should be neutral, human-readable, and accurately capture the core idea expressed in the examples.

Refined 1-Sentence Description:
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm_pipeline(
            formatted_prompt,
            max_new_tokens=60,
            do_sample=True, temperature=0.5, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
        if outputs and outputs[0].get('generated_text'):
            return outputs[0]['generated_text'].strip().split('\n')[0]
    except Exception as e:
        logger_enrich.error(f"LLM description refinement failed: {e}")
    return motif.get("description", "")

logger_enrich.info("Pipeline 2 (Enrichment) Helper Functions Defined.")

# @title Cell 2.3: Enrichment - Orchestration
# This cell runs the main enrichment pipeline.

def main_enrichment():
    """The main function to orchestrate the enrichment pipeline."""
    script_version = "Thematic Intelligence Engine - Enrichment v1.0"
    logger_enrich.info(f"--- Running: {script_version} ---")

    # --- 1. Load Data ---
    if not os.path.exists(RAW_ANALYSIS_FILE):
        logger_enrich.critical(f"FATAL: Raw analysis file not found: {RAW_ANALYSIS_FILE}"); return
    with open(RAW_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    verbatim_data = load_original_verbatim_responses(VERBATIM_RESPONSES_FILE)
    question_map = create_question_map(VERBATIM_RESPONSES_FILE)

    # --- 2. Initialize LLM (if needed) ---
    llm_pipeline, tokenizer = None, None
    if REFINE_DESCRIPTIONS_WITH_LLM:
        logger_enrich.info("Initializing LLM for description refinement...")
        # Note: This reuses the `initialize_llm_pipeline` function from the analysis helpers
        # Ensure it's defined or copy it into the enrichment helpers cell.
        from Cell_1_2__Analysis_-_Helper_Functions import initialize_llm_pipeline
        llm_pipeline, tokenizer = initialize_llm_pipeline(LLM_MODEL_ID_ENRICH, USE_QUANTIZATION_ENRICH)
        if not llm_pipeline:
            logger_enrich.warning("LLM initialization failed. Proceeding without description refinement.")

    # --- 3. Per-QID Enrichment Loop ---
    enriched_results = []
    for qid_result in raw_data.get("results_by_qid", []):
        qid = qid_result["qid"]
        logger_enrich.info(f"--- Enriching QID: {qid} ---")

        individual_responses = verbatim_data.get(qid, [])
        if not individual_responses:
            logger_enrich.warning(f"  No verbatim text found for QID {qid}. Skipping enrichment."); continue

        full_corpus = "\n\n<RSP_SEP>\n\n".join(individual_responses)
        question_text = question_map.get(qid, "Question text not found.")

        enriched_themes = []
        for motif in qid_result.get("final_motifs", []):
            # a. Calculate Frequency Metrics
            freq_metrics = calculate_frequency_metrics(full_corpus, individual_responses, motif)

            # b. Extract Exemplary Quotations
            quotes = extract_exemplary_quotations(full_corpus, motif, QUOTATIONS_TO_EXTRACT)

            # c. (Optional) Refine Description
            description = motif.get("description", "")
            if REFINE_DESCRIPTIONS_WITH_LLM and llm_pipeline:
                description = refine_description_with_llm(motif, question_text, quotes, llm_pipeline, tokenizer)

            enriched_themes.append({
                **motif, # Keep original label, sfs
                "description": description,
                "frequency_metrics": freq_metrics,
                "exemplary_quotations": quotes,
            })

        # d. Rank enriched themes by respondent penetration
        sorted_themes = sorted(
            enriched_themes,
            key=lambda t: t["frequency_metrics"]["percent_responses_hit"],
            reverse=True
        )

        # --- 4. Assemble & Store Result for QID ---
        enriched_results.append({
            "qid": qid,
            "question_text": question_text,
            "corpus_summary": {"num_responses": len(individual_responses)},
            "mdl_metrics": qid_result.get("mdl_metrics", {}),
            "themes": [{**theme, "rank": i+1} for i, theme in enumerate(sorted_themes)]
        })

    # --- 5. Final Output ---
    final_output = {
        "metadata": {
            "script_version": script_version,
            "enrichment_timestamp": time.asctime(),
            "source_analysis_file": os.path.basename(RAW_ANALYSIS_FILE),
            "llm_description_refinement_used": bool(llm_pipeline)
        },
        "results_by_qid": enriched_results
    }

    with open(ENRICHED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger_enrich.info(f"\n✅ Enrichment complete. Enriched analysis saved to: {ENRICHED_OUTPUT_FILE}")


# --- Execute the Pipeline ---
if __name__ == "__main__":
    main_enrichment()

# Pipeline 3: Synthesis (Application)
Purpose: To use the final, enriched themes as a foundation for a generative model to create new, high-fidelity text that embodies the discovered concepts, tone, and vocabulary.

Input: The "enriched analysis" JSON file from Pipeline 2.

Output: Synthetic text responses printed to the console.

# @title Cell 3.1: Synthesis - Configuration & Imports
# ==============================================================================
# STAGE 3: SYNTHESIS PIPELINE (APPLICATION)
# ==============================================================================
# This pipeline uses the enriched themes to generate new, synthetic data.
#
# PURPOSE:
#   - To demonstrate the engine's understanding by generating new, high-fidelity
#     text responses that are grounded in the discovered themes, their meaning,
#     and their voice.
#
# INPUT: The "enriched analysis" JSON file from Pipeline 2.
# OUTPUT: Synthetic text, printed to the console for inspection.
# ==============================================================================

import os
import json
import random
import textwrap
import logging
from typing import List, Dict

# --- Try to import heavy libraries, with user-friendly errors ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
except ImportError as e:
    print(f"ERROR: A required library is missing: {e}")
    print("Please install the necessary packages by running:")
    print("!pip install torch transformers bitsandbytes accelerate")
    raise

# --- Project Configuration ---
if 'BASE_PROJECT_DIR' not in locals():
    BASE_PROJECT_DIR = '/content/drive/MyDrive/Colab Notebooks/Legal/'
    os.makedirs(BASE_PROJECT_DIR, exist_ok=True)

# --- Input File ---
# IMPORTANT: Update this filename to match the output of your Enrichment Pipeline run.
ENRICHED_ANALYSIS_FILE = os.path.join(BASE_PROJECT_DIR, f"enriched_thematic_analysis_{time.strftime('%Y%m%d')}.json")

# --- Interactive Session Configuration ---
# These will be used when you run the synthesis orchestration cell.
SYNTHESIS_QID = "Q4" # The QID you want to generate responses for.
NUM_SYNTHETIC_RESPONSES = 3 # How many new responses to generate.
PERSONA = "a concerned community member" # The persona for the LLM to adopt.

# --- LLM Generation Configuration ---
LLM_MODEL_ID_SYNTH = 'google/gemma-2b-it'
USE_QUANTIZATION_SYNTH = True
SYNTH_LLM_TEMPERATURE = 0.75 # Higher value = more creative/random
SYNTH_LLM_TOP_P = 0.95       # Nucleus sampling parameter
SYNTH_LLM_MAX_NEW_TOKENS = 250 # Max length of the generated response

# --- Logger Setup ---
logger_synth = logging.getLogger("ThematicSynthesis")
if not logger_synth.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger_synth.addHandler(handler)
    logger_synth.setLevel(logging.INFO)

logger_synth.info("Pipeline 3 (Synthesis) Configuration Loaded.")

# @title Cell 3.2: Synthesis - Helper Functions

def create_synthesis_prompt(themes: List[Dict], question: str, persona: str) -> str:
    """Creates a detailed prompt for generating a synthetic response."""
    theme_context = ""
    for i, theme in enumerate(themes):
        theme_context += f"\n--- \nTheme {i+1}: {theme.get('label', '[NO_LABEL]')}\n"
        theme_context += f"Meaning: {theme.get('description', 'No description.')}\n"
        quotes = [q.replace('**', '') for q in theme.get('exemplary_quotations', [])]
        if quotes:
            theme_context += "Real examples of how people expressed this:\n"
            for q in quotes:
                theme_context += f'- "{q}"\n'

    prompt = f"""You are a creative assistant. Your task is to write a new, authentic survey response.
Your persona is: {persona}.

You are answering the question: "{question}"

Your goal is to write a new response that naturally combines the core ideas from the themes below. Study the meaning and the real example quotes to capture the correct tone and vocabulary. Do NOT just copy the quotes. Write a completely new response.

{theme_context}
---
Your synthetic response:
"""
    return prompt.strip()

def generate_synthetic_response(prompt: str, llm_pipeline, tokenizer) -> str:
    """Calls the LLM with sampling enabled to generate one synthetic response."""
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm_pipeline(
            formatted_prompt,
            max_new_tokens=SYNTH_LLM_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=SYNTH_LLM_TEMPERATURE,
            top_p=SYNTH_LLM_TOP_P,
            pad_token_id=tokenizer.pad_token_id
        )
        if outputs and outputs[0].get('generated_text'):
            return outputs[0]['generated_text'].strip()
    except Exception as e:
        logger_synth.error(f"Synthesis LLM call failed: {e}")
    return "Generation failed."

logger_synth.info("Pipeline 3 (Synthesis) Helper Functions Defined.")

# @title Cell 3.3: Synthesis - Orchestration (Interactive)
# This cell runs the synthesis process. You can change the configuration
# variables in Cell 3.1 and re-run this cell to generate different responses.

def main_synthesis():
    """Main function to drive the interactive synthesis process."""
    script_version = "Thematic Intelligence Engine - Synthesis v1.0"
    logger_synth.info(f"--- Running: {script_version} ---")

    # --- 1. Load Enriched Data ---
    if not os.path.exists(ENRICHED_ANALYSIS_FILE):
        logger_synth.critical(f"FATAL: Enriched analysis file not found: {ENRICHED_ANALYSIS_FILE}"); return
    with open(ENRICHED_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        enriched_data = json.load(f)

    # --- 2. Find the Target QID Data ---
    target_qid_data = next((item for item in enriched_data.get("results_by_qid", []) if item["qid"] == SYNTHESIS_QID), None)
    if not target_qid_data:
        logger_synth.critical(f"FATAL: QID '{SYNTHESIS_QID}' not found in the enriched data file."); return

    all_themes = target_qid_data.get("themes", [])
    question_text = target_qid_data.get("question_text", "N/A")
    if not all_themes:
        logger_synth.critical(f"FATAL: No themes found for QID '{SYNTHESIS_QID}' to synthesize from."); return

    # --- 3. Initialize LLM ---
    logger_synth.info("Initializing LLM for synthesis...")
    # This reuses the `initialize_llm_pipeline` function from the analysis helpers
    from Cell_1_2__Analysis_-_Helper_Functions import initialize_llm_pipeline
    llm_pipeline, tokenizer = initialize_llm_pipeline(LLM_MODEL_ID_SYNTH, USE_QUANTIZATION_SYNTH)
    if not llm_pipeline:
        logger_synth.critical("LLM for synthesis failed to initialize. Aborting."); return

    # --- 4. Generation Loop ---
    print("\n" + "="*80)
    print(f"Generating {NUM_SYNTHETIC_RESPONSES} responses for QID '{SYNTHESIS_QID}'")
    print(f"Question: {question_text}")
    print(f"Persona: {PERSONA}")
    print("="*80 + "\n")

    for i in range(NUM_SYNTHETIC_RESPONSES):
        # Randomly select 1-3 themes to combine for each response
        num_themes_to_combine = random.randint(1, min(3, len(all_themes)))
        selected_themes = random.sample(all_themes, num_themes_to_combine)

        selected_theme_labels = [t['label'] for t in selected_themes]
        print(f"--- Synthetic Response #{i+1} (combining themes: {selected_theme_labels}) ---\n")

        prompt = create_synthesis_prompt(selected_themes, question_text, PERSONA)
        generated_text = generate_synthetic_response(prompt, llm_pipeline, tokenizer)

        # Print wrapped text for readability
        print(textwrap.fill(generated_text, width=80))
        print("\n" + "-"*80 + "\n")

    logger_synth.info("Synthesis complete.")

# --- Execute the Pipeline ---
if __name__ == "__main__":
    main_synthesis()
