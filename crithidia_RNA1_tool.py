# -*- coding: utf-8 -*-
"""
Crithidia_RNAi_tool.py

Streamlit application to design RNAi primers targeting CDS for Crithidia fasciculata,
check for potential off-targets using k-mer analysis, format primers
with overhangs, optionally design flanking qPCR primers, and visualize results on gene model.

Version 23.4: Added checkbox to make qPCR primer design optional (default=True).
              Fixed SVG label bug where size was not displayed correctly.
              Implemented Option C: Check for qPCR/RNAi overlap based on relative CDS
              coordinates. If overlap occurs, treat qPCR design as failed and advise user.
              Removed qPCR design debugging statements.

Inputs (Assumed in the same flat directory as the script):
- Transcriptome FASTA file (with UTRs)
- Annotation GFF file (with UTRs/CDS)
- Genome FASTA file
- User-specified Gene ID and primer parameters via Streamlit UI.
- Optional logo.png file.

Requires: streamlit, primer3-py, pyfaidx, pandas, collections, pickle, gffutils
Assumes primer3-core (external tool) is installed and accessible.
"""

# --- Standard Library Imports ---
import os
import sys
import re
from collections import defaultdict, namedtuple
import time
import textwrap
import pickle
import traceback # For detailed error printing in GFF parsing

# --- Third-Party Library Imports ---
import streamlit as st
import primer3 # Interface to primer3-core
import pyfaidx # For reading FASTA efficiently
import pandas as pd # For displaying off-targets table
import gffutils # Imported, but custom parsing used for core logic

# --- Constants and Configuration ---

# File paths (assuming flat directory structure)
TRANSCRIPTOME_FASTA = "full_transcripts_coverage_aware_v2.fasta" # Used for k-mer index
ANNOTATION_GFF = "utr_annotations_coverage_aware_v2.gff" # Used for coordinates/visualization
GENOME_FASTA = 'TriTrypDB-68_CfasciculataCfCl_Genome.fasta' # Used for CDS sequence extraction
KMER_INDEX_PICKLE_FILE = "crithidia_kmer_index.pkl" # Cache file for k-mer index

# Default Primer3 settings for RNAi primers (adjustable via UI)
DEFAULT_TM_MIN = 58.0
DEFAULT_TM_OPT = 60.0
DEFAULT_TM_MAX = 62.0
DEFAULT_SIZE_MIN = 150
DEFAULT_SIZE_OPT = 250 # Calculated as average of min/max in UI
DEFAULT_SIZE_MAX = 500

# K-mer analysis settings
KMER_LENGTH = 21 # Length of k-mers for off-target analysis
MAX_MISMATCHES = 1 # Currently checks for 0 and 1 mismatch hits
BASES = ['A', 'C', 'G', 'T'] # Allowed bases in k-mers

# Overhang sequences and UI options
PROMOTER_SEQS = {
    "T7": "taatacgactcactataggg",
    "SP6": "atttaggtgacactatag",
    "T3": "aattaaccctcactaaaggg"
}
PROMOTER_OPTIONS = ["None", "T7", "SP6", "T3", "Custom"] # Options for dropdowns

# --- Helper Functions ---

def generate_kmers(sequence, k):
    """
    Generates all overlapping k-mers from a given sequence.

    Args:
        sequence (str): The input DNA sequence.
        k (int): The length of the k-mers to generate.

    Yields:
        str: The next k-mer found in the sequence.
    """
    sequence = sequence.upper().replace('U', 'T') # Ensure uppercase DNA
    if len(sequence) < k:
        return # Sequence too short for k-mer
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        # Ensure k-mer contains only standard DNA bases
        if all(base in BASES for base in kmer):
            yield kmer

def generate_1_mismatch_neighbors(kmer):
    """
    Generates all sequences with exactly one base mismatch compared to the original k-mer.

    Args:
        kmer (str): The original k-mer.

    Yields:
        str: A sequence differing from the kmer by one base at one position.
    """
    k = len(kmer)
    for i in range(k):
        original_base = kmer[i]
        for base in BASES:
            # Generate neighbor only if the base is different
            if base != original_base:
                neighbor = kmer[:i] + base + kmer[i+1:]
                yield neighbor

# --- Backend Functions ---

@st.cache_resource # Cache the loaded FASTA data for the duration of the session
def load_sequences(fasta_file):
    """
    Loads sequences from a FASTA file using pyfaidx.
    Creates the .fai index if it doesn't exist.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        pyfaidx.Fasta object or None: The loaded FASTA object, or None if loading fails.
    """
    start_time = time.time()
    print(f"Attempting to load sequences from {fasta_file}...") # User feedback in console
    try:
        # Check if the FASTA file exists
        if not os.path.exists(fasta_file):
            raise pyfaidx.FileNotFoundError(f"File not found: {fasta_file}")

        # Check if the index file exists, create if not
        index_file = fasta_file + '.fai'
        if not os.path.exists(index_file):
             print(f"FASTA index (.fai) not found for {fasta_file}. Creating...") # User feedback
             pyfaidx.Faidx(fasta_file) # This creates the .fai file
             print("FASTA index created.") # User feedback

        # Load the FASTA file using pyfaidx
        sequences = pyfaidx.Fasta(fasta_file, sequence_always_upper=True)
        num_seqs = len(sequences.keys())
        print(f"Loaded {num_seqs} sequences in {time.time() - start_time:.2f} sec.") # User feedback
        return sequences

    except pyfaidx.FileNotFoundError as fnf_err:
        st.error(f"{fnf_err}") # Show error in Streamlit app
        return None
    except Exception as e:
        print(f"Error loading FASTA file: {e}") # Log error to console
        st.error(f"An error occurred loading FASTA file '{fasta_file}'. Check console for details.")
        return None

@st.cache_resource # Cache the built k-mer index for the session
def build_transcriptome_kmer_index(_sequences, k, pickle_file):
    """
    Builds a k-mer index from transcriptome sequences or loads it from a pickle file if it exists.
    The index maps each k-mer to a set of transcript IDs containing that k-mer.

    Args:
        _sequences (pyfaidx.Fasta): The loaded transcriptome FASTA object.
                                     (Underscore indicates it's primarily for cache dependency).
        k (int): The k-mer length.
        pickle_file (str): The path to the pickle file for saving/loading the index.

    Returns:
        dict or None: The k-mer index (defaultdict(set)) or None if index creation fails.
    """
    if not _sequences:
        st.error("Cannot build k-mer index: Transcriptome sequences not loaded.")
        return None

    # Attempt to load the index from the pickle file first
    if os.path.exists(pickle_file):
        start_time = time.time()
        print(f"Attempting to load k-mer index from {pickle_file}...") # User feedback console
        try:
            with open(pickle_file, 'rb') as f_pkl:
                hit_kmer_index = pickle.load(f_pkl)
            # Basic validation
            if not isinstance(hit_kmer_index, dict):
                raise TypeError("Loaded object is not a dictionary.")
            print(f"Loaded k-mer index in {time.time() - start_time:.2f} sec.") # User feedback console
            return hit_kmer_index
        except Exception as e:
            print(f"Could not load k-mer index from {pickle_file}: {e}. Rebuilding.") # User feedback console
            # Proceed to rebuild if loading failed

    # Build the index if pickle file doesn't exist or loading failed
    start_time = time.time()
    print(f"Building {k}-mer index from {len(_sequences.keys())} transcripts...") # User feedback console
    hit_kmer_index = defaultdict(set) # Maps k-mer -> {transcript_id1, transcript_id2, ...}
    total_kmers_indexed = 0
    num_hits = len(_sequences.keys())
    seq_ids = list(_sequences.keys()) # Get all transcript IDs

    # Streamlit progress bar for visual feedback in the app
    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Starting k-mer index build...")

    # Iterate through each transcript in the FASTA file
    for i, seq_id in enumerate(seq_ids):
        # Extract the base gene ID (e.g., remove potential extra info after '|')
        base_id = seq_id.split('|')[0].strip()
        hit_seq = str(_sequences[seq_id]) # Get the sequence string
        kmers_in_hit = 0
        # Generate k-mers for the current sequence and add to the index
        for kmer in generate_kmers(hit_seq, k):
            hit_kmer_index[kmer].add(base_id)
            kmers_in_hit += 1
        total_kmers_indexed += kmers_in_hit

        # Update progress bar periodically
        if (i + 1) % (max(1, num_hits // 100)) == 0 or (i + 1) == num_hits: # Update every 1% or on last item
             progress_percent = (i + 1) / num_hits
             progress_bar.progress(progress_percent)
             progress_text.text(f"Indexing transcript {i+1}/{num_hits}")

    # Clean up progress elements
    progress_bar.empty()
    progress_text.empty()

    index_size = len(hit_kmer_index)
    print(f"K-mer index built in {time.time() - start_time:.2f} seconds.") # User feedback console
    print(f"Index contains {index_size} unique {k}-mers from {total_kmers_indexed} total k-mers found.") # User feedback console

    # Attempt to save the newly built index to the pickle file
    print(f"Attempting to save k-mer index to {pickle_file}...") # User feedback console
    try:
        with open(pickle_file, 'wb') as f_pkl:
            pickle.dump(hit_kmer_index, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"K-mer index saved successfully.") # User feedback console
    except Exception as e:
        print(f"Failed to save k-mer index to {pickle_file}: {e}") # User feedback console
        st.warning(f"Could not save k-mer index to {pickle_file}. Index will be rebuilt on next run.")

    return hit_kmer_index

def find_sequence(gene_id, sequences):
    """
    Retrieves a sequence string for a given gene ID from the loaded FASTA data.
    Handles potential variations in FASTA headers (e.g., with/without extra info after '|').

    Args:
        gene_id (str): The target gene ID input by the user.
        sequences (pyfaidx.Fasta): The loaded FASTA object.

    Returns:
        tuple (str or None, str or None):
            - The sequence string if found, else None.
            - The actual FASTA key (base gene ID) found, else None.
    """
    if sequences is None or not gene_id:
        return None, None

    gene_id_cleaned = gene_id.strip()

    # Try direct match first
    if gene_id_cleaned in sequences:
        return str(sequences[gene_id_cleaned]), gene_id_cleaned

    # Try matching the part before the first '|' (common in TriTrypDB headers)
    base_gene_id = gene_id_cleaned.split('|')[0].strip()
    if base_gene_id in sequences:
        return str(sequences[base_gene_id]), base_gene_id

    # Try matching with a space and pipe appended (another possible variation)
    header_variation = f"{base_gene_id} |"
    if header_variation in sequences:
        return str(sequences[header_variation]), base_gene_id

    # If no match found
    return None, None

def design_primers_with_primer3(sequence_id, sequence_template, size_min, size_opt, size_max, tm_min, tm_opt, tm_max):
    """
    Designs primer pairs using primer3-py bindings for a given sequence and parameters.
    Specifically configured for designing RNAi amplicons.

    Args:
        sequence_id (str): Identifier for the sequence (e.g., GeneID + "_CDS").
        sequence_template (str): The DNA sequence (CDS) to design primers against.
        size_min (int): Minimum desired amplicon size.
        size_opt (int): Optimal desired amplicon size (used by primer3).
        size_max (int): Maximum desired amplicon size.
        tm_min (float): Minimum desired primer Tm.
        tm_opt (float): Optimal desired primer Tm.
        tm_max (float): Maximum desired primer Tm.

    Returns:
        dict or None: A dictionary containing details of the best primer pair found,
                      or None if design fails. Includes raw primer3 results if any were generated.
    """
    sequence_template = sequence_template.upper().replace('U', 'T') # Ensure uppercase DNA

    # Arguments specific to the sequence being analyzed
    seq_args = {
        'SEQUENCE_ID': sequence_id,
        'SEQUENCE_TEMPLATE': sequence_template
    }

    # Global arguments controlling primer design parameters
    global_args = {
        'PRIMER_OPT_SIZE': 20,          # Optimal primer length
        'PRIMER_MIN_SIZE': 18,          # Minimum primer length
        'PRIMER_MAX_SIZE': 25,          # Maximum primer length
        'PRIMER_PRODUCT_SIZE_RANGE': [[size_min, size_max]], # Desired amplicon size range
        'PRIMER_OPT_TM': tm_opt,        # Optimal primer Tm
        'PRIMER_MIN_TM': tm_min,        # Minimum primer Tm
        'PRIMER_MAX_TM': tm_max,        # Maximum primer Tm
        'PRIMER_MIN_GC': 40.0,          # Minimum primer GC content (%)
        'PRIMER_MAX_GC': 60.0,          # Maximum primer GC content (%)
        'PRIMER_PICK_LEFT_PRIMER': 1,   # Must pick a left primer
        'PRIMER_PICK_INTERNAL_OLIGO': 0,# Do not pick an internal probe
        'PRIMER_PICK_RIGHT_PRIMER': 1,  # Must pick a right primer
        'PRIMER_NUM_RETURN': 5,         # Ask primer3 to find up to 5 pairs
        'PRIMER_EXPLAIN_FLAG': 1        # Request explanation if design fails
    }

    try:
        # Call the primer3 design function
        raw_results = primer3.bindings.designPrimers(seq_args, global_args) # Get raw results

        # Check how many pairs were returned
        num_returned = raw_results.get('PRIMER_PAIR_NUM_RETURNED', 0)

        if num_returned > 0:
             # Extract details for the *best* ranked pair (pair 0)
             pair_info = {
                 'PRIMER_LEFT_SEQUENCE': raw_results.get('PRIMER_LEFT_0_SEQUENCE'),
                 'PRIMER_RIGHT_SEQUENCE': raw_results.get('PRIMER_RIGHT_0_SEQUENCE'),
                 'PRIMER_LEFT_TM': raw_results.get('PRIMER_LEFT_0_TM'),
                 'PRIMER_RIGHT_TM': raw_results.get('PRIMER_RIGHT_0_TM'),
                 'PRIMER_LEFT_GC_PERCENT': raw_results.get('PRIMER_LEFT_0_GC_PERCENT'),
                 'PRIMER_RIGHT_GC_PERCENT': raw_results.get('PRIMER_RIGHT_0_GC_PERCENT'),
                 'PRIMER_PAIR_PRODUCT_SIZE': raw_results.get('PRIMER_PAIR_0_PRODUCT_SIZE'),
                 'PRIMER_LEFT_0': raw_results.get('PRIMER_LEFT_0'), # Tuple: (start_pos, length)
                 'PRIMER_RIGHT_0': raw_results.get('PRIMER_RIGHT_0') # Tuple: (start_pos, length) on reverse strand
             }
             # Basic validation that essential keys were found
             if all(pair_info.get(k) is not None for k in ['PRIMER_LEFT_SEQUENCE', 'PRIMER_RIGHT_SEQUENCE', 'PRIMER_PAIR_PRODUCT_SIZE', 'PRIMER_LEFT_0', 'PRIMER_RIGHT_0']):
                  # Add convenient 0-based start positions relative to the input template
                  pair_info['PRIMER_LEFT_POSITION'] = pair_info['PRIMER_LEFT_0'][0]
                  pair_info['PRIMER_RIGHT_POSITION'] = pair_info['PRIMER_RIGHT_0'][0]
                  pair_info['RAW_RESULTS'] = raw_results # Include raw results for potential debugging
                  return pair_info
             else:
                  st.warning(f"Primer3 returned pairs, but essential info missing for pair 0.")
                  return {'RAW_RESULTS': raw_results} # Return raw results even if parsing failed
        else:
            # No pairs found, provide explanation from primer3 if available
            explain = raw_results.get('PRIMER_EXPLAIN', 'No explanation provided.')
            st.warning(f"Primer3 did not return any suitable RNAi primer pairs. Explanation: {explain}")
            return {'RAW_RESULTS': raw_results} # Return raw results containing explanation
    except Exception as e:
        st.error(f"Error during RNAi primer design call: {e}")
        return None # Indicate failure

def design_qpcr_primers(sequence_id, sequence_template, exclude_region_list, global_args_override=None):
    """
    Designs qPCR primer pairs using primer3-py, optimized for typical qPCR parameters
    (short amplicon, specific Tm range) and excluding a specified region (e.g., the RNAi amplicon).

    Args:
        sequence_id (str): Identifier for the sequence (e.g., GeneID + "_CDS").
        sequence_template (str): The DNA sequence (CDS) to design primers against.
        exclude_region_list (list): A list of [start, length] pairs defining regions
                                     to avoid placing primers in (1-based coordinates).
        global_args_override (dict, optional): Dictionary to override default global args. Defaults to None.


    Returns:
        dict or None: A dictionary containing details of the best qPCR primer pair found,
                      or None if design fails. Includes raw results if any were generated.
    """
    sequence_template = sequence_template.upper().replace('U', 'T') # Ensure uppercase DNA

    # Arguments specific to the sequence being analyzed
    seq_args = {
        'SEQUENCE_ID': sequence_id + "_qPCR", # Distinguish from RNAi design
        'SEQUENCE_TEMPLATE': sequence_template
    }

    # Global arguments controlling primer design parameters, tuned for qPCR
    global_args = {
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_PRODUCT_SIZE_RANGE': [[80, 110]], # Typical qPCR amplicon size
        'PRIMER_OPT_TM': 58.0, # Typical qPCR optimal Tm
        'PRIMER_MIN_TM': 57.0, # Narrower Tm range for qPCR
        'PRIMER_MAX_TM': 59.0,
        'PRIMER_MIN_GC': 40.0,
        'PRIMER_MAX_GC': 60.0,
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_INTERNAL_OLIGO': 0,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_NUM_RETURN': 5,
        'PRIMER_EXPLAIN_FLAG': 1,
        'PRIMER_EXCLUDE_REGION': exclude_region_list # Critical: avoid RNAi region
    }

    # Allow overriding defaults if needed for debugging/flexibility
    if global_args_override:
        global_args.update(global_args_override)

    try:
        # Call the primer3 design function
        raw_results = primer3.bindings.designPrimers(seq_args, global_args) # Get raw results
        num_returned = raw_results.get('PRIMER_PAIR_NUM_RETURNED', 0)

        if num_returned > 0:
             # Extract details for the best ranked pair (pair 0)
             pair_info = {
                 'PRIMER_LEFT_SEQUENCE': raw_results.get('PRIMER_LEFT_0_SEQUENCE'),
                 'PRIMER_RIGHT_SEQUENCE': raw_results.get('PRIMER_RIGHT_0_SEQUENCE'),
                 'PRIMER_LEFT_TM': raw_results.get('PRIMER_LEFT_0_TM'),
                 'PRIMER_RIGHT_TM': raw_results.get('PRIMER_RIGHT_0_TM'),
                 'PRIMER_LEFT_GC_PERCENT': raw_results.get('PRIMER_LEFT_0_GC_PERCENT'),
                 'PRIMER_RIGHT_GC_PERCENT': raw_results.get('PRIMER_RIGHT_0_GC_PERCENT'),
                 'PRIMER_PAIR_PRODUCT_SIZE': raw_results.get('PRIMER_PAIR_0_PRODUCT_SIZE'),
                 'PRIMER_LEFT_0': raw_results.get('PRIMER_LEFT_0') # Tuple: (start_pos, length)
             }
             # Basic validation
             if all(pair_info.get(k) is not None for k in ['PRIMER_LEFT_SEQUENCE', 'PRIMER_RIGHT_SEQUENCE', 'PRIMER_PAIR_PRODUCT_SIZE', 'PRIMER_LEFT_0']):
                  # Add convenient 0-based start position
                  pair_info['PRIMER_LEFT_POSITION'] = pair_info['PRIMER_LEFT_0'][0]
                  pair_info['RAW_RESULTS'] = raw_results # Include raw results
                  return pair_info
             else:
                 # If essential info is missing even though pairs were returned
                 return {'RAW_RESULTS': raw_results} # Return raw results
        else:
             # No pairs found
             return {'RAW_RESULTS': raw_results} # Return raw results containing explanation
    except Exception as e:
        st.error(f"Error during qPCR primer design call: {e}")
        return None # Indicate failure

def format_primers_with_overhangs(fwd_seq, rev_seq, fwd_overhang, rev_overhang):
    """
    Concatenates overhang sequences (lowercase) with primer sequences (uppercase).

    Args:
        fwd_seq (str): Forward primer sequence.
        rev_seq (str): Reverse primer sequence.
        fwd_overhang (str): Forward overhang sequence.
        rev_overhang (str): Reverse overhang sequence.

    Returns:
        tuple (str, str): Formatted forward and reverse primer strings.
    """
    formatted_fwd = f"{fwd_overhang.lower()}{fwd_seq.upper()}"
    formatted_rev = f"{rev_overhang.lower()}{rev_seq.upper()}"
    return formatted_fwd, formatted_rev

def get_amplicon_sequence(sequence_template, primer_pair_info, cds_rel_start=0, cds_rel_end=None):
    """
    Extracts the biological amplicon sequence (region between primer binding sites)
    from the template sequence using the primer positions and product size reported by Primer3.

    Args:
        sequence_template (str): The template sequence (e.g., CDS) used for primer design.
        primer_pair_info (dict): The dictionary returned by design_primers_with_primer3.
        cds_rel_start (int): The start coordinate of the amplicon relative to the *original* sequence
                             if the template was a subsequence (default 0). Not strictly needed here.
        cds_rel_end (int or None): The end coordinate relative to the original sequence (default None).
                                   Not strictly needed here.

    Returns:
        str or None: The extracted amplicon sequence, or None if extraction fails.
    """
    sequence_template = sequence_template.upper().replace('U', 'T')
    if cds_rel_end is None:
        cds_rel_end = len(sequence_template) # Use full template length if not specified

    try:
        # Get forward primer start position (0-based) and product size
        fwd_primer_def = primer_pair_info.get('PRIMER_LEFT_0') # Tuple: (start_pos, length)
        product_size = primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE')

        if fwd_primer_def and product_size is not None:
             fwd_start_0based = fwd_primer_def[0]
             product_size = int(product_size)
             # Calculate amplicon end position (exclusive for slicing)
             amplicon_end_0based = fwd_start_0based + product_size

             # --- Boundary checks ---
             # Check relative to the provided template sequence length
             if not (0 <= fwd_start_0based < amplicon_end_0based <= len(sequence_template)):
                  st.warning(f"Calculated amplicon coordinates [{fwd_start_0based}-{amplicon_end_0based}) "
                             f"relative to template are out of bounds (template length: {len(sequence_template)}).")
                  return None

             # Extract the sequence
             return sequence_template[fwd_start_0based:amplicon_end_0based]
        else:
             # Primer info missing
             st.warning("Could not determine amplicon boundaries from primer info dictionary.")
             return None
    except Exception as e:
        st.error(f"Error extracting amplicon sequence: {e}")
        return None

def check_off_targets_kmer(amplicon_sequence, hit_kmer_index, target_gene_id):
    """
    Checks for potential off-targets by comparing k-mers from the amplicon sequence
    against the pre-built transcriptome k-mer index. Reports transcripts sharing
    perfect matches or 1-mismatch k-mers.

    Args:
        amplicon_sequence (str): The sequence of the designed RNAi amplicon (CDS portion).
        hit_kmer_index (dict): The pre-built k-mer index {kmer -> {gene_id1, gene_id2}}.
        target_gene_id (str): The ID of the intended target gene (to exclude from results).

    Returns:
        pandas.DataFrame: DataFrame listing potential off-target IDs and k-mer hit counts.
                          Empty DataFrame if no off-targets found or inputs are invalid.
    """
    if not amplicon_sequence or not hit_kmer_index:
        return pd.DataFrame() # Return empty DataFrame if inputs are missing

    start_time = time.time()
    # results format: {off_target_gene_id: [perfect_match_count, one_mismatch_hit_count]}
    results = defaultdict(lambda: [0, 0])
    # Ensure target gene ID is the base ID (without extra info) for comparison
    target_base_id = target_gene_id.split('|')[0].strip()

    # Generate unique k-mers from the amplicon sequence
    amplicon_kmers = set(generate_kmers(amplicon_sequence, KMER_LENGTH))
    num_amplicon_kmers = len(amplicon_kmers)
    processed_target_kmers = 0

    # Progress indication within Streamlit app
    progress_bar_kmer = st.progress(0)
    progress_text_kmer = st.empty()
    progress_text_kmer.text("Starting off-target k-mer check...")

    # Iterate through each unique k-mer from the amplicon
    for kmer in amplicon_kmers:
        processed_target_kmers += 1

        # --- Check for perfect matches ---
        if kmer in hit_kmer_index:
            # Iterate through all transcript IDs that contain this k-mer
            for hit_name in hit_kmer_index[kmer]:
                # Increment perfect match count if it's not the target gene itself
                if hit_name != target_base_id:
                    results[hit_name][0] += 1

        # --- Check for 1-mismatch neighbors ---
        for neighbor in generate_1_mismatch_neighbors(kmer):
            if neighbor in hit_kmer_index:
                # Iterate through all transcript IDs containing the neighbor k-mer
                for hit_name in hit_kmer_index[neighbor]:
                    # Increment mismatch count if it's not the target gene itself
                    if hit_name != target_base_id:
                        results[hit_name][1] += 1

        # Update progress bar periodically
        if num_amplicon_kmers > 0:
             progress_percent = processed_target_kmers / num_amplicon_kmers
             # Update every 10 k-mers or on the last one for smoother progress
             if processed_target_kmers % 10 == 0 or processed_target_kmers == num_amplicon_kmers:
                  progress_bar_kmer.progress(progress_percent)
                  progress_text_kmer.text(f"Checked {processed_target_kmers}/{num_amplicon_kmers} amplicon k-mers")

    # Clean up progress elements
    progress_bar_kmer.empty()
    progress_text_kmer.empty()

    # --- Format results into a DataFrame ---
    output_data = []
    for hit_name, counts in results.items():
        # Only include hits if there was at least one perfect or mismatch k-mer shared
        if sum(counts) > 0:
            output_data.append({
                'Off-Target ID': hit_name,
                'Perfect Kmer Matches': counts[0],
                '1 Mismatch Kmer Hits': counts[1]
            })

    off_target_df = pd.DataFrame(output_data)

    # Sort results for better readability (most hits first)
    if not off_target_df.empty:
        off_target_df = off_target_df.sort_values(
            by=['Perfect Kmer Matches', '1 Mismatch Kmer Hits'],
            ascending=[False, False]
        )

    print(f"Off-target check completed in {time.time() - start_time:.2f} seconds.") # Console feedback
    return off_target_df

# --- GFF Parsing Function for Visualization --- V2 (Handles Gene->mRNA->CDS linking) ---
@st.cache_data # Cache the parsed coordinates for a given gene ID and GFF file path
def get_feature_genomic_coords(_gene_id, annotation_gff_path):
    """
    Parses the annotation GFF file line-by-line to extract genomic coordinates
    for the main components (5'UTR, CDS, 3'UTR) of a given gene ID.
    Determines overall transcript boundaries, strand, and sequence ID (chromosome/contig).
    Handles Gene -> mRNA -> CDS relationships.

    Args:
        _gene_id (str): The target gene ID (base ID without extra info).
        annotation_gff_path (str): Path to the GFF annotation file.

    Returns:
        dict or None: A dictionary containing coordinates:
                      {'5UTR': (start, end), 'CDS': (start, end), '3UTR': (start, end),
                       'transcript_start': start, 'transcript_end': end,
                       'strand': '+/-', 'seqid': 'chromosome_name'}
                      Returns None if the GFF file is not found, parsing fails,
                      or essential features (like CDS or transcript boundaries) aren't found.
    """
    # Initialize dictionary to store coordinates
    coords = {'5UTR': None, 'CDS': None, '3UTR': None, 'transcript_start': None, 'transcript_end': None, 'strand': None, 'seqid': None}
    # Variables to track overall boundaries
    min_start = float('inf'); max_end = 0
    # Lists to store coordinates of potentially multiple features of the same type (e.g., CDS exons)
    cds_coords = []; utr5_coords = []; utr3_coords = []
    # Variables to store consistent strand and sequence ID
    found_strand = None; found_seqid = None; gene_found = False
    # Store mRNA IDs linked to the target gene to find associated CDS features
    target_mrna_ids = set()

    try:
        # --- File Existence Check ---
        if not os.path.exists(annotation_gff_path):
             st.error(f"Annotation GFF file not found: {annotation_gff_path}")
             return None # Return None if file not found

        # --- First pass (optional but potentially clearer): Find relevant mRNA IDs ---
        # This helps ensure we correctly link CDS features via their mRNA parent.
        with open(annotation_gff_path, 'r') as f_gff_pre:
            for line in f_gff_pre:
                if line.startswith('#'): continue # Skip comment lines
                parts = line.strip().split('\t')
                if len(parts) != 9: continue # Skip malformed lines
                featuretype = parts[2]; attributes_str = parts[8]
                # Look specifically for mRNA features
                if featuretype == 'mRNA':
                    parent_match = re.search(r'Parent=([^;]+)', attributes_str)
                    id_match = re.search(r'ID=([^;]+)', attributes_str)
                    # If mRNA's Parent is the target gene ID, store the mRNA's ID
                    if parent_match and id_match and parent_match.group(1) == _gene_id:
                        target_mrna_ids.add(id_match.group(1))

        # --- Main pass: Extract coordinates for all relevant features ---
        with open(annotation_gff_path, 'r') as f_gff:
            for line_num, line in enumerate(f_gff, 1):
                if line.startswith('#'): continue
                line_stripped = line.strip()
                parts = line_stripped.split('\t')
                if len(parts) != 9: continue
                seqid, source, featuretype, start_str, end_str, score, strand, frame, attributes_str = parts

                # Extract ID and Parent attributes using regex (handles variations)
                parent_match = re.search(r'Parent=([^;]+)', attributes_str)
                id_match = re.search(r'ID=([^;]+)', attributes_str)
                current_id = id_match.group(1) if id_match else None
                parent_id = parent_match.group(1) if parent_match else None

                # --- Determine if the current GFF line is relevant to the target gene ---
                is_relevant = False
                # 1. Is the feature the gene itself?
                if current_id == _gene_id and featuretype in ['protein_coding_gene', 'gene']:
                    is_relevant = True
                # 2. Is the feature's Parent the gene ID? (Handles mRNA, and UTRs if directly linked)
                elif parent_id == _gene_id and featuretype in ['mRNA', 'five_prime_utr', 'three_prime_utr']:
                     is_relevant = True
                # 3. Is the feature a CDS whose Parent is one of the target mRNA IDs found earlier?
                elif parent_id in target_mrna_ids and featuretype == 'CDS':
                    is_relevant = True
                # 4. Fallback: Is the feature a CDS/UTR whose Parent is the gene ID directly?
                elif parent_id == _gene_id and featuretype in ['CDS', 'five_prime_utr', 'three_prime_utr']:
                     is_relevant = True


                # --- Process relevant features ---
                if is_relevant:
                     gene_found = True # Mark that we found at least one relevant feature
                     try:
                         # Convert coordinates to integers (GFF is 1-based)
                         start_coord = int(start_str)
                         end_coord = int(end_str)

                         # Store strand and seqid, ensuring consistency across features
                         if found_strand is None: found_strand = strand
                         if found_seqid is None: found_seqid = seqid
                         elif found_strand != strand or found_seqid != seqid:
                             # If features for the same gene ID are on different strands/contigs, skip
                             continue

                         # Update overall transcript boundaries
                         min_start = min(min_start, start_coord)
                         max_end = max(max_end, end_coord)

                         # Store coordinates based on feature type
                         if featuretype == 'CDS':
                             cds_coords.append((start_coord, end_coord))
                         elif featuretype == 'five_prime_utr':
                             utr5_coords.append((start_coord, end_coord))
                         elif featuretype == 'three_prime_utr':
                             utr3_coords.append((start_coord, end_coord))

                     except ValueError:
                         # Handle cases where coordinates are not valid integers
                         continue
                     except Exception as inner_e: # Catch other potential errors during processing
                         print(f"Error processing GFF line {line_num} for {_gene_id}: {inner_e}")
                         continue # Skip this line

        # --- Finalize and return coordinates ---
        if gene_found and found_strand:
             coords['strand'] = found_strand
             coords['seqid'] = found_seqid
             # Set overall transcript boundaries
             coords['transcript_start'] = min_start if min_start != float('inf') else None
             coords['transcript_end'] = max_end if max_end != 0 else None

             # Consolidate potentially multiple CDS/UTR features into single ranges
             if cds_coords: coords['CDS'] = (min(s for s,e in cds_coords), max(e for s,e in cds_coords))
             if utr5_coords: coords['5UTR'] = (min(s for s,e in utr5_coords), max(e for s,e in utr5_coords))
             if utr3_coords: coords['3UTR'] = (min(s for s,e in utr3_coords), max(e for s,e in utr3_coords))

             # Final validation check: ensure transcript boundaries are valid
             if coords['transcript_start'] and coords['transcript_end'] and coords['transcript_start'] <= coords['transcript_end']:
                 # Check if CDS was actually found, as it's crucial for the tool
                 if coords.get('CDS') is None:
                     # Let the main code check for CDS presence
                     pass
                 return coords
             else:
                 # Invalid overall range calculated
                 return None
        else:
             # Gene ID was not found, or no valid features were processed
             return None

    except Exception as e:
        # Catch broader errors during file handling or parsing
        st.error(f"Error parsing annotation GFF {annotation_gff_path}: {e}")
        traceback.print_exc() # Print full traceback to console for detailed debugging
        return None
# --- END OF GFF PARSING FUNCTION ---


# --- SVG Function ---
def generate_gene_model_svg(gene_id, transcript_genomic_start, transcript_genomic_end, feature_coords, rnai_info, qpcr_info, padding=10):
    """
    Generates an SVG string visualizing the gene model (UTRs, CDS) and the
    positions of the designed RNAi and qPCR amplicons based on genomic coordinates.

    Args:
        gene_id (str): The ID of the gene for labeling.
        transcript_genomic_start (int): The minimum genomic coordinate for the transcript.
        transcript_genomic_end (int): The maximum genomic coordinate for the transcript.
        feature_coords (dict): Dictionary containing genomic coordinates for '5UTR', 'CDS', '3UTR'.
                               Keys may be missing or value may be None if feature not found.
        rnai_info (dict or None): Dict with {'genomic_start': int, 'size': int} for RNAi amplicon, or None.
        qpcr_info (dict or None): Dict with {'genomic_start': int, 'size': int} for qPCR amplicon, or None.
        padding (int): Pixel padding around the drawing area.

    Returns:
        str: A string containing the SVG markup, or an empty string if coordinates are invalid.
    """
    # --- SVG Dimensions and Styling ---
    svg_height = 70         # Total height of the SVG canvas
    track_height = 10       # Height of the main gene model track (UTR/CDS)
    amplicon_height = 8     # Height of the amplicon rectangles
    gene_label_y = 15       # Y position for the gene ID label
    amplicon_label_y = 30   # Y position for amplicon size labels
    cds_track_y = 42        # Y position for the top of the gene model track
    amplicon_track_y = cds_track_y + (track_height - amplicon_height) / 2 # Center amplicons vertically
    axis_label_y = cds_track_y + track_height + 10 # Y position for 5'/3' axis labels

    svg_width_px = 600      # Total width of the SVG canvas
    draw_width = svg_width_px - 2 * padding # Usable drawing width

    # --- Basic SVG Structure and Styles ---
    svg_elements = [
        f'<svg width="{svg_width_px}" height="{svg_height}" style="background-color:#f8f9fa; border-radius: 3px; margin-top: 10px; margin-bottom: 10px;">',
        # Basic CSS styles for text elements
        '<style>'
            '.small { font: italic 8px sans-serif; fill: #555; } '
            '.gene-label { font: bold 10px sans-serif; fill: #333; } '
            '.amp-label { font: 8px sans-serif; fill: #333; }'
        '</style>'
    ]

    # Calculate the total genomic span covered by the visualization
    total_genomic_span = transcript_genomic_end - transcript_genomic_start + 1
    # If coordinates are invalid, return empty SVG
    if total_genomic_span <= 0:
        print("[Error] Invalid transcript genomic span for SVG.")
        return ""

    # --- Coordinate Mapping Function ---
    def get_svg_pixels(feat_genomic_start, feat_genomic_end):
        """Maps genomic coordinates to SVG pixel coordinates."""
        if feat_genomic_start is None or feat_genomic_end is None:
            return None, None # Cannot draw if coordinates are missing

        # Calculate start position relative to the transcript start
        rel_start = feat_genomic_start - transcript_genomic_start
        # Calculate genomic length of the feature
        feat_len = feat_genomic_end - feat_genomic_start + 1

        # Scale genomic position and length to pixel values
        x_px = padding + max(0, (rel_start / total_genomic_span) * draw_width)
        w_px = max(1, (feat_len / total_genomic_span) * draw_width) # Ensure minimum width of 1px

        # Prevent drawing beyond the SVG boundaries
        if x_px + w_px > svg_width_px - padding:
            w_px = (svg_width_px - padding) - x_px

        return x_px, w_px

    # --- Draw Gene Model Backbone (Line) ---
    svg_elements.append(f'<line x1="{padding}" y1="{cds_track_y + track_height/2}" '
                        f'x2="{svg_width_px - padding}" y2="{cds_track_y + track_height/2}" '
                        'stroke="#bbb" stroke-width="1" />')

    # --- Draw UTRs and CDS Rectangles ---
    # Get coordinates from the feature_coords dictionary
    utr5_coords_g = feature_coords.get('5UTR') # Tuple (start, end) or None
    cds_coords_g = feature_coords.get('CDS')   # Tuple (start, end) or None
    utr3_coords_g = feature_coords.get('3UTR') # Tuple (start, end) or None

    # Calculate pixel positions for each feature
    utr5_x, utr5_w = get_svg_pixels(utr5_coords_g[0], utr5_coords_g[1]) if utr5_coords_g else (None, None)
    cds_x, cds_w = get_svg_pixels(cds_coords_g[0], cds_coords_g[1]) if cds_coords_g else (None, None)
    utr3_x, utr3_w = get_svg_pixels(utr3_coords_g[0], utr3_coords_g[1]) if utr3_coords_g else (None, None)

    # Draw rectangles for UTRs (light gray) and CDS (darker gray)
    # Add tooltips showing the size in base pairs
    if utr5_x is not None:
        svg_elements.append(f'<rect x="{utr5_x:.1f}" y="{cds_track_y}" width="{utr5_w:.1f}" height="{track_height}" fill="#e0e0e0">'
                            f'<title>5UTR ({utr5_coords_g[1]-utr5_coords_g[0]+1} bp)</title></rect>')
    if utr3_x is not None:
        svg_elements.append(f'<rect x="{utr3_x:.1f}" y="{cds_track_y}" width="{utr3_w:.1f}" height="{track_height}" fill="#e0e0e0">'
                            f'<title>3UTR ({utr3_coords_g[1]-utr3_coords_g[0]+1} bp)</title></rect>')
    if cds_x is not None:
        svg_elements.append(f'<rect x="{cds_x:.1f}" y="{cds_track_y}" width="{cds_w:.1f}" height="{track_height}" fill="#bdbdbd">'
                            f'<title>CDS ({cds_coords_g[1]-cds_coords_g[0]+1} bp)</title></rect>')

    # --- Draw Amplicon Rectangles ---
    rnai_label_x_center = None # Track RNAi label position to avoid qPCR label overlap

    # Draw RNAi amplicon (blue-green) if info is available
    if rnai_info:
        rnai_genomic_start = rnai_info.get('genomic_start')
        rnai_size = rnai_info.get('size')
        if rnai_genomic_start is not None and rnai_size is not None and rnai_size > 0:
             # Calculate pixel position for RNAi amplicon
             rnai_x, rnai_w = get_svg_pixels(rnai_genomic_start, rnai_genomic_start + rnai_size - 1)
             if rnai_x is not None:
                  # Draw the rectangle
                  svg_elements.append(f'<rect x="{rnai_x:.1f}" y="{amplicon_track_y}" width="{rnai_w:.1f}" '
                                      f'height="{amplicon_height}" fill="#4dd0e1" rx="1" ry="1" style="opacity:0.9;">'
                                      f'<title>RNAi Amplicon ({rnai_size} bp)</title></rect>')
                  # Add size label above the amplicon - *** SVG LABEL FIX ***
                  rnai_label_x_center = rnai_x + rnai_w / 2
                  svg_elements.append(f'<text x="{rnai_label_x_center:.1f}" y="{amplicon_label_y}" '
                                      f'class="amp-label" text-anchor="middle">{rnai_size} bp</text>') # Correctly use f-string

    # Draw qPCR amplicon (red) if info is available
    # qpcr_info will be None if the overlap check failed
    if qpcr_info:
        qpcr_genomic_start = qpcr_info.get('genomic_start')
        qpcr_size = qpcr_info.get('size')
        if qpcr_genomic_start is not None and qpcr_size is not None and qpcr_size > 0:
             # Calculate pixel position for qPCR amplicon
             qpcr_x, qpcr_w = get_svg_pixels(qpcr_genomic_start, qpcr_genomic_start + qpcr_size - 1)
             if qpcr_x is not None:
                  # Draw the rectangle
                  svg_elements.append(f'<rect x="{qpcr_x:.1f}" y="{amplicon_track_y}" width="{qpcr_w:.1f}" '
                                      f'height="{amplicon_height}" fill="#d9534f" rx="1" ry="1" style="opacity:0.9;">'
                                      f'<title>qPCR Amplicon ({qpcr_size} bp)</title></rect>')
                  # Add size label, slightly offset vertically if it overlaps RNAi label
                  label_x_center = qpcr_x + qpcr_w / 2
                  label_y_offset = 0
                  # Check if qPCR label center is close to RNAi label center
                  if rnai_label_x_center is not None and abs(label_x_center - rnai_label_x_center) < 20:
                       label_y_offset = -5 # Move qPCR label up slightly
                  # Add size label above the amplicon - *** SVG LABEL FIX ***
                  svg_elements.append(f'<text x="{label_x_center:.1f}" y="{amplicon_label_y + label_y_offset}" '
                                      f'class="amp-label" text-anchor="middle">{qpcr_size} bp</text>') # Correctly use f-string

    # --- Add Gene ID and Axis Labels ---
    svg_elements.append(f'<text x="{svg_width_px / 2}" y="{gene_label_y}" class="gene-label" text-anchor="middle">{gene_id}</text>')
    svg_elements.append(f'<text x="{padding}" y="{axis_label_y}" class="small">5\'</text>')
    svg_elements.append(f'<text x="{svg_width_px - padding}" y="{axis_label_y}" class="small" text-anchor="end">3\'</text>')

    # --- Finalize SVG ---
    svg_elements.append('</svg>')
    return "".join(svg_elements)


# --- Streamlit App UI ---

# Configure page layout (wide is generally better for this app)
st.set_page_config(layout="wide")

# --- Title and Logo ---
# Display logo in a column next to the title if logo.png exists
if os.path.exists("logo.png"):
     col_logo, col_title = st.columns([1, 6]) # Adjust column ratio as needed
     with col_logo:
         st.image("logo.png", width=80) # Adjust width as needed
     with col_title:
         st.title("*Crithidia fasciculata* RNAi Tool") # Italicized Title
else:
     # Display title only if logo is not found
     st.title("*Crithidia fasciculata* RNAi Tool") # Italicized Title

# Brief description below the title
st.markdown("Design primers for RNAi fragments targeting CDS, check off-targets (k-mer), optionally design flanking qPCR primers, visualize.")

# --- Load Data and Build/Load Index ---
# Display spinners while loading data and building index for user feedback

# Load Transcriptome Sequences (used for k-mer index)
sequences_placeholder = st.empty() # Placeholder for loading message
with st.spinner(f"Loading transcriptome sequences from {TRANSCRIPTOME_FASTA}..."):
    sequences = load_sequences(TRANSCRIPTOME_FASTA)
sequences_placeholder.empty() # Remove message once loaded or if error occurred

# Load or Build K-mer Index (depends on transcriptome sequences)
if sequences:
    kmer_index_placeholder = st.empty() # Placeholder for index message
    with st.spinner(f"Loading or building {KMER_LENGTH}-mer index (may take time on first run)..."):
        transcriptome_kmer_index = build_transcriptome_kmer_index(sequences, KMER_LENGTH, KMER_INDEX_PICKLE_FILE)
    # Progress messages are handled within build_transcriptome_kmer_index now
else:
    # Halt execution if sequences couldn't be loaded
    transcriptome_kmer_index = None
    st.error("Halting - could not load transcriptome sequences.")
    st.stop() # Stop the script execution in Streamlit

# Halt execution if k-mer index failed
if transcriptome_kmer_index is None:
    st.error("Halting - could not build or load k-mer index.")
    st.stop()

# --- Input Form ---
# Use st.form to group inputs and submit button
with st.form("rnai_design_form"):
    st.subheader("Input Parameters")

    # Target Gene ID input
    target_gene_id_input = st.text_input(
        "Target Gene ID (e.g., CFAC1_170028000)",
        key="gene_id",
        help="Enter the primary gene identifier used in FASTA/GFF files."
    )

    # Use columns for better layout of parameters
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RNAi Amplicon Parameters (within CDS):**")
        # RNAi Amplicon Size Range
        size_min = st.number_input(
            "Min Amplicon Size (bp)",
            min_value=50, max_value=1000, value=DEFAULT_SIZE_MIN, step=10,
            key="size_min", help="Minimum size of the desired RNAi amplicon."
        )
        # RNAi Primer Tm Range (Min)
        tm_min = st.number_input(
            "Min Primer Tm (°C)",
            min_value=45.0, max_value=70.0, value=DEFAULT_TM_MIN, step=0.5,
            key="tm_min", format="%.1f", help="Minimum melting temperature for RNAi primers."
        )

        st.markdown("**RNAi Primer Overhangs:**")
        # Forward Primer Overhang Selection
        fwd_overhang_choice = st.selectbox(
            "Forward Primer Overhang",
            options=PROMOTER_OPTIONS, index=0, # Default to "None"
            key="fwd_overhang", help="Select standard promoter or 'Custom'."
        )
        # Conditional input for custom forward overhang
        if fwd_overhang_choice == "Custom":
            fwd_overhang_custom = st.text_input(
                "Custom Fwd Sequence", key="fwd_custom",
                help="Enter custom 5' overhang sequence (will be lowercase)."
            ).strip()
        else:
            fwd_overhang_custom = "" # Set empty if not custom

    with col2:
        st.markdown("**RNAi Amplicon Parameters (cont.):**")
        # RNAi Amplicon Size Range (Max) - Ensure Max > Min dynamically
        dynamic_min_for_max = size_min + 10 # Ensure max is at least 10bp larger than min
        safe_default_max = max(DEFAULT_SIZE_MAX, dynamic_min_for_max) # Use default or dynamic min+10
        size_max = st.number_input(
            "Max Amplicon Size (bp)",
            min_value=dynamic_min_for_max, max_value=2000, value=safe_default_max, step=10,
            key="size_max", help="Maximum size of the desired RNAi amplicon."
        )
        # RNAi Primer Tm Range (Max) - Ensure Max > Min dynamically
        dynamic_min_tm_for_max = tm_min + 1.0 # Ensure max Tm is at least 1 degree higher
        safe_default_tm_max = max(DEFAULT_TM_MAX, dynamic_min_tm_for_max)
        tm_max = st.number_input(
            "Max Primer Tm (°C)",
            min_value=dynamic_min_tm_for_max, max_value=75.0, value=safe_default_tm_max, step=0.5,
            key="tm_max", format="%.1f", help="Maximum melting temperature for RNAi primers."
        )

        st.markdown("**RNAi Primer Overhangs (cont.):**")
        # Reverse Primer Overhang Selection
        rev_overhang_choice = st.selectbox(
            "Reverse Primer Overhang",
            options=PROMOTER_OPTIONS, index=0, # Default to "None"
            key="rev_overhang", help="Select standard promoter or 'Custom'."
        )
        # Conditional input for custom reverse overhang
        if rev_overhang_choice == "Custom":
            rev_overhang_custom = st.text_input(
                "Custom Rev Sequence", key="rev_custom",
                help="Enter custom 5' overhang sequence (will be lowercase)."
            ).strip()
        else:
            rev_overhang_custom = "" # Set empty if not custom

    # *** NEW: Checkbox for optional qPCR design ***
    st.markdown("**Options:**")
    design_qpcr = st.checkbox(
        "Design qPCR Validation Primers", value=True, key="design_qpcr",
        help="If checked, attempt to design qPCR primers outside the RNAi region."
    )

    # Calculate optimal values (used by Primer3) as midpoint of ranges
    tm_opt = (tm_min + tm_max) / 2.0
    size_opt = (size_min + size_max) // 2 # Integer division

    # Form submission button
    submitted = st.form_submit_button("Design Primers & Check Off-Targets")

# --- Processing and Output ---
# This block executes only when the form is submitted
if submitted:
    # --- Input Validation ---
    # Validate size and Tm ranges
    if size_min >= size_max or tm_min >= tm_max:
        st.error("Invalid range: Min value must be less than Max value for Size and Tm.")
        st.stop() # Halt execution if ranges are invalid

    # Check if Gene ID was entered
    if not target_gene_id_input:
        st.warning("Please enter a Target Gene ID.")
    # Check if k-mer index is available (should be loaded/built earlier)
    elif transcriptome_kmer_index is None:
        st.error("Cannot proceed: K-mer index not available (loading/building failed earlier).")
    else:
        # --- Start Processing ---
        st.subheader(f"Results for {target_gene_id_input}")

        # 1. Find the target sequence in the transcriptome FASTA (used for context/validation)
        target_full_sequence, target_gene_id_found = find_sequence(target_gene_id_input, sequences)

        if target_full_sequence and target_gene_id_found:
            st.info(f"Found transcript sequence for '{target_gene_id_found}' in transcriptome FASTA (Full Length: {len(target_full_sequence)} bp)")

            # 2. Get Feature Coordinates (CDS, UTRs, etc.) from GFF
            st.info(f"Parsing {ANNOTATION_GFF} for feature coordinates...")
            feature_coords = get_feature_genomic_coords(target_gene_id_found, ANNOTATION_GFF)

            # Validate that essential coordinates (CDS, transcript boundaries) were found
            if not feature_coords or feature_coords.get('CDS') is None or feature_coords.get('transcript_start') is None:
                 st.error(f"Could not find required CDS and/or transcript coordinates for '{target_gene_id_found}' in {ANNOTATION_GFF}. "
                          "Cannot design primers based on CDS or visualize accurately.")
                 if feature_coords:
                     st.write("Partially found coordinates:")
                     st.json(feature_coords)
                 st.stop() # Halt if essential coordinates are missing

            # Extract needed coordinates
            cds_genomic_start, cds_genomic_end = feature_coords['CDS']
            transcript_genomic_start = feature_coords['transcript_start']
            transcript_genomic_end = feature_coords['transcript_end']
            strand = feature_coords['strand']
            seqid = feature_coords['seqid'] # Chromosome/Contig ID

            # 3. Extract CDS Sequence from Genome FASTA
            st.info("Extracting CDS sequence from genome...")
            cds_sequence = None
            genome_sequences = None # Define before try block
            try:
                 # Ensure genome FASTA index exists
                 genome_fasta_index = GENOME_FASTA + '.fai'
                 if not os.path.exists(genome_fasta_index):
                     print(f"Genome FASTA index (.fai) not found for {GENOME_FASTA}. Creating...")
                     pyfaidx.Faidx(GENOME_FASTA)
                     print("Genome FASTA index created.")
                 genome_sequences = pyfaidx.Fasta(GENOME_FASTA) # Load genome fasta

                 # Use pyfaidx get_seq: 1-based coords, handles reverse complementing via rc=(strand=='-')
                 cds_sequence_obj = genome_sequences.get_seq(
                     name=seqid,
                     start=cds_genomic_start,
                     end=cds_genomic_end,
                     rc=(strand == '-') # Automatically reverse complement if strand is '-'
                 )
                 cds_sequence = cds_sequence_obj.seq # Get the sequence string
                 st.info(f"Using CDS sequence (Length: {len(cds_sequence)} bp) for primer design.")

            except Exception as e:
                 st.error(f"Failed to extract CDS sequence using coordinates {seqid}:{cds_genomic_start}-{cds_genomic_end} (Strand: {strand}): {e}")
                 if genome_sequences: genome_sequences.close() # Close file handle if opened
                 st.stop() # Halt if CDS extraction fails
            finally:
                 # Ensure genome FASTA file handle is closed
                 if genome_sequences: genome_sequences.close()


            # 4. Validate RNAi Amplicon Size against CDS Length
            if size_max > len(cds_sequence):
                st.error(f"Error: Max RNAi Amplicon Size ({size_max} bp) > CDS length ({len(cds_sequence)} bp). Please adjust parameters.");
                st.stop()
            if size_min >= len(cds_sequence):
                st.error(f"Error: Min RNAi Amplicon Size ({size_min} bp) >= CDS length ({len(cds_sequence)} bp). Please adjust parameters.");
                st.stop()

            # 5. Design RNAi Primers (using CDS sequence)
            st.markdown("---"); st.markdown("### RNAi Fragment Design (Targeting CDS)")
            with st.spinner("Designing RNAi primers using Primer3..."):
                # Pass CDS sequence to primer design function
                primer_pair_info = design_primers_with_primer3(
                    target_gene_id_found + "_CDS", cds_sequence,
                    size_min, size_opt, size_max,
                    tm_min, tm_opt, tm_max
                )

            # --- Process RNAi Primer Results ---
            qpcr_primer_pair = None # Initialize qPCR result variable

            # Check if RNAi design was successful (returned a dict with primer sequences)
            if primer_pair_info and 'PRIMER_LEFT_SEQUENCE' in primer_pair_info:
                st.success("RNAi Primer Pair Found!")

                # Format primers with selected overhangs
                fwd_oh = fwd_overhang_custom if fwd_overhang_choice == "Custom" else PROMOTER_SEQS.get(fwd_overhang_choice, "")
                rev_oh = rev_overhang_custom if rev_overhang_choice == "Custom" else PROMOTER_SEQS.get(rev_overhang_choice, "")
                fwd_formatted, rev_formatted = format_primers_with_overhangs(
                    primer_pair_info['PRIMER_LEFT_SEQUENCE'],
                    primer_pair_info['PRIMER_RIGHT_SEQUENCE'],
                    fwd_oh, rev_oh
                )
                # Display RNAi primer sequences
                st.markdown("**RNAi Primer Pair 1 (Best Ranked):**")
                st.code(f"Forward: {fwd_formatted}\nReverse: {rev_formatted}", language=None)

                # Calculate and display amplicon sizes and primer stats
                bio_amplicon_size = primer_pair_info['PRIMER_PAIR_PRODUCT_SIZE']
                total_amplicon_size = bio_amplicon_size + len(fwd_oh) + len(rev_oh)
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Biological Size", f"{bio_amplicon_size} bp", help="Size of region amplified between primer binding sites (within CDS).")
                with col2:
                     if total_amplicon_size != bio_amplicon_size: # Only show if overhangs were added
                         st.metric("Total Size w/ Overhangs", f"{total_amplicon_size} bp")
                with col3: st.metric("Fwd Tm", f"{primer_pair_info['PRIMER_LEFT_TM']:.1f} °C"); st.metric("Fwd GC%", f"{primer_pair_info['PRIMER_LEFT_GC_PERCENT']:.1f}%")
                with col4: st.metric("Rev Tm", f"{primer_pair_info['PRIMER_RIGHT_TM']:.1f} °C"); st.metric("Rev GC%", f"{primer_pair_info['PRIMER_RIGHT_GC_PERCENT']:.1f}%")

                # Get the sequence of the RNAi amplicon (biological part from CDS)
                rnai_amplicon_seq_cds = get_amplicon_sequence(cds_sequence, primer_pair_info)

                # 6. Design qPCR Primers (Conditional based on checkbox)
                # Check if the user requested qPCR primer design
                if design_qpcr:
                    st.markdown("---"); st.markdown("### qPCR Validation Primer Design (Targeting CDS)")
                    # Get RNAi amplicon position relative to CDS template
                    rnai_fwd_start_0based_cds = primer_pair_info.get('PRIMER_LEFT_POSITION')
                    rnai_amplicon_size_for_exclude = primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE') # Use reported product size

                    if rnai_fwd_start_0based_cds is not None and rnai_amplicon_size_for_exclude is not None:
                        # Calculate exclusion zone (1-based start, length) relative to CDS template
                        exclude_region_start_1based_cds = rnai_fwd_start_0based_cds + 1
                        exclude_region_length = rnai_amplicon_size_for_exclude
                        exclude_list = [[exclude_region_start_1based_cds, exclude_region_length]]

                        with st.spinner("Designing qPCR primers (Tm ~58°C, Size 80-110bp, outside RNAi region)..."):
                            # Call qPCR primer design function
                            qpcr_design_result = design_qpcr_primers(
                                target_gene_id_found + "_CDS",
                                cds_sequence,
                                exclude_region_list=exclude_list
                            )

                        # Check if qPCR design returned a potential pair
                        if qpcr_design_result and 'PRIMER_LEFT_SEQUENCE' in qpcr_design_result:
                            # Successfully found a pair, now check for overlap with RNAi region
                            qpcr_start_rel = qpcr_design_result.get('PRIMER_LEFT_POSITION')
                            qpcr_size = qpcr_design_result.get('PRIMER_PAIR_PRODUCT_SIZE')

                            # Define RNAi region relative to CDS (0-based, end-exclusive)
                            rnai_start_rel = rnai_fwd_start_0based_cds
                            rnai_end_rel = rnai_start_rel + rnai_amplicon_size_for_exclude

                            # Define qPCR region relative to CDS (0-based, end-exclusive)
                            qpcr_end_rel = qpcr_start_rel + qpcr_size

                            # Check for overlap: max(starts) < min(ends)
                            overlap_detected = max(rnai_start_rel, qpcr_start_rel) < min(rnai_end_rel, qpcr_end_rel)

                            if overlap_detected:
                                # --- Overlap Detected: Treat as Failure (Option C) ---
                                st.warning(f"qPCR Primer Design Warning: The only suitable qPCR primers found "
                                         f"(Region: {qpcr_start_rel}-{qpcr_end_rel-1} relative to CDS) overlap with the RNAi region "
                                         f"({rnai_start_rel}-{rnai_end_rel-1} relative to CDS). "
                                         f"This likely means the combination of qPCR constraints and the RNAi exclusion zone "
                                         f"was too restrictive. Try reducing the 'Max Amplicon Size' for the RNAi fragment.")
                                qpcr_primer_pair = None # Nullify the result so it's not displayed/visualized
                            else:
                                # --- No Overlap: Success ---
                                qpcr_primer_pair = qpcr_design_result # Store the valid result
                                st.success("qPCR Primer Pair Found!")
                                st.markdown("**qPCR Primer Pair 1:**")
                                # Format qPCR primers (no overhangs)
                                qpcr_fwd_formatted, qpcr_rev_formatted = format_primers_with_overhangs(
                                    qpcr_primer_pair['PRIMER_LEFT_SEQUENCE'],
                                    qpcr_primer_pair['PRIMER_RIGHT_SEQUENCE'], "", ""
                                )
                                st.code(f"Forward: {qpcr_fwd_formatted}\nReverse: {qpcr_rev_formatted}", language=None)
                                # Display qPCR stats
                                col1q, col2q, col3q = st.columns(3)
                                with col1q: st.metric("qPCR Amplicon Size", f"{qpcr_primer_pair['PRIMER_PAIR_PRODUCT_SIZE']} bp")
                                with col2q: st.metric("qPCR Fwd Tm", f"{qpcr_primer_pair['PRIMER_LEFT_TM']:.1f} °C"); st.metric("qPCR Fwd GC%", f"{qpcr_primer_pair['PRIMER_LEFT_GC_PERCENT']:.1f}%")
                                with col3q: st.metric("qPCR Rev Tm", f"{qpcr_primer_pair['PRIMER_RIGHT_TM']:.1f} °C"); st.metric("qPCR Rev GC%", f"{qpcr_primer_pair['PRIMER_RIGHT_GC_PERCENT']:.1f}%")

                        else:
                            # qPCR design failed entirely or returned no valid pairs
                            st.warning("Could not find suitable qPCR primers outside the RNAi region within the CDS using specified parameters.")
                            # Optionally display explanation from raw results if available
                            if qpcr_design_result and 'RAW_RESULTS' in qpcr_design_result:
                                 explain = qpcr_design_result['RAW_RESULTS'].get('PRIMER_EXPLAIN', 'No explanation provided.')
                                 st.info(f"Primer3 qPCR explanation: {explain}")

                    else:
                        # Could not determine RNAi region to exclude
                        st.warning("Could not define RNAi region to exclude for qPCR primer design.")
                # End of 'if design_qpcr:' block

                # 7. Generate and Display Visualization
                st.markdown("---"); st.markdown("### Gene Model and Amplicon Positions")
                svg_viz = None # Initialize SVG variable
                # Check again if feature coordinates are valid before proceeding
                if feature_coords and transcript_genomic_start is not None and transcript_genomic_end is not None:

                    # --- Calculate RNAi Amplicon Genomic Coordinates for Viz ---
                    rnai_viz_info = None
                    # **** CORRECTED LOGIC START ****
                    # Check if RNAi design succeeded AND produced necessary info for visualization
                    if primer_pair_info and primer_pair_info.get('PRIMER_LEFT_POSITION') is not None and primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE') is not None:
                        # Retrieve the necessary values directly from the successful RNAi result
                        rnai_fwd_start_0based_cds_viz = primer_pair_info['PRIMER_LEFT_POSITION'] # Use direct access [] because we checked keys
                        rnai_amplicon_size_viz = primer_pair_info['PRIMER_PAIR_PRODUCT_SIZE']

                        # Ensure CDS coordinates are also valid
                        if cds_genomic_start is not None and cds_genomic_end is not None:
                            rnai_genomic_start_viz = None
                            if strand == '+':
                                rnai_genomic_start_viz = cds_genomic_start + rnai_fwd_start_0based_cds_viz
                            else: # Reverse strand '-'
                                # Genomic start for reverse strand amplicon calculation
                                rnai_genomic_start_viz = cds_genomic_end - (rnai_fwd_start_0based_cds_viz + rnai_amplicon_size_viz) + 1

                            if rnai_genomic_start_viz is not None:
                                 # Assign the dictionary only if calculation was successful
                                rnai_viz_info = {'genomic_start': rnai_genomic_start_viz, 'size': rnai_amplicon_size_viz}
                    # **** CORRECTED LOGIC END ****

                    # --- Calculate qPCR Amplicon Genomic Coordinates for Viz ---
                    # IMPORTANT: Only calculate if qpcr_primer_pair is not None (i.e., design requested AND was successful AND did not overlap)
                    qpcr_viz_info = None
                    if qpcr_primer_pair and qpcr_primer_pair.get('PRIMER_LEFT_POSITION') is not None:
                        qpcr_rel_start_cds = qpcr_primer_pair['PRIMER_LEFT_POSITION']
                        qpcr_amplicon_size = qpcr_primer_pair.get('PRIMER_PAIR_PRODUCT_SIZE')
                        if qpcr_amplicon_size is not None and cds_genomic_start is not None and cds_genomic_end is not None:
                            qpcr_genomic_start_viz = None
                            if strand == '+':
                                qpcr_genomic_start_viz = cds_genomic_start + qpcr_rel_start_cds
                            else: # Reverse strand '-'
                                qpcr_genomic_start_viz = cds_genomic_end - (qpcr_rel_start_cds + qpcr_amplicon_size) + 1
                            if qpcr_genomic_start_viz is not None:
                                qpcr_viz_info = {'genomic_start': qpcr_genomic_start_viz, 'size': qpcr_amplicon_size}

                    # --- Generate the SVG ---
                    # Pass the potentially None rnai_viz_info and qpcr_viz_info
                    svg_viz = generate_gene_model_svg(
                        target_gene_id_found,
                        transcript_genomic_start,
                        transcript_genomic_end,
                        feature_coords,
                        rnai_viz_info,  # Use the calculated value (might be None)
                        qpcr_viz_info # Will be None if qPCR not designed or overlap check failed
                    )
                    # Display SVG
                    st.markdown(svg_viz, unsafe_allow_html=True)
                    # Display Legend
                    st.markdown('<span style="font-size: small;">LEGEND: <span style="display: inline-block; width: 10px; height: 10px; background-color: #e0e0e0; margin-left: 5px; margin-right: 3px; vertical-align: middle;"></span> UTR <span style="display: inline-block; width: 10px; height: 10px; background-color: #bdbdbd; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> CDS | <span style="display: inline-block; width: 10px; height: 8px; background-color: #4dd0e1; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> RNAi Amplicon <span style="display: inline-block; width: 10px; height: 8px; background-color: #d9534f; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> qPCR Amplicon</span>', unsafe_allow_html=True)
                    # Add Download Button
                    if svg_viz:
                        st.download_button(
                            label="Download SVG", data=svg_viz,
                            file_name=f"{target_gene_id_found}_model.svg", mime="image/svg+xml"
                        )
                else:
                    # Warning if visualization couldn't be generated due to missing GFF coords
                    st.warning(f"Could not retrieve sufficient coordinates from {ANNOTATION_GFF} to generate visualization.")

                # 8. Perform Off-Target Check
                st.markdown("---")
                if rnai_amplicon_seq_cds: # Check if RNAi amplicon sequence was extracted
                    st.markdown(f"**Potential Off-Target Check ({KMER_LENGTH}-mer based for RNAi Amplicon):**")
                    st.markdown(f"*(Shows transcripts sharing >=1 perfect or 1-mismatch {KMER_LENGTH}-mer with the RNAi amplicon)*")
                    with st.spinner(f"Checking {KMER_LENGTH}-mer off-targets..."):
                        # Call k-mer checking function
                        off_target_df = check_off_targets_kmer(
                            rnai_amplicon_seq_cds,
                            transcriptome_kmer_index,
                            target_gene_id_found # Pass the found gene ID
                        )
                    # Display off-target results
                    if not off_target_df.empty:
                        st.warning(f"Found {len(off_target_df)} potential off-target transcript(s):")
                        # Add clickable link to TriTrypDB
                        base_url = "https://tritrypdb.org/tritrypdb/app/record/gene/"
                        # Ensure 'Off-Target ID' is string type before concatenation
                        off_target_df['URL'] = base_url + off_target_df['Off-Target ID'].astype(str)
                        # Display DataFrame with configured columns
                        st.dataframe(
                            off_target_df,
                            column_config={
                                "Off-Target ID": "Off-Target ID",
                                "URL": st.column_config.LinkColumn(
                                    "Gene Page Link",
                                    help="Click to open TriTrypDB gene page in new tab",
                                    display_text="Open Link"
                                ),
                                "Perfect Kmer Matches": st.column_config.NumberColumn(format="%d"),
                                "1 Mismatch Kmer Hits": st.column_config.NumberColumn(format="%d")
                            },
                            column_order=("Off-Target ID", "URL", "Perfect Kmer Matches", "1 Mismatch Kmer Hits"),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        # Message if no off-targets found
                        st.info(f"No other transcripts found sharing perfect or 1-mismatch {KMER_LENGTH}-mers with the RNAi amplicon.")
                else:
                    # Warning if RNAi amplicon sequence wasn't available for checking
                    st.warning("Could not extract RNAi amplicon sequence to perform off-target check.")

            # else block for: if primer_pair_info and 'PRIMER_LEFT_SEQUENCE' in primer_pair_info:
            else:
                # RNAi primer design failed, message already shown by design_primers_with_primer3
                # The corrected visualization block above handles the case where primer_pair_info is bad,
                # preventing errors later. We just show the raw results explanation here if available.
                if primer_pair_info and 'RAW_RESULTS' in primer_pair_info:
                    st.info("Raw results from Primer3 (RNAi design failure):")
                    # Avoid showing the full sequence template in the json dump for brevity
                    raw_results_subset = {k: v for k, v in primer_pair_info['RAW_RESULTS'].items() if 'SEQUENCE' not in k}
                    st.json(raw_results_subset, expanded=False)


        # else block for: if target_full_sequence and target_gene_id_found:
        else:
            # Gene ID not found in transcriptome FASTA
            st.error(f"Gene ID '{target_gene_id_input}' not found in the transcriptome FASTA file: {TRANSCRIPTOME_FASTA}")

# --- Footer Info ---
# Display file paths and parameters used at the bottom of the app
st.markdown("---")
st.markdown(f"**Files Used:**")
st.markdown(f"- Transcriptome (for off-target index): `{TRANSCRIPTOME_FASTA}`")
st.markdown(f"- Annotation GFF (for coordinates/visualization): `{ANNOTATION_GFF}`")
st.markdown(f"- Genome (for sequence extraction): `{GENOME_FASTA}`")
st.markdown(f"**Parameters:**")
st.markdown(f"- K-mer length for off-target check: `{KMER_LENGTH}`")
st.markdown(f"- K-mer index cache file: `{KMER_INDEX_PICKLE_FILE}`")

# End of File
