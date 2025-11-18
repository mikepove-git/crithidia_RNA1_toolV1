# -*- coding: utf-8 -*-
"""
Crithidia_RNAi_tool.py - Cloud-Ready Version

Streamlit application to design RNAi primers targeting CDS for Crithidia fasciculata,
check for potential off-targets using k-mer analysis, format primers
with overhangs, optionally design flanking qPCR primers, and visualize results on gene model.

Version 24.0: Cloud-ready with file upload capability. Fixed FileNotFoundError bug.
              Allows users to upload data files via Streamlit interface.
              Works in browser as if running locally.

Requires: streamlit, primer3-py, pyfaidx, pandas, pickle
"""

# --- Standard Library Imports ---
import os
import sys
import re
from collections import defaultdict, namedtuple
import time
import textwrap
import pickle
import traceback
import tempfile
import io

# --- Third-Party Library Imports ---
import streamlit as st
import primer3
import pyfaidx
import pandas as pd

# --- Constants and Configuration ---

# Default Primer3 settings for RNAi primers (adjustable via UI)
DEFAULT_TM_MIN = 58.0
DEFAULT_TM_OPT = 60.0
DEFAULT_TM_MAX = 62.0
DEFAULT_SIZE_MIN = 150
DEFAULT_SIZE_OPT = 250
DEFAULT_SIZE_MAX = 500

# K-mer analysis settings
KMER_LENGTH = 21
MAX_MISMATCHES = 1
BASES = ['A', 'C', 'G', 'T']

# Overhang sequences and UI options
PROMOTER_SEQS = {
    "T7": "taatacgactcactataggg",
    "SP6": "atttaggtgacactatag",
    "T3": "aattaaccctcactaaaggg"
}
PROMOTER_OPTIONS = ["None", "T7", "SP6", "T3", "Custom"]

# --- Helper Functions ---

def generate_kmers(sequence, k):
    """Generates all overlapping k-mers from a given sequence."""
    sequence = sequence.upper().replace('U', 'T')
    if len(sequence) < k:
        return
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(base in BASES for base in kmer):
            yield kmer

def generate_1_mismatch_neighbors(kmer):
    """Generates all sequences with exactly one base mismatch."""
    k = len(kmer)
    for i in range(k):
        original_base = kmer[i]
        for base in BASES:
            if base != original_base:
                neighbor = kmer[:i] + base + kmer[i+1:]
                yield neighbor

# --- Backend Functions ---

def save_uploaded_file_to_temp(uploaded_file):
    """Saves an uploaded file to a temporary location and returns the path."""
    try:
        # Create a temporary file
        suffix = os.path.splitext(uploaded_file.name)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

@st.cache_resource
def load_sequences(fasta_file_path):
    """Loads sequences from a FASTA file using pyfaidx."""
    start_time = time.time()
    print(f"Attempting to load sequences from {fasta_file_path}...")
    try:
        if not os.path.exists(fasta_file_path):
            raise FileNotFoundError(f"File not found: {fasta_file_path}")

        index_file = fasta_file_path + '.fai'
        if not os.path.exists(index_file):
            print(f"FASTA index (.fai) not found. Creating...")
            pyfaidx.Faidx(fasta_file_path)
            print("FASTA index created.")

        sequences = pyfaidx.Fasta(fasta_file_path, sequence_always_upper=True)
        num_seqs = len(sequences.keys())
        print(f"Loaded {num_seqs} sequences in {time.time() - start_time:.2f} sec.")
        return sequences

    except FileNotFoundError as fnf_err:
        st.error(f"{fnf_err}")
        return None
    except Exception as e:
        print(f"Error loading FASTA file: {e}")
        st.error(f"An error occurred loading FASTA file. Check console for details.")
        return None

@st.cache_resource
def build_transcriptome_kmer_index(_sequences, k):
    """Builds a k-mer index from transcriptome sequences."""
    if not _sequences:
        st.error("Cannot build k-mer index: Transcriptome sequences not loaded.")
        return None

    start_time = time.time()
    print(f"Building {k}-mer index from {len(_sequences.keys())} transcripts...")
    hit_kmer_index = defaultdict(set)
    total_kmers_indexed = 0
    num_hits = len(_sequences.keys())
    seq_ids = list(_sequences.keys())

    progress_bar = st.progress(0)
    progress_text = st.empty()
    progress_text.text("Starting k-mer index build...")

    for i, seq_id in enumerate(seq_ids):
        base_id = seq_id.split('|')[0].strip()
        hit_seq = str(_sequences[seq_id])
        kmers_in_hit = 0
        for kmer in generate_kmers(hit_seq, k):
            hit_kmer_index[kmer].add(base_id)
            kmers_in_hit += 1
        total_kmers_indexed += kmers_in_hit

        if (i + 1) % (max(1, num_hits // 100)) == 0 or (i + 1) == num_hits:
            progress_percent = (i + 1) / num_hits
            progress_bar.progress(progress_percent)
            progress_text.text(f"Indexing transcript {i+1}/{num_hits}")

    progress_bar.empty()
    progress_text.empty()

    index_size = len(hit_kmer_index)
    print(f"K-mer index built in {time.time() - start_time:.2f} seconds.")
    print(f"Index contains {index_size} unique {k}-mers from {total_kmers_indexed} total k-mers found.")

    return hit_kmer_index

def find_sequence(gene_id, sequences):
    """Retrieves a sequence string for a given gene ID."""
    if sequences is None or not gene_id:
        return None, None

    gene_id_cleaned = gene_id.strip()

    if gene_id_cleaned in sequences:
        return str(sequences[gene_id_cleaned]), gene_id_cleaned

    base_gene_id = gene_id_cleaned.split('|')[0].strip()
    if base_gene_id in sequences:
        return str(sequences[base_gene_id]), base_gene_id

    header_variation = f"{base_gene_id} |"
    if header_variation in sequences:
        return str(sequences[header_variation]), base_gene_id

    return None, None

def design_primers_with_primer3(sequence_id, sequence_template, size_min, size_opt, size_max, tm_min, tm_opt, tm_max):
    """Designs primer pairs using primer3-py bindings."""
    sequence_template = sequence_template.upper().replace('U', 'T')

    seq_args = {
        'SEQUENCE_ID': sequence_id,
        'SEQUENCE_TEMPLATE': sequence_template
    }

    global_args = {
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_PRODUCT_SIZE_RANGE': [[size_min, size_max]],
        'PRIMER_OPT_TM': tm_opt,
        'PRIMER_MIN_TM': tm_min,
        'PRIMER_MAX_TM': tm_max,
        'PRIMER_MIN_GC': 40.0,
        'PRIMER_MAX_GC': 60.0,
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_INTERNAL_OLIGO': 0,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_NUM_RETURN': 5,
        'PRIMER_EXPLAIN_FLAG': 1
    }

    try:
        raw_results = primer3.bindings.designPrimers(seq_args, global_args)
        num_returned = raw_results.get('PRIMER_PAIR_NUM_RETURNED', 0)

        if num_returned > 0:
            pair_info = {
                'PRIMER_LEFT_SEQUENCE': raw_results.get('PRIMER_LEFT_0_SEQUENCE'),
                'PRIMER_RIGHT_SEQUENCE': raw_results.get('PRIMER_RIGHT_0_SEQUENCE'),
                'PRIMER_LEFT_TM': raw_results.get('PRIMER_LEFT_0_TM'),
                'PRIMER_RIGHT_TM': raw_results.get('PRIMER_RIGHT_0_TM'),
                'PRIMER_LEFT_GC_PERCENT': raw_results.get('PRIMER_LEFT_0_GC_PERCENT'),
                'PRIMER_RIGHT_GC_PERCENT': raw_results.get('PRIMER_RIGHT_0_GC_PERCENT'),
                'PRIMER_PAIR_PRODUCT_SIZE': raw_results.get('PRIMER_PAIR_0_PRODUCT_SIZE'),
                'PRIMER_LEFT_0': raw_results.get('PRIMER_LEFT_0'),
                'PRIMER_RIGHT_0': raw_results.get('PRIMER_RIGHT_0')
            }
            if all(pair_info.get(k) is not None for k in ['PRIMER_LEFT_SEQUENCE', 'PRIMER_RIGHT_SEQUENCE', 'PRIMER_PAIR_PRODUCT_SIZE', 'PRIMER_LEFT_0', 'PRIMER_RIGHT_0']):
                pair_info['PRIMER_LEFT_POSITION'] = pair_info['PRIMER_LEFT_0'][0]
                pair_info['PRIMER_RIGHT_POSITION'] = pair_info['PRIMER_RIGHT_0'][0]
                pair_info['RAW_RESULTS'] = raw_results
                return pair_info
            else:
                st.warning(f"Primer3 returned pairs, but essential info missing for pair 0.")
                return {'RAW_RESULTS': raw_results}
        else:
            explain = raw_results.get('PRIMER_EXPLAIN', 'No explanation provided.')
            st.warning(f"Primer3 did not return any suitable RNAi primer pairs. Explanation: {explain}")
            return {'RAW_RESULTS': raw_results}
    except Exception as e:
        st.error(f"Error during RNAi primer design call: {e}")
        return None

def design_qpcr_primers(sequence_id, sequence_template, exclude_region_list, global_args_override=None):
    """Designs qPCR primer pairs using primer3-py."""
    sequence_template = sequence_template.upper().replace('U', 'T')

    seq_args = {
        'SEQUENCE_ID': sequence_id + "_qPCR",
        'SEQUENCE_TEMPLATE': sequence_template
    }

    global_args = {
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_PRODUCT_SIZE_RANGE': [[80, 110]],
        'PRIMER_OPT_TM': 58.0,
        'PRIMER_MIN_TM': 57.0,
        'PRIMER_MAX_TM': 59.0,
        'PRIMER_MIN_GC': 40.0,
        'PRIMER_MAX_GC': 60.0,
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_INTERNAL_OLIGO': 0,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_NUM_RETURN': 5,
        'PRIMER_EXPLAIN_FLAG': 1,
        'PRIMER_EXCLUDE_REGION': exclude_region_list
    }

    if global_args_override:
        global_args.update(global_args_override)

    try:
        raw_results = primer3.bindings.designPrimers(seq_args, global_args)
        num_returned = raw_results.get('PRIMER_PAIR_NUM_RETURNED', 0)

        if num_returned > 0:
            pair_info = {
                'PRIMER_LEFT_SEQUENCE': raw_results.get('PRIMER_LEFT_0_SEQUENCE'),
                'PRIMER_RIGHT_SEQUENCE': raw_results.get('PRIMER_RIGHT_0_SEQUENCE'),
                'PRIMER_LEFT_TM': raw_results.get('PRIMER_LEFT_0_TM'),
                'PRIMER_RIGHT_TM': raw_results.get('PRIMER_RIGHT_0_TM'),
                'PRIMER_LEFT_GC_PERCENT': raw_results.get('PRIMER_LEFT_0_GC_PERCENT'),
                'PRIMER_RIGHT_GC_PERCENT': raw_results.get('PRIMER_RIGHT_0_GC_PERCENT'),
                'PRIMER_PAIR_PRODUCT_SIZE': raw_results.get('PRIMER_PAIR_0_PRODUCT_SIZE'),
                'PRIMER_LEFT_0': raw_results.get('PRIMER_LEFT_0')
            }
            if all(pair_info.get(k) is not None for k in ['PRIMER_LEFT_SEQUENCE', 'PRIMER_RIGHT_SEQUENCE', 'PRIMER_PAIR_PRODUCT_SIZE', 'PRIMER_LEFT_0']):
                pair_info['PRIMER_LEFT_POSITION'] = pair_info['PRIMER_LEFT_0'][0]
                pair_info['RAW_RESULTS'] = raw_results
                return pair_info
            else:
                return {'RAW_RESULTS': raw_results}
        else:
            return {'RAW_RESULTS': raw_results}
    except Exception as e:
        st.error(f"Error during qPCR primer design call: {e}")
        return None

def format_primers_with_overhangs(fwd_seq, rev_seq, fwd_overhang, rev_overhang):
    """Concatenates overhang sequences with primer sequences."""
    formatted_fwd = f"{fwd_overhang.lower()}{fwd_seq.upper()}"
    formatted_rev = f"{rev_overhang.lower()}{rev_seq.upper()}"
    return formatted_fwd, formatted_rev

def get_amplicon_sequence(sequence_template, primer_pair_info, cds_rel_start=0, cds_rel_end=None):
    """Extracts the biological amplicon sequence."""
    sequence_template = sequence_template.upper().replace('U', 'T')
    if cds_rel_end is None:
        cds_rel_end = len(sequence_template)

    try:
        fwd_primer_def = primer_pair_info.get('PRIMER_LEFT_0')
        product_size = primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE')

        if fwd_primer_def and product_size is not None:
            fwd_start_0based = fwd_primer_def[0]
            product_size = int(product_size)
            amplicon_end_0based = fwd_start_0based + product_size

            if not (0 <= fwd_start_0based < amplicon_end_0based <= len(sequence_template)):
                st.warning(f"Calculated amplicon coordinates [{fwd_start_0based}-{amplicon_end_0based}) "
                          f"relative to template are out of bounds (template length: {len(sequence_template)}).")
                return None

            return sequence_template[fwd_start_0based:amplicon_end_0based]
        else:
            st.warning("Could not determine amplicon boundaries from primer info dictionary.")
            return None
    except Exception as e:
        st.error(f"Error extracting amplicon sequence: {e}")
        return None

def check_off_targets_kmer(amplicon_sequence, hit_kmer_index, target_gene_id):
    """Checks for potential off-targets using k-mer analysis."""
    if not amplicon_sequence or not hit_kmer_index:
        return pd.DataFrame()

    start_time = time.time()
    results = defaultdict(lambda: [0, 0])
    target_base_id = target_gene_id.split('|')[0].strip()

    amplicon_kmers = set(generate_kmers(amplicon_sequence, KMER_LENGTH))
    num_amplicon_kmers = len(amplicon_kmers)
    processed_target_kmers = 0

    progress_bar_kmer = st.progress(0)
    progress_text_kmer = st.empty()
    progress_text_kmer.text("Starting off-target k-mer check...")

    for kmer in amplicon_kmers:
        processed_target_kmers += 1

        if kmer in hit_kmer_index:
            for hit_name in hit_kmer_index[kmer]:
                if hit_name != target_base_id:
                    results[hit_name][0] += 1

        for neighbor in generate_1_mismatch_neighbors(kmer):
            if neighbor in hit_kmer_index:
                for hit_name in hit_kmer_index[neighbor]:
                    if hit_name != target_base_id:
                        results[hit_name][1] += 1

        if num_amplicon_kmers > 0:
            progress_percent = processed_target_kmers / num_amplicon_kmers
            if processed_target_kmers % 10 == 0 or processed_target_kmers == num_amplicon_kmers:
                progress_bar_kmer.progress(progress_percent)
                progress_text_kmer.text(f"Checked {processed_target_kmers}/{num_amplicon_kmers} amplicon k-mers")

    progress_bar_kmer.empty()
    progress_text_kmer.empty()

    output_data = []
    for hit_name, counts in results.items():
        if sum(counts) > 0:
            output_data.append({
                'Off-Target ID': hit_name,
                'Perfect Kmer Matches': counts[0],
                '1 Mismatch Kmer Hits': counts[1]
            })

    off_target_df = pd.DataFrame(output_data)

    if not off_target_df.empty:
        off_target_df = off_target_df.sort_values(
            by=['Perfect Kmer Matches', '1 Mismatch Kmer Hits'],
            ascending=[False, False]
        )

    print(f"Off-target check completed in {time.time() - start_time:.2f} seconds.")
    return off_target_df

@st.cache_data
def get_feature_genomic_coords(_gene_id, annotation_gff_path):
    """Parses the annotation GFF file to extract genomic coordinates."""
    coords = {'5UTR': None, 'CDS': None, '3UTR': None, 'transcript_start': None, 'transcript_end': None, 'strand': None, 'seqid': None}
    min_start = float('inf')
    max_end = 0
    cds_coords = []
    utr5_coords = []
    utr3_coords = []
    found_strand = None
    found_seqid = None
    gene_found = False
    target_mrna_ids = set()

    try:
        if not os.path.exists(annotation_gff_path):
            st.error(f"Annotation GFF file not found: {annotation_gff_path}")
            return None

        with open(annotation_gff_path, 'r') as f_gff_pre:
            for line in f_gff_pre:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) != 9:
                    continue
                featuretype = parts[2]
                attributes_str = parts[8]
                if featuretype == 'mRNA':
                    parent_match = re.search(r'Parent=([^;]+)', attributes_str)
                    id_match = re.search(r'ID=([^;]+)', attributes_str)
                    if parent_match and id_match and parent_match.group(1) == _gene_id:
                        target_mrna_ids.add(id_match.group(1))

        with open(annotation_gff_path, 'r') as f_gff:
            for line_num, line in enumerate(f_gff, 1):
                if line.startswith('#'):
                    continue
                line_stripped = line.strip()
                parts = line_stripped.split('\t')
                if len(parts) != 9:
                    continue
                seqid, source, featuretype, start_str, end_str, score, strand, frame, attributes_str = parts

                parent_match = re.search(r'Parent=([^;]+)', attributes_str)
                id_match = re.search(r'ID=([^;]+)', attributes_str)
                current_id = id_match.group(1) if id_match else None
                parent_id = parent_match.group(1) if parent_match else None

                is_relevant = False
                if current_id == _gene_id and featuretype in ['protein_coding_gene', 'gene']:
                    is_relevant = True
                elif parent_id == _gene_id and featuretype in ['mRNA', 'five_prime_utr', 'three_prime_utr']:
                    is_relevant = True
                elif parent_id in target_mrna_ids and featuretype == 'CDS':
                    is_relevant = True
                elif parent_id == _gene_id and featuretype in ['CDS', 'five_prime_utr', 'three_prime_utr']:
                    is_relevant = True

                if is_relevant:
                    gene_found = True
                    try:
                        start_coord = int(start_str)
                        end_coord = int(end_str)

                        if found_strand is None:
                            found_strand = strand
                        if found_seqid is None:
                            found_seqid = seqid
                        elif found_strand != strand or found_seqid != seqid:
                            continue

                        min_start = min(min_start, start_coord)
                        max_end = max(max_end, end_coord)

                        if featuretype == 'CDS':
                            cds_coords.append((start_coord, end_coord))
                        elif featuretype == 'five_prime_utr':
                            utr5_coords.append((start_coord, end_coord))
                        elif featuretype == 'three_prime_utr':
                            utr3_coords.append((start_coord, end_coord))

                    except ValueError:
                        continue
                    except Exception as inner_e:
                        print(f"Error processing GFF line {line_num} for {_gene_id}: {inner_e}")
                        continue

        if gene_found and found_strand:
            coords['strand'] = found_strand
            coords['seqid'] = found_seqid
            coords['transcript_start'] = min_start if min_start != float('inf') else None
            coords['transcript_end'] = max_end if max_end != 0 else None

            if cds_coords:
                coords['CDS'] = (min(s for s, e in cds_coords), max(e for s, e in cds_coords))
            if utr5_coords:
                coords['5UTR'] = (min(s for s, e in utr5_coords), max(e for s, e in utr5_coords))
            if utr3_coords:
                coords['3UTR'] = (min(s for s, e in utr3_coords), max(e for s, e in utr3_coords))

            if coords['transcript_start'] and coords['transcript_end'] and coords['transcript_start'] <= coords['transcript_end']:
                return coords
            else:
                return None
        else:
            return None

    except Exception as e:
        st.error(f"Error parsing annotation GFF {annotation_gff_path}: {e}")
        traceback.print_exc()
        return None

def generate_gene_model_svg(gene_id, transcript_genomic_start, transcript_genomic_end, feature_coords, rnai_info, qpcr_info, padding=10):
    """Generates an SVG string visualizing the gene model."""
    svg_height = 70
    track_height = 10
    amplicon_height = 8
    gene_label_y = 15
    amplicon_label_y = 30
    cds_track_y = 42
    amplicon_track_y = cds_track_y + (track_height - amplicon_height) / 2
    axis_label_y = cds_track_y + track_height + 10

    svg_width_px = 600
    draw_width = svg_width_px - 2 * padding

    svg_elements = [
        f'<svg width="{svg_width_px}" height="{svg_height}" style="background-color:#f8f9fa; border-radius: 3px; margin-top: 10px; margin-bottom: 10px;">',
        '<style>'
        '.small { font: italic 8px sans-serif; fill: #555; } '
        '.gene-label { font: bold 10px sans-serif; fill: #333; } '
        '.amp-label { font: 8px sans-serif; fill: #333; }'
        '</style>'
    ]

    total_genomic_span = transcript_genomic_end - transcript_genomic_start + 1
    if total_genomic_span <= 0:
        print("[Error] Invalid transcript genomic span for SVG.")
        return ""

    def get_svg_pixels(feat_genomic_start, feat_genomic_end):
        if feat_genomic_start is None or feat_genomic_end is None:
            return None, None

        rel_start = feat_genomic_start - transcript_genomic_start
        feat_len = feat_genomic_end - feat_genomic_start + 1

        x_px = padding + max(0, (rel_start / total_genomic_span) * draw_width)
        w_px = max(1, (feat_len / total_genomic_span) * draw_width)

        if x_px + w_px > svg_width_px - padding:
            w_px = (svg_width_px - padding) - x_px

        return x_px, w_px

    svg_elements.append(f'<line x1="{padding}" y1="{cds_track_y + track_height/2}" '
                       f'x2="{svg_width_px - padding}" y2="{cds_track_y + track_height/2}" '
                       'stroke="#bbb" stroke-width="1" />')

    utr5_coords_g = feature_coords.get('5UTR')
    cds_coords_g = feature_coords.get('CDS')
    utr3_coords_g = feature_coords.get('3UTR')

    utr5_x, utr5_w = get_svg_pixels(utr5_coords_g[0], utr5_coords_g[1]) if utr5_coords_g else (None, None)
    cds_x, cds_w = get_svg_pixels(cds_coords_g[0], cds_coords_g[1]) if cds_coords_g else (None, None)
    utr3_x, utr3_w = get_svg_pixels(utr3_coords_g[0], utr3_coords_g[1]) if utr3_coords_g else (None, None)

    if utr5_x is not None:
        svg_elements.append(f'<rect x="{utr5_x:.1f}" y="{cds_track_y}" width="{utr5_w:.1f}" height="{track_height}" fill="#e0e0e0">'
                           f'<title>5UTR ({utr5_coords_g[1]-utr5_coords_g[0]+1} bp)</title></rect>')
    if utr3_x is not None:
        svg_elements.append(f'<rect x="{utr3_x:.1f}" y="{cds_track_y}" width="{utr3_w:.1f}" height="{track_height}" fill="#e0e0e0">'
                           f'<title>3UTR ({utr3_coords_g[1]-utr3_coords_g[0]+1} bp)</title></rect>')
    if cds_x is not None:
        svg_elements.append(f'<rect x="{cds_x:.1f}" y="{cds_track_y}" width="{cds_w:.1f}" height="{track_height}" fill="#bdbdbd">'
                           f'<title>CDS ({cds_coords_g[1]-cds_coords_g[0]+1} bp)</title></rect>')

    rnai_label_x_center = None

    if rnai_info:
        rnai_genomic_start = rnai_info.get('genomic_start')
        rnai_size = rnai_info.get('size')
        if rnai_genomic_start is not None and rnai_size is not None and rnai_size > 0:
            rnai_x, rnai_w = get_svg_pixels(rnai_genomic_start, rnai_genomic_start + rnai_size - 1)
            if rnai_x is not None:
                svg_elements.append(f'<rect x="{rnai_x:.1f}" y="{amplicon_track_y}" width="{rnai_w:.1f}" '
                                   f'height="{amplicon_height}" fill="#4dd0e1" rx="1" ry="1" style="opacity:0.9;">'
                                   f'<title>RNAi Amplicon ({rnai_size} bp)</title></rect>')
                rnai_label_x_center = rnai_x + rnai_w / 2
                svg_elements.append(f'<text x="{rnai_label_x_center:.1f}" y="{amplicon_label_y}" '
                                   f'class="amp-label" text-anchor="middle">{rnai_size} bp</text>')

    if qpcr_info:
        qpcr_genomic_start = qpcr_info.get('genomic_start')
        qpcr_size = qpcr_info.get('size')
        if qpcr_genomic_start is not None and qpcr_size is not None and qpcr_size > 0:
            qpcr_x, qpcr_w = get_svg_pixels(qpcr_genomic_start, qpcr_genomic_start + qpcr_size - 1)
            if qpcr_x is not None:
                svg_elements.append(f'<rect x="{qpcr_x:.1f}" y="{amplicon_track_y}" width="{qpcr_w:.1f}" '
                                   f'height="{amplicon_height}" fill="#d9534f" rx="1" ry="1" style="opacity:0.9;">'
                                   f'<title>qPCR Amplicon ({qpcr_size} bp)</title></rect>')
                label_x_center = qpcr_x + qpcr_w / 2
                label_y_offset = 0
                if rnai_label_x_center is not None and abs(label_x_center - rnai_label_x_center) < 20:
                    label_y_offset = -5
                svg_elements.append(f'<text x="{label_x_center:.1f}" y="{amplicon_label_y + label_y_offset}" '
                                   f'class="amp-label" text-anchor="middle">{qpcr_size} bp</text>')

    svg_elements.append(f'<text x="{svg_width_px / 2}" y="{gene_label_y}" class="gene-label" text-anchor="middle">{gene_id}</text>')
    svg_elements.append(f'<text x="{padding}" y="{axis_label_y}" class="small">5\'</text>')
    svg_elements.append(f'<text x="{svg_width_px - padding}" y="{axis_label_y}" class="small" text-anchor="end">3\'</text>')

    svg_elements.append('</svg>')
    return "".join(svg_elements)

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Crithidia RNAi Tool")

# --- Title ---
st.title("*Crithidia fasciculata* RNAi Tool")
st.markdown("Design primers for RNAi fragments targeting CDS, check off-targets (k-mer), optionally design flanking qPCR primers, visualize.")

# --- File Upload Section ---
st.sidebar.header("📁 Upload Data Files")
st.sidebar.markdown("Upload your required FASTA and GFF files:")

transcriptome_file = st.sidebar.file_uploader(
    "Transcriptome FASTA (with UTRs)",
    type=['fasta', 'fa', 'fna'],
    help="Full transcript sequences for k-mer index"
)

annotation_file = st.sidebar.file_uploader(
    "Annotation GFF (with UTRs/CDS)",
    type=['gff', 'gff3'],
    help="Gene annotations for coordinates/visualization"
)

genome_file = st.sidebar.file_uploader(
    "Genome FASTA",
    type=['fasta', 'fa', 'fna'],
    help="Genome sequence for CDS extraction"
)

logo_file = st.sidebar.file_uploader(
    "Logo (optional)",
    type=['png', 'jpg', 'jpeg'],
    help="Optional logo image"
)

# Display logo if uploaded
if logo_file:
    st.sidebar.image(logo_file, width=80)

# --- Check if all required files are uploaded ---
if not all([transcriptome_file, annotation_file, genome_file]):
    st.info("👈 Please upload all required files in the sidebar to begin.")
    st.markdown("""
    ### Required Files:
    1. **Transcriptome FASTA**: Full transcript sequences (with UTRs) for building the k-mer off-target index
    2. **Annotation GFF**: Gene annotations with CDS and UTR features for coordinate extraction and visualization
    3. **Genome FASTA**: Reference genome for extracting CDS sequences
    
    ### Optional:
    - **Logo**: Display your lab or institution logo
    """)
    st.stop()

# --- Save uploaded files to temporary locations ---
with st.spinner("Processing uploaded files..."):
    transcriptome_path = save_uploaded_file_to_temp(transcriptome_file)
    annotation_path = save_uploaded_file_to_temp(annotation_file)
    genome_path = save_uploaded_file_to_temp(genome_file)

if not all([transcriptome_path, annotation_path, genome_path]):
    st.error("Failed to process uploaded files. Please try again.")
    st.stop()

# --- Load Data and Build Index ---
st.markdown("---")
with st.spinner(f"Loading transcriptome sequences from {transcriptome_file.name}..."):
    sequences = load_sequences(transcriptome_path)

if sequences:
    with st.spinner(f"Building {KMER_LENGTH}-mer index (this may take a few minutes on first run)..."):
        transcriptome_kmer_index = build_transcriptome_kmer_index(sequences, KMER_LENGTH)
else:
    transcriptome_kmer_index = None
    st.error("Halting - could not load transcriptome sequences.")
    st.stop()

if transcriptome_kmer_index is None:
    st.error("Halting - could not build k-mer index.")
    st.stop()

st.success("✅ Files loaded and k-mer index built successfully!")

# --- Input Form ---
st.markdown("---")
with st.form("rnai_design_form"):
    st.subheader("Input Parameters")

    target_gene_id_input = st.text_input(
        "Target Gene ID (e.g., CFAC1_170028000)",
        key="gene_id",
        help="Enter the primary gene identifier used in FASTA/GFF files."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RNAi Amplicon Parameters (within CDS):**")
        size_min = st.number_input(
            "Min Amplicon Size (bp)",
            min_value=50, max_value=1000, value=DEFAULT_SIZE_MIN, step=10,
            key="size_min", help="Minimum size of the desired RNAi amplicon."
        )
        tm_min = st.number_input(
            "Min Primer Tm (°C)",
            min_value=45.0, max_value=70.0, value=DEFAULT_TM_MIN, step=0.5,
            key="tm_min", format="%.1f", help="Minimum melting temperature for RNAi primers."
        )

        st.markdown("**RNAi Primer Overhangs:**")
        fwd_overhang_choice = st.selectbox(
            "Forward Primer Overhang",
            options=PROMOTER_OPTIONS, index=0,
            key="fwd_overhang", help="Select standard promoter or 'Custom'."
        )
        if fwd_overhang_choice == "Custom":
            fwd_overhang_custom = st.text_input(
                "Custom Fwd Sequence", key="fwd_custom",
                help="Enter custom 5' overhang sequence (will be lowercase)."
            ).strip()
        else:
            fwd_overhang_custom = ""

    with col2:
        st.markdown("**RNAi Amplicon Parameters (cont.):**")
        dynamic_min_for_max = size_min + 10
        safe_default_max = max(DEFAULT_SIZE_MAX, dynamic_min_for_max)
        size_max = st.number_input(
            "Max Amplicon Size (bp)",
            min_value=dynamic_min_for_max, max_value=2000, value=safe_default_max, step=10,
            key="size_max", help="Maximum size of the desired RNAi amplicon."
        )
        dynamic_min_tm_for_max = tm_min + 1.0
        safe_default_tm_max = max(DEFAULT_TM_MAX, dynamic_min_tm_for_max)
        tm_max = st.number_input(
            "Max Primer Tm (°C)",
            min_value=dynamic_min_tm_for_max, max_value=75.0, value=safe_default_tm_max, step=0.5,
            key="tm_max", format="%.1f", help="Maximum melting temperature for RNAi primers."
        )

        st.markdown("**RNAi Primer Overhangs (cont.):**")
        rev_overhang_choice = st.selectbox(
            "Reverse Primer Overhang",
            options=PROMOTER_OPTIONS, index=0,
            key="rev_overhang", help="Select standard promoter or 'Custom'."
        )
        if rev_overhang_choice == "Custom":
            rev_overhang_custom = st.text_input(
                "Custom Rev Sequence", key="rev_custom",
                help="Enter custom 5' overhang sequence (will be lowercase)."
            ).strip()
        else:
            rev_overhang_custom = ""

    st.markdown("**Options:**")
    design_qpcr = st.checkbox(
        "Design qPCR Validation Primers", value=True, key="design_qpcr",
        help="If checked, attempt to design qPCR primers outside the RNAi region."
    )

    tm_opt = (tm_min + tm_max) / 2.0
    size_opt = (size_min + size_max) // 2

    submitted = st.form_submit_button("🧬 Design Primers & Check Off-Targets")

# --- Processing and Output ---
if submitted:
    if size_min >= size_max or tm_min >= tm_max:
        st.error("Invalid range: Min value must be less than Max value for Size and Tm.")
        st.stop()

    if not target_gene_id_input:
        st.warning("Please enter a Target Gene ID.")
    elif transcriptome_kmer_index is None:
        st.error("Cannot proceed: K-mer index not available.")
    else:
        st.subheader(f"Results for {target_gene_id_input}")

        target_full_sequence, target_gene_id_found = find_sequence(target_gene_id_input, sequences)

        if target_full_sequence and target_gene_id_found:
            st.info(f"Found transcript sequence for '{target_gene_id_found}' (Full Length: {len(target_full_sequence)} bp)")

            st.info(f"Parsing annotation GFF for feature coordinates...")
            feature_coords = get_feature_genomic_coords(target_gene_id_found, annotation_path)

            if not feature_coords or feature_coords.get('CDS') is None or feature_coords.get('transcript_start') is None:
                st.error(f"Could not find required CDS and/or transcript coordinates for '{target_gene_id_found}'. "
                        "Cannot design primers based on CDS or visualize accurately.")
                if feature_coords:
                    st.write("Partially found coordinates:")
                    st.json(feature_coords)
                st.stop()

            cds_genomic_start, cds_genomic_end = feature_coords['CDS']
            transcript_genomic_start = feature_coords['transcript_start']
            transcript_genomic_end = feature_coords['transcript_end']
            strand = feature_coords['strand']
            seqid = feature_coords['seqid']

            st.info("Extracting CDS sequence from genome...")
            cds_sequence = None
            genome_sequences = None
            try:
                genome_fasta_index = genome_path + '.fai'
                if not os.path.exists(genome_fasta_index):
                    print(f"Genome FASTA index (.fai) not found. Creating...")
                    pyfaidx.Faidx(genome_path)
                    print("Genome FASTA index created.")
                genome_sequences = pyfaidx.Fasta(genome_path)

                cds_sequence_obj = genome_sequences.get_seq(
                    name=seqid,
                    start=cds_genomic_start,
                    end=cds_genomic_end,
                    rc=(strand == '-')
                )
                cds_sequence = cds_sequence_obj.seq
                st.info(f"Using CDS sequence (Length: {len(cds_sequence)} bp) for primer design.")

            except Exception as e:
                st.error(f"Failed to extract CDS sequence using coordinates {seqid}:{cds_genomic_start}-{cds_genomic_end} (Strand: {strand}): {e}")
                if genome_sequences:
                    genome_sequences.close()
                st.stop()
            finally:
                if genome_sequences:
                    genome_sequences.close()

            if size_max > len(cds_sequence):
                st.error(f"Error: Max RNAi Amplicon Size ({size_max} bp) > CDS length ({len(cds_sequence)} bp). Please adjust parameters.")
                st.stop()
            if size_min >= len(cds_sequence):
                st.error(f"Error: Min RNAi Amplicon Size ({size_min} bp) >= CDS length ({len(cds_sequence)} bp). Please adjust parameters.")
                st.stop()

            st.markdown("---")
            st.markdown("### RNAi Fragment Design (Targeting CDS)")
            with st.spinner("Designing RNAi primers using Primer3..."):
                primer_pair_info = design_primers_with_primer3(
                    target_gene_id_found + "_CDS", cds_sequence,
                    size_min, size_opt, size_max,
                    tm_min, tm_opt, tm_max
                )

            qpcr_primer_pair = None

            if primer_pair_info and 'PRIMER_LEFT_SEQUENCE' in primer_pair_info:
                st.success("✅ RNAi Primer Pair Found!")

                fwd_oh = fwd_overhang_custom if fwd_overhang_choice == "Custom" else PROMOTER_SEQS.get(fwd_overhang_choice, "")
                rev_oh = rev_overhang_custom if rev_overhang_choice == "Custom" else PROMOTER_SEQS.get(rev_overhang_choice, "")
                fwd_formatted, rev_formatted = format_primers_with_overhangs(
                    primer_pair_info['PRIMER_LEFT_SEQUENCE'],
                    primer_pair_info['PRIMER_RIGHT_SEQUENCE'],
                    fwd_oh, rev_oh
                )
                st.markdown("**RNAi Primer Pair 1 (Best Ranked):**")
                st.code(f"Forward: {fwd_formatted}\nReverse: {rev_formatted}", language=None)

                bio_amplicon_size = primer_pair_info['PRIMER_PAIR_PRODUCT_SIZE']
                total_amplicon_size = bio_amplicon_size + len(fwd_oh) + len(rev_oh)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Biological Size", f"{bio_amplicon_size} bp", help="Size of region amplified between primer binding sites (within CDS).")
                with col2:
                    if total_amplicon_size != bio_amplicon_size:
                        st.metric("Total Size w/ Overhangs", f"{total_amplicon_size} bp")
                with col3:
                    st.metric("Fwd Tm", f"{primer_pair_info['PRIMER_LEFT_TM']:.1f} °C")
                    st.metric("Fwd GC%", f"{primer_pair_info['PRIMER_LEFT_GC_PERCENT']:.1f}%")
                with col4:
                    st.metric("Rev Tm", f"{primer_pair_info['PRIMER_RIGHT_TM']:.1f} °C")
                    st.metric("Rev GC%", f"{primer_pair_info['PRIMER_RIGHT_GC_PERCENT']:.1f}%")

                rnai_amplicon_seq_cds = get_amplicon_sequence(cds_sequence, primer_pair_info)

                if design_qpcr:
                    st.markdown("---")
                    st.markdown("### qPCR Validation Primer Design (Targeting CDS)")
                    rnai_fwd_start_0based_cds = primer_pair_info.get('PRIMER_LEFT_POSITION')
                    rnai_amplicon_size_for_exclude = primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE')

                    if rnai_fwd_start_0based_cds is not None and rnai_amplicon_size_for_exclude is not None:
                        exclude_region_start_1based_cds = rnai_fwd_start_0based_cds + 1
                        exclude_region_length = rnai_amplicon_size_for_exclude
                        exclude_list = [[exclude_region_start_1based_cds, exclude_region_length]]

                        with st.spinner("Designing qPCR primers (Tm ~58°C, Size 80-110bp, outside RNAi region)..."):
                            qpcr_design_result = design_qpcr_primers(
                                target_gene_id_found + "_CDS",
                                cds_sequence,
                                exclude_region_list=exclude_list
                            )

                        if qpcr_design_result and 'PRIMER_LEFT_SEQUENCE' in qpcr_design_result:
                            qpcr_start_rel = qpcr_design_result.get('PRIMER_LEFT_POSITION')
                            qpcr_size = qpcr_design_result.get('PRIMER_PAIR_PRODUCT_SIZE')

                            rnai_start_rel = rnai_fwd_start_0based_cds
                            rnai_end_rel = rnai_start_rel + rnai_amplicon_size_for_exclude

                            qpcr_end_rel = qpcr_start_rel + qpcr_size

                            overlap_detected = max(rnai_start_rel, qpcr_start_rel) < min(rnai_end_rel, qpcr_end_rel)

                            if overlap_detected:
                                st.warning(f"qPCR Primer Design Warning: The only suitable qPCR primers found "
                                         f"(Region: {qpcr_start_rel}-{qpcr_end_rel-1} relative to CDS) overlap with the RNAi region "
                                         f"({rnai_start_rel}-{rnai_end_rel-1} relative to CDS). "
                                         f"Try reducing the 'Max Amplicon Size' for the RNAi fragment.")
                                qpcr_primer_pair = None
                            else:
                                qpcr_primer_pair = qpcr_design_result
                                st.success("✅ qPCR Primer Pair Found!")
                                st.markdown("**qPCR Primer Pair 1:**")
                                qpcr_fwd_formatted, qpcr_rev_formatted = format_primers_with_overhangs(
                                    qpcr_primer_pair['PRIMER_LEFT_SEQUENCE'],
                                    qpcr_primer_pair['PRIMER_RIGHT_SEQUENCE'], "", ""
                                )
                                st.code(f"Forward: {qpcr_fwd_formatted}\nReverse: {qpcr_rev_formatted}", language=None)
                                col1q, col2q, col3q = st.columns(3)
                                with col1q:
                                    st.metric("qPCR Amplicon Size", f"{qpcr_primer_pair['PRIMER_PAIR_PRODUCT_SIZE']} bp")
                                with col2q:
                                    st.metric("qPCR Fwd Tm", f"{qpcr_primer_pair['PRIMER_LEFT_TM']:.1f} °C")
                                    st.metric("qPCR Fwd GC%", f"{qpcr_primer_pair['PRIMER_LEFT_GC_PERCENT']:.1f}%")
                                with col3q:
                                    st.metric("qPCR Rev Tm", f"{qpcr_primer_pair['PRIMER_RIGHT_TM']:.1f} °C")
                                    st.metric("qPCR Rev GC%", f"{qpcr_primer_pair['PRIMER_RIGHT_GC_PERCENT']:.1f}%")

                        else:
                            st.warning("Could not find suitable qPCR primers outside the RNAi region within the CDS using specified parameters.")
                            if qpcr_design_result and 'RAW_RESULTS' in qpcr_design_result:
                                explain = qpcr_design_result['RAW_RESULTS'].get('PRIMER_EXPLAIN', 'No explanation provided.')
                                st.info(f"Primer3 qPCR explanation: {explain}")

                    else:
                        st.warning("Could not define RNAi region to exclude for qPCR primer design.")

                st.markdown("---")
                st.markdown("### Gene Model and Amplicon Positions")
                svg_viz = None
                if feature_coords and transcript_genomic_start is not None and transcript_genomic_end is not None:

                    rnai_viz_info = None
                    if primer_pair_info and primer_pair_info.get('PRIMER_LEFT_POSITION') is not None and primer_pair_info.get('PRIMER_PAIR_PRODUCT_SIZE') is not None:
                        rnai_fwd_start_0based_cds_viz = primer_pair_info['PRIMER_LEFT_POSITION']
                        rnai_amplicon_size_viz = primer_pair_info['PRIMER_PAIR_PRODUCT_SIZE']

                        if cds_genomic_start is not None and cds_genomic_end is not None:
                            rnai_genomic_start_viz = None
                            if strand == '+':
                                rnai_genomic_start_viz = cds_genomic_start + rnai_fwd_start_0based_cds_viz
                            else:
                                rnai_genomic_start_viz = cds_genomic_end - (rnai_fwd_start_0based_cds_viz + rnai_amplicon_size_viz) + 1

                            if rnai_genomic_start_viz is not None:
                                rnai_viz_info = {'genomic_start': rnai_genomic_start_viz, 'size': rnai_amplicon_size_viz}

                    qpcr_viz_info = None
                    if qpcr_primer_pair and qpcr_primer_pair.get('PRIMER_LEFT_POSITION') is not None:
                        qpcr_rel_start_cds = qpcr_primer_pair['PRIMER_LEFT_POSITION']
                        qpcr_amplicon_size = qpcr_primer_pair.get('PRIMER_PAIR_PRODUCT_SIZE')
                        if qpcr_amplicon_size is not None and cds_genomic_start is not None and cds_genomic_end is not None:
                            qpcr_genomic_start_viz = None
                            if strand == '+':
                                qpcr_genomic_start_viz = cds_genomic_start + qpcr_rel_start_cds
                            else:
                                qpcr_genomic_start_viz = cds_genomic_end - (qpcr_rel_start_cds + qpcr_amplicon_size) + 1
                            if qpcr_genomic_start_viz is not None:
                                qpcr_viz_info = {'genomic_start': qpcr_genomic_start_viz, 'size': qpcr_amplicon_size}

                    svg_viz = generate_gene_model_svg(
                        target_gene_id_found,
                        transcript_genomic_start,
                        transcript_genomic_end,
                        feature_coords,
                        rnai_viz_info,
                        qpcr_viz_info
                    )
                    st.markdown(svg_viz, unsafe_allow_html=True)
                    st.markdown('<span style="font-size: small;">LEGEND: <span style="display: inline-block; width: 10px; height: 10px; background-color: #e0e0e0; margin-left: 5px; margin-right: 3px; vertical-align: middle;"></span> UTR <span style="display: inline-block; width: 10px; height: 10px; background-color: #bdbdbd; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> CDS | <span style="display: inline-block; width: 10px; height: 8px; background-color: #4dd0e1; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> RNAi Amplicon <span style="display: inline-block; width: 10px; height: 8px; background-color: #d9534f; margin-left: 10px; margin-right: 3px; vertical-align: middle;"></span> qPCR Amplicon</span>', unsafe_allow_html=True)
                    if svg_viz:
                        st.download_button(
                            label="📥 Download SVG", data=svg_viz,
                            file_name=f"{target_gene_id_found}_model.svg", mime="image/svg+xml"
                        )
                else:
                    st.warning(f"Could not retrieve sufficient coordinates to generate visualization.")

                st.markdown("---")
                if rnai_amplicon_seq_cds:
                    st.markdown(f"**Potential Off-Target Check ({KMER_LENGTH}-mer based for RNAi Amplicon):**")
                    st.markdown(f"*(Shows transcripts sharing >=1 perfect or 1-mismatch {KMER_LENGTH}-mer with the RNAi amplicon)*")
                    with st.spinner(f"Checking {KMER_LENGTH}-mer off-targets..."):
                        off_target_df = check_off_targets_kmer(
                            rnai_amplicon_seq_cds,
                            transcriptome_kmer_index,
                            target_gene_id_found
                        )
                    if not off_target_df.empty:
                        st.warning(f"Found {len(off_target_df)} potential off-target transcript(s):")
                        base_url = "https://tritrypdb.org/tritrypdb/app/record/gene/"
                        off_target_df['URL'] = base_url + off_target_df['Off-Target ID'].astype(str)
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
                        st.info(f"✅ No other transcripts found sharing perfect or 1-mismatch {KMER_LENGTH}-mers with the RNAi amplicon.")
                else:
                    st.warning("Could not extract RNAi amplicon sequence to perform off-target check.")

            else:
                if primer_pair_info and 'RAW_RESULTS' in primer_pair_info:
                    st.info("Raw results from Primer3 (RNAi design failure):")
                    raw_results_subset = {k: v for k, v in primer_pair_info['RAW_RESULTS'].items() if 'SEQUENCE' not in k}
                    st.json(raw_results_subset, expanded=False)

        else:
            st.error(f"Gene ID '{target_gene_id_input}' not found in the transcriptome FASTA file.")

# --- Footer Info ---
st.markdown("---")
st.markdown(f"**Parameters:**")
st.markdown(f"- K-mer length for off-target check: `{KMER_LENGTH}`")
st.markdown(f"- Files currently loaded from uploaded sources")

# End of File
