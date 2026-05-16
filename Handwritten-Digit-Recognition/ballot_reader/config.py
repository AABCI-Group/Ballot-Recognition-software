from dataclasses import dataclass, field
from typing import Optional, List, Tuple

@dataclass
class BallotConfig:
    """Configuration for the ballot reader pipeline."""
    
    # --- BLANK/NOISE GATING ---
    blank_min_ink_px_abs: int = 35
    blank_min_ink_frac: float = 0.0025
    blank_min_bbox_w_frac: float = 0.10
    blank_min_bbox_h_frac: float = 0.18
    blank_border_band_frac: float = 0.12
    blank_border_ink_ratio: float = 0.85
    blank_tiny_comp_area_frac: float = 0.0012
    blank_tiny_comp_max_dim: int = 6
    blank_speckle_field_min: int = 7
    blank_single_comp_max_area_frac: float = 0.010
    blank_density_min: float = 0.020
    blank_skeleton_min_frac: float = 0.20
    blank_perim_area_max: float = 0.55

    # --- CNN GATING ---
    cnn_max_entropy: float = 1.80
    cnn_min_margin_strict_1: float = 0.18
    cnn_min_conf_strict_1: float = 0.72
    sparse_ink_frac: float = 0.006
    min_top_conf: float = 0.60
    min_margin: float = 0.10

    # --- LAYOUT DETECTION ---
    min_y_frac: float = 0.05
    min_x_frac: float = 0.60
    min_area_frac: float = 0.005
    pad_frac: float = 0.00
    expected_boxes: Optional[int] = None

    # --- ENHANCEMENT ---
    enh_score_min_ink_frac: float = 0.002
    enh_score_max_ink_frac: float = 0.25
    enh_score_min_skeleton: int = 12

    # --- AMBIGUITY RESOLUTION (4/7) ---
    ambig_top2_set_47: bool = True
    ambig_margin_47: float = 0.22
    ambig_conf_47: float = 0.70
    seven_strong_score: float = 0.65
    four_strong_score: float = 0.65
    override_4_to_7_min_seven_score: float = 0.72
    override_4_to_7_max_junctions: int = 2
    override_4_to_7_min_topbar: float = 0.35
    override_4_to_7_max_leftstroke: float = 0.22
    strict_4_if_sevenlike_conf: float = 0.78
    strict_4_if_sevenlike_margin: float = 0.22
    strict_4_min_diag_dom: float = 0.55
    strict_4_min_topbar: float = 0.35

    # --- RECTIFICATION ---
    rectify_canonical_size: int = 128
    rectify_canny1: int = 50
    rectify_canny2: int = 150
    rectify_min_area_frac: float = 0.08
    rectify_poly_eps_frac: float = 0.02
    rectify_max_aspect_dev: float = 0.35

    # --- THIN DIGIT EXCEPTIONS ---
    thin_exc_min_ink_frac: float = 0.0028
    thin_exc_max_ink_frac: float = 0.0300
    thin_exc_min_bb_h_frac: float = 0.55
    thin_exc_min_aspect: float = 2.2
    thin_exc_max_components: int = 2
    thin_exc_skeleton_h_frac: float = 0.62
    one_allow_min_vert_dom: float = 0.48
    one_allow_min_skel_h_frac: float = 0.60
    one_rescue_p1_min: float = 0.15
    four_rescue_p4_min: float = 0.15
    # --- VOTE-BOX (BOX-FIRST) DETECTION ---
    # Detect vote boxes only in a right-side ROI to avoid interference from
    # candidate photos, party logos, and other text.
    vote_box_roi_frac: float = 0.27          # right-most ~20–30% of width
    vote_box_roi_pad_frac: float = 0.02      # extra padding to tolerate x-shift

    # Contour-based rectangle filters (applied to ROI).
    vote_box_aspect_min: float = 0.65
    vote_box_aspect_max: float = 1.55
    # Area thresholds are fractions of the full image area.
    vote_box_min_area_frac: float = 0.0020
    vote_box_max_area_frac: float = 0.045
    # approxPolyDP epsilon relative to contour perimeter.
    vote_box_poly_eps_frac: float = 0.020
    # Vote boxes should extend into the far-right column; portraits/logos can
    # form square-ish contours inside the wider right-side ROI but end earlier.
    vote_box_min_right_edge_frac: float = 0.84
    
    # --- PATHS ---
    model_path: str = "Handwritten-Digit-Recognition/tf-cnn-model.keras"
    calibration_json: str = "calibration_ballot.json"
