bl_info = {
    "name": "Humanoid Face Retopology(HFR)",
    "author": "graytutor / ChatGPT-assisted",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > HFR",
    "description": "Landmark-driven humanoid face retopology for Blender. Created with assistance from ChatGPT.",
    "category": "Mesh",
}

import bpy
import json
import os
import bmesh
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree
from mathutils.bvhtree import BVHTree
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty, IntProperty
from bpy.app.handlers import persistent

HFR_LM_DIAMETER = 0.003

# Dev-only UI gate. For release/export builds set this to False so
# the DevOption toggle and developer panels are not drawn.
HFR_SHOW_DEV_OPTIONS = True


# -----------------------------------------------------------------------------
# Naming / collections
# -----------------------------------------------------------------------------

ADDON_PREFIX = "HFR"
LM_OBJ_PREFIX = "LM_"
ANCHOR_GROUP_PREFIX = "HFR_A_"

COLL_WORK = "HumanoidRetopo_Work"
COLL_FACE = "HumanoidRetopo_LM_Face"
COLL_EAR = "HumanoidRetopo_LM_Ear"
COLL_GUIDE = "HumanoidRetopo_Guide"
COLL_MEMORY_LEGACY = "HumanoidRetopo_Memory"

TEMPLATE_ASSET_DIR = "templates"
DEFAULT_TEMPLATE_BLEND = "HFRTemplate.blend"
DEFAULT_TEMPLATE_BINDING_JSON = "HFRTemplateBinding.json"
DEFAULT_TEMPLATE_OBJECT = "FaceTemplate"
DEFAULT_TEMPLATE_CANDIDATES = (
    DEFAULT_TEMPLATE_OBJECT,
    "HFR_Base_Template",
    "HFRTemplate",
    "HFR_Template",
    "HFR_Retopo_Template",
    "HumanoidFaceRetopo_Template",
    "Face_Template",
)

# IDProperty keys are intentionally short. Keep all below 63 chars.
PID_LM = "HFR_lm"
PID_LM_ID = "HFR_lm_id"
PID_LM_GRP = "HFR_lm_grp"
PID_GUIDE = "HFR_g"
PID_GUIDE_A = "HFR_g_a"
PID_GUIDE_B = "HFR_g_b"
PID_DEF_JSON = "HFR_lm_defs"
PID_BIND_GUIDE = "HFR_bg"
PID_BIND_LM = "HFR_bg_l"
PID_BIND_OBJ = "HFR_bg_o"
PID_OUTPUT = "HFR_out"
PID_TEMPLATE = "HFR_tpl"

# Runtime-only state for live guide refresh / mirror.
_HFR_SYNC_LOCK = False
_HFR_PENDING_LM_IDS = set()
_HFR_TIMER_REGISTERED = False
_HFR_AUTO_TPL_TIMER_REGISTERED = False


# -----------------------------------------------------------------------------
# Landmark style
# RetopologyAddOn1-style approximation: small colored solid spheres + grey guide
# curves, hidden labels, separated Face/Ear/Guide collections.
# -----------------------------------------------------------------------------

GROUP_COLORS = {
    "eye":      (0.12, 0.95, 0.30, 1.0),
    "mouth":    (1.00, 0.18, 0.36, 1.0),
    "nose":     (1.00, 0.62, 0.08, 1.0),
    "cheek":    (0.15, 0.80, 0.95, 1.0),
    "chin":     (0.72, 0.36, 1.00, 1.0),
    "outer":    (0.16, 0.48, 1.00, 1.0),
    "forehead": (0.20, 0.58, 1.00, 1.0),
    "scalp":    (0.08, 0.24, 0.90, 1.0),
    "ear":      (1.00, 0.78, 0.12, 1.0),
    "neck":     (0.70, 0.70, 0.70, 1.0),
    "center":   (1.00, 1.00, 1.00, 1.0),
    "guide":    (0.55, 0.55, 0.55, 1.0),
}


# Default front-facing humanoid layout.
# Axis convention: X left/right, Y front/back, Z up/down.
# The face is assumed to look roughly toward negative Y.
LANDMARKS = [
    # Eyes: 8-point eyelid loops. Center points are intentionally omitted.
    {"id": "eye_l_inner", "grp": "eye", "co": (-0.016899, -0.080456, 1.703703), "front": True},
    {"id": "eye_l_upper_inner", "grp": "eye", "co": (-0.023425, -0.086451, 1.711952), "front": True},
    {"id": "eye_l_upper", "grp": "eye", "co": (-0.033598, -0.088020, 1.715964), "front": True},
    {"id": "eye_l_upper_outer", "grp": "eye", "co": (-0.043625, -0.080349, 1.714434), "front": True},
    {"id": "eye_l_outer", "grp": "eye", "co": (-0.053474, -0.069851, 1.705426), "front": True},
    {"id": "eye_l_lower_outer", "grp": "eye", "co": (-0.043714, -0.076961, 1.697313), "front": True},
    {"id": "eye_l_lower", "grp": "eye", "co": (-0.034118, -0.080931, 1.696542), "front": True},
    {"id": "eye_l_lower_inner", "grp": "eye", "co": (-0.022107, -0.082424, 1.698510), "front": True},
    {"id": "eye_r_inner", "grp": "eye", "co": (0.016899, -0.080456, 1.703703), "front": True},
    {"id": "eye_r_upper_inner", "grp": "eye", "co": (0.023425, -0.086451, 1.711952), "front": True},
    {"id": "eye_r_upper", "grp": "eye", "co": (0.033598, -0.088020, 1.715964), "front": True},
    {"id": "eye_r_upper_outer", "grp": "eye", "co": (0.043625, -0.080349, 1.714434), "front": True},
    {"id": "eye_r_outer", "grp": "eye", "co": (0.053474, -0.069851, 1.705426), "front": True},
    {"id": "eye_r_lower_outer", "grp": "eye", "co": (0.043714, -0.076961, 1.697313), "front": True},
    {"id": "eye_r_lower", "grp": "eye", "co": (0.034118, -0.080931, 1.696542), "front": True},
    {"id": "eye_r_lower_inner", "grp": "eye", "co": (0.022107, -0.082424, 1.698510), "front": True},

    # Brows
    {"id": "brow_l_inner", "grp": "forehead", "co": (-0.014913, -0.093804, 1.718361), "front": True},
    {"id": "brow_l_center", "grp": "forehead", "co": (-0.032455, -0.088357, 1.722462), "front": True},
    {"id": "brow_l_outer", "grp": "forehead", "co": (-0.052713, -0.074995, 1.719163), "front": True},
    {"id": "brow_r_inner", "grp": "forehead", "co": (0.014913, -0.093804, 1.718361), "front": True},
    {"id": "brow_r_center", "grp": "forehead", "co": (0.032455, -0.088357, 1.722462), "front": True},
    {"id": "brow_r_outer", "grp": "forehead", "co": (0.052713, -0.074995, 1.719163), "front": True},

    # Nose
    {"id": "nose_root", "grp": "nose", "co": (0.000000, -0.094411, 1.709559), "front": True},
    {"id": "nose_bridge_top", "grp": "nose", "co": (0.000000, -0.097915, 1.701116), "front": True},
    {"id": "nose_bridge", "grp": "nose", "co": (0.000000, -0.103418, 1.692485), "front": True},
    {"id": "nose_tip", "grp": "nose", "co": (0.000000, -0.118799, 1.662866), "front": True},
    {"id": "nose_base", "grp": "nose", "co": (0.000000, -0.104971, 1.653031), "front": True},
    {"id": "nose_l_side_upper", "grp": "nose", "co": (-0.015284, -0.088955, 1.688475), "front": True},
    {"id": "nose_r_side_upper", "grp": "nose", "co": (0.015284, -0.088955, 1.688475), "front": True},
    {"id": "nose_l_side_lower", "grp": "nose", "co": (-0.017169, -0.088120, 1.673047), "front": True},
    {"id": "nose_r_side_lower", "grp": "nose", "co": (0.017169, -0.088120, 1.673047), "front": True},
    {"id": "nose_l_alar", "grp": "nose", "co": (-0.018817, -0.089398, 1.659348), "front": True},
    {"id": "nose_r_alar", "grp": "nose", "co": (0.018817, -0.089398, 1.659348), "front": True},
    {"id": "nose_l_nostril", "grp": "nose", "co": (-0.006769, -0.099752, 1.660129), "front": True},
    {"id": "nose_r_nostril", "grp": "nose", "co": (0.006769, -0.099752, 1.660129), "front": True},

    # Mouth
    {"id": "mouth_l_corner", "grp": "mouth", "co": (-0.027385, -0.086431, 1.629276), "front": True},
    {"id": "mouth_r_corner", "grp": "mouth", "co": (0.027385, -0.086431, 1.629276), "front": True},
    {"id": "mouth_upper_mid", "grp": "mouth", "co": (0.000000, -0.099985, 1.631877), "front": True},
    {"id": "mouth_lower_mid", "grp": "mouth", "co": (0.000000, -0.099399, 1.628031), "front": True},
    {"id": "mouth_l_upper", "grp": "mouth", "co": (-0.012783, -0.096174, 1.631858), "front": True},
    {"id": "mouth_r_upper", "grp": "mouth", "co": (0.012783, -0.096174, 1.631858), "front": True},
    {"id": "mouth_l_lower", "grp": "mouth", "co": (-0.012889, -0.095804, 1.628210), "front": True},
    {"id": "mouth_r_lower", "grp": "mouth", "co": (0.012889, -0.095804, 1.628210), "front": True},

    # Cheek / outer face / jaw
    {"id": "cheek_l_center", "grp": "cheek", "co": (-0.039693, -0.076134, 1.673335), "front": True},
    {"id": "cheek_r_center", "grp": "cheek", "co": (0.039693, -0.076134, 1.673335), "front": True},
    {"id": "outer_face_l_upper", "grp": "outer", "co": (-0.065406, -0.047063, 1.688846), "front": True},
    {"id": "outer_face_r_upper", "grp": "outer", "co": (0.065406, -0.047063, 1.688846), "front": True},
    {"id": "outer_face_l_lower", "grp": "outer", "co": (-0.051869, -0.050436, 1.626351), "front": True},
    {"id": "outer_face_r_lower", "grp": "outer", "co": (0.051869, -0.050436, 1.626351), "front": True},
    {"id": "face_l_edge", "grp": "outer", "co": (-0.063117, -0.043462, 1.662483), "front": True},
    {"id": "face_r_edge", "grp": "outer", "co": (0.063117, -0.043462, 1.662483), "front": True},
    {"id": "jaw_l_edge", "grp": "chin", "co": (-0.054972, -0.030239, 1.613838), "front": True},
    {"id": "jaw_r_edge", "grp": "chin", "co": (0.054972, -0.030239, 1.613838), "front": True},

    # Chin
    {"id": "chin_center", "grp": "chin", "co": (0.000000, -0.082129, 1.582801), "front": True},
    {"id": "chin_l_outer", "grp": "chin", "co": (-0.037460, -0.057351, 1.599294), "front": True},
    {"id": "chin_r_outer", "grp": "chin", "co": (0.037460, -0.057351, 1.599294), "front": True},
    {"id": "chin_l_lower_outer", "grp": "chin", "co": (-0.030688, -0.064706, 1.594180), "front": True},
    {"id": "chin_r_lower_outer", "grp": "chin", "co": (0.030688, -0.064706, 1.594180), "front": True},
    {"id": "chin_l_lower", "grp": "chin", "co": (-0.023917, -0.072060, 1.589066), "front": True},
    {"id": "chin_r_lower", "grp": "chin", "co": (0.023917, -0.072060, 1.589066), "front": True},

    # Forehead / temple / scalp / head side
    {"id": "forehead_center", "grp": "forehead", "co": (0.000000, -0.090925, 1.745654), "front": True},
    {"id": "forehead_upper_center", "grp": "forehead", "co": (0.000000, -0.083638, 1.764708), "front": True},
    {"id": "forehead_l_upper", "grp": "forehead", "co": (-0.032710, -0.082580, 1.745051), "front": True},
    {"id": "forehead_r_upper", "grp": "forehead", "co": (0.032710, -0.082580, 1.745051), "front": True},
    {"id": "temple_l_center", "grp": "forehead", "co": (-0.064334, -0.044816, 1.707231), "front": True},
    {"id": "temple_r_center", "grp": "forehead", "co": (0.064334, -0.044816, 1.707231), "front": True},
    {"id": "head_l_side_upper", "grp": "scalp", "co": (-0.071657, -0.013679, 1.742931), "front": False},
    {"id": "head_r_side_upper", "grp": "scalp", "co": (0.071657, -0.013679, 1.742931), "front": False},
    {"id": "head_l_side_back", "grp": "scalp", "co": (-0.072173, 0.041566, 1.693771), "front": False},
    {"id": "head_r_side_back", "grp": "scalp", "co": (0.072173, 0.041566, 1.693771), "front": False},
    {"id": "scalp_front_center", "grp": "scalp", "co": (0.000000, -0.049050, 1.801812), "front": False},
    {"id": "scalp_top_center", "grp": "scalp", "co": (0.000000, 0.006247, 1.819337), "front": False},
    {"id": "scalp_back_center", "grp": "scalp", "co": (0.000000, 0.096502, 1.772391), "front": False},
    {"id": "scalp_l_front", "grp": "scalp", "co": (-0.043887, -0.032437, 1.789466), "front": False},
    {"id": "scalp_r_front", "grp": "scalp", "co": (0.043887, -0.032437, 1.789466), "front": False},
    {"id": "scalp_l_top", "grp": "scalp", "co": (-0.040937, 0.013589, 1.807022), "front": False},
    {"id": "scalp_r_top", "grp": "scalp", "co": (0.040937, 0.013589, 1.807022), "front": False},

    # Ears
    {"id": "ear_l_top", "grp": "ear", "co": (-0.086844, 0.022942, 1.715654), "front": True},
    {"id": "ear_l_front_upper", "grp": "ear", "co": (-0.076821, 0.007323, 1.704376), "front": True},
    {"id": "ear_l_front_middle", "grp": "ear", "co": (-0.073287, 0.000007, 1.680712), "front": True},
    {"id": "ear_l_front_lower", "grp": "ear", "co": (-0.068712, 0.005324, 1.653423), "front": True},
    {"id": "ear_l_back_upper", "grp": "ear", "co": (-0.086972, 0.035913, 1.711232), "front": True},
    {"id": "ear_l_back_middle", "grp": "ear", "co": (-0.084162, 0.038756, 1.676632), "front": True},
    {"id": "ear_l_back_lower", "grp": "ear", "co": (-0.079222, 0.028729, 1.660923), "front": True},
    {"id": "ear_l_inner_front_middle", "grp": "ear", "co": (-0.074654, 0.011385, 1.685902), "front": True},
    {"id": "ear_l_inner_bottom", "grp": "ear", "co": (-0.074584, 0.012068, 1.670748), "front": True},
    {"id": "ear_l_lobe", "grp": "ear", "co": (-0.072218, 0.015323, 1.653893), "front": True},
    {"id": "ear_r_top", "grp": "ear", "co": (0.086844, 0.022942, 1.715654), "front": True},
    {"id": "ear_r_front_upper", "grp": "ear", "co": (0.076821, 0.007323, 1.704376), "front": True},
    {"id": "ear_r_front_middle", "grp": "ear", "co": (0.073287, 0.000007, 1.680712), "front": True},
    {"id": "ear_r_front_lower", "grp": "ear", "co": (0.068712, 0.005324, 1.653423), "front": True},
    {"id": "ear_r_back_upper", "grp": "ear", "co": (0.086972, 0.035913, 1.711232), "front": True},
    {"id": "ear_r_back_middle", "grp": "ear", "co": (0.084162, 0.038756, 1.676632), "front": True},
    {"id": "ear_r_back_lower", "grp": "ear", "co": (0.079222, 0.028729, 1.660923), "front": True},
    {"id": "ear_r_inner_front_middle", "grp": "ear", "co": (0.074654, 0.011385, 1.685902), "front": True},
    {"id": "ear_r_inner_bottom", "grp": "ear", "co": (0.074584, 0.012068, 1.670748), "front": True},
    {"id": "ear_r_lobe", "grp": "ear", "co": (0.072218, 0.015323, 1.653893), "front": True},

    # Nape / neck
    {"id": "nape_center", "grp": "neck", "co": (0.000000, 0.088839, 1.636148), "front": False},
    {"id": "nape_l_outer", "grp": "neck", "co": (-0.027875, 0.083466, 1.633973), "front": False},
    {"id": "nape_r_outer", "grp": "neck", "co": (0.027875, 0.083466, 1.633973), "front": False},
    {"id": "neck_front_center", "grp": "neck", "co": (0.000000, -0.046883, 1.576070), "front": True},
    {"id": "neck_back_center", "grp": "neck", "co": (0.000000, 0.093426, 1.607729), "front": False},
    {"id": "neck_l_side", "grp": "neck", "co": (-0.046366, -0.004735, 1.582368), "front": True},
    {"id": "neck_r_side", "grp": "neck", "co": (0.046366, -0.004735, 1.582368), "front": True},
    {"id": "neck_top_l_front", "grp": "neck", "co": (-0.026111, -0.018234, 1.576696), "front": True},
    {"id": "neck_top_l_side", "grp": "neck", "co": (-0.064393, 0.042274, 1.595806), "front": True},
    {"id": "neck_top_l_back", "grp": "neck", "co": (-0.033655, 0.088017, 1.607524), "front": False},
    {"id": "neck_top_r_front", "grp": "neck", "co": (0.026111, -0.018234, 1.576696), "front": True},
    {"id": "neck_top_r_side", "grp": "neck", "co": (0.064393, 0.042274, 1.595806), "front": True},
    {"id": "neck_top_r_back", "grp": "neck", "co": (0.033655, 0.088017, 1.607524), "front": False},
]

LM_BY_ID = {lm["id"]: lm for lm in LANDMARKS}

# Synthetic anchors let the add-on support new structural landmarks even when
# an older template blend does not yet contain a dedicated HFR_A_* group for
# them.  The source position and affected member set are blended from nearby
# anchor groups using relative/topology-based percentages rather than absolute
# offsets.  If a real HFR_A_* group exists, it still takes precedence.
SYNTHETIC_ANCHOR_SPECS = {
    "chin_l_lower_outer": {"sources": ("chin_l_outer", "chin_l_lower"), "blend": 0.50},
    "chin_r_lower_outer": {"sources": ("chin_r_outer", "chin_r_lower"), "blend": 0.50},
}

# Ordered local feature loops used by Generate Retopology.  These are not new
# topology; they are only deformation constraints that keep eyelid/lip vertices
# between bound anchors moving along the intended loop instead of being pulled by
# unrelated nearby anchors such as nose, cheek, or brow points.
FEATURE_LOOPS = {
    "eye_l": [
        "eye_l_inner", "eye_l_upper_inner", "eye_l_upper", "eye_l_upper_outer",
        "eye_l_outer", "eye_l_lower_outer", "eye_l_lower", "eye_l_lower_inner",
    ],
    "eye_r": [
        "eye_r_inner", "eye_r_upper_inner", "eye_r_upper", "eye_r_upper_outer",
        "eye_r_outer", "eye_r_lower_outer", "eye_r_lower", "eye_r_lower_inner",
    ],
    "mouth": [
        "mouth_l_corner", "mouth_l_upper", "mouth_upper_mid", "mouth_r_upper",
        "mouth_r_corner", "mouth_r_lower", "mouth_lower_mid", "mouth_l_lower",
    ],
}

# Additional local feature loops. These are separate from the main eye/mouth
# loop controls because ear lobes and neck length often need different local
# correction strength.
EAR_FEATURE_LOOPS = {
    "ear_l_outer": [
        "ear_l_top", "ear_l_front_upper", "ear_l_front_middle", "ear_l_front_lower",
        "ear_l_lobe", "ear_l_back_lower", "ear_l_back_middle", "ear_l_back_upper",
    ],
    "ear_r_outer": [
        "ear_r_top", "ear_r_front_upper", "ear_r_front_middle", "ear_r_front_lower",
        "ear_r_lobe", "ear_r_back_lower", "ear_r_back_middle", "ear_r_back_upper",
    ],
}

NECK_FEATURE_LOOPS = {
    "neck_base": ["neck_front_center", "neck_r_side", "neck_back_center", "neck_l_side"],
    "neck_top": [
        "neck_top_l_front", "neck_top_r_front", "neck_top_r_side",
        "neck_top_r_back", "neck_top_l_back", "neck_top_l_side",
    ],
}


# Open lower-ear rails. Unlike EAR_FEATURE_LOOPS, these are not closed loops.
# They keep the lobe correction local to the lower ear and avoid the old
# behavior where a closed ear loop could pull the lobe toward the front/top.
EAR_LOWER_RAILS = {
    "ear_l_lower_rail": [
        ("ear_l_front_lower", "ear_l_lobe"),
        ("ear_l_lobe", "ear_l_back_lower"),
        ("ear_l_inner_bottom", "ear_l_lobe"),
    ],
    "ear_r_lower_rail": [
        ("ear_r_front_lower", "ear_r_lobe"),
        ("ear_r_lobe", "ear_r_back_lower"),
        ("ear_r_inner_bottom", "ear_r_lobe"),
    ],
}

# Visual attachment guides that connect the ear landmark cage to the head/face.
# They are useful as guide curves, but they must not be used as hard Guide Rail
# constraints.  On heads with a different ear depth/angle, the shortest mesh
# path from jaw/nape/head-side anchors to ear anchors can cut through the ear and
# flip the ear faces.  Keep these pairs available for display/soft guide follow,
# but exclude them from the fixed rail solver.
EAR_ATTACHMENT_GUIDES = {
    ("temple_l_center", "ear_l_front_upper"),
    ("face_l_edge", "ear_l_front_middle"),
    ("jaw_l_edge", "ear_l_front_lower"),
    ("head_l_side_back", "ear_l_back_upper"),
    ("nape_l_outer", "ear_l_back_lower"),
    ("temple_r_center", "ear_r_front_upper"),
    ("face_r_edge", "ear_r_front_middle"),
    ("jaw_r_edge", "ear_r_front_lower"),
    ("head_r_side_back", "ear_r_back_upper"),
    ("nape_r_outer", "ear_r_back_lower"),
}

# Visual glabella guides between brow-inner landmarks and nose_root.
# They help users understand the cage, but as solver rails they can route the
# mesh shortest path through the vertex just under LM_brow_*_inner and pull it
# upward into a spike.  Brow Ridge Fit now handles this band explicitly, so keep
# these pairs visual-only.
BROW_GLABELLA_GUIDES = {
    ("brow_l_inner", "nose_root"),
    ("brow_r_inner", "nose_root"),
}


def _pair_key(a, b):
    return tuple(sorted((a, b)))


def solver_guide_rail_pairs():
    excluded = {_pair_key(a, b) for a, b in EAR_ATTACHMENT_GUIDES}
    excluded.update(_pair_key(a, b) for a, b in BROW_GLABELLA_GUIDES)
    return [(a, b) for a, b in GUIDES if _pair_key(a, b) not in excluded]


def solver_soft_guide_pairs():
    # Ear attachment guides are visual-only in the solver.  They previously acted
    # as soft rails and could pull the side-head / back-ear attachment strip
    # toward the ear landmarks too aggressively.  The head/ear transition is now
    # left to the broad MLS field, attachment guards, and final snap instead.
    excluded = {_pair_key(a, b) for a, b in EAR_ATTACHMENT_GUIDES}
    excluded.update(_pair_key(a, b) for a, b in BROW_GLABELLA_GUIDES)
    return [(a, b) for a, b in GUIDES if _pair_key(a, b) not in excluded]


# -----------------------------------------------------------------------------
# Initial placement / target fitting
# -----------------------------------------------------------------------------

def _vector_min(values):
    return Vector((min(v.x for v in values), min(v.y for v in values), min(v.z for v in values)))


def _vector_max(values):
    return Vector((max(v.x for v in values), max(v.y for v in values), max(v.z for v in values)))


def default_layout_bounds():
    coords = [Vector(lm["co"]) for lm in LANDMARKS]
    return _vector_min(coords), _vector_max(coords)


def is_mesh_fit_target(obj):
    return obj is not None and obj.type == 'MESH' and not obj.get(PID_LM) and not obj.get(PID_OUTPUT) and not obj.get(PID_TEMPLATE)


def fit_target_object(context):
    if context is None:
        return None
    scene = context.scene
    target = getattr(scene, "hfr_lm_target_obj", None)
    if is_mesh_fit_target(target):
        return target
    active = getattr(context.view_layer.objects, "active", None)
    if is_mesh_fit_target(active):
        return active
    for obj in getattr(context, "selected_objects", []):
        if is_mesh_fit_target(obj):
            return obj
    return None


def world_bounds_of_object(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return _vector_min(corners), _vector_max(corners)


def fitted_target_bounds(context):
    target = fit_target_object(context)
    if not target:
        return None
    scene = context.scene
    minv, maxv = world_bounds_of_object(target)
    dims = maxv - minv

    mode = getattr(scene, "hfr_lm_fit_region", 'AUTO')
    # If a full body mesh is selected, use the upper head-like part instead of
    # stretching face landmarks over the entire body height.
    if mode == 'HEAD' or (mode == 'AUTO' and dims.z > max(dims.x, dims.y, 0.001) * 2.8):
        width = max(dims.x, dims.y, 0.001)
        head_h = min(dims.z, max(width * 1.45, dims.y * 1.8))
        minv = Vector((minv.x, minv.y, maxv.z - head_h))
        dims = maxv - minv

    margin = float(getattr(scene, "hfr_lm_fit_margin", 0.02))
    center = (minv + maxv) * 0.5
    dims = Vector((max(dims.x, 0.001), max(dims.y, 0.001), max(dims.z, 0.001)))
    dims *= max(0.05, 1.0 + margin)
    return center - dims * 0.5, center + dims * 0.5


def fitted_landmark_location(context, lm):
    bounds = fitted_target_bounds(context)
    if bounds is None:
        return Vector(lm["co"])
    dmin, dmax = default_layout_bounds()
    tmin, tmax = bounds
    dsize = dmax - dmin
    tsize = tmax - tmin
    src = Vector(lm["co"])
    nx = 0.0 if abs(dsize.x) < 1e-8 else (src.x - dmin.x) / dsize.x
    ny = 0.0 if abs(dsize.y) < 1e-8 else (src.y - dmin.y) / dsize.y
    nz = 0.0 if abs(dsize.z) < 1e-8 else (src.z - dmin.z) / dsize.z
    return Vector((tmin.x + tsize.x * nx, tmin.y + tsize.y * ny, tmin.z + tsize.z * nz))


def current_landmark_radius(scene, context=None):
    # Fixed 0.003 m landmark objects.  The mesh data is resized, so object scale
    # remains 1/1/1 even in the Transform panel.
    return HFR_LM_DIAMETER * 0.5


def set_landmark_size(obj, radius):
    # Keep object scale at 1/1/1 and resize the mesh data itself.  This prevents
    # Blender's Transform panel from showing Scale 0.1 or similar helper values.
    diameter = max(radius * 2.0, 0.001)
    if obj is None:
        return
    if getattr(obj, "type", None) == 'MESH' and obj.data and obj.data.vertices:
        obj.scale = (1.0, 1.0, 1.0)
        xs = [v.co.x for v in obj.data.vertices]
        ys = [v.co.y for v in obj.data.vertices]
        zs = [v.co.z for v in obj.data.vertices]
        minv = Vector((min(xs), min(ys), min(zs)))
        maxv = Vector((max(xs), max(ys), max(zs)))
        center = (minv + maxv) * 0.5
        max_dim = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z, 1e-8)
        factor = diameter / max_dim
        for v in obj.data.vertices:
            v.co = (v.co - center) * factor
        obj.data.update()
        obj.scale = (1.0, 1.0, 1.0)
    else:
        try:
            obj.scale = (1.0, 1.0, 1.0)
            obj.dimensions = (diameter, diameter, diameter)
            obj.scale = (1.0, 1.0, 1.0)
        except Exception:
            pass


def current_guide_bevel(scene=None, context=None):
    if scene is None:
        scene = bpy.context.scene
    return max(current_landmark_radius(scene, context) * 0.045, 0.0005)


GUIDES = [
    # Eye loops: 8-sided eyelid guide rings.
    ("eye_l_inner", "eye_l_upper_inner"), ("eye_l_upper_inner", "eye_l_upper"),
    ("eye_l_upper", "eye_l_upper_outer"), ("eye_l_upper_outer", "eye_l_outer"),
    ("eye_l_outer", "eye_l_lower_outer"), ("eye_l_lower_outer", "eye_l_lower"),
    ("eye_l_lower", "eye_l_lower_inner"), ("eye_l_lower_inner", "eye_l_inner"),
    ("eye_r_inner", "eye_r_upper_inner"), ("eye_r_upper_inner", "eye_r_upper"),
    ("eye_r_upper", "eye_r_upper_outer"), ("eye_r_upper_outer", "eye_r_outer"),
    ("eye_r_outer", "eye_r_lower_outer"), ("eye_r_lower_outer", "eye_r_lower"),
    ("eye_r_lower", "eye_r_lower_inner"), ("eye_r_lower_inner", "eye_r_inner"),

    # Brows
    ("brow_l_inner", "brow_l_center"), ("brow_l_center", "brow_l_outer"),
    ("brow_r_inner", "brow_r_center"), ("brow_r_center", "brow_r_outer"),
    ("brow_l_inner", "nose_root"), ("brow_r_inner", "nose_root"),

    # Nose
    ("nose_root", "nose_bridge_top"), ("nose_bridge_top", "nose_bridge"),
    ("nose_bridge", "nose_tip"), ("nose_tip", "nose_base"),
    ("nose_bridge", "nose_l_side_upper"), ("nose_bridge", "nose_r_side_upper"),
    ("nose_l_side_upper", "nose_l_side_lower"), ("nose_r_side_upper", "nose_r_side_lower"),
    ("nose_l_side_lower", "nose_l_alar"), ("nose_r_side_lower", "nose_r_alar"),
    ("nose_l_alar", "nose_l_nostril"), ("nose_r_alar", "nose_r_nostril"),
    ("nose_l_nostril", "nose_base"), ("nose_r_nostril", "nose_base"),

    # Mouth loop
    ("mouth_l_corner", "mouth_l_upper"), ("mouth_l_upper", "mouth_upper_mid"),
    ("mouth_upper_mid", "mouth_r_upper"), ("mouth_r_upper", "mouth_r_corner"),
    ("mouth_r_corner", "mouth_r_lower"), ("mouth_r_lower", "mouth_lower_mid"),
    ("mouth_lower_mid", "mouth_l_lower"), ("mouth_l_lower", "mouth_l_corner"),

    # Center line
    ("forehead_upper_center", "forehead_center"), ("forehead_center", "nose_root"),
    ("nose_base", "mouth_upper_mid"), ("mouth_lower_mid", "chin_center"),
    ("chin_center", "neck_front_center"),

    # Cheek / outer / jaw / chin
    ("eye_l_outer", "cheek_l_center"), ("eye_r_outer", "cheek_r_center"),
    ("cheek_l_center", "mouth_l_corner"), ("cheek_r_center", "mouth_r_corner"),
    ("cheek_l_center", "outer_face_l_upper"), ("cheek_r_center", "outer_face_r_upper"),
    ("outer_face_l_upper", "face_l_edge"), ("outer_face_r_upper", "face_r_edge"),
    ("face_l_edge", "outer_face_l_lower"), ("face_r_edge", "outer_face_r_lower"),
    ("outer_face_l_lower", "chin_l_outer"), ("outer_face_r_lower", "chin_r_outer"),
    ("chin_l_outer", "chin_l_lower_outer"), ("chin_r_outer", "chin_r_lower_outer"),
    ("chin_l_lower_outer", "chin_l_lower"), ("chin_r_lower_outer", "chin_r_lower"),
    ("chin_l_lower", "chin_center"), ("chin_r_lower", "chin_center"),
    ("chin_l_outer", "jaw_l_edge"), ("chin_r_outer", "jaw_r_edge"),
    ("jaw_l_edge", "neck_top_l_front"), ("jaw_r_edge", "neck_top_r_front"),

    # Forehead / scalp / side head
    ("forehead_center", "forehead_l_upper"), ("forehead_center", "forehead_r_upper"),
    ("forehead_l_upper", "temple_l_center"), ("forehead_r_upper", "temple_r_center"),
    ("temple_l_center", "outer_face_l_upper"), ("temple_r_center", "outer_face_r_upper"),
    ("forehead_upper_center", "scalp_front_center"), ("scalp_front_center", "scalp_top_center"),
    ("scalp_top_center", "scalp_back_center"),
    ("scalp_front_center", "scalp_l_front"), ("scalp_front_center", "scalp_r_front"),
    ("scalp_l_front", "scalp_l_top"), ("scalp_r_front", "scalp_r_top"),
    ("scalp_l_top", "head_l_side_upper"), ("scalp_r_top", "head_r_side_upper"),
    ("head_l_side_upper", "head_l_side_back"), ("head_r_side_upper", "head_r_side_back"),
    ("head_l_side_back", "nape_l_outer"), ("head_r_side_back", "nape_r_outer"),

    # Ear loops
    # v0.2.5: the lobe is now an explicit lower ear landmark.  Do not keep the
    # old direct front_lower -> back_lower shortcut; route through lobe instead.
    ("ear_l_top", "ear_l_front_upper"), ("ear_l_front_upper", "ear_l_front_middle"),
    ("ear_l_front_middle", "ear_l_front_lower"), ("ear_l_front_lower", "ear_l_lobe"),
    ("ear_l_lobe", "ear_l_back_lower"),
    ("ear_l_back_lower", "ear_l_back_middle"), ("ear_l_back_middle", "ear_l_back_upper"),
    ("ear_l_back_upper", "ear_l_top"),
    ("ear_l_inner_front_middle", "ear_l_inner_bottom"),
    ("ear_l_front_middle", "ear_l_inner_front_middle"),
    ("ear_l_inner_bottom", "ear_l_lobe"),
    ("ear_r_top", "ear_r_front_upper"), ("ear_r_front_upper", "ear_r_front_middle"),
    ("ear_r_front_middle", "ear_r_front_lower"), ("ear_r_front_lower", "ear_r_lobe"),
    ("ear_r_lobe", "ear_r_back_lower"),
    ("ear_r_back_lower", "ear_r_back_middle"), ("ear_r_back_middle", "ear_r_back_upper"),
    ("ear_r_back_upper", "ear_r_top"),
    ("ear_r_inner_front_middle", "ear_r_inner_bottom"),
    ("ear_r_front_middle", "ear_r_inner_front_middle"),
    ("ear_r_inner_bottom", "ear_r_lobe"),

    # Ear attachment guides
    ("temple_l_center", "ear_l_front_upper"), ("face_l_edge", "ear_l_front_middle"),
    ("jaw_l_edge", "ear_l_front_lower"), ("head_l_side_back", "ear_l_back_upper"),
    ("nape_l_outer", "ear_l_back_lower"),
    ("temple_r_center", "ear_r_front_upper"), ("face_r_edge", "ear_r_front_middle"),
    ("jaw_r_edge", "ear_r_front_lower"), ("head_r_side_back", "ear_r_back_upper"),
    ("nape_r_outer", "ear_r_back_lower"),

    # Neck / nape
    ("nape_l_outer", "nape_center"), ("nape_center", "nape_r_outer"),
    ("nape_l_outer", "neck_top_l_back"), ("nape_r_outer", "neck_top_r_back"),
    ("neck_l_side", "neck_top_l_side"), ("neck_l_side", "neck_top_l_front"),
    ("neck_top_l_side", "neck_top_l_back"),
    ("neck_r_side", "neck_top_r_side"), ("neck_r_side", "neck_top_r_front"),
    ("neck_top_r_side", "neck_top_r_back"),
    ("neck_front_center", "neck_top_l_front"), ("neck_front_center", "neck_top_r_front"),
    ("neck_back_center", "neck_top_l_back"), ("neck_back_center", "neck_top_r_back"),
]


def lm_obj_name(lm_id):
    return LM_OBJ_PREFIX + lm_id


def guide_obj_name(a, b):
    # Object names may be long, but keep them readable.
    return "HFR_G_" + a + "__" + b


OBSOLETE_GUIDES = {
    # v0.0.7 added routed chin / neck side guides, so these direct bypass
    # guide curves must be removed from older scenes.
    ("chin_l_outer", "chin_center"),
    ("chin_r_outer", "chin_center"),
    ("chin_l_outer", "chin_l_lower"),
    ("chin_r_outer", "chin_r_lower"),
    ("ear_l_front_lower", "ear_l_back_lower"),
    ("ear_r_front_lower", "ear_r_back_lower"),
    ("ear_l_inner_bottom", "ear_l_back_lower"),
    ("ear_r_inner_bottom", "ear_r_back_lower"),
    ("neck_top_l_front", "neck_top_l_side"),
    ("neck_l_side", "neck_top_l_back"),
    ("neck_top_r_front", "neck_top_r_side"),
    ("neck_r_side", "neck_top_r_back"),
}

OBSOLETE_ANCHOR_IDS = {
    # v0.2.9 briefly added these helper lobe landmarks, but they require more
    # control points than many existing ear templates have.  v0.2.10 returns to
    # a single lobe anchor and removes these helper groups when anchor groups are
    # refreshed.
    "ear_l_lobe_front", "ear_l_lobe_back",
    "ear_r_lobe_front", "ear_r_lobe_back",
}


def _is_obsolete_guide_pair(pair):
    if not pair or len(pair) != 2:
        return False
    a, b = pair
    return (a, b) in OBSOLETE_GUIDES or (b, a) in OBSOLETE_GUIDES


def cleanup_removed_landmarks_and_guides(remove_unused_guides=True):
    """Remove objects from older landmark layouts that no longer belong to this version."""
    valid_ids = set(LM_BY_ID.keys())
    valid_pairs = set(GUIDES)
    removed = 0
    for obj in list(bpy.data.objects):
        lm_id = obj.get(PID_LM_ID)
        if lm_id and lm_id not in valid_ids:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
            continue
        if remove_unused_guides and obj.get(PID_GUIDE):
            pair = (obj.get(PID_GUIDE_A), obj.get(PID_GUIDE_B))
            if (
                _is_obsolete_guide_pair(pair)
                or pair not in valid_pairs
                or pair[0] not in valid_ids
                or pair[1] not in valid_ids
            ):
                bpy.data.objects.remove(obj, do_unlink=True)
                removed += 1
    return removed


def ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def ensure_guide_collection():
    """Return the guide collection, migrating the legacy Memory collection."""
    guide = bpy.data.collections.get(COLL_GUIDE)
    legacy = bpy.data.collections.get(COLL_MEMORY_LEGACY)

    if guide is None and legacy is not None:
        legacy.name = COLL_GUIDE
        return legacy

    if guide is None:
        return ensure_collection(COLL_GUIDE)

    if legacy is not None and legacy != guide:
        for obj in list(legacy.objects):
            if obj.name not in guide.objects:
                guide.objects.link(obj)
            try:
                legacy.objects.unlink(obj)
            except Exception:
                pass
        if not legacy.objects:
            try:
                bpy.data.collections.remove(legacy)
            except Exception:
                pass
    return guide


def landmark_collection_for_group(group):
    if group == "ear":
        return ensure_collection(COLL_EAR)
    return ensure_collection(COLL_FACE)


def ensure_base_collections():
    ensure_collection(COLL_WORK)
    ensure_collection(COLL_FACE)
    ensure_collection(COLL_EAR)
    ensure_guide_collection()


def ensure_material(name, color):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    try:
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            if "Base Color" in bsdf.inputs:
                bsdf.inputs["Base Color"].default_value = color
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = color[3]
    except Exception:
        pass
    return mat


def material_for_group(group):
    color = GROUP_COLORS.get(group, GROUP_COLORS["center"])
    return ensure_material("HFR_LM_" + group, color)


def guide_material():
    return ensure_material("HFR_LM_Guide", GROUP_COLORS["guide"])


def unlink_from_other_collections(obj, keep_coll):
    if obj.name not in keep_coll.objects:
        keep_coll.objects.link(obj)
    for coll in list(obj.users_collection):
        if coll != keep_coll:
            try:
                coll.objects.unlink(obj)
            except Exception:
                pass


def set_obj_material(obj, mat):
    if hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def shade_smooth_landmark_object(obj):
    """Apply smooth shading to landmark mesh objects without changing geometry."""
    if obj is None or getattr(obj, "type", None) != 'MESH' or obj.data is None:
        return False
    if not obj.get(PID_LM):
        return False
    changed = False
    for poly in obj.data.polygons:
        if not poly.use_smooth:
            poly.use_smooth = True
            changed = True
    if changed:
        obj.data.update()
    return changed


def shade_smooth_all_landmarks():
    count = 0
    for obj in bpy.data.objects:
        if shade_smooth_landmark_object(obj):
            count += 1
    return count


def _hfr_deferred_landmark_smooth():
    try:
        shade_smooth_all_landmarks()
    except Exception:
        pass
    return None


def create_or_update_landmark(scene, lm, reset=False, context=None, fit_to_target=False):
    name = lm_obj_name(lm["id"])
    obj = bpy.data.objects.get(name)
    coll = landmark_collection_for_group(lm["grp"])
    mat = material_for_group(lm["grp"])
    loc = fitted_landmark_location(context, lm) if fit_to_target else Vector(lm["co"])
    radius = current_landmark_radius(scene, context)

    if obj is None:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=radius,
            location=loc,
        )
        obj = bpy.context.object
        obj.name = name
        if obj.data:
            obj.data.name = name + "_Mesh"
        unlink_from_other_collections(obj, coll)
    else:
        unlink_from_other_collections(obj, coll)
        if reset:
            obj.location = loc
        # Preserve user's manual placement unless reset is explicitly requested.

    set_landmark_size(obj, radius)
    obj[PID_LM] = True
    obj[PID_LM_ID] = lm["id"]
    obj[PID_LM_GRP] = lm["grp"]
    obj.show_name = False
    obj.show_in_front = bool(scene.hfr_lm_show_front and lm.get("front", True))
    obj.display_type = 'TEXTURED'
    set_obj_material(obj, mat)
    shade_smooth_landmark_object(obj)
    return obj


def find_landmark_obj(lm_id):
    obj = bpy.data.objects.get(lm_obj_name(lm_id))
    if obj and obj.get(PID_LM_ID) == lm_id:
        return obj
    # Fallback: search by IDProperty in case the object was renamed.
    for candidate in bpy.data.objects:
        if candidate.get(PID_LM_ID) == lm_id:
            return candidate
    return None


def landmark_location(lm_id):
    obj = find_landmark_obj(lm_id)
    if obj:
        return obj.matrix_world.translation.copy()
    lm = LM_BY_ID.get(lm_id)
    if lm:
        return Vector(lm["co"])
    return Vector((0, 0, 0))


def create_or_update_guide(a, b, recreate=False, scene=None, context=None):
    if a not in LM_BY_ID or b not in LM_BY_ID:
        return None
    coll = ensure_guide_collection()
    name = guide_obj_name(a, b)
    obj = bpy.data.objects.get(name)
    if recreate and obj:
        bpy.data.objects.remove(obj, do_unlink=True)
        obj = None

    loc_a = landmark_location(a)
    loc_b = landmark_location(b)

    if obj is None:
        curve = bpy.data.curves.new(name + "_Curve", 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 1
        curve.bevel_depth = current_guide_bevel(scene, context)
        curve.bevel_resolution = 2
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        obj = bpy.data.objects.new(name, curve)
        coll.objects.link(obj)
    else:
        if obj.type != 'CURVE':
            bpy.data.objects.remove(obj, do_unlink=True)
            return create_or_update_guide(a, b, recreate=False)
        unlink_from_other_collections(obj, coll)
        curve = obj.data
        if not curve.splines:
            spline = curve.splines.new('POLY')
            spline.points.add(1)
        else:
            spline = curve.splines[0]
            while len(spline.points) < 2:
                spline.points.add(1)

    spline = obj.data.splines[0]
    spline.points[0].co = (loc_a.x, loc_a.y, loc_a.z, 1.0)
    spline.points[1].co = (loc_b.x, loc_b.y, loc_b.z, 1.0)
    obj.data.bevel_depth = current_guide_bevel(scene, context)
    obj[PID_GUIDE] = True
    obj[PID_GUIDE_A] = a
    obj[PID_GUIDE_B] = b
    obj.show_in_front = False
    obj.hide_select = True
    set_obj_material(obj, guide_material())
    return obj


def refresh_all_guides(recreate=False, scene=None, context=None):
    ensure_base_collections()
    cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
    if recreate:
        for obj in list(bpy.data.objects):
            if obj.get(PID_GUIDE):
                bpy.data.objects.remove(obj, do_unlink=True)
    count = 0
    for a, b in GUIDES:
        if create_or_update_guide(a, b, recreate=False, scene=scene, context=context):
            count += 1
    return count


def apply_landmark_style_and_guides(scene=None, context=None):
    """Apply current HFR landmark style and refresh guides.

    Guides are always enabled in this branch.  This function is used by the
    Front property update callback and by cleanup/add/reset flows, so a separate
    UI refresh button is no longer required.
    """
    if scene is None:
        scene = bpy.context.scene
    if context is None:
        context = bpy.context
    ensure_base_collections()
    cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
    count = 0
    for lm in LANDMARKS:
        obj = find_landmark_obj(lm["id"])
        if not obj:
            continue
        obj.show_name = False
        obj.show_in_front = bool(getattr(scene, "hfr_lm_show_front", True) and lm.get("front", True))
        set_landmark_size(obj, current_landmark_radius(scene, context))
        set_obj_material(obj, material_for_group(lm["grp"]))
        shade_smooth_landmark_object(obj)
        count += 1
    refresh_all_guides(recreate=False, scene=scene, context=context)
    return count


def hfr_lm_front_update(self, context):
    # Changing Front should immediately apply style and refresh all guides.
    if _HFR_SYNC_LOCK:
        return
    apply_landmark_style_and_guides(context.scene, context)


def binding_mode_enabled(scene=None):
    if scene is None:
        scene = bpy.context.scene
    return bool(getattr(scene, "hfr_bind_mode_enabled", False))


def hfr_bind_mode_update(self, context):
    # Binding Mode is intentionally optional because live binding status/guides
    # can become expensive while many landmarks are being edited.
    scene = context.scene if context is not None else bpy.context.scene
    if scene is None:
        return
    if binding_mode_enabled(scene):
        if bool(getattr(scene, "hfr_bind_show_guides", True)):
            refresh_binding_guides(scene=scene, context=context)
    else:
        for obj in list(bpy.data.objects):
            if obj.get(PID_BIND_GUIDE):
                bpy.data.objects.remove(obj, do_unlink=True)


def _is_hfr_named_object(obj):
    name = getattr(obj, "name", "") or ""
    return name.startswith(LM_OBJ_PREFIX) or name.startswith("HFR_G_") or name.startswith("HFR_BG_")


def cleanup_hfr_collections(remove_unknown_hfr_named=True):
    """Remove obsolete HFR objects from Face/Ear/Guide collections.

    Face/Ear should contain only current landmark objects for this version.
    Guide should contain only current guide curves.  Unknown user objects are
    preserved unless they carry HFR landmark/guide properties or use HFR names.
    """
    ensure_base_collections()
    valid_ids = set(LM_BY_ID.keys())
    valid_pairs = set(GUIDES)
    valid_lm_names = {lm_obj_name(lm_id) for lm_id in valid_ids}
    valid_guide_names = {guide_obj_name(a, b) for a, b in valid_pairs}
    removed = 0
    moved = 0
    fixed = 0

    face_coll = bpy.data.collections.get(COLL_FACE)
    ear_coll = bpy.data.collections.get(COLL_EAR)
    guide_coll = bpy.data.collections.get(COLL_GUIDE)

    def remove_obj(obj):
        nonlocal removed
        if obj and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1

    # Face / Ear: keep only current landmarks that belong to that collection.
    for coll, expected_group in ((face_coll, "FACE"), (ear_coll, "EAR")):
        if not coll:
            continue
        for obj in list(coll.objects):
            lm_id = obj.get(PID_LM_ID)
            is_hfr_lm = bool(obj.get(PID_LM))
            is_hfr_guide = bool(obj.get(PID_GUIDE))
            hfr_named = _is_hfr_named_object(obj)

            if is_hfr_guide:
                remove_obj(obj)
                continue

            if is_hfr_lm or lm_id:
                if lm_id not in valid_ids:
                    remove_obj(obj)
                    continue
                lm = LM_BY_ID[lm_id]
                should_be_ear = lm.get("grp") == "ear"
                if (expected_group == "EAR") != should_be_ear:
                    target_coll = landmark_collection_for_group(lm.get("grp"))
                    unlink_from_other_collections(obj, target_coll)
                    moved += 1
                else:
                    obj[PID_LM] = True
                    obj[PID_LM_GRP] = lm.get("grp")
                    fixed += 1
                continue

            if remove_unknown_hfr_named and hfr_named and obj.name not in valid_lm_names:
                remove_obj(obj)

    # Guide: keep only current guide curves. Stale guide objects are removed.
    if guide_coll:
        for obj in list(guide_coll.objects):
            is_guide = bool(obj.get(PID_GUIDE))
            is_bind_guide = bool(obj.get(PID_BIND_GUIDE))
            pair = (obj.get(PID_GUIDE_A), obj.get(PID_GUIDE_B))
            hfr_named = _is_hfr_named_object(obj)
            if is_bind_guide:
                if obj.get(PID_BIND_LM) not in valid_ids:
                    remove_obj(obj)
                else:
                    fixed += 1
                continue
            if is_guide:
                if (
                    _is_obsolete_guide_pair(pair)
                    or pair not in valid_pairs
                    or pair[0] not in valid_ids
                    or pair[1] not in valid_ids
                ):
                    remove_obj(obj)
                else:
                    fixed += 1
                continue
            if remove_unknown_hfr_named and hfr_named:
                # Remove stale HFR_G_* guide curves even if they were created by
                # an older build before guide custom properties were written.
                if obj.name not in valid_guide_names:
                    remove_obj(obj)

    # Remove explicit obsolete guide names wherever they exist, including
    # duplicated .001/.002 objects left by previous guide topology.
    obsolete_names = set()
    for a, b in OBSOLETE_GUIDES:
        obsolete_names.add(guide_obj_name(a, b))
        obsolete_names.add(guide_obj_name(b, a))
    for obj in list(bpy.data.objects):
        base_name = obj.name.split(".")[0]
        if obj.name in obsolete_names or base_name in obsolete_names:
            remove_obj(obj)

    # Global pass for stale property-tagged objects that may be outside the three collections.
    removed += cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
    return removed, moved, fixed


def matching_landmarks_for_group(group):
    if group == 'ALL':
        return LANDMARKS
    if group == 'FACE':
        return [lm for lm in LANDMARKS if lm["grp"] not in {"ear"}]
    if group == 'EAR':
        return [lm for lm in LANDMARKS if lm["grp"] == "ear"]
    if group == 'EYE':
        return [lm for lm in LANDMARKS if lm["grp"] == "eye"]
    if group == 'MOUTH':
        return [lm for lm in LANDMARKS if lm["grp"] == "mouth"]
    if group == 'NOSE':
        return [lm for lm in LANDMARKS if lm["grp"] == "nose"]
    if group == 'SCALP':
        return [lm for lm in LANDMARKS if lm["grp"] in {"forehead", "scalp"}]
    if group == 'NECK':
        return [lm for lm in LANDMARKS if lm["grp"] == "neck"]
    return LANDMARKS


def all_lm_ids():
    return {lm["id"] for lm in LANDMARKS}


def mirror_id(lm_id, direction):
    if direction == 'L2R':
        if "_l_" in lm_id:
            return lm_id.replace("_l_", "_r_", 1)
        if lm_id.endswith("_l"):
            return lm_id[:-2] + "_r"
    else:
        if "_r_" in lm_id:
            return lm_id.replace("_r_", "_l_", 1)
        if lm_id.endswith("_r"):
            return lm_id[:-2] + "_l"
    return None


def apply_mirror(direction):
    ids = all_lm_ids()
    moved = 0
    for src_id in sorted(ids):
        dst_id = mirror_id(src_id, direction)
        if not dst_id or dst_id not in ids:
            continue
        src = find_landmark_obj(src_id)
        dst = find_landmark_obj(dst_id)
        if not src or not dst:
            continue
        loc = src.matrix_world.translation.copy()
        loc.x = -loc.x
        dst.location = loc
        moved += 1
    return moved


def apply_mirror_for_ids(changed_ids, direction):
    ids = all_lm_ids()
    moved = 0
    for src_id in sorted(set(changed_ids)):
        dst_id = mirror_id(src_id, direction)
        if not dst_id or dst_id not in ids:
            continue
        src = find_landmark_obj(src_id)
        dst = find_landmark_obj(dst_id)
        if not src or not dst:
            continue
        loc = src.matrix_world.translation.copy()
        loc.x = -loc.x
        dst.location = loc
        moved += 1
    return moved


def _schedule_live_update():
    global _HFR_TIMER_REGISTERED
    if _HFR_TIMER_REGISTERED:
        return
    _HFR_TIMER_REGISTERED = True
    bpy.app.timers.register(_hfr_live_update_timer, first_interval=0.03)


def _hfr_live_update_timer():
    global _HFR_SYNC_LOCK, _HFR_TIMER_REGISTERED
    _HFR_TIMER_REGISTERED = False
    if _HFR_SYNC_LOCK:
        return 0.05

    scene = bpy.context.scene
    if scene is None:
        _HFR_PENDING_LM_IDS.clear()
        return None

    changed_ids = set(_HFR_PENDING_LM_IDS)
    _HFR_PENDING_LM_IDS.clear()
    if not changed_ids:
        return None

    use_mirror = bool(getattr(scene, "hfr_lm_mirror_x", False))

    _HFR_SYNC_LOCK = True
    try:
        if use_mirror:
            apply_mirror_for_ids(changed_ids, getattr(scene, "hfr_lm_mirror_dir", 'L2R'))
        refresh_all_guides(recreate=False, scene=scene, context=bpy.context)
        if binding_mode_enabled(scene) and bool(getattr(scene, "hfr_bind_show_guides", True)):
            refresh_binding_guides(scene=scene, context=bpy.context)
    finally:
        _HFR_SYNC_LOCK = False
    return None


def force_landmark_mirror_sync(scene=None, context=None):
    """Flush only pending Landmark Mirror X edits before generation.

    v0.3.10 forced a full L/R landmark copy at generate time.  That could
    overwrite the side that was already in the intended position and make the
    cheek deformation worse.  This safer version applies only the landmarks that
    were actually reported as edited by the live-update queue.  If there is no
    pending edit, generation uses the current scene positions as-is.
    """
    global _HFR_SYNC_LOCK
    if scene is None:
        scene = bpy.context.scene
    if context is None:
        context = bpy.context
    if scene is None or not bool(getattr(scene, "hfr_lm_mirror_x", False)):
        _HFR_PENDING_LM_IDS.clear()
        return 0
    changed_ids = set(_HFR_PENDING_LM_IDS)
    if not changed_ids or _HFR_SYNC_LOCK:
        _HFR_PENDING_LM_IDS.clear()
        return 0
    moved = 0
    _HFR_SYNC_LOCK = True
    try:
        moved = apply_mirror_for_ids(changed_ids, getattr(scene, "hfr_lm_mirror_dir", 'L2R'))
        _HFR_PENDING_LM_IDS.clear()
        refresh_all_guides(recreate=False, scene=scene, context=context)
        if binding_mode_enabled(scene) and bool(getattr(scene, "hfr_bind_show_guides", True)):
            refresh_binding_guides(scene=scene, context=context)
    finally:
        _HFR_SYNC_LOCK = False
    return moved


@persistent
def hfr_lm_depsgraph_update(scene, depsgraph):
    if _HFR_SYNC_LOCK:
        return
    if scene is None:
        return
    found = False
    for update in depsgraph.updates:
        obj = getattr(update, "id", None)
        if not isinstance(obj, bpy.types.Object):
            continue
        if not obj.get(PID_LM):
            continue
        lm_id = obj.get(PID_LM_ID)
        if lm_id in LM_BY_ID:
            _HFR_PENDING_LM_IDS.add(lm_id)
            found = True
    if found:
        _schedule_live_update()


def ensure_live_update_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    if hfr_lm_depsgraph_update not in handlers:
        handlers.append(hfr_lm_depsgraph_update)


def remove_live_update_handler():
    handlers = bpy.app.handlers.depsgraph_update_post
    while hfr_lm_depsgraph_update in handlers:
        handlers.remove(hfr_lm_depsgraph_update)


def save_landmark_defaults_to_scene(scene):
    data = {}
    for lm in LANDMARKS:
        obj = find_landmark_obj(lm["id"])
        if obj:
            loc = obj.matrix_world.translation
            data[lm["id"]] = [loc.x, loc.y, loc.z]
    scene[PID_DEF_JSON] = json.dumps(data, separators=(",", ":"))
    return len(data)


def load_landmark_defaults_from_scene(scene):
    raw = scene.get(PID_DEF_JSON)
    if not raw:
        return 0
    try:
        data = json.loads(raw)
    except Exception:
        return 0
    count = 0
    for lm_id, co in data.items():
        if lm_id not in LM_BY_ID or not isinstance(co, list) or len(co) != 3:
            continue
        obj = find_landmark_obj(lm_id)
        if obj:
            obj.location = Vector((float(co[0]), float(co[1]), float(co[2])))
            count += 1
    return count


def landmark_position_export_payload():
    """Return copy/paste friendly JSON for replacing built-in landmark defaults."""
    items = []
    missing = []
    for lm in LANDMARKS:
        lm_id = lm["id"]
        obj = find_landmark_obj(lm_id)
        if obj is not None:
            loc = obj.matrix_world.translation.copy()
            source = "scene"
        else:
            loc = Vector(lm["co"])
            source = "fallback_default"
            missing.append(lm_id)
        items.append({
            "id": lm_id,
            "grp": lm.get("grp", ""),
            "front": bool(lm.get("front", True)),
            "co": [round(float(loc.x), 6), round(float(loc.y), 6), round(float(loc.z), 6)],
            "source": source,
        })
    return {
        "hfr_export_type": "landmark_positions",
        "addon_version": [1, 0, 0],
        "landmark_count": len(items),
        "missing_landmarks": missing,
        "landmarks": items,
    }


def write_landmark_position_export_text(context=None):
    payload = landmark_position_export_payload()
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    text = bpy.data.texts.get("HFR_Landmark_Position_Export")
    if text is None:
        text = bpy.data.texts.new("HFR_Landmark_Position_Export")
    text.clear()
    text.write(raw)
    if context is not None:
        try:
            context.window_manager.clipboard = raw
        except Exception:
            pass
    return text, payload


# -----------------------------------------------------------------------------
# Template binding utilities
# -----------------------------------------------------------------------------

def anchor_group_name(lm_id):
    return ANCHOR_GROUP_PREFIX + lm_id


def binding_guide_obj_name(lm_id):
    return "HFR_BG_" + lm_id


def _binding_status_for_lm(obj, lm_id):
    if obj is None or obj.type != 'MESH':
        return 'NO_TEMPLATE', 0
    group = obj.vertex_groups.get(anchor_group_name(lm_id))
    if group is None:
        return 'MISSING', 0
    count = len(vertex_indices_in_group(obj, group))
    if count <= 0:
        return 'EMPTY', 0
    return 'BOUND', count


def binding_status_summary(obj):
    missing = []
    empty = []
    bound = []
    for lm in LANDMARKS:
        lm_id = lm["id"]
        status, count = _binding_status_for_lm(obj, lm_id)
        if status == 'BOUND':
            bound.append((lm_id, count))
        elif status == 'EMPTY':
            empty.append(lm_id)
        else:
            missing.append(lm_id)
    return missing, empty, bound


def binding_landmark_items(self=None, context=None):
    obj = template_object(context) if context is not None else None
    items = []
    for lm in LANDMARKS:
        lm_id = lm["id"]
        status, count = _binding_status_for_lm(obj, lm_id)
        if status == 'BOUND':
            label = "[OK] LM_{} ({}v)".format(lm_id, count)
            desc = "Bound: {} template vertices".format(count)
        elif status == 'EMPTY':
            label = "[  ] LM_{} (empty)".format(lm_id)
            desc = "Anchor group exists but has no assigned vertices"
        elif status == 'MISSING':
            label = "[  ] LM_{} (missing)".format(lm_id)
            desc = "Anchor group has not been created yet"
        else:
            label = "[?] LM_{}".format(lm_id)
            desc = "Assign Template Mesh to show binding status"
        items.append((lm_id, label, desc))
    return items


def is_template_mesh(obj):
    return obj is not None and obj.type == 'MESH' and not obj.get(PID_LM) and not obj.get(PID_OUTPUT)


def template_object(context):
    if context is None:
        return None
    scene = context.scene
    obj = getattr(scene, "hfr_template_obj", None)
    if is_template_mesh(obj):
        return obj
    active = getattr(context.view_layer.objects, "active", None)
    if is_template_mesh(active):
        return active
    for candidate in getattr(context, "selected_objects", []):
        if is_template_mesh(candidate):
            return candidate
    return None

def addon_root_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def default_template_blend_path():
    return os.path.join(addon_root_dir(), TEMPLATE_ASSET_DIR, DEFAULT_TEMPLATE_BLEND)


def default_template_relative_path():
    return os.path.join(TEMPLATE_ASSET_DIR, DEFAULT_TEMPLATE_BLEND).replace("\\", "/")


def _clean_template_object_name(name):
    clean = (name or "").strip()
    return clean or DEFAULT_TEMPLATE_OBJECT


def _is_default_template_object(obj, object_name=None):
    if obj is None or getattr(obj, "type", None) != 'MESH':
        return False
    if obj.get(PID_OUTPUT) or obj.get(PID_LM):
        return False
    preferred = _clean_template_object_name(object_name)
    base = (getattr(obj, "name", "") or "").split(".")[0]
    if bool(obj.get(PID_TEMPLATE)):
        return True
    if base == preferred or base in DEFAULT_TEMPLATE_CANDIDATES:
        return True
    return False


def find_loaded_default_template(object_name=None):
    preferred = _clean_template_object_name(object_name)
    for obj in bpy.data.objects:
        if _is_default_template_object(obj, preferred):
            return obj
    return None


def remove_loaded_default_templates(object_name=None):
    removed = 0
    preferred = _clean_template_object_name(object_name)
    for obj in list(bpy.data.objects):
        if _is_default_template_object(obj, preferred):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def choose_template_asset_object_name(available_names, preferred_name=None):
    names = [str(name) for name in list(available_names or []) if str(name)]
    if not names:
        raise ValueError("Default template .blend has no objects")
    preferred = _clean_template_object_name(preferred_name)

    # Exact explicit preference first.  Existing scenes may still store the old
    # HFR_Base_Template value, so this is only a preference, not a hard failure.
    if preferred in names:
        return preferred

    for candidate in DEFAULT_TEMPLATE_CANDIDATES:
        if candidate in names:
            return candidate

    lowered = [(name, name.lower()) for name in names]

    # v0.5.3: the bundled file may contain helper objects such as Mesh_0.001
    # plus the actual template named FaceTemplate.  Any object name containing
    # "template" is now a valid auto candidate, not only HFR*/Retopo* names.
    template_named = [name for name, low in lowered if "template" in low]
    if len(template_named) == 1:
        return template_named[0]
    if template_named:
        for name in template_named:
            low = name.lower()
            if low.startswith("face") or low.startswith("hfr"):
                return name
        return template_named[0]

    # If there is exactly one object in the asset blend, use it.  It will still
    # be type-checked after append, so cameras/lights fail safely.
    if len(names) == 1:
        return names[0]

    preview = ", ".join(names[:12])
    if len(names) > 12:
        preview += ", ..."
    raise ValueError(
        "Template object not found automatically. Set Default Template Object or rename the mesh to FaceTemplate. Available: %s"
        % preview
    )


def default_template_binding_path():
    return os.path.join(addon_dir(), TEMPLATE_ASSET_DIR, DEFAULT_TEMPLATE_BINDING_JSON)


def load_bundled_template_binding_payload():
    path = default_template_binding_path()
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("hfr_export_type") != "template_binding" or not isinstance(payload.get("bindings"), dict):
        raise ValueError("Bundled template binding JSON is invalid: %s" % path)
    return payload


def apply_bundled_template_binding(context, obj, replace_existing=False):
    """Apply templates/HFRTemplateBinding.json when it is bundled.

    This is intentionally automatic for the default template path.  A shared
    add-on must not require end users to import binding data manually.  When the
    bundled blend already carries HFR_A_* groups this is normally a no-op; when
    a packaging/export path strips or replaces those groups, the JSON restores
    them.
    """
    payload = load_bundled_template_binding_payload()
    if payload is None:
        return 0, 0, False
    imported, skipped = import_template_binding_payload(context, obj, payload, replace_existing=replace_existing)
    try:
        obj["HFR_abind"] = int(imported)
    except Exception:
        pass
    return imported, skipped, True


def select_active_object(context, obj):
    if context is None or obj is None:
        return
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass
    try:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    except Exception:
        pass


def load_default_template_asset(context, replace=False, select_loaded=True):
    if context is None:
        raise ValueError("Context is not available")
    scene = context.scene
    object_name = _clean_template_object_name(getattr(scene, "hfr_tpl_obj_name", DEFAULT_TEMPLATE_OBJECT))
    blend_path = default_template_blend_path()
    if not os.path.exists(blend_path):
        raise FileNotFoundError("Default template file not found: %s" % blend_path)

    if not bool(replace):
        existing = find_loaded_default_template(object_name)
        if existing is not None:
            scene.hfr_template_obj = existing
            try:
                apply_bundled_template_binding(context, existing, replace_existing=False)
            except Exception as exc:
                print("[HFR] Bundled template binding apply skipped: %s" % exc)
            if bool(select_loaded):
                select_active_object(context, existing)
            return existing, "existing", 0

    removed = remove_loaded_default_templates(object_name) if bool(replace) else 0

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        chosen_name = choose_template_asset_object_name(data_from.objects, object_name)
        data_to.objects = [chosen_name]

    loaded = [obj for obj in data_to.objects if obj is not None]
    if not loaded:
        raise RuntimeError("Failed to append default template object from %s" % blend_path)

    obj = loaded[0]
    if obj.type != 'MESH':
        obj_name = obj.name
        bpy.data.objects.remove(obj, do_unlink=True)
        raise TypeError("Default template object must be a Mesh, but %s is %s" % (obj_name, obj.type))

    ensure_base_collections()
    work = ensure_collection(COLL_WORK)
    unlink_from_other_collections(obj, work)
    obj[PID_TEMPLATE] = True
    obj.show_name = False
    obj.show_in_front = False
    try:
        obj.display_type = 'TEXTURED'
    except Exception:
        pass
    scene.hfr_template_obj = obj
    try:
        apply_bundled_template_binding(context, obj, replace_existing=False)
    except Exception as exc:
        print("[HFR] Bundled template binding apply skipped: %s" % exc)
    if bool(select_loaded):
        select_active_object(context, obj)
    return obj, "loaded", removed


def should_auto_load_default_template(context):
    if context is None:
        return False
    scene = getattr(context, "scene", None)
    if scene is None:
        return False
    try:
        current = getattr(scene, "hfr_template_obj", None)
        if current is not None and is_template_mesh(current):
            return False
    except Exception:
        pass
    if find_loaded_default_template(getattr(scene, "hfr_tpl_obj_name", DEFAULT_TEMPLATE_OBJECT)) is not None:
        return True
    if not os.path.exists(default_template_blend_path()):
        return False
    # Avoid injecting the template into completely unrelated empty scenes when
    # possible.  A selected/active target mesh or an assigned Target Mesh is a
    # strong signal that the user is applying HFR to the current scene.
    if fit_target_object(context) is not None:
        return True
    for obj in getattr(scene, "objects", []):
        if is_mesh_fit_target(obj):
            return True
    return False


def auto_load_default_template_if_needed(context=None):
    if context is None:
        context = bpy.context
    if not should_auto_load_default_template(context):
        return 0
    try:
        obj, _state, _removed = load_default_template_asset(context, replace=False, select_loaded=False)
        return 1 if obj is not None else 0
    except Exception as exc:
        # Silent UI failure is worse during development, but operator-style
        # reports are not available from a timer. Keep this as a console notice.
        print("[HFR] Auto Load Default Template skipped: %s" % exc)
        return 0


def _hfr_deferred_auto_load_template():
    global _HFR_AUTO_TPL_TIMER_REGISTERED
    _HFR_AUTO_TPL_TIMER_REGISTERED = False
    try:
        auto_load_default_template_if_needed(bpy.context)
    except Exception as exc:
        print("[HFR] Auto Load Default Template failed: %s" % exc)
    return None


def schedule_auto_load_default_template(delay=0.20):
    global _HFR_AUTO_TPL_TIMER_REGISTERED
    if _HFR_AUTO_TPL_TIMER_REGISTERED:
        return
    _HFR_AUTO_TPL_TIMER_REGISTERED = True
    try:
        bpy.app.timers.register(_hfr_deferred_auto_load_template, first_interval=max(0.01, float(delay)))
    except Exception:
        _HFR_AUTO_TPL_TIMER_REGISTERED = False


def ensure_anchor_group(obj, lm_id):
    if obj is None or obj.type != 'MESH' or lm_id not in LM_BY_ID:
        return None
    name = anchor_group_name(lm_id)
    group = obj.vertex_groups.get(name)
    if group is None:
        group = obj.vertex_groups.new(name=name)
    return group


def vertex_indices_in_group(obj, group):
    if obj is None or obj.type != 'MESH' or group is None:
        return []
    result = []
    gidx = group.index
    for vert in obj.data.vertices:
        for item in vert.groups:
            if item.group == gidx:
                result.append((vert.index, float(item.weight)))
                break
    return result


def clear_vertex_group(obj, group):
    if obj is None or obj.type != 'MESH' or group is None:
        return
    indices = [v.index for v in obj.data.vertices]
    if indices:
        try:
            group.remove(indices)
        except RuntimeError:
            pass


def active_or_selected_landmark_id(context):
    active = getattr(context.view_layer.objects, "active", None)
    if active and active.get(PID_LM_ID) in LM_BY_ID:
        return active.get(PID_LM_ID)
    for obj in getattr(context, "selected_objects", []):
        if obj.get(PID_LM_ID) in LM_BY_ID:
            return obj.get(PID_LM_ID)
    lm_id = getattr(context.scene, "hfr_bind_lm_id", None)
    if lm_id in LM_BY_ID:
        return lm_id
    return LANDMARKS[0]["id"] if LANDMARKS else ""


def selected_template_vertex_indices(context, obj):
    if obj is None or obj.type != 'MESH':
        return []
    if obj.mode == 'EDIT':
        # This is valid when the template object is the active edit object.
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        return [v.index for v in bm.verts if v.select]
    return [v.index for v in obj.data.vertices if v.select]


def _activate_obj(context, obj):
    if obj is None:
        return None
    prev_active = context.view_layer.objects.active
    try:
        obj.select_set(True)
        context.view_layer.objects.active = obj
    except Exception:
        pass
    return prev_active


def _object_mode_for_group_edit(context, obj):
    prev_active = _activate_obj(context, obj)
    prev_mode = getattr(obj, "mode", 'OBJECT')
    if prev_mode != 'OBJECT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return prev_active, prev_mode


def _restore_mode_after_group_edit(context, obj, prev_active, prev_mode):
    try:
        if obj and obj.name in bpy.data.objects:
            context.view_layer.objects.active = obj
            if prev_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
    except Exception:
        pass
    try:
        if prev_active and prev_active.name in bpy.data.objects:
            context.view_layer.objects.active = prev_active
    except Exception:
        pass


def bind_vertices_to_landmark(context, obj, lm_id, indices, replace=True):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    if lm_id not in LM_BY_ID:
        raise ValueError("Active Landmark is invalid")
    clean_indices = sorted({int(i) for i in indices if 0 <= int(i) < len(obj.data.vertices)})
    if not clean_indices:
        raise ValueError("No selected template vertices")
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    try:
        group = ensure_anchor_group(obj, lm_id)
        if replace:
            clear_vertex_group(obj, group)
        group.add(clean_indices, 1.0, 'ADD')
        obj.data.update()
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    return len(clean_indices)


def clear_anchor_binding(context, obj, lm_id):
    if obj is None or obj.type != 'MESH' or lm_id not in LM_BY_ID:
        return 0
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    try:
        group = ensure_anchor_group(obj, lm_id)
        before = len(vertex_indices_in_group(obj, group))
        clear_vertex_group(obj, group)
        obj.data.update()
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    remove_binding_guide(lm_id)
    return before


def export_template_binding_payload(context, obj):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    try:
        bindings = {}
        missing = []
        empty = []
        bound = []
        for lm in LANDMARKS:
            lm_id = lm["id"]
            group_name = anchor_group_name(lm_id)
            group = obj.vertex_groups.get(group_name)
            if group is None:
                missing.append(lm_id)
                continue
            members = vertex_indices_in_group(obj, group)
            if not members:
                empty.append(lm_id)
                continue
            items = [[int(idx), round(float(weight), 6)] for idx, weight in sorted(members, key=lambda item: int(item[0]))]
            bindings[lm_id] = {
                "group": group_name,
                "vertices": items,
            }
            bound.append(lm_id)
        payload = {
            "hfr_export_type": "template_binding",
            "addon_version": [1, 0, 0],
            "template_object": obj.name,
            "mesh_name": obj.data.name,
            "vertex_count": len(obj.data.vertices),
            "edge_count": len(obj.data.edges),
            "face_count": len(obj.data.polygons),
            "landmark_count": len(LANDMARKS),
            "bound_count": len(bound),
            "missing_count": len(missing),
            "empty_count": len(empty),
            "missing": missing,
            "empty": empty,
            "bindings": bindings,
        }
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    return payload


def write_template_binding_export(context, obj):
    payload = export_template_binding_payload(context, obj)
    blob = json.dumps(payload, indent=2, ensure_ascii=False)
    text = bpy.data.texts.get("HFR_Template_Binding_Export")
    if text is None:
        text = bpy.data.texts.new("HFR_Template_Binding_Export")
    text.clear()
    text.write(blob)
    if context is not None:
        try:
            context.window_manager.clipboard = blob
        except Exception:
            pass
    return payload, text


def _binding_payload_from_context(context):
    blob = ""
    if context is not None:
        try:
            blob = context.window_manager.clipboard or ""
        except Exception:
            blob = ""
    if not blob.strip():
        text = bpy.data.texts.get("HFR_Template_Binding_Export")
        if text is not None:
            blob = text.as_string()
    if not blob.strip():
        raise ValueError("Clipboard or HFR_Template_Binding_Export text has no binding JSON")
    payload = json.loads(blob)
    if payload.get("hfr_export_type") != "template_binding" or not isinstance(payload.get("bindings"), dict):
        raise ValueError("Binding JSON is not an HFR template_binding export")
    return payload


def import_template_binding_payload(context, obj, payload, replace_existing=True):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Binding payload has no bindings dictionary")
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    imported = 0
    skipped = 0
    try:
        for raw_key, raw_value in bindings.items():
            lm_id = str(raw_key)
            if lm_id.startswith(ANCHOR_GROUP_PREFIX):
                lm_id = lm_id[len(ANCHOR_GROUP_PREFIX):]
            if lm_id not in LM_BY_ID:
                skipped += 1
                continue
            if isinstance(raw_value, dict):
                vertices = raw_value.get("vertices", [])
            else:
                vertices = raw_value
            if not isinstance(vertices, (list, tuple)):
                skipped += 1
                continue
            group = ensure_anchor_group(obj, lm_id)
            if replace_existing:
                clear_vertex_group(obj, group)
            added_for_group = 0
            for item in vertices:
                try:
                    idx = int(item[0])
                    weight = float(item[1]) if len(item) > 1 else 1.0
                except Exception:
                    skipped += 1
                    continue
                if idx < 0 or idx >= len(obj.data.vertices):
                    skipped += 1
                    continue
                weight = max(0.0, min(1.0, weight))
                if weight <= 0.0:
                    continue
                group.add([idx], weight, 'ADD')
                added_for_group += 1
            if added_for_group:
                imported += 1
        obj.data.update()
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    refresh_binding_guides(scene=context.scene if context is not None else None, context=context)
    return imported, skipped


def select_anchor_vertices(context, obj, lm_id):
    if obj is None or obj.type != 'MESH' or lm_id not in LM_BY_ID:
        return 0
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    try:
        group = obj.vertex_groups.get(anchor_group_name(lm_id))
        indices = {idx for idx, _w in vertex_indices_in_group(obj, group)}
        for vert in obj.data.vertices:
            vert.select = vert.index in indices
        obj.data.update()
    finally:
        try:
            context.view_layer.objects.active = obj
            if prev_mode == 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            else:
                # Switching to Edit Mode makes the selected anchor visible and ready to adjust.
                bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    return len(indices)


def anchor_group_centroid_world(obj, lm_id):
    if obj is None or obj.type != 'MESH' or lm_id not in LM_BY_ID:
        return None
    group = obj.vertex_groups.get(anchor_group_name(lm_id))
    members = vertex_indices_in_group(obj, group)
    if not members:
        return None
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for idx, weight in members:
        w = max(float(weight), 0.0001)
        acc += (obj.matrix_world @ obj.data.vertices[idx].co) * w
        total += w
    if total <= 0.0:
        return None
    return acc / total


def binding_guide_material():
    return ensure_material("HFR_LM_BindGuide", (1.0, 0.82, 0.12, 1.0))


def remove_binding_guide(lm_id):
    name = binding_guide_obj_name(lm_id)
    for obj in list(bpy.data.objects):
        base = obj.name.split(".")[0]
        if obj.name == name or base == name or (obj.get(PID_BIND_GUIDE) and obj.get(PID_BIND_LM) == lm_id):
            bpy.data.objects.remove(obj, do_unlink=True)


def create_or_update_binding_guide(lm_id, template, scene=None, context=None):
    if lm_id not in LM_BY_ID or template is None or template.type != 'MESH':
        return None
    loc_a = landmark_location(lm_id)
    loc_b = anchor_group_centroid_world(template, lm_id)
    if loc_b is None:
        remove_binding_guide(lm_id)
        return None
    coll = ensure_guide_collection()
    name = binding_guide_obj_name(lm_id)
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != 'CURVE':
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        curve = bpy.data.curves.new(name + "_Curve", 'CURVE')
        curve.dimensions = '3D'
        curve.resolution_u = 1
        curve.bevel_depth = max(current_guide_bevel(scene, context) * 0.70, 0.00035)
        curve.bevel_resolution = 1
        spline = curve.splines.new('POLY')
        spline.points.add(1)
        obj = bpy.data.objects.new(name, curve)
        coll.objects.link(obj)
    else:
        unlink_from_other_collections(obj, coll)
        curve = obj.data
        if not curve.splines:
            spline = curve.splines.new('POLY')
            spline.points.add(1)
        else:
            spline = curve.splines[0]
            while len(spline.points) < 2:
                spline.points.add(1)
    spline = obj.data.splines[0]
    spline.points[0].co = (loc_a.x, loc_a.y, loc_a.z, 1.0)
    spline.points[1].co = (loc_b.x, loc_b.y, loc_b.z, 1.0)
    obj.data.bevel_depth = max(current_guide_bevel(scene, context) * 0.70, 0.00035)
    obj[PID_BIND_GUIDE] = True
    obj[PID_BIND_LM] = lm_id
    obj[PID_BIND_OBJ] = template.name[:40]
    obj.show_in_front = False
    obj.hide_select = True
    set_obj_material(obj, binding_guide_material())
    return obj


def refresh_binding_guides(scene=None, context=None):
    if scene is None:
        scene = bpy.context.scene
    if context is None:
        context = bpy.context
    template = template_object(context)
    show = binding_mode_enabled(scene) and bool(getattr(scene, "hfr_bind_show_guides", True))
    if not show or template is None:
        for obj in list(bpy.data.objects):
            if obj.get(PID_BIND_GUIDE):
                bpy.data.objects.remove(obj, do_unlink=True)
        return 0
    count = 0
    for lm in LANDMARKS:
        if create_or_update_binding_guide(lm["id"], template, scene=scene, context=context):
            count += 1
    # Remove stale binding guide objects for landmarks that no longer exist.
    valid_ids = set(LM_BY_ID.keys())
    for obj in list(bpy.data.objects):
        if obj.get(PID_BIND_GUIDE) and obj.get(PID_BIND_LM) not in valid_ids:
            bpy.data.objects.remove(obj, do_unlink=True)
    return count


def mirror_anchor_groups(context, obj, direction='L2R', tolerance=0.01):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    quality_warnings = binding_quality_warnings(obj)
    side_warnings = binding_side_warnings(obj)
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    groups_done = 0
    verts_done = 0
    misses = 0
    try:
        verts = list(obj.data.vertices)
        coords = [v.co.copy() for v in verts]
        all_indices = [v.index for v in verts]
        for src_lm in sorted(all_lm_ids()):
            dst_lm = mirror_id(src_lm, direction)
            if not dst_lm or dst_lm not in LM_BY_ID:
                continue
            src_group = obj.vertex_groups.get(anchor_group_name(src_lm))
            if src_group is None:
                continue
            src_members = vertex_indices_in_group(obj, src_group)
            if not src_members:
                continue
            dst_group = ensure_anchor_group(obj, dst_lm)
            clear_vertex_group(obj, dst_group)
            assign_weights = {}
            for idx, weight in src_members:
                src_co = coords[idx]
                target = Vector((-src_co.x, src_co.y, src_co.z))
                best_i = None
                best_d = None
                for cand_i, cand_co in enumerate(coords):
                    d = (cand_co - target).length
                    if best_d is None or d < best_d:
                        best_i = all_indices[cand_i]
                        best_d = d
                if best_i is None or (tolerance > 0.0 and best_d is not None and best_d > tolerance):
                    misses += 1
                    continue
                assign_weights[best_i] = max(float(weight), assign_weights.get(best_i, 0.0))
            for vi, w in assign_weights.items():
                dst_group.add([vi], max(min(w, 1.0), 0.0), 'REPLACE')
                verts_done += 1
            groups_done += 1
        obj.data.update()
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)
    refresh_binding_guides(scene=context.scene, context=context)
    return groups_done, verts_done, misses




def binding_quality_warnings(obj):
    """Return non-fatal binding quality warnings.

    The template fitting can technically run with any non-empty anchor group,
    but sparse features such as the current ear lobe are sensitive to how many
    vertices are bound.  This helper reports cases that are likely to produce
    spikes even when the binding is formally complete.
    """
    warnings = []
    if obj is None or obj.type != 'MESH':
        return warnings

    def _count(lm_id):
        group = obj.vertex_groups.get(anchor_group_name(lm_id))
        if group is None:
            return None
        return len(vertex_indices_in_group(obj, group))

    for side in ("l", "r"):
        lm_id = f"ear_{side}_lobe"
        count = _count(lm_id)
        if count is None:
            continue
        group_name = anchor_group_name(lm_id)
        if count > 2:
            warnings.append(
                "%s has %d vertices. Current sparse ear-lobe topology should use 1-2 vertices; 2 is recommended."
                % (group_name, count)
            )
        elif count == 1:
            warnings.append(
                "%s has 1 vertex. This is valid, but 2 vertices usually keeps the lobe edge less spike-prone."
                % group_name
            )
    return warnings


def _binding_centroid_local(obj, lm_id):
    if obj is None or obj.type != 'MESH' or lm_id not in LM_BY_ID:
        return None
    group = obj.vertex_groups.get(anchor_group_name(lm_id))
    members = vertex_indices_in_group(obj, group)
    if not members:
        return None
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for idx, weight in members:
        if 0 <= idx < len(obj.data.vertices):
            w = max(float(weight), 0.0001)
            acc += obj.data.vertices[idx].co.copy() * w
            total += w
    if total <= 0.0:
        return None
    return acc / total


def _binding_symmetry_center_x(obj):
    center_ids = [
        "nose_root", "nose_bridge_top", "nose_bridge", "nose_tip", "nose_base",
        "mouth_upper_mid", "mouth_lower_mid", "chin_center", "forehead_center",
        "forehead_upper_center", "scalp_front_center", "scalp_top_center",
        "scalp_back_center", "nape_center", "neck_front_center", "neck_back_center",
    ]
    vals = []
    for lm_id in center_ids:
        co = _binding_centroid_local(obj, lm_id)
        if co is not None:
            vals.append(co.x)
    if vals:
        return sum(vals) / float(len(vals))
    try:
        xs = [v.co.x for v in obj.data.vertices]
        if xs:
            return (min(xs) + max(xs)) * 0.5
    except Exception:
        pass
    return 0.0


def binding_side_warnings(obj, tol=0.0):
    """Report suspicious left/right anchor binding asymmetry.

    A formally complete binding can still deform asymmetrically if a right-side
    HFR_A_* group is assigned to the wrong side or if its local centroid does not
    mirror the matching left group.  These warnings do not block generation; they
    point to binding issues that solver settings cannot fix reliably.
    """
    warnings = []
    if obj is None or obj.type != 'MESH':
        return warnings
    try:
        diag = max(float(obj.dimensions.length), 1.0e-6)
    except Exception:
        diag = 1.0
    limit = max(float(tol), diag * 0.012, 0.004)
    center_x = _binding_symmetry_center_x(obj)

    for lm in LANDMARKS:
        lm_id = lm["id"]
        co = _binding_centroid_local(obj, lm_id)
        if co is None:
            continue
        if "_l_" in lm_id and co.x > center_x + limit:
            warnings.append("%s centroid is on the right side of template center X." % anchor_group_name(lm_id))
        elif "_r_" in lm_id and co.x < center_x - limit:
            warnings.append("%s centroid is on the left side of template center X." % anchor_group_name(lm_id))

    checked = set()
    for lm in LANDMARKS:
        left_id = lm["id"]
        right_id = mirror_id(left_id, 'L2R')
        if not right_id or right_id not in LM_BY_ID:
            continue
        key = tuple(sorted((left_id, right_id)))
        if key in checked:
            continue
        checked.add(key)
        lco = _binding_centroid_local(obj, left_id)
        rco = _binding_centroid_local(obj, right_id)
        if lco is None or rco is None:
            continue
        mirrored_l = Vector((2.0 * center_x - lco.x, lco.y, lco.z))
        dist = (mirrored_l - rco).length
        if dist > limit * 2.5:
            warnings.append(
                "%s / %s anchor centroids are not mirrored. Local delta %.5f > %.5f."
                % (anchor_group_name(left_id), anchor_group_name(right_id), dist, limit * 2.5)
            )
    return warnings

def validate_template_binding(context, obj):
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    quality_warnings = binding_quality_warnings(obj)
    side_warnings = binding_side_warnings(obj)
    prev_active, prev_mode = _object_mode_for_group_edit(context, obj)
    try:
        missing = []
        empty = []
        bound = []
        for lm in LANDMARKS:
            lm_id = lm["id"]
            group = obj.vertex_groups.get(anchor_group_name(lm_id))
            if group is None:
                missing.append(lm_id)
                continue
            members = vertex_indices_in_group(obj, group)
            if not members:
                empty.append(lm_id)
            else:
                bound.append((lm_id, len(members)))
    finally:
        _restore_mode_after_group_edit(context, obj, prev_active, prev_mode)

    text = bpy.data.texts.get("HFR_Template_Binding_Report")
    if text is None:
        text = bpy.data.texts.new("HFR_Template_Binding_Report")
    text.clear()
    text.write("HFR Template Binding Report\n")
    text.write("Template Mesh: %s\n" % obj.name)
    text.write("Landmarks: %d\n" % len(LANDMARKS))
    text.write("Bound Groups: %d\n" % len(bound))
    text.write("Missing Groups: %d\n" % len(missing))
    text.write("Empty Groups: %d\n" % len(empty))
    text.write("Quality Warnings: %d\n" % len(quality_warnings))
    text.write("Side Binding Warnings: %d\n\n" % len(side_warnings))
    if quality_warnings:
        text.write("Quality Warnings\n")
        for warning in quality_warnings:
            text.write("- %s\n" % warning)
        text.write("\n")
    if side_warnings:
        text.write("Side Binding Warnings\n")
        for warning in side_warnings:
            text.write("- %s\n" % warning)
        text.write("\n")
    if missing:
        text.write("Missing Groups\n")
        for lm_id in missing:
            text.write("- %s\n" % anchor_group_name(lm_id))
        text.write("\n")
    if empty:
        text.write("Empty Groups\n")
        for lm_id in empty:
            text.write("- %s\n" % anchor_group_name(lm_id))
        text.write("\n")
    if bound:
        text.write("Bound Groups\n")
        for lm_id, count in bound:
            text.write("- %s : %d vertices\n" % (anchor_group_name(lm_id), count))
    refresh_binding_guides(scene=context.scene, context=context)
    return missing, empty, bound



# -----------------------------------------------------------------------------
# Template generation / deformation utilities
# -----------------------------------------------------------------------------

def generate_target_object(context):
    if context is None:
        return None
    scene = context.scene
    obj = getattr(scene, "hfr_lm_target_obj", None)
    if is_template_mesh(obj):
        return obj
    return None


def output_collection():
    ensure_base_collections()
    return ensure_collection(COLL_WORK)


def remove_existing_output_by_name(name):
    removed = 0
    for obj in list(bpy.data.objects):
        base = obj.name.split(".")[0]
        if obj.get(PID_OUTPUT) or obj.name == name or base == name:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def make_retopo_output_from_template(context, template, output_name="HFR_Retopo", replace=True, display_wire=True, show_in_front=False):
    if template is None or template.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    clean_name = (output_name or "HFR_Retopo").strip() or "HFR_Retopo"
    if replace:
        remove_existing_output_by_name(clean_name)
    out_obj = template.copy()
    out_obj.data = template.data.copy()
    out_obj.name = clean_name
    out_obj.data.name = clean_name + "_Mesh"
    out_obj[PID_OUTPUT] = True
    if PID_TEMPLATE in out_obj:
        try:
            del out_obj[PID_TEMPLATE]
        except Exception:
            pass
    out_obj.show_name = False
    out_obj.show_in_front = bool(show_in_front)
    try:
        out_obj.display_type = 'WIRE' if bool(display_wire) else 'TEXTURED'
    except Exception:
        pass
    output_collection().objects.link(out_obj)
    for coll in list(out_obj.users_collection):
        if coll.name != COLL_WORK:
            try:
                coll.objects.unlink(out_obj)
            except Exception:
                pass
    return out_obj


def _weighted_source_from_members(positions, members):
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for idx, weight in members:
        if 0 <= idx < len(positions):
            w = max(float(weight), 0.0001)
            acc += positions[idx].copy() * w
            total += w
    if total <= 0.0:
        return None
    return acc / total


def _merge_member_sets(member_sets, scales):
    merged = {}
    for members, scale in zip(member_sets, scales):
        s = max(float(scale), 0.0)
        if s <= 0.0:
            continue
        for idx, weight in members:
            if idx < 0:
                continue
            w = max(float(weight), 0.0001) * s
            merged[idx] = merged.get(idx, 0.0) + w
    return sorted((idx, w) for idx, w in merged.items() if w > 0.0)


def _synthetic_anchor_record(obj, lm_id, inv, source_positions=None):
    spec = SYNTHETIC_ANCHOR_SPECS.get(lm_id)
    if not spec:
        return None
    source_ids = tuple(spec.get("sources", ()))
    if len(source_ids) != 2:
        return None
    blend = max(0.0, min(1.0, float(spec.get("blend", 0.5))))
    positions = source_positions if source_positions and len(source_positions) == len(obj.data.vertices) else [v.co.copy() for v in obj.data.vertices]
    member_sets = []
    source_points = []
    for src_id in source_ids:
        group = obj.vertex_groups.get(anchor_group_name(src_id))
        members = vertex_indices_in_group(obj, group)
        if not members:
            return None
        src = _weighted_source_from_members(positions, members)
        if src is None:
            return None
        member_sets.append(members)
        source_points.append(src)
    source = source_points[0].lerp(source_points[1], blend)
    members = _merge_member_sets(member_sets, (1.0 - blend, blend))
    if not members:
        return None
    target = inv @ landmark_location(lm_id)
    return {
        "lm_id": lm_id,
        "source": source,
        "target": target,
        "delta": target - source,
        "members": members,
        "synthetic": True,
    }


def anchor_records_for_template(obj):
    """Return all bound anchor records in the object's local space."""
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    inv = obj.matrix_world.inverted()
    records = []
    positions = [v.co.copy() for v in obj.data.vertices]
    for lm in LANDMARKS:
        lm_id = lm["id"]
        group = obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(obj, group)
        if not members:
            synth = _synthetic_anchor_record(obj, lm_id, inv, source_positions=positions)
            if synth is not None:
                records.append(synth)
            continue
        source = _weighted_source_from_members(positions, members)
        if source is None:
            continue
        target = inv @ landmark_location(lm_id)
        records.append({
            "lm_id": lm_id,
            "source": source,
            "target": target,
            "delta": target - source,
            "members": members,
        })
    return records


def anchor_records_for_template_with_source_positions(obj, source_positions):
    """Return anchor records using supplied original/source vertex positions.

    This is used for post-snap feature re-locks. After snapping, current anchor
    positions are no longer the canonical template source positions, so using
    anchor_records_for_template() would often produce tiny deltas and make the
    post solver appear to do nothing.
    """
    if obj is None or obj.type != 'MESH':
        raise ValueError("Template Mesh is not assigned")
    if not source_positions or len(source_positions) != len(obj.data.vertices):
        return anchor_records_for_template(obj)
    inv = obj.matrix_world.inverted()
    records = []
    for lm in LANDMARKS:
        lm_id = lm["id"]
        group = obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(obj, group)
        if not members:
            synth = _synthetic_anchor_record(obj, lm_id, inv, source_positions=source_positions)
            if synth is not None:
                records.append(synth)
            continue
        source = _weighted_source_from_members(source_positions, members)
        if source is None:
            continue
        target = inv @ landmark_location(lm_id)
        records.append({
            "lm_id": lm_id,
            "source": source,
            "target": target,
            "delta": target - source,
            "members": members,
        })
    return records


def _idw_delta_for_point(co, records, power=2.0, nearest_count=12):
    if not records:
        return Vector((0.0, 0.0, 0.0))
    scored = []
    min_d = None
    min_rec = None
    for rec in records:
        d = (co - rec["source"]).length
        if min_d is None or d < min_d:
            min_d = d
            min_rec = rec
        scored.append((d, rec))
    if min_d is not None and min_d < 1.0e-8:
        return min_rec["delta"].copy()
    if nearest_count and nearest_count > 0 and nearest_count < len(scored):
        scored.sort(key=lambda item: item[0])
        scored = scored[:nearest_count]
    p = max(float(power), 0.01)
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for d, rec in scored:
        w = 1.0 / ((max(d, 1.0e-6) ** p) + 1.0e-9)
        acc += rec["delta"] * w
        total += w
    if total <= 0.0:
        return Vector((0.0, 0.0, 0.0))
    return acc / total


def _outer3(a, b):
    return Matrix((
        (a.x * b.x, a.x * b.y, a.x * b.z),
        (a.y * b.x, a.y * b.y, a.y * b.z),
        (a.z * b.x, a.z * b.y, a.z * b.z),
    ))


def _zero3x3():
    return Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def _mls_delta_for_point(co, records, power=2.0, nearest_count=18):
    """Weighted local affine deformation from landmark sources to targets.

    IDW averages landmark deltas and can leave wide areas between edited
    landmarks under-constrained. MLS solves a local affine map from nearby
    source anchors to target anchors, so intermediate vertices follow the
    implied local scale/shear/translation of the landmark cage.
    """
    if not records:
        return Vector((0.0, 0.0, 0.0))

    scored = []
    min_d = None
    min_rec = None
    for rec in records:
        d = (co - rec["source"]).length
        if min_d is None or d < min_d:
            min_d = d
            min_rec = rec
        scored.append((d, rec))

    if min_d is not None and min_d < 1.0e-8:
        return min_rec["delta"].copy()

    if nearest_count and nearest_count > 0 and nearest_count < len(scored):
        scored.sort(key=lambda item: item[0])
        scored = scored[:max(4, int(nearest_count))]

    p = max(float(power), 0.01)
    weighted = []
    total = 0.0
    for d, rec in scored:
        w = 1.0 / ((max(d, 1.0e-6) ** p) + 1.0e-9)
        weighted.append((w, rec))
        total += w
    if total <= 0.0:
        return _idw_delta_for_point(co, records, power=power, nearest_count=nearest_count)

    pbar = Vector((0.0, 0.0, 0.0))
    qbar = Vector((0.0, 0.0, 0.0))
    for w, rec in weighted:
        pbar += rec["source"] * w
        qbar += rec["target"] * w
    pbar /= total
    qbar /= total

    cov = _zero3x3()
    cross = _zero3x3()
    for w, rec in weighted:
        dp = rec["source"] - pbar
        dq = rec["target"] - qbar
        cov += _outer3(dp, dp) * w
        cross += _outer3(dq, dp) * w

    trace = abs(cov[0][0]) + abs(cov[1][1]) + abs(cov[2][2])
    reg = max(trace * 1.0e-5, 1.0e-8)
    cov[0][0] += reg
    cov[1][1] += reg
    cov[2][2] += reg

    try:
        inv = cov.inverted()
    except Exception:
        return _idw_delta_for_point(co, records, power=power, nearest_count=nearest_count)

    affine = cross @ inv
    target = qbar + (affine @ (co - pbar))
    return target - co


def apply_fixed_displacement_constraints(displacements, fixed_map):
    if not fixed_map:
        return 0
    changed = 0
    count = len(displacements)
    for idx, delta in fixed_map.items():
        if 0 <= idx < count:
            displacements[idx] = delta.copy()
            changed += 1
    return changed


def spread_fixed_displacement_constraints(obj, displacements, fixed_map, spread_steps=1, spread_strength=0.65):
    if obj is None or obj.type != 'MESH' or not fixed_map:
        return 0
    steps = max(0, int(spread_steps))
    strength = max(0.0, min(float(spread_strength), 1.0))
    if steps <= 0 or strength <= 0.0:
        return 0
    adj = build_mesh_adjacency(obj)
    fixed_keys = {idx for idx in fixed_map.keys() if 0 <= idx < len(displacements)}
    visited = set(fixed_keys)
    frontier = set(fixed_keys)
    changed = 0
    for ring in range(1, steps + 1):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in visited:
                    continue
                visited.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        ring_strength = strength * (1.0 - (ring - 1) / float(max(1, steps + 1)))
        for nb in nxt:
            direct = [d for r_idx, d in fixed_map.items() if 0 <= r_idx < len(displacements) and r_idx in adj[nb]]
            source = direct if direct else [d for i, d in fixed_map.items() if 0 <= i < len(displacements)]
            if not source:
                continue
            acc = Vector((0.0, 0.0, 0.0))
            for d in source:
                acc += d
            target = acc / float(len(source))
            displacements[nb] = displacements[nb].lerp(target, ring_strength)
            changed += 1
        frontier = nxt
    return changed


def mls_refine_displacements(original, displacements, records, power=2.0, nearest_count=18, mls_strength=0.75):
    strength = max(0.0, min(float(mls_strength), 1.0))
    if strength <= 0.0 or not records:
        return 0

    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0
    for vidx, co in enumerate(original):
        if vidx in fixed:
            continue
        mls_delta = _mls_delta_for_point(co, records, power=power, nearest_count=nearest_count)
        displacements[vidx] = displacements[vidx].lerp(mls_delta, strength)
        changed += 1
    return changed


def build_mesh_adjacency(obj):
    """Return vertex adjacency from the mesh edge graph."""
    count = len(obj.data.vertices)
    adj = [set() for _ in range(count)]
    for e in obj.data.edges:
        a, b = e.vertices
        if 0 <= a < count and 0 <= b < count:
            adj[a].add(b)
            adj[b].add(a)
    return [tuple(items) for items in adj]


def _anchor_delta_by_vertex(records, vert_count):
    """Build fixed anchor displacement constraints per vertex."""
    accum = {}
    for rec in records:
        delta = rec["delta"]
        for idx, weight in rec["members"]:
            if 0 <= idx < vert_count:
                w = max(float(weight), 0.0001)
                if idx not in accum:
                    accum[idx] = [Vector((0.0, 0.0, 0.0)), 0.0]
                accum[idx][0] += delta * w
                accum[idx][1] += w
    fixed = {}
    for idx, (vec, total) in accum.items():
        if total > 0.0:
            fixed[idx] = vec / total
    return fixed


def _closest_segment_factor(point, a, b):
    ab = b - a
    denom = ab.dot(ab)
    if denom <= 1.0e-12:
        return 0.0, (point - a).length
    t = max(0.0, min(1.0, (point - a).dot(ab) / denom))
    closest = a + ab * t
    return t, (point - closest).length


def build_boundary_adjacency(obj):
    """Return vertex adjacency using only open boundary edges."""
    if obj is None or obj.type != 'MESH':
        return []
    mesh = obj.data
    adj = [set() for _ in mesh.vertices]
    edge_counts = {}
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        count = len(verts)
        for i in range(count):
            a = int(verts[i])
            b = int(verts[(i + 1) % count])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    for a, b in mesh.edges.keys():
        a = int(a); b = int(b)
        key = (a, b) if a < b else (b, a)
        if edge_counts.get(key, 0) <= 1:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _boundary_component_from_seeds(boundary_adj, seeds, max_steps=96):
    if not boundary_adj:
        return set()
    seeds = [idx for idx in seeds if 0 <= idx < len(boundary_adj)]
    if not seeds:
        return set()
    max_steps = max(1, int(max_steps))
    seen = set(seeds)
    frontier = [(idx, 0) for idx in seeds]
    head = 0
    while head < len(frontier):
        cur, d = frontier[head]
        head += 1
        if d >= max_steps:
            continue
        for nb in boundary_adj[cur]:
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, d + 1))
    return seen


def _eye_loop_stats(loop_recs):
    sources = [rec["source"] for rec in loop_recs]
    if not sources:
        return 0.004, 0, 0, 0, 0
    seg_total = 0.0
    seg_count = 0
    n = len(sources)
    for i in range(n):
        seg = (sources[(i + 1) % n] - sources[i]).length
        if seg > 1.0e-8:
            seg_total += seg
            seg_count += 1
    avg_len = (seg_total / float(seg_count)) if seg_count else 0.004
    y_min = min(v.y for v in sources)
    y_max = max(v.y for v in sources)
    z_min = min(v.z for v in sources)
    z_max = max(v.z for v in sources)
    return avg_len, y_min, y_max, z_min, z_max


def _bfs_shortest_path_masked(adj, seed_indices, goal_indices, allowed_indices):
    seeds = [idx for idx in seed_indices if 0 <= idx < len(adj)]
    goals = set(idx for idx in goal_indices if 0 <= idx < len(adj))
    allowed = set(idx for idx in allowed_indices if 0 <= idx < len(adj)) if allowed_indices is not None else None
    if not seeds or not goals:
        return []
    from collections import deque
    q = deque()
    prev = {}
    for s in seeds:
        if allowed is not None and s not in allowed:
            continue
        prev[s] = None
        q.append(s)
    if not q:
        return []
    found = None
    while q:
        cur = q.popleft()
        if cur in goals:
            found = cur
            break
        for nb in adj[cur]:
            if allowed is not None and nb not in allowed:
                continue
            if nb in prev:
                continue
            prev[nb] = cur
            q.append(nb)
    if found is None:
        return []
    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _eye_loop_local_allowed_indices(original, loop_recs, side_sign, pad_scale=1.05):
    if not original or not loop_recs:
        return set()
    avg_len, y_min, y_max, z_min, z_max = _eye_loop_stats(loop_recs)
    pad = max(avg_len * max(float(pad_scale), 0.25), 0.005)
    sources = [rec["source"] for rec in loop_recs]
    segs = []
    for i in range(len(sources)):
        j = (i + 1) % len(sources)
        segs.append((sources[i], sources[j]))
    allowed = set()
    max_dist = max(avg_len * 1.15, 0.006)
    x_vals = [co.x for co in sources]
    x_min = min(x_vals) - pad
    x_max = max(x_vals) + pad
    for vidx, co in enumerate(original):
        if side_sign < 0.0 and co.x > x_max:
            continue
        if side_sign > 0.0 and co.x < x_min:
            continue
        if co.y < y_min - pad or co.y > y_max + pad:
            continue
        if co.z < z_min - pad or co.z > z_max + pad:
            continue
        best_dist = None
        for a, b in segs:
            _t, dist = _closest_segment_factor(co, a, b)
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is not None and best_dist <= max_dist:
            allowed.add(vidx)
    return allowed


def build_eye_member_path_constraints(obj, original, records, path_strength=1.0, max_path_len=24):
    """Constrain the visible eyelid support-row paths between eye landmark-bound vertices."""
    strength = max(0.0, min(float(path_strength), 1.0))
    if obj is None or obj.type != 'MESH' or strength <= 0.0 or not records:
        return {}
    adj = build_mesh_adjacency(obj)
    vert_count = len(original)
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    anchor_fixed = _anchor_delta_by_vertex(records, vert_count)
    brow_protect = brow_preserve_region_vertex_indices(obj, original, records, expand_steps=2)
    max_len = max(0, int(max_path_len))
    accum = {}

    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        allowed = _eye_loop_local_allowed_indices(original, loop_recs, side_sign, pad_scale=1.05)
        n = len(loop_recs)
        for i in range(n):
            j = (i + 1) % n
            a = loop_recs[i]
            b = loop_recs[j]
            a_seeds = _group_indices_from_record(a, vert_count)
            b_seeds = _group_indices_from_record(b, vert_count)
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            if brow_protect:
                local_allowed.difference_update(brow_protect)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                path = _bfs_shortest_path(adj, a_seeds, b_seeds)
                if local_allowed:
                    path = [idx for idx in path if idx in local_allowed or idx in a_seeds or idx in b_seeds]
            if len(path) < 2:
                continue
            if max_len > 0 and len(path) > max_len:
                continue
            seg_lengths = []
            total_len = 0.0
            for p in range(len(path) - 1):
                seg = (original[path[p + 1]] - original[path[p]]).length
                seg_lengths.append(seg)
                total_len += seg
            if total_len <= 1.0e-10:
                total_len = float(max(len(path) - 1, 1))
                seg_lengths = [1.0 for _ in range(max(len(path) - 1, 1))]
            travelled = 0.0
            for p, vidx in enumerate(path):
                if p > 0:
                    travelled += seg_lengths[p - 1]
                if vidx in anchor_fixed or vidx in brow_protect:
                    continue
                t = max(0.0, min(1.0, travelled / total_len))
                delta = a["delta"].lerp(b["delta"], t)
                interior = 1.0 - abs(t - 0.5) * 2.0
                w = strength * (0.82 + 0.18 * max(0.0, interior))
                if w <= 0.0:
                    continue
                if vidx not in accum:
                    accum[vidx] = [Vector((0.0, 0.0, 0.0)), 0.0]
                accum[vidx][0] += delta * w
                accum[vidx][1] += w

    fixed = {}
    for idx, (vec, total) in accum.items():
        if total > 0.0:
            fixed[idx] = vec / total
    return fixed


def apply_eye_member_path_fit(out_obj, original_positions, records, eye_strength=1.0, max_path_len=24):
    """Post-fit the visible eyelid support rows between eye landmarks."""
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(eye_strength), 1.0))
    if strength <= 0.0:
        return 0
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    brow_protect = brow_preserve_region_vertex_indices(out_obj, original_positions, records, expand_steps=2)
    max_len = max(0, int(max_path_len))
    changed = 0

    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        allowed = _eye_loop_local_allowed_indices(original_positions, loop_recs, side_sign, pad_scale=1.05)
        n = len(loop_recs)
        for i in range(n):
            j = (i + 1) % n
            a = loop_recs[i]
            b = loop_recs[j]
            a_seeds = _group_indices_from_record(a, len(original_positions))
            b_seeds = _group_indices_from_record(b, len(original_positions))
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            if brow_protect:
                local_allowed.difference_update(brow_protect)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                path = _bfs_shortest_path(adj, a_seeds, b_seeds)
                if local_allowed:
                    path = [idx for idx in path if idx in local_allowed or idx in a_seeds or idx in b_seeds]
            if len(path) < 2:
                continue
            if max_len > 0 and len(path) > max_len:
                continue
            seg_lengths = []
            total_len = 0.0
            for p in range(len(path) - 1):
                seg = (original_positions[path[p + 1]] - original_positions[path[p]]).length
                seg_lengths.append(seg)
                total_len += seg
            if total_len <= 1.0e-10:
                total_len = float(max(len(path) - 1, 1))
                seg_lengths = [1.0 for _ in range(max(len(path) - 1, 1))]
            travelled = 0.0
            for p, vidx in enumerate(path):
                if p > 0:
                    travelled += seg_lengths[p - 1]
                if vidx in fixed or vidx in brow_protect:
                    continue
                t = max(0.0, min(1.0, travelled / total_len))
                target_delta = a["delta"].lerp(b["delta"], t)
                target = original_positions[vidx] + target_delta
                weight = strength * (0.85 + 0.15 * (1.0 - abs(t - 0.5) * 2.0))
                verts[vidx].co = verts[vidx].co.lerp(target, min(1.0, weight))
                changed += 1
    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eyepth"] = int(changed)
    except Exception:
        pass
    return changed


def _eye_non_eye_anchor_blockers(records, vert_count):
    blockers = set()
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if lm_id.startswith("eye_"):
            continue
        for idx, _w in rec.get("members", []):
            if 0 <= idx < vert_count:
                blockers.add(idx)
    return blockers


def _eye_topology_band_candidates(obj, original, records, loop_recs, side_sign, steps=3, radius_scale=1.45):
    """Collect a local topological support band around same-side eye anchors."""
    if obj is None or obj.type != 'MESH' or not original or not loop_recs:
        return set()
    adj = build_mesh_adjacency(obj)
    vert_count = len(original)
    steps = max(1, int(steps))
    radius_scale = max(float(radius_scale), 0.10)
    seeds = []
    for rec in loop_recs:
        seeds.extend(_group_indices_from_record(rec, vert_count))
    seeds = [idx for idx in seeds if 0 <= idx < vert_count]
    if not seeds:
        return set()
    blockers = _eye_non_eye_anchor_blockers(records, vert_count)
    blockers.update(brow_preserve_region_vertex_indices(obj, original, records, expand_steps=2))
    seen = set(seeds)
    frontier = [(idx, 0) for idx in seeds]
    head = 0
    while head < len(frontier):
        cur, depth = frontier[head]
        head += 1
        if depth >= steps:
            continue
        for nb in adj[cur]:
            if nb in seen or nb in blockers:
                continue
            seen.add(nb)
            frontier.append((nb, depth + 1))
    avg_len, y_min, y_max, z_min, z_max = _eye_loop_stats(loop_recs)
    sources = [rec["source"] for rec in loop_recs]
    segs = []
    for i in range(len(loop_recs)):
        j = (i + 1) % len(loop_recs)
        seg_len = (sources[j] - sources[i]).length
        if seg_len > 1.0e-8:
            segs.append((i, j, seg_len))
    if not segs:
        return set()
    radius = max(avg_len * radius_scale, 0.006)
    pad_y = max(avg_len * 1.50, radius * 1.15, 0.006)
    pad_z = max(avg_len * 1.50, radius * 1.15, 0.006)
    x_vals = [co.x for co in sources]
    x_min = min(x_vals) - max(avg_len * 1.25, 0.006)
    x_max = max(x_vals) + max(avg_len * 1.25, 0.006)
    fixed = _anchor_delta_by_vertex(records, vert_count)
    candidates = set()
    for vidx in seen:
        if vidx in fixed or vidx in blockers:
            continue
        co = original[vidx]
        if side_sign < 0.0 and co.x > x_max:
            continue
        if side_sign > 0.0 and co.x < x_min:
            continue
        if co.y < y_min - pad_y or co.y > y_max + pad_y:
            continue
        if co.z < z_min - pad_z or co.z > z_max + pad_z:
            continue
        best_dist = None
        for i, j, _seg_len in segs:
            _t, dist = _closest_segment_factor(co, sources[i], sources[j])
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is not None and best_dist <= radius:
            candidates.add(vidx)
    return candidates


def eye_topology_band_refine_displacements(obj, original, displacements, records, eye_strength=1.0, band_steps=3, band_radius=1.45):
    """Move local eyelid support-band vertices using nearest eye-loop segment deltas."""
    strength = max(0.0, min(float(eye_strength), 1.0))
    if obj is None or obj.type != 'MESH' or strength <= 0.0 or not records:
        return 0
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        candidates = _eye_topology_band_candidates(obj, original, records, loop_recs, side_sign, steps=band_steps, radius_scale=band_radius)
        if not candidates:
            continue
        sources = [rec["source"] for rec in loop_recs]
        deltas = [rec["delta"] for rec in loop_recs]
        segs = []
        for i in range(len(loop_recs)):
            j = (i + 1) % len(loop_recs)
            seg_len = (sources[j] - sources[i]).length
            if seg_len > 1.0e-8:
                segs.append((i, j, seg_len))
        for vidx in candidates:
            if vidx in fixed:
                continue
            co = original[vidx]
            best = None
            for i, j, _seg_len in segs:
                t, dist = _closest_segment_factor(co, sources[i], sources[j])
                if best is None or dist < best[0]:
                    best = (dist, t, i, j)
            if best is None:
                continue
            _dist, t, i, j = best
            target_delta = deltas[i].lerp(deltas[j], t)
            displacements[vidx] = displacements[vidx].lerp(target_delta, strength)
            changed += 1
    return changed


def apply_eye_topology_band_fit(out_obj, original_positions, records, eye_strength=1.0, band_steps=3, band_radius=1.45):
    """Post-fit the local eyelid support band after broad deformation."""
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(eye_strength), 1.0))
    if strength <= 0.0:
        return 0
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    changed = 0
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        candidates = _eye_topology_band_candidates(out_obj, original_positions, records, loop_recs, side_sign, steps=band_steps, radius_scale=band_radius)
        if not candidates:
            continue
        sources = [rec["source"] for rec in loop_recs]
        deltas = [rec["delta"] for rec in loop_recs]
        segs = []
        for i in range(len(loop_recs)):
            j = (i + 1) % len(loop_recs)
            seg_len = (sources[j] - sources[i]).length
            if seg_len > 1.0e-8:
                segs.append((i, j, seg_len))
        for vidx in candidates:
            if vidx in fixed:
                continue
            co = original_positions[vidx]
            best = None
            for i, j, _seg_len in segs:
                t, dist = _closest_segment_factor(co, sources[i], sources[j])
                if best is None or dist < best[0]:
                    best = (dist, t, i, j)
            if best is None:
                continue
            _dist, t, i, j = best
            target_delta = deltas[i].lerp(deltas[j], t)
            verts[vidx].co = verts[vidx].co.lerp(co + target_delta, strength)
            changed += 1
    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eyeband"] = int(changed)
    except Exception:
        pass
    return changed


def _eye_boundary_seeds(boundary_adj, original, rec, side_sign, max_dist, y_min=None, y_max=None, z_min=None, z_max=None):
    if rec is None or not boundary_adj:
        return []
    vert_count = len(original)
    direct = []
    for idx, _w in rec.get("members", []):
        if 0 <= idx < len(boundary_adj) and boundary_adj[idx]:
            direct.append(idx)
    if direct:
        return direct

    src = rec.get("source")
    if src is None:
        return []
    best_idx = None
    best_dist = None
    pad = max(float(max_dist), 0.002)
    for idx, nbs in enumerate(boundary_adj):
        if not nbs or idx < 0 or idx >= vert_count:
            continue
        co = original[idx]
        if side_sign * co.x < -pad * 0.35:
            continue
        if y_min is not None and (co.y < y_min - pad or co.y > y_max + pad):
            continue
        if z_min is not None and (co.z < z_min - pad or co.z > z_max + pad):
            continue
        d = (co - src).length
        if best_dist is None or d < best_dist:
            best_dist = d
            best_idx = idx
    if best_idx is None:
        return []
    # A generous but finite limit: if the anchor was bound to a support row, the
    # nearest eye boundary can still be a few segment lengths away.  Avoid using
    # some unrelated open boundary elsewhere on the head.
    if best_dist is not None and best_dist <= max(float(max_dist) * 3.0, 0.015):
        return [best_idx]
    return []


def build_eye_boundary_path_constraints(obj, original, records, path_strength=1.0, max_path_len=48):
    """Constrain actual eye-hole boundary vertices between eye landmarks.

    Unlike v0.4.13/v0.4.14, this does not assume the landmark-bound vertices are
    themselves on the hole boundary.  If an anchor group was bound to a support
    row, the nearest same-side eye boundary vertex is used as the path endpoint.
    This directly addresses cases where intermediate eyelid boundary vertices
    remain static between the green eye landmarks.
    """
    strength = max(0.0, min(float(path_strength), 1.0))
    if obj is None or obj.type != 'MESH' or strength <= 0.0 or not records:
        return {}
    vert_count = len(original)
    boundary_adj = build_boundary_adjacency(obj)
    if not boundary_adj or not any(boundary_adj):
        return {}
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    anchor_fixed = _anchor_delta_by_vertex(records, vert_count)
    max_len = max(0, int(max_path_len))
    accum = {}

    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        avg_len, y_min, y_max, z_min, z_max = _eye_loop_stats(loop_recs)
        search_dist = max(avg_len * 2.2, 0.003)
        seeds_by_i = []
        for rec in loop_recs:
            seeds = _eye_boundary_seeds(
                boundary_adj, original, rec, side_sign, search_dist,
                y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
            )
            seeds_by_i.append(seeds)
        n = len(loop_recs)
        for i in range(n):
            j = (i + 1) % n
            a = loop_recs[i]
            b = loop_recs[j]
            a_seeds = seeds_by_i[i]
            b_seeds = seeds_by_i[j]
            if not a_seeds or not b_seeds:
                continue
            path = _bfs_shortest_path(boundary_adj, a_seeds, b_seeds)
            if len(path) < 2:
                continue
            if max_len > 0 and len(path) > max_len:
                continue
            seg_lengths = []
            total_len = 0.0
            for p in range(len(path) - 1):
                seg = (original[path[p + 1]] - original[path[p]]).length
                seg_lengths.append(seg)
                total_len += seg
            if total_len <= 1.0e-10:
                total_len = float(max(len(path) - 1, 1))
                seg_lengths = [1.0 for _p in range(max(len(path) - 1, 1))]
            travelled = 0.0
            for p, vidx in enumerate(path):
                if p > 0:
                    travelled += seg_lengths[p - 1]
                if vidx in anchor_fixed:
                    continue
                t = max(0.0, min(1.0, travelled / total_len))
                delta = a["delta"].lerp(b["delta"], t)
                interior = 1.0 - abs(t - 0.5) * 2.0
                w = strength * (0.78 + 0.22 * max(0.0, interior))
                if w <= 0.0:
                    continue
                if vidx not in accum:
                    accum[vidx] = [Vector((0.0, 0.0, 0.0)), 0.0]
                accum[vidx][0] += delta * w
                accum[vidx][1] += w

    fixed = {}
    for idx, (vec, total) in accum.items():
        if total > 0.0:
            fixed[idx] = vec / total
    return fixed


def apply_eye_boundary_loop_fit(out_obj, original_positions, records, eye_strength=1.0, eye_steps=96):
    """Post-fit only the eye-hole boundary component.

    This is intentionally boundary-only.  Brow / upper-lid support vertices are
    not candidates, so the v0.4.15 brow-pull regression is avoided.  Landmark
    anchor groups may sit on a support row; in that case nearest same-side eye
    boundary vertices are used to identify the component and interpolate deltas.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(eye_strength), 1.0))
    if strength <= 0.0:
        return 0
    boundary_adj = build_boundary_adjacency(out_obj)
    if not boundary_adj or not any(boundary_adj):
        return 0
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    changed = 0

    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        avg_len, y_min, y_max, z_min, z_max = _eye_loop_stats(loop_recs)
        search_dist = max(avg_len * 2.2, 0.003)
        seed_indices = []
        for rec in loop_recs:
            seed_indices.extend(_eye_boundary_seeds(
                boundary_adj, original_positions, rec, side_sign, search_dist,
                y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
            ))
        boundary_candidates = _boundary_component_from_seeds(boundary_adj, seed_indices, max_steps=max(16, int(eye_steps)))
        if not boundary_candidates:
            continue

        sources = [rec["source"] for rec in loop_recs]
        deltas = [rec["delta"] for rec in loop_recs]
        segs = []
        for i in range(len(loop_recs)):
            j = (i + 1) % len(loop_recs)
            seg_len = (sources[j] - sources[i]).length
            if seg_len > 1.0e-8:
                segs.append((i, j, seg_len))
        if not segs:
            continue
        pad = max(search_dist * 2.0, 0.006)
        for vidx in boundary_candidates:
            if vidx in fixed or vidx < 0 or vidx >= len(original_positions):
                continue
            src_co = original_positions[vidx]
            if side_sign * src_co.x < -pad * 0.25:
                continue
            if src_co.y < y_min - pad or src_co.y > y_max + pad:
                continue
            if src_co.z < z_min - pad or src_co.z > z_max + pad:
                continue
            best_dist = None
            best_delta = None
            for i, j, _seg_len in segs:
                t, dist = _closest_segment_factor(src_co, sources[i], sources[j])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_delta = deltas[i].lerp(deltas[j], t)
            if best_delta is None:
                continue
            # Boundary candidate is already on the hole rim, so use a strong fit.
            target = src_co + best_delta
            verts[vidx].co = verts[vidx].co.lerp(target, strength)
            changed += 1
    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eyebnd"] = int(changed)
    except Exception:
        pass
    return changed


def eye_boundary_region_vertex_indices(out_obj, steps=96):
    """Return current eye-hole boundary region for snap guarding."""
    if out_obj is None or out_obj.type != 'MESH':
        return set()
    try:
        records = anchor_records_for_template(out_obj)
    except Exception:
        return set()
    if not records:
        return set()
    boundary_adj = build_boundary_adjacency(out_obj)
    if not boundary_adj or not any(boundary_adj):
        return set()
    original = [v.co.copy() for v in out_obj.data.vertices]
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    result = set()
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        avg_len, y_min, y_max, z_min, z_max = _eye_loop_stats(loop_recs)
        search_dist = max(avg_len * 2.2, 0.003)
        seeds = []
        for rec in loop_recs:
            seeds.extend(_eye_boundary_seeds(
                boundary_adj, original, rec, side_sign, search_dist,
                y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
            ))
        result.update(_boundary_component_from_seeds(boundary_adj, seeds, max_steps=max(16, int(steps))))
    return result


def eye_support_region_vertex_indices(out_obj, steps=96, band_steps=3, band_radius=1.45, max_path_len=24):
    """Return both the actual eye-hole boundary and the visible eyelid support-row region.

    Earlier eye snap guarding only protected the open boundary component. That
    preserves the hole itself, but the visible intermediate eyelid row between
    eye landmarks can still be snapped back toward the target surface, which is
    exactly the row the user sees lagging behind. This helper expands the guard
    to include:

    - the real eye-hole boundary component,
    - masked shortest paths between adjacent eye landmark-bound groups on the
      visible eyelid row,
    - the small same-side topology band around those eye anchors.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return set()
    try:
        records = anchor_records_for_template(out_obj)
    except Exception:
        return set()
    if not records:
        return set()
    original = [v.co.copy() for v in out_obj.data.vertices]
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    result = set()
    result.update(eye_boundary_region_vertex_indices(out_obj, steps=steps))
    adj = build_mesh_adjacency(out_obj)
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        allowed = _eye_loop_local_allowed_indices(original, loop_recs, side_sign, pad_scale=1.05)
        band = _eye_topology_band_candidates(out_obj, original, records, loop_recs, side_sign, steps=band_steps, radius_scale=band_radius)
        result.update(band)
        n = len(loop_recs)
        for i in range(n):
            j = (i + 1) % n
            a_seeds = _group_indices_from_record(loop_recs[i], len(original))
            b_seeds = _group_indices_from_record(loop_recs[j], len(original))
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            local_allowed.update(band)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                continue
            if max_path_len > 0 and len(path) > max(8, int(max_path_len)):
                continue
            result.update(path)
    return result


def apply_eye_direct_loop_fit(out_obj, original_positions, records,
                              eye_strength=1.0, eye_radius=0.90):
    """Direct spatial fit for the visible eye loop band.

    This pass ignores topology path selection and uses source-space distance to
    the ordered eye landmark loop. It targets the row the user actually sees
    lagging between green eye landmarks, while keeping the band narrow enough to
    avoid the brow pull regression from v0.4.15.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(eye_strength), 1.0))
    radius_scale = max(float(eye_radius), 0.05)
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))

    protected = set(brow_preserve_region_vertex_indices(out_obj, original_positions, records, expand_steps=2))
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if not lm_id.startswith("eye_"):
            for idx, _w in rec.get("members", []):
                if 0 <= idx < len(verts):
                    protected.add(idx)

    changed = 0
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        sources = [rec["source"] for rec in loop_recs]
        deltas = [rec["delta"] for rec in loop_recs]
        segs = []
        total_len = 0.0
        for i in range(len(loop_recs)):
            j = (i + 1) % len(loop_recs)
            seg_len = (sources[j] - sources[i]).length
            if seg_len <= 1.0e-8:
                continue
            segs.append((i, j, seg_len))
            total_len += seg_len
        if not segs:
            continue
        avg_len = total_len / float(len(segs))
        radius = max(avg_len * radius_scale, 0.0035)
        pad = max(radius * 1.80, avg_len * 0.85, 0.005)
        x_vals = [co.x for co in sources]
        y_vals = [co.y for co in sources]
        z_vals = [co.z for co in sources]
        x_min = min(x_vals) - pad
        x_max = max(x_vals) + pad
        y_min = min(y_vals) - pad
        y_max = max(y_vals) + pad
        z_min = min(z_vals) - pad
        z_max = max(z_vals) + pad

        for vidx, src_co in enumerate(original_positions):
            if vidx in fixed or vidx in protected:
                continue
            if side_sign < 0.0 and src_co.x > x_max:
                continue
            if side_sign > 0.0 and src_co.x < x_min:
                continue
            if src_co.y < y_min or src_co.y > y_max:
                continue
            if src_co.z < z_min or src_co.z > z_max:
                continue
            best = None
            for i, j, _seg_len in segs:
                t, dist = _closest_segment_factor(src_co, sources[i], sources[j])
                if best is None or dist < best[0]:
                    best = (dist, t, i, j)
            if best is None:
                continue
            dist, t, i, j = best
            if dist > radius:
                continue
            prox = max(0.0, 1.0 - (dist / radius))
            w = strength * prox * prox * (3.0 - 2.0 * prox)
            if w <= 0.0:
                continue
            target_delta = deltas[i].lerp(deltas[j], t)
            target = src_co + target_delta
            verts[vidx].co = verts[vidx].co.lerp(target, min(1.0, w))
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eyedir"] = int(out_obj.get("HFR_eyedir", 0)) + int(changed)
    except Exception:
        pass
    return changed


def _hfr_set_debug_vertex_group(obj, name, indices):
    """Create/update a diagnostic vertex group. Does not affect deformation."""
    if obj is None or obj.type != 'MESH':
        return 0
    safe_name = str(name)[:63]
    try:
        old = obj.vertex_groups.get(safe_name)
        if old is not None:
            obj.vertex_groups.remove(old)
        vg = obj.vertex_groups.new(name=safe_name)
        clean = sorted({int(i) for i in indices if 0 <= int(i) < len(obj.data.vertices)})
        if clean:
            vg.add(clean, 1.0, 'ADD')
        return len(clean)
    except Exception:
        return 0


def _eye_direct_candidate_indices(original_positions, records, eye_radius=0.90):
    """Return the exact source-space candidates that Eye Direct Fit would affect."""
    if not original_positions or not records:
        return {"eye_l": set(), "eye_r": set()}
    radius_scale = max(float(eye_radius), 0.05)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    protected = set()
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if not lm_id.startswith("eye_"):
            for idx, _w in rec.get("members", []):
                if 0 <= idx < len(original_positions):
                    protected.add(idx)
    result = {"eye_l": set(), "eye_r": set()}
    for loop_name in ("eye_l", "eye_r"):
        ids = FEATURE_LOOPS.get(loop_name)
        if not ids:
            continue
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        side_sign = -1.0 if loop_name.endswith("_l") else 1.0
        sources = [rec["source"] for rec in loop_recs]
        segs = []
        total_len = 0.0
        for i in range(len(loop_recs)):
            j = (i + 1) % len(loop_recs)
            seg_len = (sources[j] - sources[i]).length
            if seg_len <= 1.0e-8:
                continue
            segs.append((i, j, seg_len))
            total_len += seg_len
        if not segs:
            continue
        avg_len = total_len / float(len(segs))
        radius = max(avg_len * radius_scale, 0.0035)
        pad = max(radius * 1.80, avg_len * 0.85, 0.005)
        x_vals = [co.x for co in sources]
        y_vals = [co.y for co in sources]
        z_vals = [co.z for co in sources]
        x_min = min(x_vals) - pad
        x_max = max(x_vals) + pad
        y_min = min(y_vals) - pad
        y_max = max(y_vals) + pad
        z_min = min(z_vals) - pad
        z_max = max(z_vals) + pad
        for vidx, src_co in enumerate(original_positions):
            if vidx in fixed or vidx in protected:
                continue
            if side_sign < 0.0 and src_co.x > x_max:
                continue
            if side_sign > 0.0 and src_co.x < x_min:
                continue
            if src_co.y < y_min or src_co.y > y_max:
                continue
            if src_co.z < z_min or src_co.z > z_max:
                continue
            best_dist = None
            for i, j, _seg_len in segs:
                _t, dist = _closest_segment_factor(src_co, sources[i], sources[j])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
            if best_dist is not None and best_dist <= radius:
                result[loop_name].add(vidx)
    return result


def create_eye_brow_debug_groups(out_obj, original_positions, records, eye_radius=0.90):
    """Create vertex groups that show which vertices the current eye/brow solvers see."""
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    total = 0
    try:
        eye_candidates = _eye_direct_candidate_indices(original_positions, records, eye_radius=eye_radius)
        total += _hfr_set_debug_vertex_group(out_obj, "HFR_DBG_eye_l_direct", eye_candidates.get("eye_l", set()))
        total += _hfr_set_debug_vertex_group(out_obj, "HFR_DBG_eye_r_direct", eye_candidates.get("eye_r", set()))
    except Exception:
        pass
    try:
        # These are current-output index regions used by snap guards / brow rails.
        total += _hfr_set_debug_vertex_group(out_obj, "HFR_DBG_eye_snap", eye_support_region_vertex_indices(out_obj))
    except Exception:
        pass
    try:
        total += _hfr_set_debug_vertex_group(out_obj, "HFR_DBG_brow_rail", brow_rail_region_vertex_indices(out_obj))
    except Exception:
        pass
    try:
        out_obj["HFR_dbg"] = int(total)
    except Exception:
        pass
    return total


def feature_loop_refine_displacements(original, displacements, records, loop_strength=0.85, loop_radius=1.15, loops=None):
    """Pull vertices near eye/mouth anchor loops by segment-interpolated deltas.

    Global IDW + topology diffusion is intentionally broad.  For eyelids and
    lips that can still leave the vertices between anchors too influenced by
    nearby non-loop anchors.  This local pass finds vertices close to the
    original eye/lip anchor polylines and blends their displacement toward the
    linear delta of the closest loop segment.  It makes the visible hole/lip
    boundary follow the user's 8-point eye loop and mouth loop more cleanly.
    """
    strength = max(0.0, min(float(loop_strength), 1.0))
    radius_scale = max(float(loop_radius), 0.05)
    if strength <= 0.0 or not records:
        return 0

    rec_by_id = {rec["lm_id"]: rec for rec in records}
    changed = 0
    loop_map = loops if loops is not None else FEATURE_LOOPS
    for _loop_name, ids in loop_map.items():
        loop_recs = [rec_by_id.get(lm_id) for lm_id in ids]
        if any(rec is None for rec in loop_recs):
            continue
        n = len(loop_recs)
        if n < 3:
            continue
        sources = [rec["source"] for rec in loop_recs]
        deltas = [rec["delta"] for rec in loop_recs]
        segs = []
        total_len = 0.0
        for i in range(n):
            j = (i + 1) % n
            seg_len = (sources[j] - sources[i]).length
            if seg_len <= 1.0e-8:
                continue
            segs.append((i, j, seg_len))
            total_len += seg_len
        if not segs:
            continue
        avg_len = total_len / len(segs)
        radius = max(avg_len * radius_scale, 0.004)

        for vidx, co in enumerate(original):
            best_dist = None
            best_delta = None
            for i, j, _seg_len in segs:
                t, dist = _closest_segment_factor(co, sources[i], sources[j])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_delta = deltas[i].lerp(deltas[j], t)
            if best_dist is None or best_dist > radius or best_delta is None:
                continue
            x = max(0.0, 1.0 - (best_dist / radius))
            # Smooth falloff: strong on the actual feature-loop boundary, weak on
            # the surrounding cheek/lid/lip support geometry.
            w = strength * x * x * (3.0 - 2.0 * x)
            if w <= 0.0:
                continue
            displacements[vidx] = displacements[vidx].lerp(best_delta, w)
            changed += 1
    return changed


def stabilize_ear_lobe_records(records, y_strength=0.85):
    """Prevent ear_lobe anchors from drifting forward/back when used mainly as
    vertical lobe-length controls.

    The lobe landmark is usually pulled down in World Z.  If the bound lobe
    vertex centroid is slightly off the intended lower-ear point, its raw
    anchor delta can contain a large local-Y component and create a forward/back
    spike.  This pass keeps the lobe target Y close to the neighboring lower
    ear anchors while preserving its X/Z target.
    """
    strength = max(0.0, min(float(y_strength), 1.0))
    if strength <= 0.0 or not records:
        return 0
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    changed = 0
    for side in ("l", "r"):
        lobe = rec_by_id.get(f"ear_{side}_lobe")
        if lobe is None:
            continue
        refs = []
        for lm_id in (f"ear_{side}_front_lower", f"ear_{side}_back_lower", f"ear_{side}_inner_bottom"):
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                refs.append(rec["target"].y)
        if len(refs) < 2:
            continue
        ref_y = sum(refs) / len(refs)
        old_target = lobe["target"]
        new_target = old_target.copy()
        new_target.y = old_target.y * (1.0 - strength) + ref_y * strength
        lobe["target"] = new_target
        lobe["delta"] = new_target - lobe["source"]
        changed += 1
    return changed


def solve_ear_lobe_relative_records(records, solve_strength=1.0, xy_strength=1.0):
    """Solve ear_lobe targets relative to neighboring lower-ear anchors.

    Pulling LM_ear_*_lobe down should primarily lengthen the lobe downward.
    v0.2.6 still allowed the lobe to behave like an isolated free point, so a
    small source/target mismatch could turn that edit into a forward/up spike.
    This pass uses front_lower/back_lower/inner_bottom as the lower-ear frame:
    X/Y follows that frame, while Z preserves the user's lobe height relative to
    the same frame.
    """
    if not records:
        return 0
    strength = max(0.0, min(float(solve_strength), 1.0))
    xy_s = max(0.0, min(float(xy_strength), 1.0))
    if strength <= 0.0:
        return 0
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    changed = 0
    for side in ("l", "r"):
        lobe = rec_by_id.get(f"ear_{side}_lobe")
        if lobe is None:
            continue
        refs = []
        for lm_id in (f"ear_{side}_front_lower", f"ear_{side}_back_lower", f"ear_{side}_inner_bottom"):
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                refs.append(rec)
        if len(refs) < 2:
            continue
        src_avg = Vector((0.0, 0.0, 0.0))
        tgt_avg = Vector((0.0, 0.0, 0.0))
        delta_avg = Vector((0.0, 0.0, 0.0))
        for rec in refs:
            src_avg += rec["source"]
            tgt_avg += rec["target"]
            delta_avg += rec["delta"]
        inv_count = 1.0 / float(len(refs))
        src_avg *= inv_count
        tgt_avg *= inv_count
        delta_avg *= inv_count
        raw_target = lobe["target"].copy()
        raw_delta = lobe["delta"].copy()
        src_rel_z = lobe["source"].z - src_avg.z
        tgt_rel_z = raw_target.z - tgt_avg.z
        rel_delta_z = tgt_rel_z - src_rel_z
        solved_delta = raw_delta.copy()
        solved_delta.x = raw_delta.x * (1.0 - xy_s) + delta_avg.x * xy_s
        solved_delta.y = raw_delta.y * (1.0 - xy_s) + delta_avg.y * xy_s
        solved_delta.z = delta_avg.z + rel_delta_z
        new_delta = raw_delta.lerp(solved_delta, strength)
        lobe["target"] = lobe["source"] + new_delta
        lobe["delta"] = new_delta
        changed += 1
    return changed


def feature_line_refine_displacements(original, displacements, records, lines=None, line_strength=0.85, line_radius=1.0):
    """Refine displacements along open feature rails.

    This is intentionally not a closed loop. The lower ear/lobe path is an open
    rail, so closing it can pull lower-ear vertices toward unrelated upper/front
    ear anchors.
    """
    strength = max(0.0, min(float(line_strength), 1.0))
    radius_scale = max(float(line_radius), 0.05)
    if strength <= 0.0 or not records:
        return 0
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    rail_map = lines if lines is not None else EAR_LOWER_RAILS
    changed = 0
    for _rail_name, pairs in rail_map.items():
        segs = []
        total_len = 0.0
        for a_id, b_id in pairs:
            a = rec_by_id.get(a_id)
            b = rec_by_id.get(b_id)
            if a is None or b is None:
                continue
            seg_len = (b["source"] - a["source"]).length
            if seg_len <= 1.0e-8:
                continue
            segs.append((a, b, seg_len))
            total_len += seg_len
        if not segs:
            continue
        avg_len = total_len / float(len(segs))
        radius = max(avg_len * radius_scale, 0.003)
        for vidx, co in enumerate(original):
            best_dist = None
            best_delta = None
            for a, b, _seg_len in segs:
                t, dist = _closest_segment_factor(co, a["source"], b["source"])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_delta = a["delta"].lerp(b["delta"], t)
            if best_dist is None or best_dist > radius or best_delta is None:
                continue
            x = max(0.0, 1.0 - (best_dist / radius))
            w = strength * x * x * (3.0 - 2.0 * x)
            if w <= 0.0:
                continue
            displacements[vidx] = displacements[vidx].lerp(best_delta, w)
            changed += 1
    return changed


def guide_follow_refine_displacements(original, displacements, records, guide_pairs=None, guide_strength=0.55, guide_radius=1.10):
    """Refine vertices near landmark guides by interpolating between guide endpoints.

    The broad IDW + topology passes are anchor-centric: vertices directly bound to
    landmarks move correctly, but wide regions between two edited landmarks can
    stay comparatively static.  This pass treats the existing landmark guide
    graph as a soft deformation rail network.  Vertices close to a guide segment
    are nudged toward the delta obtained by linearly interpolating the two guide
    endpoint deltas.

    In practice this helps the forehead/scalp, cheek, jaw, and other between-
    landmark areas follow the user edits instead of only moving near the anchor
    points themselves.
    """
    strength = max(0.0, min(float(guide_strength), 1.0))
    radius_scale = max(float(guide_radius), 0.05)
    if strength <= 0.0 or not records:
        return 0

    rec_by_id = {rec["lm_id"]: rec for rec in records}
    pairs = guide_pairs if guide_pairs is not None else GUIDES
    segs = []
    for a_id, b_id in pairs:
        a = rec_by_id.get(a_id)
        b = rec_by_id.get(b_id)
        if a is None or b is None:
            continue
        seg_len = (b["source"] - a["source"]).length
        if seg_len <= 1.0e-8:
            continue
        segs.append((a, b, seg_len))
    if not segs:
        return 0

    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0
    min_radius = 0.004
    for vidx, co in enumerate(original):
        if vidx in fixed:
            continue
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        best = 0.0
        for a, b, seg_len in segs:
            radius = max(seg_len * radius_scale, min_radius)
            t, dist = _closest_segment_factor(co, a["source"], b["source"])
            if dist > radius:
                continue
            x = max(0.0, 1.0 - (dist / radius))
            prox = x * x * (3.0 - 2.0 * x)
            # Bias slightly toward the interior of each guide so the *between*
            # region moves more than it did in the anchor-only solve, while the
            # endpoints still remain continuous with the anchor-locked vertices.
            mid = 1.0 - abs(t - 0.5) * 2.0
            mid_boost = 0.45 + 0.55 * max(0.0, mid)
            w = prox * mid_boost
            if w <= 0.0:
                continue
            acc += a["delta"].lerp(b["delta"], t) * w
            total += w
            if w > best:
                best = w
        if total <= 0.0:
            continue
        target = acc / total
        blend = strength * min(1.0, max(best, total / 3.0))
        if blend <= 0.0:
            continue
        displacements[vidx] = displacements[vidx].lerp(target, blend)
        changed += 1
    return changed


def nose_web_refine_displacements(original, displacements, records, nose_strength=1.0, nose_radius=1.25, nose_samples=18):
    """Refine the nose side web between the bridge/tip rail and side rails.

    The broad solvers can move the explicit nose anchors correctly while leaving
    the small surface patch between LM_nose_bridge -> LM_nose_tip and
    LM_nose_l/r_side_upper -> LM_nose_l/r_side_lower comparatively static.  This
    pass treats each side of the nose as a narrow quad strip and blends vertices
    close to that strip toward bilinear landmark deltas.
    """
    strength = max(0.0, min(float(nose_strength), 1.0))
    radius_scale = max(float(nose_radius), 0.05)
    samples = max(4, int(nose_samples))
    if strength <= 0.0 or not records:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    bridge = rec_by_id.get("nose_bridge")
    tip = rec_by_id.get("nose_tip")
    if bridge is None or tip is None:
        return 0

    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0

    for side in ("l", "r"):
        upper = rec_by_id.get(f"nose_{side}_side_upper")
        lower = rec_by_id.get(f"nose_{side}_side_lower")
        if upper is None or lower is None:
            continue

        # Pre-sample the strip so the band still works when the bridge/tip rail
        # and side rail are not parallel in the template.
        strip = []
        avg_width = 0.0
        for i in range(samples + 1):
            t = float(i) / float(samples)
            c_src = bridge["source"].lerp(tip["source"], t)
            s_src = upper["source"].lerp(lower["source"], t)
            c_delta = bridge["delta"].lerp(tip["delta"], t)
            s_delta = upper["delta"].lerp(lower["delta"], t)
            width = (s_src - c_src).length
            avg_width += width
            strip.append((c_src, s_src, c_delta, s_delta))
        avg_width /= float(len(strip))
        if avg_width <= 1.0e-8:
            continue
        radius = max(avg_width * radius_scale, 0.0025)
        side_sign = -1.0 if side == "l" else 1.0

        for vidx, co in enumerate(original):
            if vidx in fixed:
                continue
            # Keep the pass on its own side.  The centerline is allowed only as a
            # small overlap so the bridge vertices remain continuous.
            if side_sign * co.x < -radius * 0.25:
                continue

            best_dist = None
            best_delta = None
            best_u = 0.0
            for c_src, s_src, c_delta, s_delta in strip:
                u, dist = _closest_segment_factor(co, c_src, s_src)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_u = u
                    best_delta = c_delta.lerp(s_delta, u)
            if best_dist is None or best_delta is None or best_dist > radius:
                continue

            prox = max(0.0, 1.0 - (best_dist / radius))
            u_band = 0.35 + 0.65 * (1.0 - abs(best_u - 0.5) * 0.35)
            w = strength * prox * prox * (3.0 - 2.0 * prox) * u_band
            if w <= 0.0:
                continue
            displacements[vidx] = displacements[vidx].lerp(best_delta, min(1.0, w))
            changed += 1
    return changed


def _segment_factor_unclamped(point, a, b):
    ab = b - a
    denom = ab.dot(ab)
    if denom <= 1.0e-12:
        return 0.0
    return (point - a).dot(ab) / denom


def _sample_three_point_strip(rec_a, rec_b, rec_c, t):
    t = max(0.0, min(1.0, float(t)))
    if t <= 0.5:
        local_t = t * 2.0
        src = rec_a["source"].lerp(rec_b["source"], local_t)
        delta = rec_a["delta"].lerp(rec_b["delta"], local_t)
    else:
        local_t = (t - 0.5) * 2.0
        src = rec_b["source"].lerp(rec_c["source"], local_t)
        delta = rec_b["delta"].lerp(rec_c["delta"], local_t)
    return src, delta


def _brow_rail_local_allowed_indices(original, brow_recs, side_sign, pad_scale=0.95):
    if not original or not brow_recs:
        return set()
    sources = [rec["source"] for rec in brow_recs]
    segs = []
    total = 0.0
    for i in range(len(sources) - 1):
        seg_len = (sources[i + 1] - sources[i]).length
        if seg_len > 1.0e-8:
            segs.append((sources[i], sources[i + 1]))
            total += seg_len
    if not segs:
        return set()
    avg_len = total / float(len(segs))
    pad = max(avg_len * max(float(pad_scale), 0.25), 0.003)
    radius = max(avg_len * 0.80, 0.0035)
    x_vals = [co.x for co in sources]
    y_vals = [co.y for co in sources]
    z_vals = [co.z for co in sources]
    x_min = min(x_vals) - pad
    x_max = max(x_vals) + pad
    y_min = min(y_vals) - pad * 1.35
    y_max = max(y_vals) + pad * 1.25
    z_min = min(z_vals) - pad * 1.35
    z_max = max(z_vals) + pad * 0.90
    allowed = set()
    for vidx, co in enumerate(original):
        if side_sign < 0.0 and co.x > x_max:
            continue
        if side_sign > 0.0 and co.x < x_min:
            continue
        if co.y < y_min or co.y > y_max:
            continue
        if co.z < z_min or co.z > z_max:
            continue
        best_dist = None
        for a, b in segs:
            _t, dist = _closest_segment_factor(co, a, b)
            if best_dist is None or dist < best_dist:
                best_dist = dist
        if best_dist is not None and best_dist <= radius:
            allowed.add(vidx)
    return allowed


def brow_rail_refine_displacements(obj, original, displacements, records, brow_strength=0.85, max_path_len=18):
    """Move the visible brow rail between brow landmarks so outer brow anchors do not leave stretched face fans behind."""
    strength = max(0.0, min(float(brow_strength), 1.0))
    if obj is None or obj.type != 'MESH' or strength <= 0.0 or not records:
        return 0
    adj = build_mesh_adjacency(obj)
    vert_count = len(original)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, vert_count)
    changed = 0
    for side in ("l", "r"):
        brow_recs = [rec_by_id.get(f"brow_{side}_inner"), rec_by_id.get(f"brow_{side}_center"), rec_by_id.get(f"brow_{side}_outer")]
        if any(rec is None for rec in brow_recs):
            continue
        side_sign = -1.0 if side == "l" else 1.0
        allowed = _brow_rail_local_allowed_indices(original, brow_recs, side_sign, pad_scale=0.95)
        for i in range(len(brow_recs) - 1):
            a = brow_recs[i]
            b = brow_recs[i + 1]
            a_seeds = _group_indices_from_record(a, vert_count)
            b_seeds = _group_indices_from_record(b, vert_count)
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                continue
            if max_path_len > 0 and len(path) > max(6, int(max_path_len)):
                continue
            seg_lengths = []
            total_len = 0.0
            for p in range(len(path) - 1):
                seg = (original[path[p + 1]] - original[path[p]]).length
                seg_lengths.append(seg)
                total_len += seg
            if total_len <= 1.0e-10:
                total_len = float(max(len(path) - 1, 1))
                seg_lengths = [1.0 for _ in range(max(len(path) - 1, 1))]
            travelled = 0.0
            for p, vidx in enumerate(path):
                if p > 0:
                    travelled += seg_lengths[p - 1]
                if vidx in fixed:
                    continue
                t = max(0.0, min(1.0, travelled / total_len))
                target_delta = a["delta"].lerp(b["delta"], t)
                weight = strength * (0.82 + 0.18 * (1.0 - abs(t - 0.5) * 2.0))
                displacements[vidx] = displacements[vidx].lerp(target_delta, min(1.0, weight))
                changed += 1
    return changed


def apply_brow_rail_support_fit(out_obj, original_positions, records, brow_strength=0.85, max_path_len=18):
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(brow_strength), 1.0))
    if strength <= 0.0:
        return 0
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    changed = 0
    for side in ("l", "r"):
        brow_recs = [rec_by_id.get(f"brow_{side}_inner"), rec_by_id.get(f"brow_{side}_center"), rec_by_id.get(f"brow_{side}_outer")]
        if any(rec is None for rec in brow_recs):
            continue
        side_sign = -1.0 if side == "l" else 1.0
        allowed = _brow_rail_local_allowed_indices(original_positions, brow_recs, side_sign, pad_scale=0.95)
        for i in range(len(brow_recs) - 1):
            a = brow_recs[i]
            b = brow_recs[i + 1]
            a_seeds = _group_indices_from_record(a, len(original_positions))
            b_seeds = _group_indices_from_record(b, len(original_positions))
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                continue
            if max_path_len > 0 and len(path) > max(6, int(max_path_len)):
                continue
            seg_lengths = []
            total_len = 0.0
            for p in range(len(path) - 1):
                seg = (original_positions[path[p + 1]] - original_positions[path[p]]).length
                seg_lengths.append(seg)
                total_len += seg
            if total_len <= 1.0e-10:
                total_len = float(max(len(path) - 1, 1))
                seg_lengths = [1.0 for _ in range(max(len(path) - 1, 1))]
            travelled = 0.0
            for p, vidx in enumerate(path):
                if p > 0:
                    travelled += seg_lengths[p - 1]
                if vidx in fixed:
                    continue
                t = max(0.0, min(1.0, travelled / total_len))
                target_delta = a["delta"].lerp(b["delta"], t)
                target = original_positions[vidx] + target_delta
                weight = strength * (0.84 + 0.16 * (1.0 - abs(t - 0.5) * 2.0))
                verts[vidx].co = verts[vidx].co.lerp(target, min(1.0, weight))
                changed += 1
    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_brpth"] = int(changed)
    except Exception:
        pass
    return changed


def brow_rail_region_vertex_indices(out_obj, max_path_len=18):
    if out_obj is None or out_obj.type != 'MESH':
        return set()
    try:
        records = anchor_records_for_template(out_obj)
    except Exception:
        return set()
    if not records:
        return set()
    original = [v.co.copy() for v in out_obj.data.vertices]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    result = set()
    for side in ("l", "r"):
        brow_recs = [rec_by_id.get(f"brow_{side}_inner"), rec_by_id.get(f"brow_{side}_center"), rec_by_id.get(f"brow_{side}_outer")]
        if any(rec is None for rec in brow_recs):
            continue
        side_sign = -1.0 if side == "l" else 1.0
        allowed = _brow_rail_local_allowed_indices(original, brow_recs, side_sign, pad_scale=0.95)
        for i in range(len(brow_recs) - 1):
            a_seeds = _group_indices_from_record(brow_recs[i], len(original))
            b_seeds = _group_indices_from_record(brow_recs[i + 1], len(original))
            if not a_seeds or not b_seeds:
                continue
            local_allowed = set(allowed)
            local_allowed.update(a_seeds)
            local_allowed.update(b_seeds)
            path = _bfs_shortest_path_masked(adj, a_seeds, b_seeds, local_allowed)
            if len(path) < 2:
                continue
            if max_path_len > 0 and len(path) > max(6, int(max_path_len)):
                continue
            result.update(path)
    return result


def brow_preserve_region_vertex_indices(obj, original_positions, records, expand_steps=2):
    """Protected brow band + immediate under-brow wedge.

    New direction after repeated eye-only attempts: do not keep trying to make
    eye passes also repair brow. Instead, explicitly protect the accepted brow
    band so v0.4.24-style eye passes cannot drag brow vertices or the inner
    under-brow wedge. This is used only as an exclusion mask for eye-specific
    passes; it does not add new brow deformation on its own.
    """
    if obj is None or obj.type != 'MESH' or not original_positions or not records:
        return set()
    adj = build_mesh_adjacency(obj)
    base = brow_rail_region_vertex_indices(obj, max_path_len=18)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    sources = []
    for side in ("l", "r"):
        for key in (f"brow_{side}_inner", f"brow_{side}_center", f"brow_{side}_outer", f"eye_{side}_upper", f"eye_{side}_outer", f"eye_{side}_inner"):
            rec = rec_by_id.get(key)
            if rec is not None:
                sources.append(rec["source"])
    if not sources:
        return set(base)
    x_vals = [co.x for co in sources]
    y_vals = [co.y for co in sources]
    z_vals = [co.z for co in sources]
    span_x = max(x_vals) - min(x_vals)
    span_y = max(y_vals) - min(y_vals)
    span_z = max(z_vals) - min(z_vals)
    pad_x = max(span_x * 0.35, 0.008)
    pad_y = max(span_y * 0.90, 0.010)
    pad_z = max(span_z * 0.85, 0.010)
    x_min = min(x_vals) - pad_x
    x_max = max(x_vals) + pad_x
    y_min = min(y_vals) - pad_y
    y_max = max(y_vals) + pad_y * 0.35
    z_min = min(z_vals) - pad_z
    z_max = max(z_vals) + pad_z * 0.50
    seen = set(idx for idx in base if 0 <= idx < len(original_positions))
    frontier = [(idx, 0) for idx in seen]
    head = 0
    max_depth = max(0, int(expand_steps))
    while head < len(frontier):
        cur, depth = frontier[head]
        head += 1
        if depth >= max_depth:
            continue
        for nb in adj[cur]:
            if nb in seen or not (0 <= nb < len(original_positions)):
                continue
            co = original_positions[nb]
            if co.x < x_min or co.x > x_max:
                continue
            if co.y < y_min or co.y > y_max:
                continue
            if co.z < z_min or co.z > z_max:
                continue
            seen.add(nb)
            frontier.append((nb, depth + 1))
    return seen


def brow_ridge_refine_displacements(original, displacements, records, brow_strength=0.80, brow_radius=1.15, brow_samples=20):
    """Refine the supraorbital band between the eye-upper rail and brow rail.

    Broad MLS / guide passes can pull the eye-upper area sharply toward brow
    landmarks, especially when the target has a strong brow ridge.  This pass
    treats the eyebrow region as a narrow strip between two ordered rails:

    - lower rail: eye_upper_inner -> eye_upper -> eye_upper_outer
    - upper rail: brow_inner -> brow_center -> brow_outer

    Vertices inside the strip follow the interpolated landmark deltas, while
    explicitly bound eye/brow anchors remain fixed.  The goal is to make the
    brow band follow without turning the eyelid loop into a hard crease.
    """
    strength = max(0.0, min(float(brow_strength), 1.0))
    radius_scale = max(float(brow_radius), 0.05)
    samples = max(6, int(brow_samples))
    if strength <= 0.0 or not records:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0

    for side in ("l", "r"):
        lower_a = rec_by_id.get(f"eye_{side}_upper_inner")
        lower_b = rec_by_id.get(f"eye_{side}_upper")
        lower_c = rec_by_id.get(f"eye_{side}_upper_outer")
        upper_a = rec_by_id.get(f"brow_{side}_inner")
        upper_b = rec_by_id.get(f"brow_{side}_center")
        upper_c = rec_by_id.get(f"brow_{side}_outer")
        if None in (lower_a, lower_b, lower_c, upper_a, upper_b, upper_c):
            continue

        strip = []
        avg_gap = 0.0
        z_vals = []
        y_vals = []
        for i in range(samples + 1):
            t = float(i) / float(samples)
            low_src, low_delta = _sample_three_point_strip(lower_a, lower_b, lower_c, t)
            up_src, up_delta = _sample_three_point_strip(upper_a, upper_b, upper_c, t)
            strip.append((low_src, up_src, low_delta, up_delta))
            avg_gap += (up_src - low_src).length
            z_vals.extend((low_src.z, up_src.z))
            y_vals.extend((low_src.y, up_src.y))
        avg_gap /= float(len(strip))
        if avg_gap <= 1.0e-8:
            continue

        radius = max(avg_gap * min(radius_scale, 1.35), 0.0012)
        z_margin = max(avg_gap * 0.65, radius * 1.1, 0.0015)
        y_margin = max(avg_gap * 0.80, radius * 1.3, 0.0015)
        side_sign = -1.0 if side == "l" else 1.0
        z_min = min(z_vals) - z_margin
        z_max = max(z_vals) + z_margin
        y_min = min(y_vals) - y_margin
        y_max = max(y_vals) + y_margin

        for vidx, src_co in enumerate(original):
            if vidx in fixed:
                continue
            if src_co.z < z_min or src_co.z > z_max:
                continue
            if src_co.y < y_min or src_co.y > y_max:
                continue
            inner_abs = min(abs(lower_a["source"].x), abs(upper_a["source"].x))
            if side_sign * src_co.x < max(inner_abs * 0.72, avg_gap * 0.18):
                continue

            best = None
            for low_src, up_src, low_delta, up_delta in strip:
                raw_u = _segment_factor_unclamped(src_co, low_src, up_src)
                # Do not let the brow pass grab vertices just below the
                # eye-upper rail / brow-inner transition.  Those vertices belong
                # to the glabella or eyelid support, and were the source of the
                # sharp upward spike under LM_brow_*_inner.
                if raw_u < 0.02 or raw_u > 1.04:
                    continue
                u = max(0.0, min(1.0, raw_u))
                closest = low_src.lerp(up_src, u)
                dist = (src_co - closest).length
                if dist > radius:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, u, low_delta, up_delta)
            if best is None:
                continue

            dist, u, low_delta, up_delta = best
            prox = max(0.0, 1.0 - (dist / radius))
            ridge = 0.55 + 0.45 * (1.0 - abs(u - 0.52) * 1.35)
            w = strength * prox * prox * (3.0 - 2.0 * prox) * max(0.25, ridge)
            if w <= 0.0:
                continue
            target_delta = low_delta.lerp(up_delta, u)
            displacements[vidx] = displacements[vidx].lerp(target_delta, min(1.0, w))
            changed += 1
    return changed


def apply_brow_inner_support_fit(out_obj, original_positions, records,
                                 support_strength=0.70, support_steps=2, support_radius=1.10):
    """Move the small under-brow/glabella support fan with brow_inner.

    Brow Ridge Fit intentionally excludes the innermost glabella fan so it will
    not collapse into the nose_root guide. On strong brow edits, however, the
    vertex directly connected under LM_brow_*_inner can then remain behind while
    the bound brow vertex moves, producing a small upward spike. This pass fills
    that gap with a very small same-side topological support solve.

    It moves only non-anchor vertices close to brow_inner in the source template
    and blends their target from brow_inner, eye_upper_inner, nose_root, and
    brow_center landmark deltas. It does not re-enable the old brow_inner ->
    nose_root guide rail.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0

    strength = max(0.0, min(float(support_strength), 1.0))
    steps = max(1, int(support_steps))
    radius_scale = max(float(support_radius), 0.05)
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    adj = build_mesh_adjacency(out_obj)
    changed = 0

    for side in ("l", "r"):
        brow = rec_by_id.get(f"brow_{side}_inner")
        brow_center = rec_by_id.get(f"brow_{side}_center")
        eye_upper = rec_by_id.get(f"eye_{side}_upper_inner")
        nose_root = rec_by_id.get("nose_root")
        if brow is None or eye_upper is None or nose_root is None:
            continue

        seeds = [idx for idx, _w in brow.get("members", []) if 0 <= idx < len(verts)]
        if not seeds:
            continue

        dist_map = {}
        frontier = []
        for idx in seeds:
            dist_map[idx] = 0
            frontier.append(idx)
        head = 0
        while head < len(frontier):
            cur = frontier[head]
            head += 1
            d = dist_map[cur]
            if d >= steps:
                continue
            for nb in adj[cur]:
                if nb not in dist_map:
                    dist_map[nb] = d + 1
                    frontier.append(nb)

        support_recs = [brow, eye_upper, nose_root]
        support_bias = [2.40, 1.00, 0.75]
        if brow_center is not None:
            support_recs.append(brow_center)
            support_bias.append(0.65)

        bsrc = brow["source"]
        esrc = eye_upper["source"]
        nsrc = nose_root["source"]
        csrc = brow_center["source"] if brow_center is not None else bsrc
        side_sign = -1.0 if side == "l" else 1.0
        inner_width = max(abs(bsrc.x - nsrc.x), abs(csrc.x - bsrc.x), 0.003)
        radius = max((bsrc - esrc).length, (bsrc - nsrc).length, inner_width) * radius_scale
        radius = max(radius, 0.003)
        z_min = min(bsrc.z, esrc.z, nsrc.z) - radius * 0.85
        z_max = max(bsrc.z, esrc.z, nsrc.z, csrc.z) + radius * 0.75
        y_min = min(bsrc.y, esrc.y, nsrc.y, csrc.y) - radius * 0.85
        y_max = max(bsrc.y, esrc.y, nsrc.y, csrc.y) + radius * 0.85
        x_max = max(abs(csrc.x), abs(esrc.x), abs(bsrc.x), inner_width) + radius * 0.35

        for vidx, topod in dist_map.items():
            if topod <= 0 or vidx in fixed:
                continue
            src_co = original_positions[vidx]
            sx = side_sign * src_co.x
            if sx < -radius * 0.10 or sx > x_max:
                continue
            if src_co.z < z_min or src_co.z > z_max:
                continue
            if src_co.y < y_min or src_co.y > y_max:
                continue
            db = (src_co - bsrc).length
            if db > radius:
                continue

            acc = Vector((0.0, 0.0, 0.0))
            total = 0.0
            for rec, bias in zip(support_recs, support_bias):
                d = (src_co - rec["source"]).length
                w = float(bias) / max(d * d, 1.0e-7)
                acc += rec["delta"] * w
                total += w
            if total <= 0.0:
                continue
            target_delta = acc / total
            target_co = src_co + target_delta

            topo_w = max(0.0, 1.0 - (float(topod) / float(steps + 1)))
            topo_w = topo_w * topo_w * (3.0 - 2.0 * topo_w)
            geo_w = max(0.0, 1.0 - (db / radius))
            geo_w = geo_w * geo_w * (3.0 - 2.0 * geo_w)
            w = strength * max(topo_w, geo_w * 0.70)
            if target_co.z < verts[vidx].co.z:
                w *= 1.12
            w = min(1.0, max(0.0, w))
            if w <= 0.0:
                continue
            verts[vidx].co = verts[vidx].co.lerp(target_co, w)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_brinn"] = int(changed)
    except Exception:
        pass
    return changed


def apply_brow_ridge_surface_fit(out_obj, original_positions, records, brow_strength=0.80, brow_radius=1.15, brow_samples=20, brow_smooth=0.22):
    """Post-fit and smooth the brow band without collapsing the eyelid loop.

    This pass re-uses the same eye-upper rail / brow rail strip as the broad
    brow refine, but applies the safe v0.4.2-style correction: original source
    positions are mapped to interpolated landmark deltas rather than projected
    onto a line.  A light restricted smoothing pass then removes harsh ridges in
    the middle of the band while leaving the explicit eye/brow anchors intact.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0

    strength = max(0.0, min(float(brow_strength), 1.0))
    radius_scale = max(float(brow_radius), 0.05)
    samples = max(6, int(brow_samples))
    smooth_strength = max(0.0, min(float(brow_smooth), 1.0))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original_positions))

    changed = 0
    smooth_candidates = {}

    for side in ("l", "r"):
        lower_a = rec_by_id.get(f"eye_{side}_upper_inner")
        lower_b = rec_by_id.get(f"eye_{side}_upper")
        lower_c = rec_by_id.get(f"eye_{side}_upper_outer")
        upper_a = rec_by_id.get(f"brow_{side}_inner")
        upper_b = rec_by_id.get(f"brow_{side}_center")
        upper_c = rec_by_id.get(f"brow_{side}_outer")
        if None in (lower_a, lower_b, lower_c, upper_a, upper_b, upper_c):
            continue

        strip = []
        avg_gap = 0.0
        z_vals = []
        y_vals = []
        for i in range(samples + 1):
            t = float(i) / float(samples)
            low_src, low_delta = _sample_three_point_strip(lower_a, lower_b, lower_c, t)
            up_src, up_delta = _sample_three_point_strip(upper_a, upper_b, upper_c, t)
            strip.append((low_src, up_src, low_delta, up_delta))
            avg_gap += (up_src - low_src).length
            z_vals.extend((low_src.z, up_src.z))
            y_vals.extend((low_src.y, up_src.y))
        avg_gap /= float(len(strip))
        if avg_gap <= 1.0e-8:
            continue

        radius = max(avg_gap * min(radius_scale, 1.35), 0.0012)
        z_margin = max(avg_gap * 0.65, radius * 1.1, 0.0015)
        y_margin = max(avg_gap * 0.80, radius * 1.3, 0.0015)
        side_sign = -1.0 if side == "l" else 1.0
        z_min = min(z_vals) - z_margin
        z_max = max(z_vals) + z_margin
        y_min = min(y_vals) - y_margin
        y_max = max(y_vals) + y_margin

        for vidx, src_co in enumerate(original_positions):
            if vidx in fixed:
                continue
            if src_co.z < z_min or src_co.z > z_max:
                continue
            if src_co.y < y_min or src_co.y > y_max:
                continue
            inner_abs = min(abs(lower_a["source"].x), abs(upper_a["source"].x))
            if side_sign * src_co.x < max(inner_abs * 0.72, avg_gap * 0.18):
                continue

            best = None
            for low_src, up_src, low_delta, up_delta in strip:
                raw_u = _segment_factor_unclamped(src_co, low_src, up_src)
                # Do not let the brow pass grab vertices just below the
                # eye-upper rail / brow-inner transition.  Those vertices belong
                # to the glabella or eyelid support, and were the source of the
                # sharp upward spike under LM_brow_*_inner.
                if raw_u < 0.02 or raw_u > 1.04:
                    continue
                u = max(0.0, min(1.0, raw_u))
                closest = low_src.lerp(up_src, u)
                dist = (src_co - closest).length
                if dist > radius:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, u, low_delta, up_delta)
            if best is None:
                continue

            dist, u, low_delta, up_delta = best
            prox = max(0.0, 1.0 - (dist / radius))
            ridge = 0.55 + 0.45 * (1.0 - abs(u - 0.52) * 1.35)
            w = strength * prox * prox * (3.0 - 2.0 * prox) * max(0.25, ridge)
            if w <= 0.0:
                continue
            target_delta = low_delta.lerp(up_delta, u)
            target_co = src_co + target_delta
            verts[vidx].co = verts[vidx].co.lerp(target_co, min(1.0, w))
            smooth_candidates[vidx] = max(smooth_candidates.get(vidx, 0.0), u)
            changed += 1

    smooth_changed = 0
    if smooth_strength > 0.0 and smooth_candidates:
        adj = build_mesh_adjacency(out_obj)
        for _iter in range(2):
            new_positions = {}
            for vidx, u in smooth_candidates.items():
                if u <= 0.10 or u >= 0.90:
                    continue
                neighbors = [n for n in adj[vidx] if n in smooth_candidates]
                if len(neighbors) < 2:
                    continue
                avg = Vector((0.0, 0.0, 0.0))
                for nb in neighbors:
                    avg += verts[nb].co
                avg /= float(len(neighbors))
                band = max(0.0, 1.0 - abs(u - 0.5) * 2.0)
                w = smooth_strength * band
                if w <= 0.0:
                    continue
                new_positions[vidx] = verts[vidx].co.lerp(avg, min(0.45, w))
            if not new_positions:
                break
            for vidx, co in new_positions.items():
                verts[vidx].co = co
                smooth_changed += 1
        try:
            out_obj["HFR_brow_sm"] = int(smooth_changed)
        except Exception:
            pass

    if changed or smooth_changed:
        out_obj.data.update()
    try:
        out_obj["HFR_brow_post"] = int(changed)
    except Exception:
        pass
    return changed + smooth_changed


def apply_nose_web_surface_fit(out_obj, original_positions, records, nose_strength=1.0, nose_radius=0.60, nose_samples=24):
    """Post-refine only the narrow nose web without collapsing it onto guide rails.

    v0.4.1 pulled candidate vertices directly onto the bilinear strip between the
    bridge/tip rail and the side nose rail.  That proved too aggressive: broad
    radius + direct target projection could catch forehead, cheek, and mouth
    vertices and collapse them into spike-like sheets.

    v0.4.2 keeps the useful part of the idea but changes the solve to a safe
    localized delta fit:

    - candidates must lie inside the original narrow web between the center rail
      and the same-side side rail,
    - candidates outside the segment span are rejected rather than accepted from
      extended guide lines,
    - the correction applies the interpolated landmark *delta* to the vertex's
      original position instead of moving the vertex onto the guide strip itself,
    - vertices owned by non-nose feature anchors and the eye/mouth/forehead bands
      are protected.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0

    strength = max(0.0, min(float(nose_strength), 1.0))
    radius_scale = max(float(nose_radius), 0.05)
    samples = max(6, int(nose_samples))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    bridge = rec_by_id.get("nose_bridge")
    tip = rec_by_id.get("nose_tip")
    if bridge is None or tip is None:
        return 0

    fixed = _anchor_delta_by_vertex(records, len(original_positions))

    # Protect non-nose feature anchor neighborhoods.  The post-fit should not
    # ever grab eyelid, brow/forehead, mouth, chin, cheek, outer face, ear, or
    # neck vertices even when those vertices are geometrically near an extended
    # nose guide line in the flattened template.
    protected_ids = []
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if not lm_id.startswith("nose_") and lm_id not in {"nose_root", "nose_bridge_top", "nose_bridge", "nose_tip", "nose_base"}:
            protected_ids.append(lm_id)
    protected = set()
    for lm_id in protected_ids:
        rec = rec_by_id.get(lm_id)
        if rec is None:
            continue
        for idx, _w in rec.get("members", []):
            if 0 <= idx < len(verts):
                protected.add(idx)

    relevant_src = [bridge["source"], tip["source"]]
    for side in ("l", "r"):
        for lm_id in (f"nose_{side}_side_upper", f"nose_{side}_side_lower"):
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                relevant_src.append(rec["source"])
    z_min = min(v.z for v in relevant_src)
    z_max = max(v.z for v in relevant_src)
    y_min = min(v.y for v in relevant_src)
    y_max = max(v.y for v in relevant_src)

    accum = {}

    for side in ("l", "r"):
        upper = rec_by_id.get(f"nose_{side}_side_upper")
        lower = rec_by_id.get(f"nose_{side}_side_lower")
        if upper is None or lower is None:
            continue

        strips = []
        avg_width = 0.0
        for i in range(samples + 1):
            t = float(i) / float(samples)
            c_src = bridge["source"].lerp(tip["source"], t)
            s_src = upper["source"].lerp(lower["source"], t)
            c_delta = bridge["delta"].lerp(tip["delta"], t)
            s_delta = upper["delta"].lerp(lower["delta"], t)
            width = (s_src - c_src).length
            avg_width += width
            strips.append((t, c_src, s_src, c_delta, s_delta, width))
        avg_width /= float(len(strips))
        if avg_width <= 1.0e-8:
            continue

        # The web is narrow.  Clamp the effective radius so a high stored scene
        # value from v0.4.1 cannot pull forehead/mouth/cheek regions into the
        # solve.  The UI value still expands the band within this safe range.
        radius = max(avg_width * min(radius_scale, 0.75), 0.0015)
        z_margin = max(avg_width * 0.55, radius * 1.2, 0.002)
        y_margin = max(avg_width * 0.65, radius * 1.2, 0.002)
        side_sign = -1.0 if side == "l" else 1.0

        for vidx, src_co in enumerate(original_positions):
            if vidx in fixed or vidx in protected:
                continue
            if src_co.z < z_min - z_margin or src_co.z > z_max + z_margin:
                continue
            if src_co.y < y_min - y_margin or src_co.y > y_max + y_margin:
                continue
            # Keep each side on its own half.  Centerline overlap is intentionally
            # tiny; bridge/tip anchors already handle the center rail.
            if side_sign * src_co.x < -avg_width * 0.08:
                continue

            best = None
            for t, c_src, s_src, c_delta, s_delta, _width in strips:
                raw_u = _segment_factor_unclamped(src_co, c_src, s_src)
                # v0.4.2: do not use extended segment influence.  Only vertices
                # actually between the center rail and side rail may be moved.
                if raw_u < -0.04 or raw_u > 1.04:
                    continue
                u = max(0.0, min(1.0, raw_u))
                closest = c_src.lerp(s_src, u)
                dist = (src_co - closest).length
                if dist > radius:
                    continue
                if best is None or dist < best[0]:
                    best = (dist, u, c_delta, s_delta)
            if best is None:
                continue

            dist, u, c_delta, s_delta = best
            prox = max(0.0, 1.0 - (dist / radius))
            falloff = _smoothstep01(prox)
            # Strongest near the center of the web, weaker at both rails so the
            # explicit anchor lock and existing topology solve remain dominant.
            middle = 1.0 - abs(u - 0.5) * 2.0
            middle_bias = 0.45 + 0.55 * max(0.0, middle)
            w = min(1.0, strength * falloff * middle_bias)
            if w <= 0.0:
                continue
            delta = c_delta.lerp(s_delta, u)
            desired = src_co + delta
            if vidx not in accum:
                accum[vidx] = [Vector((0.0, 0.0, 0.0)), 0.0]
            accum[vidx][0] += desired * w
            accum[vidx][1] += w

    changed = 0
    for vidx, (vec, total) in accum.items():
        if total <= 0.0:
            continue
        target = vec / total
        blend = min(1.0, total)
        current = verts[vidx].co.copy()
        verts[vidx].co = current.lerp(target, blend)
        changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_npost"] = int(changed)
    except Exception:
        pass
    return changed


def _nose_alar_patch_records(rec_by_id, side):
    lower = rec_by_id.get(f"nose_{side}_side_lower")
    alar = rec_by_id.get(f"nose_{side}_alar")
    nostril = rec_by_id.get(f"nose_{side}_nostril")
    base = rec_by_id.get("nose_base")
    if lower is None or alar is None or nostril is None or base is None:
        return None
    return lower, alar, nostril, base


def _nose_alar_candidate(original_co, side, lower, alar, nostril, base, radius_scale):
    """Return local alar patch weight and local delta for one source vertex.

    The alar patch is intentionally small: it only covers the nostril wing area
    bounded by side_lower / alar / nostril / nose_base.  It does not project
    vertices onto a guide surface; it only returns an interpolated local delta.
    """
    pts = [lower["source"], alar["source"], nostril["source"], base["source"]]
    local_span = max((a - b).length for a in pts for b in pts)
    if local_span <= 1.0e-8:
        return 0.0, None

    radius_scale = max(0.05, min(float(radius_scale), 1.65))
    radius = max(local_span * 0.44 * radius_scale, 0.0020)
    margin = max(radius * 1.10, local_span * 0.18, 0.0020)

    z_min = min(p.z for p in pts) - margin
    z_max = max(p.z for p in pts) + margin
    y_min = min(p.y for p in pts) - margin
    y_max = max(p.y for p in pts) + margin
    x_min = min(p.x for p in pts) - margin
    x_max = max(p.x for p in pts) + margin
    if original_co.z < z_min or original_co.z > z_max:
        return 0.0, None
    if original_co.y < y_min or original_co.y > y_max:
        return 0.0, None
    if original_co.x < x_min or original_co.x > x_max:
        return 0.0, None

    side_sign = -1.0 if side == "l" else 1.0
    # Allow a small center overlap for nose_base-adjacent vertices, but reject
    # the opposite wing.  This prevents left alar edits from grabbing right alar
    # surface vertices before Output Mirror Finish runs.
    if side_sign * original_co.x < -margin * 0.25:
        return 0.0, None

    segs = [
        (lower, alar, 1.00),
        (alar, nostril, 1.15),
        (nostril, base, 0.75),
        (lower, nostril, 0.70),
        (alar, base, 0.55),
    ]
    best_dist = None
    best_pair = None
    best_t = 0.0
    best_bias = 1.0
    for a, b, bias in segs:
        t, dist = _closest_segment_factor(original_co, a["source"], b["source"])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pair = (a, b)
            best_t = t
            best_bias = bias
    if best_dist is None or best_pair is None or best_dist > radius:
        return 0.0, None

    prox = max(0.0, 1.0 - (best_dist / radius))
    falloff = _smoothstep01(prox)
    a, b = best_pair
    seg_delta = a["delta"].lerp(b["delta"], best_t)

    # Blend with a four-anchor local IDW so the actual alar point has stronger
    # influence, while side_lower / nostril / base keep the wing attached.
    local_records = [lower, alar, nostril, base]
    idw_delta = _idw_delta_for_point(original_co, local_records, power=2.4, nearest_count=4)
    alar_dist = (original_co - alar["source"]).length
    alar_prox = max(0.0, 1.0 - (alar_dist / max(radius * 1.35, 1.0e-6)))
    alar_w = 0.35 + 0.45 * _smoothstep01(alar_prox)
    local_delta = seg_delta.lerp(idw_delta, alar_w)

    # Encourage the wing/nostril fold to follow alar, but keep influence small
    # enough that this pass cannot collapse the nose like the old v0.4.1 strip
    # projection did.
    weight = min(1.0, falloff * best_bias * (0.65 + 0.35 * _smoothstep01(alar_prox)))
    return weight, local_delta


def nose_alar_refine_displacements(original, displacements, records, alar_strength=0.85, alar_radius=1.0, alar_samples=12):
    """Pre-fit nostril wing vertices around LM_nose_*_alar.

    This pass is deliberately narrower than Nose Web Fit.  It handles the local
    wing patch bounded by side_lower / alar / nostril / nose_base so moving the
    alar landmark also moves the nearby unbound alar surface vertices.
    """
    strength = max(0.0, min(float(alar_strength), 1.0))
    if strength <= 0.0 or not records:
        return 0
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    fixed = _anchor_delta_by_vertex(records, len(original))
    changed = 0
    for side in ("l", "r"):
        patch = _nose_alar_patch_records(rec_by_id, side)
        if patch is None:
            continue
        lower, alar, nostril, base = patch
        for vidx, src_co in enumerate(original):
            if vidx in fixed:
                continue
            weight, delta = _nose_alar_candidate(src_co, side, lower, alar, nostril, base, alar_radius)
            if weight <= 0.0 or delta is None:
                continue
            w = min(1.0, strength * weight)
            displacements[vidx] = displacements[vidx].lerp(delta, w)
            changed += 1
    return changed


def apply_nose_alar_surface_fit(out_obj, original_positions, records, alar_strength=0.85, alar_radius=1.0, alar_samples=12):
    """Post-lock local nostril-wing fit around LM_nose_*_alar.

    Anchor lock can leave the explicitly bound alar point correct while the
    neighboring unbound surface is still weak.  This post pass repeats the same
    small local solve after anchor lock, blending current vertices toward their
    original position plus local alar/nostril/base deltas.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0
    strength = max(0.0, min(float(alar_strength), 1.0))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    local_ids = {
        "nose_base",
        "nose_l_side_lower", "nose_l_alar", "nose_l_nostril",
        "nose_r_side_lower", "nose_r_alar", "nose_r_nostril",
    }
    fixed = _anchor_delta_by_vertex(records, len(original_positions))
    protected = set()
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if lm_id in local_ids:
            continue
        # Keep all other feature anchors out of the post correction, including
        # nose bridge/tip/side_upper because Nose Web Fit already owns that area.
        for idx, _w in rec.get("members", []):
            if 0 <= idx < len(verts):
                protected.add(idx)

    accum = {}
    for side in ("l", "r"):
        patch = _nose_alar_patch_records(rec_by_id, side)
        if patch is None:
            continue
        lower, alar, nostril, base = patch
        for vidx, src_co in enumerate(original_positions):
            if vidx in fixed or vidx in protected:
                continue
            weight, delta = _nose_alar_candidate(src_co, side, lower, alar, nostril, base, alar_radius)
            if weight <= 0.0 or delta is None:
                continue
            w = min(1.0, strength * weight)
            desired = src_co + delta
            if vidx not in accum:
                accum[vidx] = [Vector((0.0, 0.0, 0.0)), 0.0]
            accum[vidx][0] += desired * w
            accum[vidx][1] += w

    changed = 0
    for vidx, (vec, total) in accum.items():
        if total <= 0.0:
            continue
        target = vec / total
        blend = min(1.0, total)
        current = verts[vidx].co.copy()
        verts[vidx].co = current.lerp(target, blend)
        changed += 1
    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_nalar_post"] = int(changed)
    except Exception:
        pass
    return changed


def local_anchor_patch_refine_displacements(obj, original, displacements, records, lm_ids=None, patch_steps=4, patch_strength=0.85):
    """Blend a topological neighborhood around specific anchors toward their delta.

    This is used for ear lobes so a single bound lobe vertex does not become a
    spike while nearby lower-ear vertices remain mostly static.
    """
    strength = max(0.0, min(float(patch_strength), 1.0))
    steps = max(0, int(patch_steps))
    if strength <= 0.0 or steps <= 0 or not records:
        return 0
    ids = list(lm_ids or [])
    if not ids:
        return 0
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    adj = build_mesh_adjacency(obj)
    changed = 0
    for lm_id in ids:
        rec = rec_by_id.get(lm_id)
        if rec is None:
            continue
        seeds = [idx for idx, _w in rec.get("members", []) if 0 <= idx < len(original)]
        if not seeds:
            continue
        dist = {}
        frontier = []
        for idx in seeds:
            dist[idx] = 0
            frontier.append(idx)
        head = 0
        while head < len(frontier):
            cur = frontier[head]
            head += 1
            d = dist[cur]
            if d >= steps:
                continue
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = d + 1
                    frontier.append(nb)
        for idx, d in dist.items():
            x = max(0.0, 1.0 - (float(d) / float(steps + 1)))
            w = strength * x * x * (3.0 - 2.0 * x)
            if w <= 0.0:
                continue
            displacements[idx] = displacements[idx].lerp(rec["delta"], w)
            changed += 1
    return changed


def build_guide_rail_constraints(obj, original, records, guide_pairs=None, rail_strength=1.0, max_path_len=80):
    """Create explicit displacement constraints along template edge paths between landmarks.

    Guide Follow/MLS are broad field refinements.  They can still be too weak when
    a visible strip of template vertices sits between two bound landmarks: the
    endpoints move, but the actual edge path between them is not constrained.  This
    helper finds the mesh shortest path between each pair of bound landmarks in the
    guide graph and gives the intermediate vertices a linearly interpolated delta.

    These rail constraints are then treated like fixed values during topology
    propagation, so the vertex chain between two landmarks follows the landmarks
    instead of waiting for indirect diffusion.
    """
    strength = max(0.0, min(float(rail_strength), 1.0))
    if obj is None or obj.type != 'MESH' or strength <= 0.0 or not records:
        return {}
    vert_count = len(original)
    if vert_count <= 0:
        return {}
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    anchor_fixed = _anchor_delta_by_vertex(records, vert_count)
    adj = build_mesh_adjacency(obj)
    pairs = guide_pairs if guide_pairs is not None else GUIDES
    max_len = max(0, int(max_path_len))
    accum = {}

    for a_id, b_id in pairs:
        a = rec_by_id.get(a_id)
        b = rec_by_id.get(b_id)
        if a is None or b is None:
            continue
        # No need to add an all-zero rail.
        if a["delta"].length <= 1.0e-10 and b["delta"].length <= 1.0e-10:
            continue
        a_seeds = _group_indices_from_record(a, vert_count)
        b_seeds = _group_indices_from_record(b, vert_count)
        path = _bfs_shortest_path(adj, a_seeds, b_seeds)
        if len(path) < 3:
            continue
        if max_len > 0 and len(path) > max_len:
            # Very long paths usually mean the landmark guide does not map to a
            # local template edge chain.  Skip rather than pulling half the mesh.
            continue

        seg_lengths = []
        total_len = 0.0
        for i in range(len(path) - 1):
            seg = (original[path[i + 1]] - original[path[i]]).length
            seg_lengths.append(seg)
            total_len += seg
        if total_len <= 1.0e-10:
            total_len = float(len(path) - 1)
            seg_lengths = [1.0 for _ in range(len(path) - 1)]

        travelled = 0.0
        for i, vidx in enumerate(path):
            if i > 0:
                travelled += seg_lengths[i - 1]
            if vidx in anchor_fixed:
                continue
            t = max(0.0, min(1.0, travelled / total_len))
            delta = a["delta"].lerp(b["delta"], t)
            # Slightly emphasize the interior; endpoints are already held by anchors.
            interior = 1.0 - abs(t - 0.5) * 2.0
            w = strength * (0.65 + 0.35 * max(0.0, interior))
            if w <= 0.0:
                continue
            if vidx not in accum:
                accum[vidx] = [Vector((0.0, 0.0, 0.0)), 0.0]
            accum[vidx][0] += delta * w
            accum[vidx][1] += w

    fixed = {}
    for idx, (vec, total) in accum.items():
        if total > 0.0:
            fixed[idx] = vec / total
    return fixed


def topology_propagate_displacements(obj, original, records, power=2.0, nearest_count=12, topo_iters=36, topo_strength=0.65, extra_fixed=None):
    """Create a displacement field that respects mesh connectivity.

    v0.2.0/0.2.1 used only direct IDW deltas and then corrected the explicit
    anchor vertices.  That made eyelid/lip vertices sitting between two bound
    anchors look almost static.  This pass treats anchor deltas as fixed
    constraints and relaxes the displacement field along the template edge graph,
    so intermediate vertices travel with the local eye/mouth loops.
    """
    vert_count = len(original)
    fixed = _anchor_delta_by_vertex(records, vert_count)
    if extra_fixed:
        for idx, delta in extra_fixed.items():
            if 0 <= idx < vert_count and idx not in fixed:
                fixed[idx] = delta.copy()
    disp = [Vector((0.0, 0.0, 0.0)) for _ in range(vert_count)]

    # IDW remains a useful initial global fit before topology relaxation.
    for i, co in enumerate(original):
        if i in fixed:
            disp[i] = fixed[i].copy()
        else:
            disp[i] = _idw_delta_for_point(co, records, power=power, nearest_count=nearest_count)

    iters = max(0, int(topo_iters))
    strength = max(0.0, min(float(topo_strength), 1.0))
    if iters <= 0 or strength <= 0.0:
        return disp, len(fixed)

    adj = build_mesh_adjacency(obj)
    for _ in range(iters):
        new_disp = [d.copy() for d in disp]
        for i, neighbors in enumerate(adj):
            if i in fixed:
                new_disp[i] = fixed[i].copy()
                continue
            if not neighbors:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for n in neighbors:
                avg += disp[n]
            avg /= len(neighbors)
            new_disp[i] = disp[i].lerp(avg, strength)
        # Keep constraints exact after every relaxation pass.
        for idx, delta in fixed.items():
            new_disp[idx] = delta.copy()
        disp = new_disp
    return disp, len(fixed)




def _group_indices_from_record(rec, vert_count):
    return [idx for idx, _w in rec.get("members", []) if 0 <= idx < vert_count]


def _current_group_centroid(verts, rec):
    members = [(idx, weight) for idx, weight in rec.get("members", []) if 0 <= idx < len(verts)]
    if not members:
        return None
    acc = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for idx, weight in members:
        w = max(float(weight), 0.0001)
        acc += verts[idx].co * w
        total += w
    if total <= 0.0:
        return None
    return acc / total


def _bfs_shortest_path(adj, seed_indices, goal_indices):
    seeds = [idx for idx in seed_indices if 0 <= idx < len(adj)]
    goals = set(idx for idx in goal_indices if 0 <= idx < len(adj))
    if not seeds or not goals:
        return []
    from collections import deque
    q = deque()
    prev = {}
    for s in seeds:
        prev[s] = None
        q.append(s)
        if s in goals:
            return [s]
    found = None
    while q:
        cur = q.popleft()
        if cur in goals:
            found = cur
            break
        for nb in adj[cur]:
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    if found is None:
        return []
    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def post_lock_fit_ear_lower_strip(out_obj, records, fit_strength=0.85, y_lock_strength=1.0):
    """Directly fit the lower-ear strip after anchor lock.

    Earlier ear-lobe attempts only modified the broad displacement field. The
    final anchor-lock step then pulled the explicitly bound lobe vertex back to
    its exact landmark target, recreating the spike and making the lobe options
    look ineffective. This post-lock solve takes a different route: it reshapes
    the lower-ear strip *after* anchor lock by fitting the shortest mesh paths
    between lobe and its neighboring lower-ear anchors.
    """
    strength = max(0.0, min(float(fit_strength), 1.0))
    y_strength = max(0.0, min(float(y_lock_strength), 1.0))
    if out_obj is None or out_obj.type != 'MESH' or strength <= 0.0 or not records:
        return 0
    verts = out_obj.data.vertices
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    changed = 0
    for side in ("l", "r"):
        lobe = rec_by_id.get(f"ear_{side}_lobe")
        front = rec_by_id.get(f"ear_{side}_front_lower")
        back = rec_by_id.get(f"ear_{side}_back_lower")
        inner = rec_by_id.get(f"ear_{side}_inner_bottom")
        if lobe is None or front is None or back is None or inner is None:
            continue
        lobe_ids = _group_indices_from_record(lobe, len(verts))
        front_ids = _group_indices_from_record(front, len(verts))
        back_ids = _group_indices_from_record(back, len(verts))
        inner_ids = _group_indices_from_record(inner, len(verts))
        if not lobe_ids or not front_ids or not back_ids or not inner_ids:
            continue
        lobe_center = _current_group_centroid(verts, lobe)
        front_center = _current_group_centroid(verts, front)
        back_center = _current_group_centroid(verts, back)
        inner_center = _current_group_centroid(verts, inner)
        if None in (lobe_center, front_center, back_center, inner_center):
            continue
        support_y = (front_center.y + back_center.y + inner_center.y) / 3.0
        corrected_lobe = lobe_center.copy()
        corrected_lobe.y = corrected_lobe.y * (1.0 - y_strength) + support_y * y_strength
        # First, pull the bound lobe centroid back onto the lower-ear Y frame.
        lobe_residual = (corrected_lobe - lobe_center) * strength
        if lobe_residual.length > 1.0e-12:
            for idx in lobe_ids:
                verts[idx].co += lobe_residual
                changed += 1
        rails = [
            (front_ids, _current_group_centroid(verts, front)),
            (back_ids, _current_group_centroid(verts, back)),
            (inner_ids, _current_group_centroid(verts, inner)),
        ]
        lobe_set = set(lobe_ids)
        fixed_support = set(front_ids) | set(back_ids) | set(inner_ids)
        for seed_ids, support_co in rails:
            if support_co is None:
                continue
            path = _bfs_shortest_path(adj, lobe_ids, seed_ids)
            if len(path) < 3:
                continue
            if path[0] not in lobe_set:
                path = list(reversed(path))
            seg_count = len(path) - 1
            if seg_count <= 0:
                continue
            for step, vidx in enumerate(path):
                if vidx in lobe_set or vidx in fixed_support:
                    continue
                t = float(step) / float(seg_count)
                desired = corrected_lobe.lerp(support_co, t)
                current = verts[vidx].co.copy()
                desired.y = current.y * (1.0 - y_strength) + desired.y * y_strength
                verts[vidx].co = current.lerp(desired, strength)
                changed += 1
    if changed:
        out_obj.data.update()
    return changed



def apply_directional_lobe_stretch(out_obj, records, lm_ids=("ear_l_lobe", "ear_r_lobe"), steps=2, strength=1.0, falloff=0.65):
    """Propagate the ear-lobe landmark delta outward in the same direction.

    This is intentionally *directional* rather than corrective.  If the user
    drags the lobe down, forward, or diagonally, neighboring lower-ear
    vertices follow in that same direction with a simple topological falloff.
    The goal is not to keep the shape natural; the goal is to stretch in the
    dragged direction and roughly by the dragged amount.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    steps = max(0, int(steps))
    strength = max(0.0, float(strength))
    falloff = max(0.0, min(float(falloff), 1.0))
    if steps <= 0 or strength <= 0.0:
        return 0
    verts = out_obj.data.vertices
    adj = build_mesh_adjacency(out_obj)
    anchor_owner = {}
    for rec in records:
        for idx, _w in rec.get("members", []):
            if 0 <= idx < len(verts):
                anchor_owner[idx] = rec.get("lm_id")
    changed = 0
    for rec in records:
        lm_id = rec.get("lm_id")
        if lm_id not in lm_ids:
            continue
        delta = rec.get("target", Vector((0.0, 0.0, 0.0))) - rec.get("source", Vector((0.0, 0.0, 0.0)))
        if delta.length <= 1.0e-10:
            continue
        seeds = [idx for idx, _w in rec.get("members", []) if 0 <= idx < len(verts)]
        if not seeds:
            continue
        visited = set(seeds)
        frontier = set(seeds)
        for ring in range(1, steps + 1):
            next_frontier = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in visited:
                        continue
                    visited.add(nb)
                    owner = anchor_owner.get(nb)
                    # v0.2.15: do not let directional lobe stretch travel *through*
                    # vertices that belong to other landmarks.  In v0.2.14 those
                    # foreign-anchor vertices were skipped for movement, but they still
                    # entered the BFS frontier, so the stretch leaked past ear-front /
                    # face-edge boundary anchors into the front-face strip.
                    if owner and owner != lm_id:
                        continue
                    next_frontier.add(nb)
            if not next_frontier:
                break
            influence = strength * (falloff ** (ring - 1))
            for nb in next_frontier:
                verts[nb].co += delta * influence
                changed += 1
            frontier = next_frontier
    if changed:
        out_obj.data.update()
    return changed


def enforce_sparse_ear_lobe_plane(out_obj, records, y_strength=1.0, neighbor_blend=0.35):
    """Stabilize sparse ear-lobe topology after anchor lock.

    When the template has only one or two vertices available for the lower ear,
    broad loop/rail fitting tends to turn the lobe into a forward spike.  This
    post step does *not* try to create new lobe volume.  Instead, it keeps the
    explicitly bound lobe vertices on the local lower-ear Y plane defined by the
    neighboring supports (front_lower/back_lower/inner_bottom), and softly blends
    their immediate topological neighbors toward the same Y so the result stays
    attached instead of forming a dart-like triangle.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    y_strength = max(0.0, min(float(y_strength), 1.0))
    neighbor_blend = max(0.0, min(float(neighbor_blend), 1.0))
    if y_strength <= 0.0:
        return 0
    verts = out_obj.data.vertices
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec["lm_id"]: rec for rec in records}
    changed = 0
    for side in ("l", "r"):
        lobe = rec_by_id.get(f"ear_{side}_lobe")
        front = rec_by_id.get(f"ear_{side}_front_lower")
        back = rec_by_id.get(f"ear_{side}_back_lower")
        inner = rec_by_id.get(f"ear_{side}_inner_bottom")
        if lobe is None or front is None or back is None or inner is None:
            continue
        lobe_ids = _group_indices_from_record(lobe, len(verts))
        front_ids = _group_indices_from_record(front, len(verts))
        back_ids = _group_indices_from_record(back, len(verts))
        inner_ids = _group_indices_from_record(inner, len(verts))
        if not lobe_ids or not (front_ids or back_ids or inner_ids):
            continue
        support_vals = []
        for group in (front_ids, back_ids, inner_ids):
            for vidx in group:
                support_vals.append(verts[vidx].co.y)
        if not support_vals:
            continue
        support_y = sum(support_vals) / len(support_vals)
        lobe_set = set(lobe_ids)
        support_set = set(front_ids) | set(back_ids) | set(inner_ids)
        for vidx in lobe_ids:
            v = verts[vidx]
            v.co.y = v.co.y * (1.0 - y_strength) + support_y * y_strength
            changed += 1
            if neighbor_blend > 0.0:
                for nb in adj[vidx]:
                    if nb in lobe_set or nb in support_set:
                        continue
                    nv = verts[nb]
                    nv.co.y = nv.co.y * (1.0 - neighbor_blend) + support_y * neighbor_blend
                    changed += 1
    if changed:
        out_obj.data.update()
    return changed

def _expand_vertex_set(seed_set, adj, steps=1):
    cur = set(seed_set)
    frontier = set(seed_set)
    for _ in range(max(0, int(steps))):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb not in cur:
                    cur.add(nb)
                    nxt.add(nb)
        if not nxt:
            break
        frontier = nxt
    return cur


def _smoothstep01(t):
    t = max(0.0, min(float(t), 1.0))
    return t * t * (3.0 - 2.0 * t)


def apply_head_round_fit(out_obj, records, original_positions=None,
                         anchor_ids=("scalp_top_center", "scalp_front_center", "scalp_back_center",
                                     "scalp_l_front", "scalp_r_front", "scalp_l_top", "scalp_r_top",
                                     "head_l_side_upper", "head_r_side_upper", "head_l_side_back",
                                     "head_r_side_back"),
                         region_steps=7, smooth_strength=0.70, smooth_iters=2, z_margin=0.30):
    """Fair the upper-head region after deformation so tall scalp edits stay rounder.

    v0.2.20 switched to a scalp-only correction, which stopped the full-face melt,
    but the direct height redistribution could still create a pointed crown and
    pull the forehead band backward.  v0.2.21 changes strategy again:

    - work only on a protected upper-head/scalp region,
    - fair that region by topology-based smoothing,
    - keep the face / temples / ear roots protected,
    - preserve the front-vs-back Y profile from the landmark targets so the
      forehead does not get dragged rearward,
    - keep crown anchors as *soft* constraints rather than absolute spikes,
    - let the upper silhouette move instead of pinning the top boundary into side spikes.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    if original_positions is None or len(original_positions) != len(verts):
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(smooth_strength), 1.0))
    if strength <= 0.0:
        return 0
    region_steps = max(1, int(region_steps))
    smooth_iters = max(0, int(smooth_iters))
    effective_margin = max(0.0, min(float(z_margin), 0.18))

    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}

    def _members_for(lm_ids):
        out = set()
        for lm_id in lm_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            out.update(idx for idx, _w in rec.get("members", []) if 0 <= idx < len(verts))
        return out

    def _avg_component(ids, key, axis):
        vals = []
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            vals.append(getattr(rec[key], axis))
        if not vals:
            return None
        return sum(vals) / float(len(vals))

    scalp_seed = _members_for(anchor_ids)
    if len(scalp_seed) < 4:
        return 0

    # These anchors should define the scalp region but must not become harsh spikes.
    # v0.2.22: include the upper side anchors as *soft* anchors too, so the top silhouette
    # can actually round instead of staying pinned into sideways corners.
    soft_anchor_ids = [
        "scalp_top_center", "scalp_l_top", "scalp_r_top",
        "scalp_front_center", "scalp_back_center",
        "scalp_l_front", "scalp_r_front",
        "head_l_side_upper", "head_r_side_upper",
    ]
    soft_set = _members_for(soft_anchor_ids)

    # Strong protection around the face and ear roots.
    protect_ids = [
        "ear_l_top", "ear_r_top",
        "ear_l_front_upper", "ear_r_front_upper",
        "ear_l_front_middle", "ear_r_front_middle",
        "ear_l_back_upper", "ear_r_back_upper",
        "ear_l_back_middle", "ear_r_back_middle",
        "ear_l_back_lower", "ear_r_back_lower",
        "temple_l_center", "temple_r_center",
        "brow_l_inner", "brow_l_center", "brow_l_outer",
        "brow_r_inner", "brow_r_center", "brow_r_outer",
        "eye_l_inner", "eye_l_upper", "eye_l_outer",
        "eye_r_inner", "eye_r_upper", "eye_r_outer",
        "nose_root", "nose_bridge_top",
        "cheek_l_center", "cheek_r_center",
        "outer_face_l_upper", "outer_face_r_upper",
        "face_l_edge", "face_r_edge",
    ]
    protect = _expand_vertex_set(_members_for(protect_ids), adj, steps=1)

    floor_ids = [
        "forehead_upper_center", "forehead_l_upper", "forehead_r_upper",
        "head_l_side_upper", "head_r_side_upper",
        "head_l_side_back", "head_r_side_back",
        "temple_l_center", "temple_r_center",
    ]
    floor_src_z_vals = [rec_by_id[lm_id]["source"].z for lm_id in floor_ids if lm_id in rec_by_id]
    floor_tgt_z_vals = [rec_by_id[lm_id]["target"].z for lm_id in floor_ids if lm_id in rec_by_id]
    if not floor_src_z_vals or not floor_tgt_z_vals:
        return 0
    floor_src_z = min(floor_src_z_vals) - effective_margin
    floor_tgt_z = sum(floor_tgt_z_vals) / float(len(floor_tgt_z_vals))

    # Front/back profile references for preserving forehead depth.
    front_ids = ["forehead_upper_center", "forehead_l_upper", "forehead_r_upper",
                 "scalp_front_center", "scalp_l_front", "scalp_r_front"]
    back_ids = ["scalp_back_center", "head_l_side_back", "head_r_side_back"]
    front_src_y = _avg_component(front_ids, "source", "y")
    front_tgt_y = _avg_component(front_ids, "target", "y")
    back_src_y = _avg_component(back_ids, "source", "y")
    back_tgt_y = _avg_component(back_ids, "target", "y")
    if None in (front_src_y, front_tgt_y, back_src_y, back_tgt_y):
        front_src_y = front_tgt_y = back_src_y = back_tgt_y = None

    # Region: scalp seeds expanded upward, but never below the upper-face boundary.
    region = set(idx for idx in scalp_seed if original_positions[idx].z >= floor_src_z and idx not in protect)
    frontier = set(region)
    for _ in range(region_steps):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in region or nb in protect:
                    continue
                src = original_positions[nb]
                if src.z < floor_src_z:
                    continue
                region.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        frontier = nxt

    if len(region) < 8:
        return 0

    # Height normalization is used both for fairing and for deciding how much of the
    # scalp boundary may move.  v0.2.21 fixed every region-boundary vertex, which left
    # the upper silhouette pinned and caused the sideways protrusions the user reported.
    crown_src_ids = ["scalp_top_center", "scalp_l_top", "scalp_r_top"]
    crown_src_z_vals = [rec_by_id[lm_id]["source"].z for lm_id in crown_src_ids if lm_id in rec_by_id]
    top_src_z = max(crown_src_z_vals) if crown_src_z_vals else max(original_positions[idx].z for idx in region)
    src_h = max(top_src_z - floor_src_z, 1.0e-6)
    lower_guard_z = floor_src_z + src_h * 0.38

    # Boundary protection: keep the lower boundary / face-adjacent shell fixed, but allow
    # the upper silhouette to be fairable so the crown can actually become rounder.
    hard_fixed = set(protect)
    for idx in list(region):
        src = original_positions[idx]
        if src.z < floor_src_z:
            hard_fixed.add(idx)
            continue
        touching_protect = False
        touching_outside = False
        for nb in adj[idx]:
            if nb in protect:
                touching_protect = True
                break
            if nb not in region:
                touching_outside = True
        if touching_protect:
            hard_fixed.add(idx)
            continue
        if touching_outside and src.z <= lower_guard_z:
            hard_fixed.add(idx)
            continue

    # Preserve only the lower forehead/back scaffold as hard constraints.
    # The upper side anchors are intentionally *not* hard-fixed anymore.
    scaffold_ids = [
        "forehead_upper_center", "forehead_l_upper", "forehead_r_upper",
        "head_l_side_back", "head_r_side_back",
        "scalp_back_center",
    ]
    hard_fixed.update(_members_for(scaffold_ids))
    hard_fixed &= region
    soft_set = (soft_set & region) - hard_fixed
    movable = sorted(idx for idx in region if idx not in hard_fixed and idx not in soft_set)
    if not movable and not soft_set:
        return 0

    base_positions = [v.co.copy() for v in verts]
    current = [v.co.copy() for v in verts]

    def _target_profile_y(src_y):
        if None in (front_src_y, front_tgt_y, back_src_y, back_tgt_y):
            return None
        denom = (front_src_y - back_src_y)
        if abs(denom) <= 1.0e-8:
            t = 0.5
        else:
            t = (src_y - back_src_y) / denom
        t = max(0.0, min(1.0, t))
        delta_front = front_tgt_y - front_src_y
        delta_back = back_tgt_y - back_src_y
        return src_y + (delta_back * (1.0 - t) + delta_front * t)

    iter_count = max(4, smooth_iters * 3)
    changed = 0
    for _ in range(iter_count):
        updates = {}
        # First fair free vertices.
        for idx in movable:
            nbs = [nb for nb in adj[idx] if nb in region and nb not in protect]
            if len(nbs) < 2:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for nb in nbs:
                avg += current[nb]
            avg /= float(len(nbs))

            src = original_positions[idx]
            h = _smoothstep01((src.z - floor_src_z) / src_h)
            lower_band = _smoothstep01((src.z - floor_src_z) / max(src_h * 0.55, 1.0e-6))
            local = strength * (0.18 + 0.62 * h) * (0.35 + 0.65 * lower_band)
            co = current[idx].lerp(avg, local)

            # Preserve front/back depth profile so the forehead does not get dragged rearward.
            target_y = _target_profile_y(src.y)
            if target_y is not None:
                y_keep = 0.18 + 0.30 * (1.0 - h)
                co.y = co.y * (1.0 - y_keep) + target_y * y_keep

            updates[idx] = co

        # Then fair crown-anchor vertices softly, so they influence the shape but do not form spikes.
        for idx in soft_set:
            nbs = [nb for nb in adj[idx] if nb in region and nb not in protect]
            if len(nbs) < 2:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for nb in nbs:
                avg += current[nb]
            avg /= float(len(nbs))
            src = original_positions[idx]
            h = _smoothstep01((src.z - floor_src_z) / src_h)
            local = strength * (0.20 + 0.55 * h)
            # Soft anchors retain much of their deformed position, but not all of it.
            co = current[idx].lerp(avg, local * 0.65)
            co = co.lerp(base_positions[idx], 0.58)
            target_y = _target_profile_y(src.y)
            if target_y is not None:
                y_keep = 0.12 + 0.20 * (1.0 - h)
                co.y = co.y * (1.0 - y_keep) + target_y * y_keep
            updates[idx] = co

        if not updates:
            break
        for idx, co in updates.items():
            current[idx] = co
        changed += len(updates)

    # Commit only the fairing region.
    for idx in movable:
        verts[idx].co = current[idx]
    for idx in soft_set:
        verts[idx].co = current[idx]

    # Small final Z-only smoothing pass inside the dome region to soften residual vertical channels.
    final_iters = max(1, smooth_iters)
    for _ in range(final_iters):
        z_updates = {}
        for idx in movable:
            nbs = [nb for nb in adj[idx] if nb in region and nb not in protect]
            if len(nbs) < 2:
                continue
            avg_z = sum(verts[nb].co.z for nb in nbs) / float(len(nbs))
            src = original_positions[idx]
            h = _smoothstep01((src.z - floor_src_z) / src_h)
            z_w = min(0.18, 0.05 + 0.13 * h)
            z_updates[idx] = verts[idx].co.z * (1.0 - z_w) + avg_z * z_w
        for idx, z in z_updates.items():
            co = verts[idx].co.copy()
            co.z = z
            verts[idx].co = co
            changed += 1

    if changed:
        out_obj.data.update()
    return changed



def apply_output_mirror_finish(out_obj, original_positions, direction='L2R', center_epsilon=0.0005, max_pair_dist=0.0):
    """Mirror the generated output from the working side to the opposite side.

    This is a final symmetry pass, not another broad deformation solver.  When
    Landmark Mirror X is being used, the user's intent is usually that the edited
    side and the followed side match.  If one cheek surface still under-follows
    because guide rails or MLS did not reach it, this pass uses the symmetric
    template vertex pairs and copies the already-correct source side to the
    destination side.

    Direction follows the UI setting:
      L2R: copy LM_*_l / negative-X side to LM_*_r / positive-X side.
      R2L: copy the positive-X side to the negative-X side.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    if not original_positions or len(original_positions) != len(verts):
        return 0

    try:
        xs = [co.x for co in original_positions]
        center_x = (min(xs) + max(xs)) * 0.5
        diag = (Vector((max(xs), 0.0, 0.0)) - Vector((min(xs), 0.0, 0.0))).length
    except Exception:
        center_x = 0.0
        diag = 1.0

    eps = max(float(center_epsilon), 1.0e-8)
    if direction == 'R2L':
        def is_src(co): return co.x > center_x + eps
        def is_dst(co): return co.x < center_x - eps
    else:
        def is_src(co): return co.x < center_x - eps
        def is_dst(co): return co.x > center_x + eps

    src_indices = [i for i, co in enumerate(original_positions) if is_src(co)]
    dst_indices = [i for i, co in enumerate(original_positions) if is_dst(co)]
    if not src_indices or not dst_indices:
        return 0

    kd = KDTree(len(src_indices))
    for slot, idx in enumerate(src_indices):
        kd.insert(original_positions[idx], idx)
    kd.balance()

    limit = float(max_pair_dist)
    if limit <= 0.0:
        # Symmetric templates should pair almost exactly, but keep a practical
        # fallback for tiny modeling offsets.
        all_min = Vector((min(co.x for co in original_positions), min(co.y for co in original_positions), min(co.z for co in original_positions)))
        all_max = Vector((max(co.x for co in original_positions), max(co.y for co in original_positions), max(co.z for co in original_positions)))
        limit = max((all_max - all_min).length * 0.035, 0.003)

    current = [v.co.copy() for v in verts]
    changed = 0
    for dst_idx in dst_indices:
        dst_orig = original_positions[dst_idx]
        mirror_orig = Vector((2.0 * center_x - dst_orig.x, dst_orig.y, dst_orig.z))
        _co, src_idx, dist = kd.find(mirror_orig)
        if src_idx is None or dist > limit:
            continue
        src_now = current[src_idx]
        mirrored = Vector((2.0 * center_x - src_now.x, src_now.y, src_now.z))
        verts[dst_idx].co = mirrored
        changed += 1

    # Keep the center seam exactly on the mirror plane where applicable.
    for idx, orig in enumerate(original_positions):
        if abs(orig.x - center_x) <= eps:
            co = verts[idx].co.copy()
            co.x = center_x
            verts[idx].co = co
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_mfix"] = int(changed)
        out_obj["HFR_mdir"] = str(direction)
    except Exception:
        pass
    return changed



# Side-specific ear landmark sets used by the local ear solve.  The global guide
# field is still useful for the head/face, but the ear is a thin folded part and
# needs to preserve its own local loop order.
def ear_side_landmark_ids(side):
    return [
        f"ear_{side}_top",
        f"ear_{side}_front_upper",
        f"ear_{side}_front_middle",
        f"ear_{side}_front_lower",
        f"ear_{side}_lobe",
        f"ear_{side}_back_lower",
        f"ear_{side}_back_middle",
        f"ear_{side}_back_upper",
        f"ear_{side}_inner_front_middle",
        f"ear_{side}_inner_bottom",
    ]


def _record_member_indices(rec, vert_count):
    return {idx for idx, _w in rec.get("members", []) if 0 <= idx < vert_count}


def _min_distance_to_points(co, points):
    best = None
    for pt in points:
        d = (co - pt).length
        if best is None or d < best:
            best = d
    return best if best is not None else 1.0e9


def _expanded_vertex_set(seed_set, adj, steps=1):
    region = set(seed_set)
    frontier = set(seed_set)
    for _ in range(max(0, int(steps))):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in region:
                    continue
                region.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        frontier = nxt
    return region


def ear_attachment_guard_vertex_indices(obj, records, original_positions, side,
                                        expand_steps=1, max_path_len=38, include_lower=True):
    """Protect the side-head / back-ear attachment strip from ear-local solvers.

    The ear shell is a thin folded part, but the vertices between
    head_side_upper/head_side_back and head_side_back/back-ear landmarks belong
    to the head attachment surface. If Ear Local Fit treats these vertices as ear
    shell members, the side of the head is pulled into the ear and the attachment
    becomes visibly kinked.

    v0.5.8 used the same guard for the lower-ear transition fan as well. That
    repeated an old lower-lobe failure mode: the guard blocked the back_lower /
    nape side of the small lower-ear fan, while the lobe directional stretch and
    anchor lock still moved the lobe. The unsolved fan then collapsed into a
    red flipped triangle.

    v0.5.9 keeps the full guard available for ear-shell solvers, but lets callers
    disable the lower attachment rails when they intentionally solve the
    lower-ear transition fan.
    """
    if obj is None or obj.type != 'MESH' or not records:
        return set()
    vert_count = len(obj.data.vertices)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in obj.data.vertices]

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(obj)
    pairs = [
        (f"head_{side}_side_upper", f"head_{side}_side_back"),
        (f"head_{side}_side_back", f"ear_{side}_back_upper"),
    ]
    if include_lower:
        pairs.extend([
            (f"head_{side}_side_back", f"ear_{side}_back_lower"),
            (f"nape_{side}_outer", f"ear_{side}_back_lower"),
        ])
    result = set()
    side_sign = -1.0 if side == "l" else 1.0

    for a_id, b_id in pairs:
        a = rec_by_id.get(a_id)
        b = rec_by_id.get(b_id)
        if a is None or b is None:
            continue
        a_inds = _record_member_indices(a, vert_count)
        b_inds = _record_member_indices(b, vert_count)
        if not a_inds or not b_inds:
            continue
        a_src = a["source"]
        b_src = b["source"]
        seg_len = max((b_src - a_src).length, 1.0e-6)
        radius = max(seg_len * 0.32, 0.0035)
        margin = max(radius * 1.65, 0.0045)
        x_min = min(a_src.x, b_src.x) - margin
        x_max = max(a_src.x, b_src.x) + margin
        y_min = min(a_src.y, b_src.y) - margin
        y_max = max(a_src.y, b_src.y) + margin
        z_min = min(a_src.z, b_src.z) - margin
        z_max = max(a_src.z, b_src.z) + margin

        allowed = set(a_inds) | set(b_inds)
        near_band = set()
        for idx, co in enumerate(original_positions):
            if idx < 0 or idx >= vert_count:
                continue
            if side_sign * co.x < -margin * 0.25:
                continue
            if co.x < x_min or co.x > x_max or co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
                continue
            _t, dist = _closest_segment_factor(co, a_src, b_src)
            if dist <= radius:
                allowed.add(idx)
                near_band.add(idx)

        path = _bfs_shortest_path_masked(adj, a_inds, b_inds, allowed)
        if path and len(path) <= max(4, int(max_path_len)):
            result.update(path)
        # Keep only the tighter spatial band as fallback. The wider allowed set
        # is just for path routing and would protect too much of the side head.
        result.update(near_band)

    if result and expand_steps > 0:
        result = _expanded_vertex_set(result, adj, steps=max(0, int(expand_steps)))
    return result


def ear_side_region_vertex_indices(obj, records, original_positions, side, steps=4):
    """Collect only the real ear shell for one side.

    v0.4.7 improved the ear itself, but its topological expansion was still too
    loose: it entered the ear-under / nape / side-head attachment strips and
    pulled those vertices into the ear local MLS frame.  The ear local solve must
    affect the thin ear shell, not the surrounding head attachment surface.

    This region therefore uses three guards at once:
    - same-side ear anchor BFS seeds,
    - expanded non-ear anchors as hard blockers,
    - source-space tests against same-side ear anchors and attachment/protect
      anchors so the search cannot leak into nape, jaw, cheek, or side head.
    """
    if obj is None or obj.type != 'MESH' or not records:
        return set()
    vert_count = len(obj.data.vertices)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in obj.data.vertices]

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    side_ids = set(ear_side_landmark_ids(side))
    side_recs = [rec_by_id[lm_id] for lm_id in side_ids if lm_id in rec_by_id]
    if len(side_recs) < 4:
        return set()

    adj = build_mesh_adjacency(obj)
    seeds = set()
    non_ear_anchor_members = set()
    protect_anchor_members = set()

    # Anchors adjacent to the ear attachment must block the local ear shell fit.
    # They are allowed to guide the general face/head deformation, but they must
    # not be pulled into the ear frame.
    side_protect_ids = {
        f"temple_{side}_center",
        f"face_{side}_edge",
        f"jaw_{side}_edge",
        f"head_{side}_side_back",
        f"head_{side}_side_upper",
        f"nape_{side}_outer",
        f"neck_top_{side}_side",
        f"neck_top_{side}_back",
        f"neck_{side}_side",
        f"outer_face_{side}_upper",
        f"outer_face_{side}_lower",
        f"cheek_{side}_center",
    }

    for rec in records:
        members = _record_member_indices(rec, vert_count)
        lm_id = rec.get("lm_id")
        if lm_id in side_ids:
            seeds.update(members)
        else:
            non_ear_anchor_members.update(members)
            if lm_id in side_protect_ids or (lm_id and lm_id.startswith(f"ear_{'r' if side == 'l' else 'l'}_")):
                protect_anchor_members.update(members)

    if not seeds:
        return set()

    # Do not allow BFS to pass through or immediately around non-ear anchors.
    # The extra ring around protect anchors specifically prevents ear-under and
    # nape/side-head strips from being re-solved as if they were part of the ear.
    blocked = set(non_ear_anchor_members)
    blocked.update(_expanded_vertex_set(protect_anchor_members, adj, steps=1))
    attachment_guard = ear_attachment_guard_vertex_indices(
        obj, records, original_positions, side, expand_steps=1
    )
    blocked.update(attachment_guard)

    ear_points = [rec["source"] for rec in side_recs]
    protect_points = [rec_by_id[lm_id]["source"] for lm_id in side_protect_ids if lm_id in rec_by_id]

    center = Vector((0.0, 0.0, 0.0))
    for co in ear_points:
        center += co
    center /= float(len(ear_points))

    xs = [p.x for p in ear_points]
    ys = [p.y for p in ear_points]
    zs = [p.z for p in ear_points]
    span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    max_span = max(span.x, span.y, span.z, 1.0e-6)

    # Compact shell gate.  v0.4.7 used ~2.15x anchor radius; that let the solve
    # enter the head attachment fan.  This is deliberately tighter and includes
    # only a small margin around the original ear-anchor box.
    margin = max_span * 0.42
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    min_z, max_z = min(zs) - margin * 0.55, max(zs) + margin * 0.55
    radial_limit = max(max_span * 1.18, 0.006)

    def _is_ear_shell_candidate(idx):
        if idx in blocked:
            return False
        if idx < 0 or idx >= len(original_positions):
            return False
        co = original_positions[idx]
        # Same side only, with a small seam tolerance for unusual templates.
        if side == "l" and co.x > 0.002:
            return False
        if side == "r" and co.x < -0.002:
            return False
        if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
            return False
        ear_d = _min_distance_to_points(co, ear_points)
        if ear_d > radial_limit:
            return False
        if protect_points:
            protect_d = _min_distance_to_points(co, protect_points)
            # If the vertex is at least as much an attachment/head vertex as an
            # ear vertex, leave it to the global face/head solve.
            if protect_d <= ear_d * 1.12:
                return False
        return True

    region = set(idx for idx in seeds if 0 <= idx < vert_count)
    frontier = set(region)
    max_steps = min(max(0, int(steps)), 4)
    for _ in range(max_steps):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in region:
                    continue
                if not _is_ear_shell_candidate(nb):
                    continue
                region.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        frontier = nxt

    # Keep explicit same-side ear anchors exact even if an anchor lies right on a
    # guard boundary.  Those anchors define the local frame.
    region.update(idx for idx in seeds if 0 <= idx < vert_count)
    return region

def apply_ear_local_frame_fit(out_obj, records, original_positions=None,
                              strength=1.0, steps=7, nearest_count=0):
    """Re-fit each ear from same-side ear anchors only.

    The previous failure was not primarily Snap: the generated ear shell was
    already folded before/inside the snap-protected area. This post pass
    overwrites the ear-local patch with an ear-only MLS field computed from the
    original template ear anchors to the current landmark targets. It preserves
    the local front/back/top/lobe ordering of the ear instead of letting global
    jaw/nape/face rails decide the shortest path through the ear.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    if original_positions is None or len(original_positions) != len(verts):
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    if strength <= 0.0:
        return 0
    nearest = int(nearest_count)
    changed = 0
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    vert_count = len(verts)

    for side in ("l", "r"):
        side_records = [rec_by_id[lm_id] for lm_id in ear_side_landmark_ids(side) if lm_id in rec_by_id]
        if len(side_records) < 4:
            continue
        region = ear_side_region_vertex_indices(
            out_obj,
            records,
            original_positions,
            side,
            steps=steps,
        )
        if not region:
            continue
        fixed = _anchor_delta_by_vertex(side_records, vert_count)
        for idx in sorted(region):
            src = original_positions[idx]
            if idx in fixed:
                target = src + fixed[idx]
                # Keep explicit ear anchors exact. They already represent the
                # user's landmark placement and define the ear frame.
                verts[idx].co = target
                changed += 1
                continue
            delta = _mls_delta_for_point(
                src,
                side_records,
                power=2.0,
                nearest_count=nearest,
            )
            target = src + delta
            verts[idx].co = verts[idx].co.lerp(target, strength)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_earlf"] = int(changed)
    except Exception:
        pass
    return changed



def ear_upper_inner_support_fit(out_obj, records, original_positions=None,
                                strength=0.36, steps=1, radius_scale=0.68):
    """Conservatively move the immediate upper-ear inner support row.

    v0.5.6 used an ear-only MLS solve for this patch. On the current template
    that was too free: the one-ring inner vertices could shear across the rim
    frame and make the top/back ear normals worse.  This version deliberately
    avoids affine/shear fitting.  It only gives the nearest one-ring upper-ear
    support vertices an IDW-blended *translation delta* from the same-side upper
    ear landmarks, while leaving all anchor vertices, lower ear, and head
    attachment vertices untouched.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    if original_positions is None or len(original_positions) != len(verts):
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 2))
    radius_scale = max(0.10, float(radius_scale))
    if strength <= 0.0:
        return 0

    vert_count = len(verts)
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    all_fixed = _anchor_delta_by_vertex(records, vert_count)
    changed = 0

    for side in ("l", "r"):
        upper_ids = [
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
        ]
        support_records = [rec_by_id[lm_id] for lm_id in upper_ids if lm_id in rec_by_id]
        if len(support_records) < 4:
            continue

        seed_ids = [
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
        ]
        seeds = set()
        for lm_id in seed_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                seeds.update(_record_member_indices(rec, vert_count))
        if not seeds:
            continue

        # Keep this pass off the attachment fan and the lower ear.  The lower
        # ear has its own transition solve, and the head attachment should be
        # governed by scalp/head/side-face deformation plus snap.
        blocker_ids = {
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"neck_{side}_side",
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
            f"ear_{side}_inner_bottom",
            f"ear_{'r' if side == 'l' else 'l'}_top",
        }
        blockers = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blockers.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blockers.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1
        ))
        blockers.update(_expanded_vertex_set(blockers, adj, steps=1))

        support_points = [rec["source"] for rec in support_records]
        xs = [p.x for p in support_points]
        ys = [p.y for p in support_points]
        zs = [p.z for p in support_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.44, 0.003)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.75, max(zs) + margin * 0.55
        radial_limit = max(max_span * radius_scale, 0.0045)

        candidates = set()
        frontier = set(seeds)
        visited = set(seeds)
        topo_depth = {}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in visited or nb in blockers or nb in all_fixed:
                        continue
                    if nb < 0 or nb >= vert_count:
                        continue
                    co = original_positions[nb]
                    if side == "l" and co.x > 0.002:
                        continue
                    if side == "r" and co.x < -0.002:
                        continue
                    if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                        continue
                    ear_d = _min_distance_to_points(co, support_points)
                    if ear_d > radial_limit:
                        continue
                    if blocker_points:
                        block_d = _min_distance_to_points(co, blocker_points)
                        if block_d <= ear_d * 1.18:
                            continue
                    visited.add(nb)
                    candidates.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        for idx in sorted(candidates):
            src = original_positions[idx]
            # Translation-only IDW.  This keeps the local rim/support offset
            # instead of allowing a free local affine transform to fold it.
            delta = _idw_delta_for_point(src, support_records, power=2.6, nearest_count=4)
            target = src + delta
            d = _min_distance_to_points(src, support_points)
            geo = max(0.0, 1.0 - min(1.0, d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.55)
            if w <= 0.0:
                continue
            verts[idx].co = verts[idx].co.lerp(target, min(0.72, w))
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_earup"] = int(changed)
    except Exception:
        pass
    return changed

def ear_inner_lower_fan_ids(side):
    """Ear-shell landmarks that own the concha/lower inner fan.

    This is the zone that caused the v0.5.9 regression: front_middle and
    inner_front_middle were treated as blockers by the lower attachment solve,
    while lobe/back_lower/inner_bottom moved in a different frame. The result
    was an unsolved wedge around back_lower, lobe, inner_front_middle, and
    front_middle. Keep this as an ear-only fan and solve it separately from the
    head attachment strip.
    """
    return [
        f"ear_{side}_front_middle",
        f"ear_{side}_inner_front_middle",
        f"ear_{side}_front_lower",
        f"ear_{side}_inner_bottom",
        f"ear_{side}_lobe",
        f"ear_{side}_back_lower",
        f"ear_{side}_back_middle",
    ]


def apply_ear_inner_lower_fan_fit(out_obj, records, original_positions=None,
                                  strength=0.78, steps=2, radius_scale=1.05):
    """Conservatively re-fit the inner/lower ear fan.

    This pass addresses the specific lower-ear fold that can appear when the
    upper ear is protected correctly but the concha/lower fan is not solved as
    one region. It includes LM_ear_*_front_middle,
    LM_ear_*_inner_front_middle, LM_ear_*_front_lower, LM_ear_*_inner_bottom,
    LM_ear_*_lobe, LM_ear_*_back_lower, and LM_ear_*_back_middle as a single
    local ear-shell frame.

    It deliberately uses translation-only IDW instead of local affine MLS. That
    preserves the template fan ordering and avoids the old failure where a free
    ear-local affine solve folded the sparse lower ear into red inverted
    triangles. Foreign landmark anchors are hard blockers and are not entered by
    BFS, matching the v0.2.15 lobe-stretch leak fix direction.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 4))
    radius_scale = max(0.10, float(radius_scale))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        fan_ids = [lm_id for lm_id in ear_inner_lower_fan_ids(side) if lm_id in rec_by_id]
        fan_records = [rec_by_id[lm_id] for lm_id in fan_ids]
        if len(fan_records) < 4:
            continue

        fan_anchor_members = set()
        for rec in fan_records:
            fan_anchor_members.update(_record_member_indices(rec, vert_count))
        if not fan_anchor_members:
            continue

        # The fan may touch the upper/back ear and lower lobe, but it must not
        # enter the head attachment surface or the opposite ear/face.  Do not
        # treat front_middle / inner_front_middle as blockers here; they are the
        # upper boundary of this local fan.
        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"neck_{side}_side",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_top",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        # Keep the upper side-head/back-ear strip protected, but do not use the
        # lower attachment rails as a hard blocker for this inner ear fan. The
        # fan must be allowed to connect into back_lower/lobe/inner_bottom.
        upper_guard = ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        )
        upper_guard.difference_update(fan_anchor_members)
        blocker_members.update(upper_guard)
        # Block one ring around non-fan anchor vertices so the search cannot
        # travel through jaw/head/nape anchors and leak into the side-head strip.
        blocker_members.update(_expanded_vertex_set(blocker_members, adj, steps=1))
        blocker_members.difference_update(fan_anchor_members)

        fan_points = [rec["source"] for rec in fan_records]
        xs = [p.x for p in fan_points]
        ys = [p.y for p in fan_points]
        zs = [p.z for p in fan_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.42, 0.0035)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.70, max(zs) + margin * 0.65
        radial_limit = max(max_span * radius_scale, 0.006)

        side_sign = -1.0 if side == "l" else 1.0

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            # Non-fan anchors are fixed boundaries.  They must not be moved or
            # used as BFS pass-through points.
            if idx in all_anchor_members and idx not in fan_anchor_members:
                return False
            co = original_positions[idx]
            if side_sign * co.x < -0.002:
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            fan_d = _min_distance_to_points(co, fan_points)
            if fan_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                # If it is visually closer to head/upper-boundary blockers than
                # to the inner/lower ear fan, leave it out.
                if block_d <= fan_d * 0.92:
                    return False
            return True

        region = set(idx for idx in fan_anchor_members if 0 <= idx < vert_count)
        frontier = set(region)
        topo_depth = {idx: 0 for idx in region}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        fixed = _anchor_delta_by_vertex(fan_records, vert_count)
        for idx in sorted(region):
            if idx < 0 or idx >= vert_count:
                continue
            src = original_positions[idx]
            if idx in fixed:
                verts[idx].co = src + fixed[idx]
                changed += 1
                continue
            delta = _idw_delta_for_point(src, fan_records, power=2.4, nearest_count=5)
            target = src + delta
            fan_d = _min_distance_to_points(src, fan_points)
            geo = max(0.0, 1.0 - min(1.0, fan_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.65)
            # This pass should repair the local fan, not overwrite the broader
            # side-head solve. Keep the final blend bounded.
            verts[idx].co = verts[idx].co.lerp(target, min(0.82, max(0.0, w)))
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_earfn"] = int(changed)
    except Exception:
        pass
    return changed


def ear_lower_transition_ids(side):
    """Landmarks that define the lower-ear to head transition patch.

    This is not the ear shell itself. It is the small fan under/behind the ear
    that must connect ear_front_lower/lobe/back_lower to jaw/nape/neck anchors
    without being pulled into the ear local frame.
    """
    return [
        f"ear_{side}_front_lower",
        f"ear_{side}_lobe",
        f"ear_{side}_back_lower",
        f"ear_{side}_inner_bottom",
        f"jaw_{side}_edge",
        f"nape_{side}_outer",
        f"neck_top_{side}_side",
        f"neck_top_{side}_back",
    ]


def apply_ear_lower_transition_fit(out_obj, records, original_positions=None,
                                   strength=0.70, steps=3, nearest_count=0):
    """Stabilize the lower ear attachment fan after Ear Local Fit.

    v0.4.8 correctly restricted Ear Local Fit to the ear shell, but that also
    left the lower attachment fan between the lobe/front-lower/back-lower and
    jaw/nape/neck anchors to the broad global solve. On different heads that fan
    can bunch into a crushed triangle under the ear. This pass solves only that
    transition fan from lower-ear + jaw/nape/neck landmarks, keeping ear shell
    and non-ear anchors exact while making the in-between vertices interpolate
    through a local lower-ear frame instead of collapsing.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    if original_positions is None or len(original_positions) != len(verts):
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    if strength <= 0.0:
        return 0
    steps = max(0, min(int(steps), 5))
    nearest = int(nearest_count)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    vert_count = len(verts)
    adj = build_mesh_adjacency(out_obj)
    changed = 0

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    for side in ("l", "r"):
        ids = [lm_id for lm_id in ear_lower_transition_ids(side) if lm_id in rec_by_id]
        side_records = [rec_by_id[lm_id] for lm_id in ids]
        if len(side_records) < 4:
            continue

        allowed_anchor_members = set()
        for rec in side_records:
            allowed_anchor_members.update(_record_member_indices(rec, vert_count))
        if not allowed_anchor_members:
            continue

        # Upper ear / cheek / head-side anchors are blockers.  The lower
        # transition fit should not crawl upward into the ear shell or sideways
        # across the cheek/head surface.
        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        attachment_guard = ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        )
        # v0.5.9: keep the upper side-head/back-ear strip protected, but do not
        # block the lower back_lower/lobe/nape fan here. v0.5.8 blocked that fan
        # completely and recreated the old lower-ear flipped-triangle failure.
        attachment_guard.difference_update(allowed_anchor_members)
        blocker_members.update(attachment_guard)

        lower_points = [rec["source"] for rec in side_records]
        xs = [p.x for p in lower_points]
        ys = [p.y for p in lower_points]
        zs = [p.z for p in lower_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.55, 0.004)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.70, max(zs) + margin * 0.45
        radial_limit = max(max_span * 1.35, 0.008)

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            # Other anchors are fixed boundaries, not transition surface.
            if idx in all_anchor_members and idx not in allowed_anchor_members:
                return False
            co = original_positions[idx]
            if side == "l" and co.x > 0.002:
                return False
            if side == "r" and co.x < -0.002:
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            lower_d = _min_distance_to_points(co, lower_points)
            if lower_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                # Keep vertices that visually belong to the upper ear shell or
                # cheek/head attachment out of the lower transition pass.
                if block_d < lower_d * 1.05:
                    return False
            return True

        region = set(allowed_anchor_members)
        frontier = set(allowed_anchor_members)
        for _ in range(steps):
            nxt = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        fixed = _anchor_delta_by_vertex(side_records, vert_count)
        for idx in sorted(region):
            src = original_positions[idx]
            if idx in fixed:
                target = src + fixed[idx]
                verts[idx].co = target
                changed += 1
                continue
            delta = _mls_delta_for_point(
                src,
                side_records,
                power=2.0,
                nearest_count=nearest,
            )
            target = src + delta
            # Softer toward the outer boundary of the transition patch.  This
            # avoids replacing the side-head strip; it only un-crushes the small
            # under-ear fan.
            lower_d = _min_distance_to_points(src, lower_points)
            fall = max(0.0, 1.0 - min(1.0, lower_d / radial_limit))
            w = strength * (0.35 + 0.65 * fall)
            verts[idx].co = verts[idx].co.lerp(target, w)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_earlo"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_lower_attachment_height_guard(out_obj, records, original_positions=None,
                                            strength=0.82, steps=3, radius_scale=1.20,
                                            z_pad_ratio=0.10):
    """Prevent the lower ear/head attachment fan from being lifted by upper-ear solves.

    The v0.5.10 inner/lower fan solve correctly included front_middle and
    inner_front_middle as the local upper boundary, but on the current template
    some vertices that visually belong to the lower attachment strip near
    ear_front_lower / ear_back_lower were still allowed to follow the higher
    front_middle/back_middle frame.  From the inside this appears as the
    back_lower/front_lower connection climbing upward into a folded triangle.

    This pass runs after the normal lower transition fit.  It collects only the
    small same-side lower attachment region and clamps vertices that ended above
    their lower-ear/head-frame target.  It never raises vertices, never moves
    explicit anchor vertices, and treats upper-ear/head-side anchors as blockers.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 5))
    radius_scale = max(0.15, float(radius_scale))
    z_pad_ratio = max(0.0, min(float(z_pad_ratio), 0.60))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        lower_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
            f"jaw_{side}_edge",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ]
        lower_records = [rec_by_id[lm_id] for lm_id in lower_ids if lm_id in rec_by_id]
        if len(lower_records) < 4:
            continue

        lower_anchor_members = set()
        for rec in lower_records:
            lower_anchor_members.update(_record_member_indices(rec, vert_count))
        if not lower_anchor_members:
            continue

        # These higher anchors must be boundaries only.  If lower attachment
        # vertices blend toward them directly, the under/back-lower fan lifts up.
        upper_blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_top",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in upper_blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.update(_expanded_vertex_set(blocker_members, adj, steps=1))
        blocker_members.difference_update(lower_anchor_members)

        lower_points = [rec["source"] for rec in lower_records]
        xs = [p.x for p in lower_points]
        ys = [p.y for p in lower_points]
        zs = [p.z for p in lower_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.50, 0.004)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.65, max(zs) + margin * 0.50
        radial_limit = max(max_span * radius_scale, 0.007)
        z_pad = max(max_span * z_pad_ratio, 0.00055)
        side_sign = -1.0 if side == "l" else 1.0

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            if idx in all_anchor_members and idx not in lower_anchor_members:
                return False
            co = original_positions[idx]
            if side_sign * co.x < -0.002:
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            lower_d = _min_distance_to_points(co, lower_points)
            if lower_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                # This is a height guard for lower attachment vertices only.
                # Anything as close to the upper ear/head-side boundary as the
                # lower frame is left to the upper/head solvers.
                if block_d <= lower_d * 1.08:
                    return False
            return True

        region = set(idx for idx in lower_anchor_members if 0 <= idx < vert_count)
        frontier = set(region)
        topo_depth = {idx: 0 for idx in region}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        fixed = _anchor_delta_by_vertex(lower_records, vert_count)
        for idx in sorted(region):
            if idx < 0 or idx >= vert_count:
                continue
            if idx in all_anchor_members:
                # Do not rewrite anchor locations here.  This pass only controls
                # the in-between lower attachment vertices.
                continue
            src = original_positions[idx]
            delta = _idw_delta_for_point(src, lower_records, power=2.8, nearest_count=4)
            target = src + delta
            cur = verts[idx].co.copy()
            # Only correct upward overshoot.  If the broader solve already placed
            # the vertex at or below the lower frame, preserve it.
            overshoot = cur.z - (target.z + z_pad)
            if overshoot <= 0.0:
                continue
            lower_d = _min_distance_to_points(src, lower_points)
            geo = max(0.0, 1.0 - min(1.0, lower_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.70)
            if w <= 0.0:
                continue
            desired = cur.copy()
            desired.z = target.z + z_pad
            # A small XY blend helps the folded dart relax, but Z is the main
            # correction.  Keep XY conservative so the side-head guard remains.
            desired.x = cur.x * 0.82 + target.x * 0.18
            desired.y = cur.y * 0.82 + target.y * 0.18
            verts[idx].co = cur.lerp(desired, min(1.0, w))
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elhgt"] = int(changed)
    except Exception:
        pass
    return changed


def _hfr_world_z(obj, local_co):
    try:
        return (obj.matrix_world @ local_co).z
    except Exception:
        return float(local_co.z)


def _hfr_local_delta_for_world_z(obj, dz):
    """Return a local-space vector that changes world Z by dz.

    The template output may be rotated/scaled, so local Z is not always Blender
    World-Z.  The ear lower height fixes must operate in the user's visual
    up/down axis, therefore they use this helper instead of modifying local z.
    """
    try:
        inv3 = obj.matrix_world.to_3x3().inverted()
        return inv3 @ Vector((0.0, 0.0, float(dz)))
    except Exception:
        return Vector((0.0, 0.0, float(dz)))


def _hfr_world_x(obj, local_co):
    try:
        return (obj.matrix_world @ local_co).x
    except Exception:
        return float(local_co.x)


def _hfr_local_delta_for_world_x(obj, dx):
    """Return a local-space vector that changes Blender World-X by dx."""
    try:
        inv3 = obj.matrix_world.to_3x3().inverted()
        return inv3 @ Vector((float(dx), 0.0, 0.0))
    except Exception:
        return Vector((float(dx), 0.0, 0.0))


def _hfr_world_y(obj, local_co):
    try:
        return (obj.matrix_world @ local_co).y
    except Exception:
        return float(local_co.y)


def _hfr_local_delta_for_world_y(obj, dy):
    """Return a local-space vector that changes Blender World-Y by dy."""
    try:
        inv3 = obj.matrix_world.to_3x3().inverted()
        return inv3 @ Vector((0.0, float(dy), 0.0))
    except Exception:
        return Vector((0.0, float(dy), 0.0))


def apply_ear_inner_lower_inward_guard(out_obj, records, original_positions=None,
                                       strength=0.86, steps=2, world_x_pad=0.0009):
    """Push lower/inner ear connector vertices back out from the head side.

    v0.5.13 fixed the same lower-ear connector fan in World-Z, but the next
    visible problem is a different axis: vertices around front_lower/lobe,
    inner_bottom, and inner_front_middle can still tuck inward toward the head
    center.  This pass uses Blender World-X as the user's left/right axis and
    only pushes non-anchor connector vertices outward when the local ear frame
    says they should be farther from the centerline.

    The pass is intentionally one-sided and conservative:
    - lower/inner ear anchors are not moved,
    - upper-ear/head/face anchors are hard blockers,
    - correction is applied only along World-X,
    - only vertices reachable from the lower/inner fan are candidates.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 4))
    world_x_pad = max(0.0, float(world_x_pad))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        # Anchors that define the lower/inner ear pocket the user pointed to.
        support_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_lower",
        ]
        support_records = [rec_by_id.get(lm_id) for lm_id in support_ids if rec_by_id.get(lm_id) is not None]
        if len(support_records) < 4:
            continue

        seed_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
        ]
        seed_members = set()
        support_anchor_members = set()
        for rec in support_records:
            support_anchor_members.update(_record_member_indices(rec, vert_count))
        for lm_id in seed_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                seed_members.update(_record_member_indices(rec, vert_count))
        seed_members = {idx for idx in seed_members if 0 <= idx < vert_count}
        if not seed_members:
            continue

        # Do not enter the upper ear or head attachment surface while repairing
        # the lower/inner pocket.  front_middle/inner_front_middle are not
        # blockers here because they are part of the requested problem boundary.
        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"neck_{side}_side",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_top",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.update(_expanded_vertex_set(blocker_members, adj, steps=1))
        blocker_members.difference_update(support_anchor_members)

        support_points = [rec["source"] for rec in support_records]
        xs = [p.x for p in support_points]
        ys = [p.y for p in support_points]
        zs = [p.z for p in support_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.55, 0.0045)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.80, max(zs) + margin * 0.80
        radial_limit = max(max_span * 1.25, 0.0075)
        side_sign = -1.0 if side == "l" else 1.0

        def _same_side_world(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            if idx in all_anchor_members and idx not in support_anchor_members:
                return False
            co = original_positions[idx]
            if not _same_side_world(co):
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            support_d = _min_distance_to_points(co, support_points)
            if support_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                if block_d <= support_d * 0.86:
                    return False
            return True

        region = set(seed_members)
        frontier = set(seed_members)
        topo_depth = {idx: 0 for idx in seed_members}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        for idx in sorted(region):
            if idx < 0 or idx >= vert_count:
                continue
            # Never rewrite anchors in this guard.  It only handles the in-between
            # pocket vertices that are visually slipping inward.
            if idx in all_anchor_members:
                continue
            src = original_positions[idx]
            if not _candidate(idx):
                continue
            target_delta = _idw_delta_for_point(src, support_records, power=2.8, nearest_count=5)
            target = src + target_delta
            cur = verts[idx].co.copy()
            cur_out = side_sign * _hfr_world_x(out_obj, cur)
            target_out = side_sign * _hfr_world_x(out_obj, target)
            inward = (target_out + world_x_pad) - cur_out
            if inward <= 0.0:
                continue

            support_d = _min_distance_to_points(src, support_points)
            geo = max(0.0, 1.0 - min(1.0, support_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.72)
            if w <= 0.0:
                continue
            dx = side_sign * inward * min(1.0, w)
            verts[idx].co = cur + _hfr_local_delta_for_world_x(out_obj, dx)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eilig"] = int(changed)
    except Exception:
        pass
    return changed



def apply_ear_inner_pocket_depth_guard(out_obj, records, original_positions=None,
                                       strength=0.90, steps=3,
                                       world_y_pad=0.0010,
                                       world_x_pad=0.0010):
    """Keep the lower/inner ear pocket from folding into the head surface.

    v0.5.14 showed that an X-only inward guard is not enough.  The diagnostic
    selected two kinds of practical failures:
      - a non-anchor front_lower/lobe connector that stays slightly too close to
        the head centerline, and
      - a non-anchor inner_bottom/inner_front_middle connector that is pulled
        too far in Blender World-Y, producing a small inward spike.

    This guard therefore works in Blender world axes, not template local axes:
      - World-X is used only as a side/outward minimum for front-lower/lobe
        connector vertices,
      - World-Y is used as a depth cap for the inner-bottom/inner-front pocket.

    Landmark anchor vertices are not moved here.  If LM_ear_*_inner_front_middle
    itself is selected/problematic, that is a true landmark placement issue;
    this pass only stabilizes the surrounding fan so small placement differences
    do not create inverted connector triangles.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 4))
    world_y_pad = max(0.0, float(world_y_pad))
    world_x_pad = max(0.0, float(world_x_pad))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0

        seed_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
        ]
        support_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_lower",
        ]
        cap_y_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
        ]
        lower_x_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
        ]

        support_records = [rec_by_id.get(lm_id) for lm_id in support_ids if rec_by_id.get(lm_id) is not None]
        if len(support_records) < 4:
            continue

        seed_members = set()
        support_anchor_members = set()
        cap_y_members = set()
        lower_x_members = set()
        for rec in support_records:
            support_anchor_members.update(_record_member_indices(rec, vert_count))
        for lm_id in seed_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                seed_members.update(_record_member_indices(rec, vert_count))
        for lm_id in cap_y_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                cap_y_members.update(_record_member_indices(rec, vert_count))
        for lm_id in lower_x_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                lower_x_members.update(_record_member_indices(rec, vert_count))

        seed_members = {idx for idx in seed_members if 0 <= idx < vert_count}
        cap_y_members = {idx for idx in cap_y_members if 0 <= idx < vert_count}
        lower_x_members = {idx for idx in lower_x_members if 0 <= idx < vert_count}
        if not seed_members or not cap_y_members:
            continue

        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"neck_{side}_side",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_top",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.update(_expanded_vertex_set(blocker_members, adj, steps=1))
        blocker_members.difference_update(support_anchor_members)

        support_points = [rec["source"] for rec in support_records]
        xs = [p.x for p in support_points]
        ys = [p.y for p in support_points]
        zs = [p.z for p in support_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.58, 0.0045)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.90, max(zs) + margin * 0.90
        radial_limit = max(max_span * 1.35, 0.0080)

        def _same_side_world(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            if idx in all_anchor_members and idx not in support_anchor_members:
                return False
            co = original_positions[idx]
            if not _same_side_world(co):
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            support_d = _min_distance_to_points(co, support_points)
            if support_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                if block_d <= support_d * 0.84:
                    return False
            return True

        region = set(seed_members)
        frontier = set(seed_members)
        topo_depth = {idx: 0 for idx in seed_members}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        y_cap = max(_hfr_world_y(out_obj, verts[idx].co) for idx in cap_y_members) + world_y_pad
        x_min_out = None
        if lower_x_members:
            x_min_out = min(side_sign * _hfr_world_x(out_obj, verts[idx].co) for idx in lower_x_members) + world_x_pad

        for idx in sorted(region):
            if idx < 0 or idx >= vert_count:
                continue
            if idx in all_anchor_members:
                continue
            if not _candidate(idx):
                continue

            src = original_positions[idx]
            support_d = _min_distance_to_points(src, support_points)
            geo = max(0.0, 1.0 - min(1.0, support_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.76)
            if w <= 0.0:
                continue

            cur = verts[idx].co.copy()
            delta_local = Vector((0.0, 0.0, 0.0))
            cur_y = _hfr_world_y(out_obj, cur)
            if cur_y > y_cap:
                dy = (y_cap - cur_y) * min(1.0, w)
                delta_local += _hfr_local_delta_for_world_y(out_obj, dy)

            if x_min_out is not None:
                cur_out = side_sign * _hfr_world_x(out_obj, cur)
                if cur_out < x_min_out:
                    dx = side_sign * (x_min_out - cur_out) * min(1.0, w)
                    delta_local += _hfr_local_delta_for_world_x(out_obj, dx)

            if delta_local.length <= 1.0e-10:
                continue
            verts[idx].co = cur + delta_local
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eipdg"] = int(changed)
    except Exception:
        pass
    return changed

def apply_ear_inner_sheet_outward_guard(out_obj, records, original_positions=None,
                                       strength=0.84, steps=3, world_x_pad=0.00065,
                                       max_world_x_push=0.0028):
    """Stabilize the lower/inner ear sheet using the current anchor frame.

    v0.5.15 still left a small inward pocket around the selected lower/inner
    ear quads.  The important difference from the earlier inward/depth guards
    is that this pass reads the *current* anchor vertex positions after all ear
    local fits and clamps have run.  It then gives nearby connector vertices an
    outward World-X floor inferred from the surrounding ear anchors.

    This catches vertices such as the selected inner-front connector that are
    no longer high in World-Z and may pass the previous depth cap, but are still
    visibly tucked toward the head centerline.  Landmark anchors themselves are
    not moved; only connector vertices reachable from the lower/inner ear sheet
    are corrected.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 4))
    world_x_pad = max(0.0, float(world_x_pad))
    max_world_x_push = max(0.0001, float(max_world_x_push))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        support_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_lower",
        ]
        support_records = [rec_by_id.get(lm_id) for lm_id in support_ids if rec_by_id.get(lm_id) is not None]
        if len(support_records) < 4:
            continue

        support_anchor_members = set()
        for rec in support_records:
            support_anchor_members.update(_record_member_indices(rec, vert_count))
        support_anchor_members = {idx for idx in support_anchor_members if 0 <= idx < vert_count}
        if not support_anchor_members:
            continue

        # Current anchor positions are the frame of reference for this final
        # micro-guard.  Earlier passes may have already clamped or fitted them,
        # so record/source deltas alone are no longer enough for the remaining
        # connector sheet.
        support_local = [verts[idx].co.copy() for idx in sorted(support_anchor_members)]
        support_world_out = [side_sign * _hfr_world_x(out_obj, co) for co in support_local]
        if not support_local:
            continue

        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"neck_{side}_side",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_top",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.update(_expanded_vertex_set(blocker_members, adj, steps=1))
        blocker_members.difference_update(support_anchor_members)

        xs = [p.x for p in support_local]
        ys = [p.y for p in support_local]
        zs = [p.z for p in support_local]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.64, 0.0048)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.95, max(zs) + margin * 0.95
        radial_limit = max(max_span * 1.42, 0.0085)

        def _same_side_world(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            if idx in all_anchor_members and idx not in support_anchor_members:
                return False
            co = original_positions[idx]
            if not _same_side_world(co):
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            support_d = _min_distance_to_points(co, support_local)
            if support_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                if block_d <= support_d * 0.82:
                    return False
            return True

        region = set(support_anchor_members)
        frontier = set(support_anchor_members)
        topo_depth = {idx: 0 for idx in support_anchor_members}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        def _expected_outward_floor(local_co):
            weighted = []
            for p, out_val in zip(support_local, support_world_out):
                d = max((local_co - p).length, 1.0e-6)
                weighted.append((d, out_val))
            weighted.sort(key=lambda item: item[0])
            nearest = weighted[:5]
            denom = 0.0
            accum = 0.0
            for d, out_val in nearest:
                w = 1.0 / (d ** 2.8 + 1.0e-12)
                denom += w
                accum += w * out_val
            if denom <= 1.0e-12:
                return None
            return accum / denom + world_x_pad

        for idx in sorted(region):
            if idx < 0 or idx >= vert_count:
                continue
            # This pass is for the sheet between landmarks, not the landmarks.
            if idx in all_anchor_members:
                continue
            if not _candidate(idx):
                continue
            cur = verts[idx].co.copy()
            floor_out = _expected_outward_floor(cur)
            if floor_out is None:
                continue
            cur_out = side_sign * _hfr_world_x(out_obj, cur)
            need = floor_out - cur_out
            if need <= 0.0:
                continue

            src = original_positions[idx]
            support_d = _min_distance_to_points(src, support_local)
            geo = max(0.0, 1.0 - min(1.0, support_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps)) / float(steps + 1)))
            w = strength * max(_smoothstep01(geo), _smoothstep01(topo) * 0.72)
            if w <= 0.0:
                continue
            dx = side_sign * min(need, max_world_x_push) * min(1.0, w)
            verts[idx].co = cur + _hfr_local_delta_for_world_x(out_obj, dx)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eisog"] = int(changed)
    except Exception:
        pass
    return changed

def apply_ear_lower_selected_vertex_clamp(out_obj, records, original_positions=None,
                                          strength=0.92, world_z_pad=0.0022):
    """Clamp the exact lower-ear connector vertices that the diagnostic exposed.

    v0.5.12 added the vertex diagnostic because the previous guards were looking
    at the wrong practical axis.  The selected problem vertices were not high in
    local Z; they were high in Blender World-Z because this template's local Y/Z
    axes are rotated.  This pass therefore uses world height.

    The affected topology is the one-ring connector around:
      ear_front_lower, ear_back_lower, and ear_lobe.

    It does not move anchors.  It also excludes one-ring vertices directly owned
    by upper ear landmarks such as front_middle/back_middle, so the v0.5.8 upper
    ear improvement is preserved.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    world_z_pad = max(0.0, float(world_z_pad))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    total_candidates = 0

    for side in ("l", "r"):
        lower_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_back_lower",
            f"ear_{side}_lobe",
        ]
        lower_records = [rec_by_id.get(lm_id) for lm_id in lower_ids if rec_by_id.get(lm_id) is not None]
        if len(lower_records) < 3:
            continue

        lower_anchor_members = set()
        for rec in lower_records:
            lower_anchor_members.update(_record_member_indices(rec, vert_count))
        lower_anchor_members = {idx for idx in lower_anchor_members if 0 <= idx < vert_count}
        if not lower_anchor_members:
            continue

        # Upper/inner-middle ear landmarks are visual boundaries. Their direct
        # neighbor fan should not be lowered by this clamp.
        upper_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"nape_{side}_outer",
        }
        upper_members = set()
        for lm_id in upper_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            upper_members.update(_record_member_indices(rec, vert_count))
        upper_block = set(upper_members)
        for idx in list(upper_members):
            if 0 <= idx < vert_count:
                upper_block.update(adj[idx])

        # The diagnostic-selected vertices are exactly in this lower one-ring:
        # 878/1128/1444 on the current template, and the mirrored right-side
        # equivalents on the other half.
        candidates = set()
        for idx in lower_anchor_members:
            for nb in adj[idx]:
                if nb in lower_anchor_members:
                    continue
                if nb in all_anchor_members:
                    continue
                if nb in upper_block:
                    continue
                candidates.add(nb)

        if not candidates:
            continue

        lower_cap_z = max(_hfr_world_z(out_obj, verts[idx].co) for idx in lower_anchor_members)
        z_cap = lower_cap_z + world_z_pad

        # Keep the correction local to the lower-ear connector fan.  A vertex may
        # be one-ring from a lower anchor but actually belong to the upper
        # concha; reject those if their source-space distance is closer to upper
        # boundary records than to the lower frame.
        lower_points = [rec["source"] for rec in lower_records]
        upper_points = []
        for lm_id in upper_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                upper_points.append(rec["source"])

        for idx in sorted(candidates):
            if idx < 0 or idx >= vert_count:
                continue
            src = original_positions[idx]
            lower_d = _min_distance_to_points(src, lower_points)
            if upper_points:
                upper_d = _min_distance_to_points(src, upper_points)
                if upper_d <= lower_d * 0.72:
                    continue

            cur = verts[idx].co.copy()
            cur_wz = _hfr_world_z(out_obj, cur)
            overshoot = cur_wz - z_cap
            if overshoot <= 0.0:
                continue

            # Blend a little stronger for vertices nearest the lower anchors.
            local_strength = strength
            try:
                # If the vertex is directly between two lower anchors, use the
                # full clamp.  If it is a looser one-ring side fan, keep it softer.
                lower_neighbors = sum(1 for nb in adj[idx] if nb in lower_anchor_members)
                if lower_neighbors <= 1:
                    local_strength *= 0.78
            except Exception:
                pass

            dz = -overshoot * max(0.0, min(1.0, local_strength))
            verts[idx].co = cur + _hfr_local_delta_for_world_z(out_obj, dz)
            changed += 1
            total_candidates += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_ellwc"] = int(changed)
        out_obj["HFR_ellwn"] = int(total_candidates)
    except Exception:
        pass
    return changed


def apply_ear_lower_front_connector_height_guard(out_obj, records, original_positions=None,
                                                 strength=0.88, steps=2,
                                                 world_z_pad=0.00045,
                                                 depth_z_bias=0.00125,
                                                 max_world_z_drop=0.0085):
    """Lower the front-lower ear connector sheet that can climb into the head.

    v0.5.16 fixed the lower/inner ear pocket mostly by pushing selected sheets
    outward and by clamping the direct one-ring lower anchors in World-Z.  The
    new diagnostic, however, selected the edge 1128-1350: 1128 is one-ring from
    ear_front_lower/lobe, while 1350 is the next connector row toward the head.
    That second-row connector is not caught by the direct one-ring clamp, and the
    first-row cap was still too high because it used the highest lower anchor.

    This pass handles only the lower front attachment fan:
      - seed from ear_front_lower / ear_lobe / ear_back_lower,
      - expand at most two rings through non-anchor vertices,
      - do not move anchors,
      - use Blender World-Z, not template local Z,
      - only lower vertices that sit above a weighted lower-anchor height cap.

    The cap is intentionally derived from the local lower-ear anchor frame rather
    than from the highest lower anchor, so the front-lower/lobe connector cannot
    remain lifted into an inward fold.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 3))
    world_z_pad = max(0.0, float(world_z_pad))
    depth_z_bias = max(0.0, float(depth_z_bias))
    max_world_z_drop = max(0.0001, float(max_world_z_drop))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        frame_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
        ]
        frame_records = [rec_by_id.get(lm_id) for lm_id in frame_ids if rec_by_id.get(lm_id) is not None]
        if len(frame_records) < 3:
            continue

        frame_anchor_members = set()
        anchor_z_samples = []
        source_points = []
        for rec in frame_records:
            members = _record_member_indices(rec, vert_count)
            frame_anchor_members.update(members)
            source_points.append(rec["source"])
            for idx in members:
                if 0 <= idx < vert_count:
                    anchor_z_samples.append((rec["source"].copy(), _hfr_world_z(out_obj, verts[idx].co)))
        frame_anchor_members = {idx for idx in frame_anchor_members if 0 <= idx < vert_count}
        if not frame_anchor_members or not anchor_z_samples:
            continue

        # Boundary anchors.  The guard may look past lower anchors by two rings,
        # but it should not climb into inner/front-middle, upper ear, face, or
        # nape/neck rows.
        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        # Keep the side-head attachment guard as a boundary, but do not let it
        # remove the lower frame anchors themselves.
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.difference_update(frame_anchor_members)

        xs = [p.x for p in source_points]
        ys = [p.y for p in source_points]
        zs = [p.z for p in source_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.72, 0.0060)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.85, max(zs) + margin * 0.85
        radial_limit = max(max_span * 1.70, 0.0100)

        def _same_side(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            # This guard corrects connector vertices only.  Anchors are fixed.
            if idx in all_anchor_members:
                return False
            co = original_positions[idx]
            if not _same_side(co):
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            lower_d = _min_distance_to_points(co, source_points)
            if lower_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                if block_d < lower_d * 0.78:
                    return False
            return True

        region = set()
        frontier = set(frame_anchor_members)
        topo_depth = {}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region or nb in frame_anchor_members:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        if not region:
            continue

        def _weighted_lower_cap_world_z(src, topo_depth_value):
            weighted = []
            for p, wz in anchor_z_samples:
                d = max((src - p).length, 1.0e-6)
                w = 1.0 / (d ** 2.6)
                weighted.append((w, wz))
            weighted.sort(reverse=True, key=lambda item: item[0])
            # The two nearest lower anchors define the local connector height
            # better than the highest of the whole lower frame.
            top = weighted[:2] if len(weighted) >= 2 else weighted
            wsum = sum(w for w, _ in top)
            if wsum <= 1.0e-12:
                base = min(wz for _, wz in anchor_z_samples)
            else:
                base = sum(w * wz for w, wz in top) / wsum
            # The second ring toward the head has to sit slightly lower than the
            # first ring; otherwise the sheet stays tilted upward and cuts into
            # the surface from the inside view.
            depth_bias = depth_z_bias * max(0, int(topo_depth_value) - 1)
            return base + world_z_pad - depth_bias

        for idx in sorted(region):
            src = original_positions[idx]
            cur = verts[idx].co.copy()
            cur_wz = _hfr_world_z(out_obj, cur)
            cap_z = _weighted_lower_cap_world_z(src, topo_depth.get(idx, steps))
            overshoot = cur_wz - cap_z
            if overshoot <= 0.0:
                continue
            lower_d = _min_distance_to_points(src, source_points)
            geo = max(0.0, 1.0 - min(1.0, lower_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps) - 1) / float(max(1, steps))))
            w = strength * max(0.42, _smoothstep01(geo)) * (0.70 + 0.30 * topo)
            dz = -min(overshoot * max(0.0, min(1.0, w)), max_world_z_drop)
            if dz >= 0.0:
                continue
            verts[idx].co = cur + _hfr_local_delta_for_world_z(out_obj, dz)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elfhg"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_lower_front_connector_inset_guard(out_obj, records, original_positions=None,
                                                strength=0.58, steps=2,
                                                world_x_inset=0.00075,
                                                second_ring_boost=1.35,
                                                max_world_x_inset=0.00145):
    """Inset the already-lowered front/lobe connector slightly toward the head.

    v0.5.17 lowered the selected 1128/1350 style lower-front connector fan in
    Blender World-Z, which reduced the inward piercing.  The next inspection
    shows that the same fan should sit a little farther inside the ear/head
    attachment, not merely lower.  This pass is intentionally narrow:

      - seed from ear_front_lower / ear_lobe / ear_back_lower,
      - expand only through non-anchor connector vertices for up to two rings,
      - do not move any landmark anchor,
      - move only in Blender World-X toward the head center,
      - do not touch the upper-ear / inner-front-middle / face / nape blockers.

    For the left ear, inward means +World-X; for the right ear, inward means
    -World-X.  The second connector row gets a slightly stronger inset because
    the diagnostic edge 1128-1350 showed that row 2 remains visually too far out
    after the height clamp.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 3))
    world_x_inset = max(0.0, float(world_x_inset))
    second_ring_boost = max(1.0, min(float(second_ring_boost), 2.25))
    max_world_x_inset = max(0.0001, float(max_world_x_inset))
    if strength <= 0.0 or world_x_inset <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        frame_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
        ]
        frame_records = [rec_by_id.get(lm_id) for lm_id in frame_ids if rec_by_id.get(lm_id) is not None]
        if len(frame_records) < 3:
            continue

        frame_anchor_members = set()
        source_points = []
        anchor_world_out = []
        for rec in frame_records:
            source_points.append(rec["source"])
            members = _record_member_indices(rec, vert_count)
            frame_anchor_members.update(members)
            for idx in members:
                if 0 <= idx < vert_count:
                    anchor_world_out.append(side_sign * _hfr_world_x(out_obj, verts[idx].co))
        frame_anchor_members = {idx for idx in frame_anchor_members if 0 <= idx < vert_count}
        if not frame_anchor_members or not source_points:
            continue

        # Keep this pass in the lower-front/lobe attachment only.  In particular,
        # inner_front_middle and inner_bottom belong to the upper/inner pocket and
        # should not be pulled inward by this lower-front corrective pass.
        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blocker_members.update(ear_attachment_guard_vertex_indices(
            out_obj, records, original_positions, side, expand_steps=1, include_lower=False
        ))
        blocker_members.difference_update(frame_anchor_members)

        xs = [p.x for p in source_points]
        ys = [p.y for p in source_points]
        zs = [p.z for p in source_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.74, 0.0060)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.90, max(zs) + margin * 0.90
        radial_limit = max(max_span * 1.70, 0.0100)

        def _same_side(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in blocker_members:
                return False
            if idx in all_anchor_members:
                return False
            co = original_positions[idx]
            if not _same_side(co):
                return False
            if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                return False
            lower_d = _min_distance_to_points(co, source_points)
            if lower_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(co, blocker_points)
                if block_d < lower_d * 0.78:
                    return False
            return True

        region = set()
        frontier = set(frame_anchor_members)
        topo_depth = {}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region or nb in frame_anchor_members:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        if not region:
            continue

        # Do not let the inset push a connector past the lower anchor frame too
        # far toward the face center.  A tiny inward margin is sufficient; the
        # goal is to tuck the selected lower sheet, not collapse it.
        if anchor_world_out:
            min_anchor_out = min(anchor_world_out)
            inward_limit_out = max(0.0, min_anchor_out - 0.0012)
        else:
            inward_limit_out = 0.0

        for idx in sorted(region):
            cur = verts[idx].co.copy()
            cur_out = side_sign * _hfr_world_x(out_obj, cur)
            depth = topo_depth.get(idx, steps)
            lower_d = _min_distance_to_points(original_positions[idx], source_points)
            geo = max(0.0, 1.0 - min(1.0, lower_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(depth - 1) / float(max(1, steps))))
            boost = second_ring_boost if depth >= 2 else 1.0
            desired = world_x_inset * boost * strength * (0.60 + 0.40 * _smoothstep01(geo)) * (0.70 + 0.30 * topo)
            desired = min(desired, max_world_x_inset)
            if desired <= 0.0:
                continue
            # Inward means reducing the side-signed outward coordinate.
            if cur_out - desired < inward_limit_out:
                desired = max(0.0, cur_out - inward_limit_out)
            if desired <= 0.0:
                continue
            dx = -side_sign * desired
            verts[idx].co = cur + _hfr_local_delta_for_world_x(out_obj, dx)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elfig"] = int(changed)
    except Exception:
        pass
    return changed



def apply_ear_lower_back_nape_direction_guard(out_obj, records, original_positions=None,
                                             slide_strength=0.42, dot_threshold=0.72,
                                             strong_dot=0.90):
    """Tune the lower-back ear connector to the nape-directed slide target.

    The v0.5.18 diagnostic selected vertex 878 on the left lower ear.  Visually,
    the desired placement matched sliding that connector about 0.42 toward
    LM_nape_l_outer.  This is not a Blender vertex-slide operation; it is a
    formulaic post-fit guard:

      - seed only from ear_back_lower anchors,
      - move only non-anchor one-ring connector vertices,
      - choose the neighbor edge whose world direction best points toward the
        same-side nape_outer landmark,
      - lerp along that existing edge by the requested slide_strength.

    On the current template this selects the 878 -> 1131 edge rather than pulling
    the point directly across space to the nape landmark.  That reproduces the
    requested slide-like placement while preserving local topology and avoiding
    the earlier over-pull of the broader ear attachment strip.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    dot_threshold = max(-1.0, min(float(dot_threshold), 1.0))
    strong_dot = max(dot_threshold, min(float(strong_dot), 1.0))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _world_co(local_co):
        try:
            return out_obj.matrix_world @ local_co
        except Exception:
            return local_co.copy()

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        back_rec = rec_by_id.get(f"ear_{side}_back_lower")
        nape_rec = rec_by_id.get(f"nape_{side}_outer")
        if back_rec is None or nape_rec is None:
            continue

        back_members = {idx for idx in _record_member_indices(back_rec, vert_count) if 0 <= idx < vert_count}
        nape_members = {idx for idx in _record_member_indices(nape_rec, vert_count) if 0 <= idx < vert_count}
        if not back_members or not nape_members:
            continue

        # Use the actual solved anchor position, not the source landmark record.
        nape_world = Vector((0.0, 0.0, 0.0))
        for idx in nape_members:
            nape_world += _world_co(verts[idx].co)
        nape_world /= float(len(nape_members))

        # Boundary anchors that should not be the slide target.  The selected
        # lower connector should follow the lower-back/nape direction through the
        # connector sheet, not collapse onto ear_back_lower/back_middle/lobe.
        hard_block_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"temple_{side}_center",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
        }
        hard_block = set(back_members)
        for lm_id in hard_block_ids:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                hard_block.update(_record_member_indices(rec, vert_count))

        # Candidate scope is intentionally one-ring from ear_back_lower.  This
        # catches the exposed 878-style connector and its mirrored counterpart,
        # but avoids the wider side-head attachment strip that v0.5.8 had to
        # protect.
        candidates = set()
        for idx in back_members:
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in all_anchor_members:
                    continue
                candidates.add(nb)

        for idx in sorted(candidates):
            cur = verts[idx].co.copy()
            cur_world = _world_co(cur)
            if side_sign * cur_world.x <= -0.002:
                continue
            to_nape = nape_world - cur_world
            if to_nape.length <= 1.0e-10:
                continue
            to_nape.normalize()

            best_nb = None
            best_dot = -2.0
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in hard_block:
                    continue
                # Do not slide onto any landmark anchor; this guard is for the
                # connector sheet only.
                if nb in all_anchor_members:
                    continue
                nb_world = _world_co(verts[nb].co)
                edge_vec = nb_world - cur_world
                if edge_vec.length <= 1.0e-10:
                    continue
                edge_dir = edge_vec.normalized()
                d = edge_dir.dot(to_nape)
                if d > best_dot:
                    best_dot = d
                    best_nb = nb

            if best_nb is None or best_dot < dot_threshold:
                continue

            # When the edge direction strongly matches the nape direction, use
            # the requested 0.42 factor.  Near-threshold matches are softened so
            # the mirrored or neighboring sheet does not get over-pulled.
            if strong_dot > dot_threshold:
                match = max(0.0, min(1.0, (best_dot - dot_threshold) / (strong_dot - dot_threshold)))
            else:
                match = 1.0
            local_slide = slide_strength * (0.55 + 0.45 * _smoothstep01(match))
            if best_dot >= strong_dot:
                local_slide = slide_strength
            local_slide = max(0.0, min(1.0, local_slide))
            if local_slide <= 0.0:
                continue

            target = cur.lerp(verts[best_nb].co, local_slide)
            if (target - cur).length <= 1.0e-10:
                continue
            verts[idx].co = target
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elnbg"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_lobe_upper_connector_lift_guard(out_obj, records, original_positions=None,
                                             strength=0.96, steps=3,
                                             world_z_pad=0.00004,
                                             upper_blend_bias=0.24,
                                             max_world_z_lift=0.0105):
    """Raise only the outside upper-lobe connector sheet.

    v0.5.20 used the inner-bottom / inner-front-middle landmarks both as height
    references and as topology seeds.  That raised some inside concha connector
    vertices again, which reproduced the inner-side fold the user had already
    accepted as fixed.  This pass keeps those inner landmarks as optional height
    references, but does not seed from them and blocks their direct 1-ring fan.

    The intended target is the outer sheet just above the lobe/back-lower area,
    including diagnostic vertices such as 1456 and 1468.  The position is still
    computed procedurally from the current lower and upper ear frame; no fixed
    world-coordinate move or hard-coded vertex index is used.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    strength = max(0.0, min(float(strength), 1.0))
    steps = max(1, min(int(steps), 4))
    world_z_pad = max(0.0, float(world_z_pad))
    upper_blend_bias = max(0.0, min(float(upper_blend_bias), 0.45))
    max_world_z_lift = max(0.0001, float(max_world_z_lift))
    if strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _samples_for_ids(ids):
        samples = []
        points = []
        members = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            points.append(rec["source"].copy())
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    members.add(idx)
                    samples.append((rec["source"].copy(), _hfr_world_z(out_obj, verts[idx].co)))
        return samples, points, members

    def _idw_z(src, samples, power=2.35):
        if not samples:
            return None
        weighted = []
        for p, wz in samples:
            d = max((src - p).length, 1.0e-6)
            weighted.append((1.0 / (d ** power), wz))
        total = sum(w for w, _ in weighted)
        if total <= 1.0e-12:
            return None
        return sum(w * wz for w, wz in weighted) / total

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        lower_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
        ]
        outer_upper_ids = [
            f"ear_{side}_front_middle",
            f"ear_{side}_back_middle",
        ]
        inner_ref_ids = [
            f"ear_{side}_inner_bottom",
            f"ear_{side}_inner_front_middle",
        ]
        lower_samples, lower_points, lower_members = _samples_for_ids(lower_ids)
        outer_samples, outer_points, outer_members = _samples_for_ids(outer_upper_ids)
        inner_samples, inner_points, inner_members = _samples_for_ids(inner_ref_ids)
        if not lower_samples or not outer_samples:
            continue

        # Use inner landmarks only as height references.  They must not become
        # topology seeds, otherwise the concha-side vertices move again.
        upper_samples = list(outer_samples) + list(inner_samples)
        upper_points = list(outer_points) + list(inner_points)
        frame_points = list(lower_points) + list(outer_points)
        reference_points = list(frame_points) + list(inner_points)
        seed_members = lower_members | outer_members
        if not seed_members or not frame_points or not upper_points:
            continue

        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
            f"ear_{'r' if side == 'l' else 'l'}_lobe",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"].copy())

        # Protect the inside concha sheet: anchor members plus their direct
        # 1-ring are blocked from this outer-lobe lift.  Vertices like 1468 are
        # still reachable from the lower/back-middle side, but direct
        # inner-bottom / inner-front-middle neighbors are left untouched.
        inner_block = set(inner_members)
        for idx in list(inner_members):
            if 0 <= idx < vert_count:
                inner_block.update(adj[idx])
        blocker_members.update(inner_block)
        blocker_members.difference_update(seed_members)

        xs = [p.x for p in reference_points]
        ys = [p.y for p in reference_points]
        zs = [p.z for p in reference_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.86, 0.0075)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.85, max(zs) + margin * 0.85
        radial_limit = max(max_span * 1.80, 0.0125)

        def _same_side(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in all_anchor_members:
                return False
            if idx in blocker_members:
                return False
            src = original_positions[idx]
            if not _same_side(src):
                return False
            if src.x < min_x or src.x > max_x or src.y < min_y or src.y > max_y or src.z < min_z or src.z > max_z:
                return False
            frame_d = _min_distance_to_points(src, frame_points)
            if frame_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(src, blocker_points)
                if block_d < frame_d * 0.58:
                    return False
            return True

        region = set()
        frontier = set(seed_members)
        topo_depth = {}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region or nb in seed_members:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        if not region:
            continue

        for idx in sorted(region):
            src = original_positions[idx]
            lower_d = _min_distance_to_points(src, lower_points)
            outer_d = _min_distance_to_points(src, outer_points)
            upper_d = _min_distance_to_points(src, upper_points)
            denom = lower_d + upper_d
            if denom <= 1.0e-9:
                continue

            lower_z = _idw_z(src, lower_samples)
            upper_z = _idw_z(src, upper_samples)
            if lower_z is None or upper_z is None:
                continue

            base_t = lower_d / denom
            frame_d = _min_distance_to_points(src, frame_points)
            geo = max(0.0, 1.0 - min(1.0, frame_d / max(radial_limit, 1.0e-6)))
            topo = max(0.0, 1.0 - (float(topo_depth.get(idx, steps) - 1) / float(max(1, steps))))
            outer_bias = 0.10 if outer_d <= lower_d * 1.35 else 0.0
            upper_t = max(0.28, min(0.86, base_t + upper_blend_bias * (0.65 + 0.35 * _smoothstep01(geo)) + outer_bias))
            floor_z = lower_z * (1.0 - upper_t) + upper_z * upper_t - world_z_pad

            cur = verts[idx].co.copy()
            cur_wz = _hfr_world_z(out_obj, cur)
            lift = floor_z - cur_wz
            if lift <= 0.0:
                continue

            local_strength = strength * (0.84 + 0.16 * _smoothstep01(geo)) * (0.88 + 0.12 * topo)
            dz = min(lift * max(0.0, min(1.0, local_strength)), max_world_z_lift)
            if dz <= 0.0:
                continue
            verts[idx].co = cur + _hfr_local_delta_for_world_z(out_obj, dz)
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elulg"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_inner_lower_negative_z_slide_guard(out_obj, records, original_positions=None,
                                                slide_strength=0.60, steps=2,
                                                z_drop_threshold=0.0030,
                                                lower_height_pad=0.0022,
                                                max_slide_strength=0.60):
    """Move the inner lower-ear connector toward the same result as a 0.6
    negative-World-Z vertex slide.

    This is intentionally a topology/edge-based correction, not a raw vertical
    coordinate edit.  It targets the 1350-style connector: a non-anchor vertex
    in the lower/inner ear sheet that sits above the local front-lower/lobe/
    back-lower frame.  The chosen destination is the adjacent edge whose endpoint
    is most strongly lower in Blender World-Z; the vertex is then lerped along
    that edge by the requested 0.6 slide amount.  Direct landmark anchors and the
    accepted inner-front/upper concha fan are excluded.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    max_slide_strength = max(0.0, min(float(max_slide_strength), 1.0))
    steps = max(2, min(int(steps), 3))
    z_drop_threshold = max(0.0, float(z_drop_threshold))
    lower_height_pad = max(0.0, float(lower_height_pad))
    if slide_strength <= 0.0 or max_slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members_and_points(ids):
        members = set()
        points = []
        samples = []
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            points.append(rec["source"].copy())
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    members.add(idx)
                    samples.append((rec["source"].copy(), _hfr_world_z(out_obj, verts[idx].co)))
        return members, points, samples

    def _idw_z(src, samples, power=2.1):
        if not samples:
            return None
        weighted = []
        for p, wz in samples:
            d = max((src - p).length, 1.0e-6)
            weighted.append((1.0 / (d ** power), wz))
        total = sum(w for w, _ in weighted)
        if total <= 1.0e-12:
            return None
        return sum(w * wz for w, wz in weighted) / total

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0

        lower_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
        ]
        frame_ids = lower_ids + [
            f"ear_{side}_inner_bottom",
            f"ear_{side}_front_middle",
            f"ear_{side}_back_middle",
        ]
        seed_members, lower_points, lower_samples = _members_and_points(lower_ids)
        _, frame_points, _ = _members_and_points(frame_ids)
        if not seed_members or not lower_points or not lower_samples:
            continue

        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_inner_front_middle",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        }
        blocker_members = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blocker_members.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"].copy())

        # Keep the upper/inner-front concha fan from being re-touched.  The
        # selected 1350-style vertex remains reachable from the lower anchors,
        # but direct inner_front_middle neighbors are protected.
        inner_front_rec = rec_by_id.get(f"ear_{side}_inner_front_middle")
        if inner_front_rec is not None:
            inner_members = _record_member_indices(inner_front_rec, vert_count)
            blocker_members.update(inner_members)
            for idx in list(inner_members):
                if 0 <= idx < vert_count:
                    blocker_members.update(adj[idx])

        # The lower anchors are seeds, not candidates.  Remove them from blockers
        # in case a landmark is also listed as a nearby protected guide.
        blocker_members.difference_update(seed_members)

        if frame_points:
            xs = [p.x for p in frame_points]
            ys = [p.y for p in frame_points]
            zs = [p.z for p in frame_points]
            span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
            max_span = max(span.x, span.y, span.z, 1.0e-6)
            margin = max(max_span * 0.95, 0.0080)
            min_x, max_x = min(xs) - margin, max(xs) + margin
            min_y, max_y = min(ys) - margin, max(ys) + margin
            min_z, max_z = min(zs) - margin, max(zs) + margin
            radial_limit = max(max_span * 1.65, 0.0130)
        else:
            min_x = min_y = min_z = -1.0e12
            max_x = max_y = max_z = 1.0e12
            radial_limit = 0.020

        def _same_side(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        def _candidate(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in all_anchor_members:
                return False
            if idx in blocker_members:
                return False
            src = original_positions[idx]
            if not _same_side(src):
                return False
            if src.x < min_x or src.x > max_x or src.y < min_y or src.y > max_y or src.z < min_z or src.z > max_z:
                return False
            frame_d = _min_distance_to_points(src, frame_points) if frame_points else 0.0
            if frame_d > radial_limit:
                return False
            if blocker_points:
                block_d = _min_distance_to_points(src, blocker_points)
                if block_d < frame_d * 0.62:
                    return False
            return True

        region = set()
        frontier = set(seed_members)
        topo_depth = {}
        for depth in range(1, steps + 1):
            nxt = set()
            for vidx in frontier:
                if vidx < 0 or vidx >= vert_count:
                    continue
                for nb in adj[vidx]:
                    if nb in region or nb in seed_members:
                        continue
                    if not _candidate(nb):
                        continue
                    region.add(nb)
                    topo_depth[nb] = depth
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt

        if not region:
            continue

        for idx in sorted(region):
            # Only the second ring is treated as the "inside connector" problem.
            # First-ring lower connectors were tuned in earlier passes and should
            # not be pulled down again.
            if int(topo_depth.get(idx, 1)) < 2:
                continue

            cur = verts[idx].co.copy()
            cur_wz = _hfr_world_z(out_obj, cur)
            lower_z = _idw_z(original_positions[idx], lower_samples)
            if lower_z is None:
                continue
            if cur_wz <= lower_z + lower_height_pad:
                continue

            best_nb = None
            best_drop = 0.0
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in all_anchor_members:
                    continue
                if nb in blocker_members:
                    continue
                nb_wz = _hfr_world_z(out_obj, verts[nb].co)
                drop = cur_wz - nb_wz
                if drop > best_drop:
                    best_drop = drop
                    best_nb = nb

            if best_nb is None or best_drop < z_drop_threshold:
                continue

            # Strongly matching 1350-style cases should land at the requested
            # 0.6 slide target.  We slightly soften only marginal cases so the
            # neighboring lower sheet does not get dragged down accidentally.
            denom = max(z_drop_threshold * 2.5, 1.0e-6)
            match = max(0.0, min(1.0, (best_drop - z_drop_threshold) / denom))
            local_slide = slide_strength * (0.72 + 0.28 * _smoothstep01(match))
            if best_drop >= z_drop_threshold * 3.0:
                local_slide = slide_strength
            local_slide = min(local_slide, max_slide_strength)
            if local_slide <= 0.0:
                continue

            target = cur.lerp(verts[best_nb].co, local_slide)
            target_wz = _hfr_world_z(out_obj, target)
            if target_wz >= cur_wz:
                continue
            verts[idx].co = target
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eilzsg"] = int(changed)
    except Exception:
        pass
    return changed



def apply_ear_lower_front_first_ring_z_slide_guard(out_obj, records, original_positions=None,
                                                   slide_strength=0.35,
                                                   z_drop_threshold=0.0018,
                                                   min_lower_anchor_links=2):
    """Tune the lower-front/lobe first-ring connector toward a 0.35
    negative-World-Z vertex-slide-equivalent target.

    This pass is for the 1128-style vertex: a non-anchor first-ring connector
    shared by LM_ear_*_front_lower and LM_ear_*_lobe.  Earlier passes tune the
    second-ring inner connector.  Here we intentionally keep the region narrower
    so the accepted inner concha sheet is not re-touched.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    z_drop_threshold = max(0.0, float(z_drop_threshold))
    min_lower_anchor_links = max(1, int(min_lower_anchor_links))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        points = []
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            points.append(rec["source"].copy())
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}, points

    changed = 0
    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        lower_ids = [
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_lower",
        ]
        lower_members, lower_points = _members(lower_ids)
        if len(lower_members) < 2 or not lower_points:
            continue

        frame_ids = lower_ids + [
            f"ear_{side}_front_middle",
            f"ear_{side}_inner_bottom",
        ]
        _, frame_points = _members(frame_ids)
        if not frame_points:
            frame_points = lower_points
        xs = [p.x for p in frame_points]
        ys = [p.y for p in frame_points]
        zs = [p.z for p in frame_points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.90, 0.0075)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin, max(zs) + margin

        protected_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_inner_front_middle",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        }
        protected_members = set()
        for lm_id in protected_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            protected_members.update(_record_member_indices(rec, vert_count))
        # Do not let this narrow lower-front adjustment leak into the accepted
        # inner-front concha fan.
        inner_front_rec = rec_by_id.get(f"ear_{side}_inner_front_middle")
        if inner_front_rec is not None:
            for idx in _record_member_indices(inner_front_rec, vert_count):
                if 0 <= idx < vert_count:
                    protected_members.add(idx)
                    protected_members.update(adj[idx])
        protected_members.difference_update(lower_members)

        def _same_side(co):
            return side_sign * _hfr_world_x(out_obj, co) > -0.002

        for idx in range(vert_count):
            if idx in all_anchor_members:
                continue
            if idx in protected_members:
                continue
            src = original_positions[idx]
            if not _same_side(src):
                continue
            if src.x < min_x or src.x > max_x or src.y < min_y or src.y > max_y or src.z < min_z or src.z > max_z:
                continue

            linked_lower = [nb for nb in adj[idx] if nb in lower_members]
            if len(linked_lower) < min_lower_anchor_links:
                continue

            # The 1128-style connector sits between front_lower and lobe.  Requiring
            # front_lower+lobe keeps this pass off the outer/back-lower sheet that
            # was already tuned by the nape blend guard.
            front_lower_members, _ = _members([f"ear_{side}_front_lower"])
            lobe_members, _ = _members([f"ear_{side}_lobe"])
            if not any(nb in front_lower_members for nb in linked_lower):
                continue
            if not any(nb in lobe_members for nb in linked_lower):
                continue

            cur = verts[idx].co.copy()
            cur_wz = _hfr_world_z(out_obj, cur)
            best_nb = None
            best_score = 0.0
            best_drop = 0.0
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in protected_members:
                    continue
                # Lower anchors are allowed as slide targets; they are endpoints,
                # not edited vertices.  Other landmark anchors are excluded.
                if nb in all_anchor_members and nb not in lower_members:
                    continue
                nb_co = verts[nb].co
                nb_wz = _hfr_world_z(out_obj, nb_co)
                drop = cur_wz - nb_wz
                if drop <= z_drop_threshold:
                    continue
                edge_len = max((nb_co - cur).length, 1.0e-8)
                score = drop / edge_len
                if score > best_score:
                    best_score = score
                    best_drop = drop
                    best_nb = nb

            if best_nb is None:
                continue

            # Match the user's requested slide-equivalent ratio.  The selected
            # vertex should not be raw-Z translated; it follows the best existing
            # negative-Z edge by 0.35 of that edge.
            target = cur.lerp(verts[best_nb].co, slide_strength)
            if _hfr_world_z(out_obj, target) >= cur_wz:
                continue
            verts[idx].co = target
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_elfzsg"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_lobe_upper_z_slide_guard(out_obj, records, original_positions=None,
                                       slide_strength=0.50,
                                       z_drop_threshold=0.0011):
    """Slide the upper-lobe bridge downward along the existing topology.

    The user's intended edit for the selected upper-lobe bridge pair is a
    negative World-Z slide, not an upward lift.  The target topology is the
    stable sheet:

        ear_back_lower anchor -> bridge A -> bridge B -> ear_lobe anchor

    Bridge A is slid toward the adjacent back_lower anchor and bridge B is slid
    toward the adjacent lobe anchor.  Anchors and accepted inner/front-lower
    vertices are used only as read-only slide endpoints; they are never edited.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    z_drop_threshold = max(0.0, float(z_drop_threshold))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _path_lerp(points, t):
        if not points:
            return None
        if len(points) == 1:
            return points[0].copy()
        seg_lengths = []
        total = 0.0
        for i in range(len(points) - 1):
            length = (points[i + 1] - points[i]).length
            seg_lengths.append(length)
            total += length
        if total <= 1.0e-12:
            return points[-1].copy()
        remain = max(0.0, min(1.0, float(t))) * total
        for i, length in enumerate(seg_lengths):
            if remain <= length or i == len(seg_lengths) - 1:
                if length <= 1.0e-12:
                    return points[i + 1].copy()
                return points[i].lerp(points[i + 1], remain / length)
            remain -= length
        return points[-1].copy()

    changed = 0
    current_snapshot = [v.co.copy() for v in verts]

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        back_lower_members = _members([f"ear_{side}_back_lower"])
        lobe_members = _members([f"ear_{side}_lobe"])
        if not back_lower_members or not lobe_members:
            continue

        front_lower_members = _members([f"ear_{side}_front_lower"])
        inner_members = _members([
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
        ])

        # Accepted inner/front-lower vertices remain fixed.  These vertices may
        # still be read-only slide endpoints only when they are lower/lobe frame
        # anchors; otherwise they are not edited or selected as intermediate
        # editable bridge vertices.
        frozen_edit = set(front_lower_members) | set(inner_members)
        for idx in list(front_lower_members) + list(inner_members):
            if 0 <= idx < vert_count:
                frozen_edit.update(adj[idx])

        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        frozen_edit.update(hard_blockers)

        def _same_side(idx):
            if idx < 0 or idx >= vert_count:
                return False
            try:
                return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
            except Exception:
                return True

        # Structural search for:
        #   back_lower anchor -> bridge A -> bridge B -> lobe anchor
        bridge_pairs = set()
        for back_idx in sorted(back_lower_members):
            if back_idx < 0 or back_idx >= vert_count:
                continue
            for a in adj[back_idx]:
                if a in all_anchor_members or not _same_side(a):
                    continue
                for b in adj[a]:
                    if b == back_idx:
                        continue
                    if b in all_anchor_members or not _same_side(b):
                        continue
                    if any(lobe_idx in adj[b] for lobe_idx in lobe_members):
                        bridge_pairs.add((a, b))

        if not bridge_pairs:
            continue

        # Each bridge endpoint gets its own lower endpoint.  This avoids the
        # previous upward target-selection mistake: A goes toward back_lower,
        # B goes toward lobe.
        target_hints = {}
        for a, b in bridge_pairs:
            target_hints.setdefault(a, set()).update(back_lower_members)
            target_hints.setdefault(b, set()).update(lobe_members)

        for idx in sorted(target_hints.keys()):
            if idx in all_anchor_members or idx in frozen_edit:
                continue
            if not _same_side(idx):
                continue

            cur = current_snapshot[idx]
            cur_wz = _hfr_world_z(out_obj, cur)
            best_path = None
            best_drop = 0.0
            best_len = 0.0

            # Prefer the intended direct slide edge to the lower anchor endpoint.
            for target_idx in sorted(target_hints.get(idx, set())):
                if target_idx not in adj[idx]:
                    continue
                if target_idx < 0 or target_idx >= vert_count:
                    continue
                target_co = current_snapshot[target_idx]
                target_wz = _hfr_world_z(out_obj, target_co)
                drop = cur_wz - target_wz
                if drop <= z_drop_threshold:
                    continue
                length = (target_co - cur).length
                if (drop > best_drop + 1.0e-10) or (abs(drop - best_drop) <= 1.0e-10 and length > best_len):
                    best_path = [idx, target_idx]
                    best_drop = drop
                    best_len = length

            # Fallback: if a mirrored/template variant lacks a direct anchor edge,
            # search a short downward path, but still reject upward endpoints.
            if best_path is None:
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count:
                        continue
                    if nb in all_anchor_members and nb not in back_lower_members and nb not in lobe_members:
                        continue
                    if nb in hard_blockers:
                        continue
                    if not _same_side(nb):
                        continue
                    nb_wz = _hfr_world_z(out_obj, current_snapshot[nb])
                    path_options = [[idx, nb]]
                    for nb2 in adj[nb]:
                        if nb2 < 0 or nb2 >= vert_count or nb2 == idx:
                            continue
                        if nb2 in all_anchor_members and nb2 not in back_lower_members and nb2 not in lobe_members:
                            continue
                        if nb2 in hard_blockers:
                            continue
                        if not _same_side(nb2):
                            continue
                        nb2_wz = _hfr_world_z(out_obj, current_snapshot[nb2])
                        if nb2_wz < nb_wz - (z_drop_threshold * 0.25):
                            path_options.append([idx, nb, nb2])
                    for path in path_options:
                        end_co = current_snapshot[path[-1]]
                        drop = cur_wz - _hfr_world_z(out_obj, end_co)
                        if drop <= z_drop_threshold:
                            continue
                        length = 0.0
                        for pi in range(len(path) - 1):
                            length += (current_snapshot[path[pi + 1]] - current_snapshot[path[pi]]).length
                        if (drop > best_drop + 1.0e-10) or (abs(drop - best_drop) <= 1.0e-10 and length > best_len):
                            best_path = path
                            best_drop = drop
                            best_len = length

            if best_path is None:
                continue

            points = [current_snapshot[p] for p in best_path]
            target = _path_lerp(points, slide_strength)
            if target is None:
                continue
            if _hfr_world_z(out_obj, target) >= cur_wz - (z_drop_threshold * 0.25):
                continue

            verts[idx].co = target
            changed += 1

    if changed:
        out_obj.data.update()
    try:
        out_obj["HFR_eluzsg"] = int(changed)
    except Exception:
        pass
    return changed

def apply_ear_lobe_upper_to_lobe_slide_guard(out_obj, records, original_positions=None,
                                             slide_strength=0.46,
                                             z_rise_threshold=0.0025):
    """Slide the next upper-lobe connector toward the lobe anchor.

    The intended selected vertex is the upper connector directly above the
    lowered lobe bridge.  On the current template this is the topology pattern:

        ear_lobe anchor -> lower lobe bridge -> upper lobe connector

    The upper connector is moved along that existing 2-edge path toward the
    lobe anchor by the requested slide-equivalent amount.  Lower anchors and
    accepted front/inner ear vertices are read-only endpoints only.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    z_rise_threshold = max(0.0, float(z_rise_threshold))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _path_lerp(points, t):
        if not points:
            return None
        if len(points) == 1:
            return points[0].copy()
        seg_lengths = []
        total = 0.0
        for i in range(len(points) - 1):
            length = (points[i + 1] - points[i]).length
            seg_lengths.append(length)
            total += length
        if total <= 1.0e-12:
            return points[-1].copy()
        remain = max(0.0, min(1.0, float(t))) * total
        for i, length in enumerate(seg_lengths):
            if remain <= length or i == len(seg_lengths) - 1:
                if length <= 1.0e-12:
                    return points[i + 1].copy()
                return points[i].lerp(points[i + 1], remain / length)
            remain -= length
        return points[-1].copy()

    changed = 0

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        lobe_members = _members([f"ear_{side}_lobe"])
        if not lobe_members:
            continue

        lower_frame_members = _members([
            f"ear_{side}_front_lower",
            f"ear_{side}_back_lower",
        ])
        inner_members = _members([
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
        ])
        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])

        # Preserve the already accepted inner/front-lower area as editable
        # geometry.  Those vertices may still act as fixed path endpoints only.
        protected_edit = set(inner_members)
        for idx in list(inner_members):
            if 0 <= idx < vert_count:
                protected_edit.update(adj[idx])
        protected_edit.update(hard_blockers)

        def _same_side(idx):
            if idx < 0 or idx >= vert_count:
                return False
            try:
                return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
            except Exception:
                return True

        # Candidate pattern:
        #   lobe anchor -> lower bridge -> upper connector
        # The selected 1457-style vertex is the upper connector.  We exclude
        # lower bridge vertices that touch the back/front lower anchors, so the
        # previous 1444/1445 correction remains in place and this only adjusts
        # the row immediately above it.
        target_paths = {}
        for lobe_idx in sorted(lobe_members):
            if lobe_idx < 0 or lobe_idx >= vert_count:
                continue
            for bridge_idx in adj[lobe_idx]:
                if bridge_idx < 0 or bridge_idx >= vert_count:
                    continue
                if bridge_idx in all_anchor_members or not _same_side(bridge_idx):
                    continue
                # Keep the accepted lower/front-lower and inner-lower bridge
                # from becoming a source path for this upper-lobe adjustment.
                # The intended bridge is the lobe-side connector, not the
                # front/back lower frame itself.
                if any(anchor_idx in adj[bridge_idx] for anchor_idx in lower_frame_members):
                    continue
                bridge_wz = _hfr_world_z(out_obj, current_snapshot[bridge_idx])
                for upper_idx in adj[bridge_idx]:
                    if upper_idx == lobe_idx:
                        continue
                    if upper_idx < 0 or upper_idx >= vert_count:
                        continue
                    if upper_idx in all_anchor_members or upper_idx in protected_edit:
                        continue
                    if not _same_side(upper_idx):
                        continue
                    # Do not re-edit the bridge endpoints themselves or lower
                    # frame direct connectors.  This keeps 1444/1445 and the
                    # accepted front-lower/lobe frame stable.
                    if any(anchor_idx in adj[upper_idx] for anchor_idx in lower_frame_members):
                        continue
                    if any(anchor_idx in adj[upper_idx] for anchor_idx in lobe_members):
                        continue
                    upper_wz = _hfr_world_z(out_obj, current_snapshot[upper_idx])
                    if upper_wz <= bridge_wz + z_rise_threshold:
                        continue
                    path = [upper_idx, bridge_idx, lobe_idx]
                    # When more than one lobe member is available, choose the
                    # path with the strongest downward/lobeward effect.
                    drop = upper_wz - _hfr_world_z(out_obj, current_snapshot[lobe_idx])
                    length = ((current_snapshot[upper_idx] - current_snapshot[bridge_idx]).length +
                              (current_snapshot[bridge_idx] - current_snapshot[lobe_idx]).length)
                    old = target_paths.get(upper_idx)
                    if old is None or drop > old[0] + 1.0e-10 or (abs(drop - old[0]) <= 1.0e-10 and length > old[1]):
                        target_paths[upper_idx] = (drop, length, path)

        for idx, (_, _, path) in sorted(target_paths.items()):
            cur_wz = _hfr_world_z(out_obj, current_snapshot[idx])
            points = [current_snapshot[p] for p in path]
            target = _path_lerp(points, slide_strength)
            if target is None:
                continue
            # This requested move must go toward the lobe/downward frame, not
            # toward the upper support row.
            if _hfr_world_z(out_obj, target) >= cur_wz:
                continue
            verts[idx].co = target
            changed += 1

    if changed:
        out_obj.data.update()
    return changed


def apply_ear_front_inner_bridge_slide_guard(out_obj, records, original_positions=None,
                                            slide_strength=0.53,
                                            max_front_dist=3,
                                            outward_x_pad=0.00035):
    """Slide the 1450/1452-style front-inner ear bridge toward the
    LM_ear_*_front_middle -> LM_ear_*_front_lower rail.

    The requested edit is a negative vertex-slide-equivalent on the selected
    edge just behind the front lower/middle rail.  In the current template this
    is detected as a parallel edge behind a front rail edge:

        front rail U -- front rail V
              |              |
        bridge A ------ bridge B

    For the left diagnostic, U/V are the 1438/1440-style rail vertices and
    A/B are the selected 1450/1452-style bridge vertices.  A and B are moved
    toward U and V by the requested slide amount.  Front/inner/lobe anchors are
    read-only endpoints and are never edited by this pass.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    max_front_dist = max(1, int(max_front_dist))
    outward_x_pad = max(0.0, float(outward_x_pad))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _edge_pairs():
        for i, nbs in enumerate(adj):
            if i < 0 or i >= vert_count:
                continue
            for j in nbs:
                if i < j < vert_count:
                    yield i, j

    def _dist_from(starts, blocked, same_side_fn):
        dist = {}
        queue = []
        for s in starts:
            if 0 <= s < vert_count and same_side_fn(s):
                dist[s] = 0
                queue.append(s)
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            d = dist[cur]
            if d >= max_front_dist:
                continue
            for nb in adj[cur]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in dist or nb in blocked:
                    continue
                if not same_side_fn(nb):
                    continue
                dist[nb] = d + 1
                queue.append(nb)
        return dist

    changed = 0
    target_map = {}

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        front_middle_members = _members([f"ear_{side}_front_middle"])
        front_lower_members = _members([f"ear_{side}_front_lower"])
        if not front_middle_members or not front_lower_members:
            continue

        inner_bottom_members = _members([f"ear_{side}_inner_bottom"])
        inner_front_members = _members([f"ear_{side}_inner_front_middle"])
        lobe_members = _members([f"ear_{side}_lobe"])
        back_lower_members = _members([f"ear_{side}_back_lower"])
        lower_frame_members = front_lower_members | lobe_members | back_lower_members

        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        # These are editable blockers, not path endpoints.  We deliberately do
        # not block the one-ring around inner_bottom because the 1452-style
        # bridge vertex touches inner_bottom and is the intended target.
        edit_blocked = set(hard_blockers) | set(front_middle_members) | set(front_lower_members) | set(inner_bottom_members) | set(inner_front_members) | set(lobe_members) | set(back_lower_members)

        def _same_side(idx):
            if idx < 0 or idx >= vert_count:
                return False
            try:
                return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
            except Exception:
                return True

        # Front rail search is constrained to the middle/lower front area and
        # kept away from the lobe/back/inner pocket frame so the accepted lobe
        # and inner concha corrections remain fixed.
        rail_blocked = set(hard_blockers) | set(inner_bottom_members) | set(inner_front_members) | set(lobe_members) | set(back_lower_members)
        dist_mid = _dist_from(front_middle_members, rail_blocked, _same_side)
        dist_low = _dist_from(front_lower_members, rail_blocked, _same_side)
        front_dist = {i: min(dist_mid.get(i, 999), dist_low.get(i, 999)) for i in set(dist_mid) | set(dist_low)}

        def _is_front_rail_vertex(idx):
            if idx < 0 or idx >= vert_count:
                return False
            if idx in rail_blocked:
                return False
            if front_dist.get(idx, 999) > max_front_dist:
                return False
            # Exclude rail candidates that directly lead into lobe/back-lower.
            if any(nb in lower_frame_members for nb in adj[idx] if nb not in front_lower_members):
                return False
            return _same_side(idx)

        for u, v in _edge_pairs():
            if not _is_front_rail_vertex(u) or not _is_front_rail_vertex(v):
                continue
            # At least one endpoint should be close to the front-lower side;
            # this avoids taking the upper front-middle parallel row when the
            # intended edit is the lower/middle selected bridge edge.
            if min(dist_low.get(u, 999), dist_low.get(v, 999)) > 2:
                continue
            for a in adj[u]:
                if a == v or a < 0 or a >= vert_count:
                    continue
                if a in all_anchor_members or a in edit_blocked or not _same_side(a):
                    continue
                # The bridge sits deeper/outward from the front rail on the
                # side axis.  This keeps the slide on the selected inner bridge,
                # not on the rail itself.
                if side_sign * (_hfr_world_x(out_obj, current_snapshot[a]) - _hfr_world_x(out_obj, current_snapshot[u])) <= outward_x_pad:
                    continue
                for b in adj[v]:
                    if b == u or b == a or b < 0 or b >= vert_count:
                        continue
                    if b not in adj[a]:
                        continue
                    if b in all_anchor_members or b in edit_blocked or not _same_side(b):
                        continue
                    if side_sign * (_hfr_world_x(out_obj, current_snapshot[b]) - _hfr_world_x(out_obj, current_snapshot[v])) <= outward_x_pad:
                        continue
                    # The requested pair is the lower front-inner bridge; one
                    # of the two bridge vertices touches the inner-bottom row.
                    # This excludes the immediately upper parallel edge while
                    # still handling left/right and mirrored topology.
                    if not (any(nb in inner_bottom_members for nb in adj[a]) or any(nb in inner_bottom_members for nb in adj[b])):
                        continue
                    score = 10.0
                    score += max(0.0, 4.0 - min(dist_low.get(u, 999), dist_low.get(v, 999)))
                    score += side_sign * (_hfr_world_x(out_obj, current_snapshot[a]) - _hfr_world_x(out_obj, current_snapshot[u]))
                    score += side_sign * (_hfr_world_x(out_obj, current_snapshot[b]) - _hfr_world_x(out_obj, current_snapshot[v]))
                    for idx, rail_idx in ((a, u), (b, v)):
                        old = target_map.get(idx)
                        if old is None or score > old[0]:
                            target_map[idx] = (score, rail_idx)

    for idx, (_, rail_idx) in sorted(target_map.items()):
        if idx < 0 or idx >= vert_count or rail_idx < 0 or rail_idx >= vert_count:
            continue
        cur = verts[idx].co.copy()
        target = cur.lerp(current_snapshot[rail_idx], slide_strength)
        verts[idx].co = target
        changed += 1

    if changed:
        out_obj.data.update()
    return changed


def apply_ear_inner_bottom_opposite_slide_guard(out_obj, records, original_positions=None,
                                                slide_strength=0.42,
                                                z_drop_threshold=0.0012,
                                                alignment_threshold=0.45):
    """Slide the 1462-style inner-bottom adjacent vertex in the direction
    opposite to the LM_ear_*_inner_bottom -> LM_ear_*_inner_front_middle rail.

    The selected diagnostic vertex sits directly next to inner_bottom and on the
    row opposite the inner-front-middle direction.  For the current topology the
    stable slide edge is:

        selected connector -> LM_ear_*_inner_bottom

    The inner_bottom anchor is used only as a read-only slide endpoint.  The
    anchor itself and the accepted front/lobe/lower frame remain fixed.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    verts = out_obj.data.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    z_drop_threshold = max(0.0, float(z_drop_threshold))
    alignment_threshold = max(-1.0, min(float(alignment_threshold), 1.0))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _world_vec(idx_a, idx_b):
        try:
            return ((out_obj.matrix_world @ current_snapshot[idx_b]) -
                    (out_obj.matrix_world @ current_snapshot[idx_a]))
        except Exception:
            return current_snapshot[idx_b] - current_snapshot[idx_a]

    changed = 0
    target_map = {}

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        inner_bottom_members = _members([f"ear_{side}_inner_bottom"])
        inner_front_members = _members([f"ear_{side}_inner_front_middle"])
        if not inner_bottom_members or not inner_front_members:
            continue

        front_middle_members = _members([f"ear_{side}_front_middle"])
        front_lower_members = _members([f"ear_{side}_front_lower"])
        lobe_members = _members([f"ear_{side}_lobe"])
        back_lower_members = _members([f"ear_{side}_back_lower"])

        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        anchor_blocked = (set(inner_bottom_members) | set(inner_front_members) |
                          set(front_middle_members) | set(front_lower_members) |
                          set(lobe_members) | set(back_lower_members) |
                          set(hard_blockers))

        def _same_side(idx):
            if idx < 0 or idx >= vert_count:
                return False
            try:
                return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
            except Exception:
                return True

        # Use the shortest inner_bottom -> inner_front_middle vector as the
        # rail direction.  The requested move is the opposite of that rail, so a
        # candidate is accepted only when its slide edge toward inner_bottom is
        # aligned with inner_front_middle -> inner_bottom and also moves downward
        # in Blender World-Z.  This isolates the 1462-style vertex and excludes
        # the already accepted 1452/1450 front bridge.
        rail_vectors = []
        for bottom_idx in sorted(inner_bottom_members):
            if bottom_idx < 0 or bottom_idx >= vert_count:
                continue
            for front_idx in sorted(inner_front_members):
                if front_idx < 0 or front_idx >= vert_count:
                    continue
                vec = _world_vec(front_idx, bottom_idx)  # opposite of bottom -> front
                length = vec.length
                if length <= 1.0e-12:
                    continue
                rail_vectors.append((length, vec.normalized()))
        if not rail_vectors:
            continue
        rail_vectors.sort(key=lambda item: item[0])
        opposite_dir = rail_vectors[0][1]

        for bottom_idx in sorted(inner_bottom_members):
            if bottom_idx < 0 or bottom_idx >= vert_count:
                continue
            bottom_wz = _hfr_world_z(out_obj, current_snapshot[bottom_idx])
            for idx in sorted(adj[bottom_idx]):
                if idx < 0 or idx >= vert_count:
                    continue
                if idx in all_anchor_members or idx in anchor_blocked:
                    continue
                if not _same_side(idx):
                    continue
                cur_wz = _hfr_world_z(out_obj, current_snapshot[idx])
                drop = cur_wz - bottom_wz
                if drop <= z_drop_threshold:
                    continue
                slide_vec = _world_vec(idx, bottom_idx)
                slide_len = slide_vec.length
                if slide_len <= 1.0e-12:
                    continue
                align = slide_vec.normalized().dot(opposite_dir)
                if align < alignment_threshold:
                    continue
                # Avoid re-editing the front-rail bridge and lobe/back-lower
                # frame: the intended vertex has a clean downward edge toward
                # inner_bottom and is not directly tied to front_lower/lobe/back.
                if any(nb in front_lower_members for nb in adj[idx]):
                    continue
                if any(nb in lobe_members for nb in adj[idx]):
                    continue
                if any(nb in back_lower_members for nb in adj[idx]):
                    continue
                score = align * 10.0 + drop + slide_len
                old = target_map.get(idx)
                if old is None or score > old[0]:
                    target_map[idx] = (score, bottom_idx)

    for idx, (_, bottom_idx) in sorted(target_map.items()):
        if idx < 0 or idx >= vert_count or bottom_idx < 0 or bottom_idx >= vert_count:
            continue
        cur = current_snapshot[idx]
        target = cur.lerp(current_snapshot[bottom_idx], slide_strength)
        # The requested direction is the opposite/downward side of the inner
        # bottom/front-middle rail, so do not accept accidental upward results.
        if _hfr_world_z(out_obj, target) >= _hfr_world_z(out_obj, cur) - (z_drop_threshold * 0.25):
            continue
        verts[idx].co = target
        changed += 1

    if changed:
        out_obj.data.update()
    return changed



def apply_ear_inner_pocket_face_inward_guard(out_obj, records, original_positions=None,
                                             slide_strength=0.22,
                                             bottom_fan_steps=1,
                                             min_anchor_links=2):
    """Pull the selected 1478/1481/1485/1486-style inner pocket face inward.

    The diagnostic face is the non-anchor quad immediately behind the
    inner-front-middle / inner-bottom pocket rail.  This pass detects that
    topology instead of using fixed vertex indices:

        inner_front_middle anchor -- upper pocket fan
                  |                    |
             selected quad vertices around the inner pocket
                  |                    |
        inner_bottom one-ring / lower inner fan

    Each detected quad vertex is moved a small percentage toward the local
    inner_bottom -> inner_front_middle rail.  The anchors themselves are used
    only as read-only endpoints and are not edited.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    mesh = out_obj.data
    verts = mesh.vertices
    if not verts or not mesh.polygons:
        return 0
    vert_count = len(verts)
    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    bottom_fan_steps = max(1, min(int(bottom_fan_steps), 3))
    min_anchor_links = max(1, int(min_anchor_links))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _same_side(side_sign, idx):
        if idx < 0 or idx >= vert_count:
            return False
        try:
            return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
        except Exception:
            return True

    def _closest_point_on_segment(pt, a, b):
        ab = b - a
        denom = ab.length_squared
        if denom <= 1.0e-14:
            return a.copy()
        t = (pt - a).dot(ab) / denom
        t = max(0.0, min(1.0, float(t)))
        return a + ab * t

    def _best_inner_rail_for_point(pt, bottoms, fronts):
        best = None
        for bidx in bottoms:
            if bidx < 0 or bidx >= vert_count:
                continue
            bco = current_snapshot[bidx]
            for fidx in fronts:
                if fidx < 0 or fidx >= vert_count:
                    continue
                fco = current_snapshot[fidx]
                proj = _closest_point_on_segment(pt, bco, fco)
                score = (pt - proj).length_squared
                if best is None or score < best[0]:
                    best = (score, proj)
        return None if best is None else best[1]

    changed_indices = set()

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        inner_front_members = _members([f"ear_{side}_inner_front_middle"])
        inner_bottom_members = _members([f"ear_{side}_inner_bottom"])
        if not inner_front_members or not inner_bottom_members:
            continue

        front_lower_members = _members([f"ear_{side}_front_lower"])
        front_middle_members = _members([f"ear_{side}_front_middle"])
        lobe_members = _members([f"ear_{side}_lobe"])
        back_lower_members = _members([f"ear_{side}_back_lower"])
        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        protected = (set(inner_front_members) | set(inner_bottom_members) |
                     set(front_lower_members) | set(front_middle_members) |
                     set(lobe_members) | set(back_lower_members) |
                     set(hard_blockers))

        bottom_fan = set(inner_bottom_members)
        frontier = set(inner_bottom_members)
        for _ in range(bottom_fan_steps):
            nxt = set()
            for idx in frontier:
                if idx < 0 or idx >= vert_count:
                    continue
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count or nb in bottom_fan:
                        continue
                    if not _same_side(side_sign, nb):
                        continue
                    if nb in hard_blockers or nb in front_lower_members or nb in lobe_members or nb in back_lower_members:
                        continue
                    bottom_fan.add(nb)
                    nxt.add(nb)
            frontier = nxt

        face_targets = set()
        for poly in mesh.polygons:
            face = list(poly.vertices)
            if len(face) != 4:
                continue
            if any((idx < 0 or idx >= vert_count) for idx in face):
                continue
            if any(idx in all_anchor_members for idx in face):
                continue
            if any(idx in protected for idx in face):
                continue
            if not all(_same_side(side_sign, idx) for idx in face):
                continue

            front_links = sum(1 for idx in face if any(nb in inner_front_members for nb in adj[idx]))
            bottom_links = sum(1 for idx in face if any(nb in bottom_fan for nb in adj[idx]))
            if front_links < 1 or bottom_links < 1:
                continue
            if front_links + bottom_links < min_anchor_links:
                continue

            # Keep the edit inside the inner pocket.  The intended selected
            # quad has both a front-middle side and a lower inner-bottom side,
            # while avoiding front/lobe/back frame faces.
            if any(any(nb in front_lower_members for nb in adj[idx]) for idx in face):
                continue
            if any(any(nb in lobe_members for nb in adj[idx]) for idx in face):
                continue
            if any(any(nb in back_lower_members for nb in adj[idx]) for idx in face):
                continue

            face_targets.update(face)

        for idx in face_targets:
            if idx < 0 or idx >= vert_count:
                continue
            if idx in all_anchor_members or idx in protected:
                continue
            cur = current_snapshot[idx]
            rail_target = _best_inner_rail_for_point(cur, inner_bottom_members, inner_front_members)
            if rail_target is None:
                continue
            target = cur.lerp(rail_target, slide_strength)
            # Keep this as a small inward pocket pull, not a vertical collapse.
            verts[idx].co = target
            changed_indices.add(idx)

    changed = len(changed_indices)
    if changed:
        mesh.update()
    try:
        out_obj["HFR_eipfg"] = int(changed)
    except Exception:
        pass
    return changed



def apply_ear_inner_pocket_outward_relief_guard(out_obj, records, original_positions=None,
                                               outward_strength=0.34,
                                               pocket_steps=2,
                                               include_inner_front_anchor=True):
    """Move the inner pocket cap slightly outward from the head center.

    v0.5.34 tried to pull the 1478/1481/1485/1486-style quad toward the
    inner-bottom / inner-front-middle rail.  On the actual generated mesh this
    was often visually cancelled by the anchor-connected pocket frame.  The
    user's manual test showed that moving the corresponding inner-front landmark
    outward prevents the fold, so this pass emulates that result on the output
    mesh: the inner-front-middle anchor vertex and the immediately supported
    non-anchor inner pocket cap are shifted outward by a proportional amount.

    The amount is not an absolute world distance.  It is derived from the local
    ear landmark spacing between inner_front_middle and front_middle/front_lower,
    then applied along the side's outward X direction.  Frame landmarks such as
    front_lower/lobe/back_lower/top/head/neck are protected from direct edits.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    mesh = out_obj.data
    verts = mesh.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    outward_strength = max(0.0, min(float(outward_strength), 1.0))
    pocket_steps = max(1, min(int(pocket_steps), 4))
    if outward_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _same_side(side_sign, idx):
        if idx < 0 or idx >= vert_count:
            return False
        try:
            return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
        except Exception:
            return True

    def _avg_co(indices):
        pts = [current_snapshot[i] for i in indices if 0 <= i < vert_count]
        if not pts:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        for p in pts:
            acc += p
        return acc / float(len(pts))

    # Object-local representation of world X.  The output mesh is normally only
    # scaled/rotated as its object transform, but this keeps the direction valid
    # if the user keeps the output under a transformed parent/object.
    try:
        local_x_dir = out_obj.matrix_world.inverted().to_3x3() @ Vector((1.0, 0.0, 0.0))
        if local_x_dir.length <= 1.0e-12:
            local_x_dir = Vector((1.0, 0.0, 0.0))
        else:
            local_x_dir.normalize()
    except Exception:
        local_x_dir = Vector((1.0, 0.0, 0.0))

    changed_indices = set()

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        inner_front_members = _members([f"ear_{side}_inner_front_middle"])
        inner_bottom_members = _members([f"ear_{side}_inner_bottom"])
        if not inner_front_members or not inner_bottom_members:
            continue

        front_middle_members = _members([f"ear_{side}_front_middle"])
        front_lower_members = _members([f"ear_{side}_front_lower"])
        lobe_members = _members([f"ear_{side}_lobe"])
        back_lower_members = _members([f"ear_{side}_back_lower"])
        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        protected_frame = (set(front_lower_members) | set(lobe_members) |
                           set(back_lower_members) | set(hard_blockers))

        inner_front_co = _avg_co(inner_front_members)
        if inner_front_co is None:
            continue
        ref_lengths = []
        for ref_set in (front_middle_members, front_lower_members, inner_bottom_members):
            ref_co = _avg_co(ref_set)
            if ref_co is None:
                continue
            # Use only the side-axis gap when possible; this keeps the relief
            # proportional to the local ear width instead of an absolute offset.
            side_gap = abs(float(inner_front_co.x - ref_co.x))
            full_gap = (inner_front_co - ref_co).length
            if side_gap > 1.0e-8:
                ref_lengths.append(side_gap)
            elif full_gap > 1.0e-8:
                ref_lengths.append(full_gap * 0.35)
        if not ref_lengths:
            continue
        ref_len = max(ref_lengths)
        move_vec = local_x_dir * (side_sign * ref_len * outward_strength)
        if move_vec.length <= 1.0e-12:
            continue

        # Build a small pocket support area.  This deliberately starts from the
        # inner-front anchor and inner-bottom fan so it still catches the target
        # quad even when that quad is constrained by adjacent anchor-connected
        # faces and did not pass the stricter v0.5.34 face filter.
        support = set(inner_front_members) | set(inner_bottom_members)
        frontier = set(support)
        for _ in range(pocket_steps):
            nxt = set()
            for idx in frontier:
                if idx < 0 or idx >= vert_count:
                    continue
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count or nb in support:
                        continue
                    if not _same_side(side_sign, nb):
                        continue
                    if nb in protected_frame:
                        continue
                    support.add(nb)
                    nxt.add(nb)
            frontier = nxt

        face_targets = set()
        for poly in mesh.polygons:
            face = list(poly.vertices)
            if len(face) != 4:
                continue
            if any((idx < 0 or idx >= vert_count) for idx in face):
                continue
            if not all(_same_side(side_sign, idx) for idx in face):
                continue
            if any(idx in protected_frame for idx in face):
                continue
            # Prefer the 1478/1481/1485/1486 cap: a non-anchor face with at
            # least two vertices in the local support area and at least one
            # connection to the inner-front anchor side.
            support_count = sum(1 for idx in face if idx in support)
            front_link = any(any(nb in inner_front_members for nb in adj[idx]) for idx in face)
            bottom_link = any(any(nb in inner_bottom_members or nb in support for nb in adj[idx]) for idx in face)
            if support_count >= 2 and (front_link or bottom_link):
                for idx in face:
                    if idx not in all_anchor_members and idx not in protected_frame:
                        face_targets.add(idx)

        edit_targets = set(face_targets)
        if include_inner_front_anchor:
            edit_targets.update(inner_front_members)

        # Avoid pulling the whole concha sheet: keep the edit close to the local
        # pocket by requiring proximity to the inner-front/bottom support.
        for idx in sorted(edit_targets):
            if idx < 0 or idx >= vert_count:
                continue
            if idx in protected_frame:
                continue
            if idx in all_anchor_members and idx not in inner_front_members:
                continue
            if idx not in inner_front_members:
                near_support = any(nb in support or nb in inner_front_members or nb in inner_bottom_members for nb in adj[idx])
                if not near_support:
                    continue
            cur = current_snapshot[idx]
            target = cur + move_vec
            # Accept only actual outward movement on the relevant side.
            try:
                cur_wx = _hfr_world_x(out_obj, cur)
                tar_wx = _hfr_world_x(out_obj, target)
                if side_sign * tar_wx <= side_sign * cur_wx + 1.0e-7:
                    continue
            except Exception:
                pass
            verts[idx].co = target
            changed_indices.add(idx)

    changed = len(changed_indices)
    if changed:
        mesh.update()
    try:
        out_obj["HFR_eipog"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_back_middle_spoke_slide_guard(out_obj, records, original_positions=None,
                                           slide_strength=0.60):
    """Slide the 1443-style back-middle spoke toward LM_ear_*_back_middle.

    The selected diagnostic case is a non-anchor spoke vertex directly connected
    to LM_ear_l_back_middle and to the lower-lobe bridge.  The user now requested
    this vertex to move toward LM_ear_l_back_middle by about 0.60 vertex-slide
    equivalent.  Anchor vertices are used only as slide endpoints and are not
    edited.

        LM_ear_*_back_middle anchor -> spoke vertex -> lower-lobe bridge

    This replaces the v0.5.39 support-edge direction for this spoke.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    mesh = out_obj.data
    verts = mesh.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    if original_positions is None or len(original_positions) != vert_count:
        original_positions = [v.co.copy() for v in verts]

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _same_side(side_sign, idx):
        if idx < 0 or idx >= vert_count:
            return False
        try:
            return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
        except Exception:
            return True

    changed_indices = set()

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        back_middle_members = _members([f"ear_{side}_back_middle"])
        back_lower_members = _members([f"ear_{side}_back_lower"])
        lobe_members = _members([f"ear_{side}_lobe"])
        if not back_middle_members or not back_lower_members or not lobe_members:
            continue

        # Reuse the lower-lobe bridge structure as the locator.  In the left
        # diagnostic case this finds the lower bridge endpoint adjacent to the
        # selected spoke vertex, so the edit targets the same topology position
        # without depending on a fixed vertex index.
        lower_bridge_a = set()
        for back_idx in sorted(back_lower_members):
            if back_idx < 0 or back_idx >= vert_count:
                continue
            for a in adj[back_idx]:
                if a in all_anchor_members or not _same_side(side_sign, a):
                    continue
                for b in adj[a]:
                    if b == back_idx:
                        continue
                    if b in all_anchor_members or not _same_side(side_sign, b):
                        continue
                    if any(lobe_idx in adj[b] for lobe_idx in lobe_members):
                        lower_bridge_a.add(a)
        if not lower_bridge_a:
            continue

        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_front_lower",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])

        targets = {}
        for back_mid_idx in sorted(back_middle_members):
            if back_mid_idx < 0 or back_mid_idx >= vert_count:
                continue
            if not _same_side(side_sign, back_mid_idx):
                continue
            for idx in adj[back_mid_idx]:
                if idx in all_anchor_members or idx in hard_blockers:
                    continue
                if not _same_side(side_sign, idx):
                    continue
                if not any(nb in lower_bridge_a for nb in adj[idx]):
                    continue

                cur = current_snapshot[idx]
                anchor = current_snapshot[back_mid_idx]
                edge_len = (anchor - cur).length
                if edge_len <= 1.0e-9:
                    continue

                prev = targets.get(idx)
                if prev is None or edge_len < prev[1]:
                    targets[idx] = (back_mid_idx, edge_len)

        for idx, (target_idx, _edge_len) in sorted(targets.items()):
            if idx in all_anchor_members or idx in hard_blockers:
                continue
            if target_idx < 0 or target_idx >= vert_count:
                continue
            cur = current_snapshot[idx]
            target = current_snapshot[target_idx]
            new_co = cur.lerp(target, slide_strength)
            verts[idx].co = new_co
            changed_indices.add(idx)

    changed = len(changed_indices)
    if changed:
        mesh.update()
    try:
        out_obj["HFR_ebmsg"] = int(changed)
    except Exception:
        pass
    return changed


def apply_ear_inner_bottom_away_cap_slide_guard(out_obj, records, original_positions=None,
                                                slide_strength=0.63,
                                                min_bottom_steps=2,
                                                max_bottom_steps=4,
                                                min_front_steps=3):
    """Slide the 1475-style inner pocket cap away from LM_ear_*_inner_bottom.

    The selected diagnostic vertex is a non-anchor cap vertex in the inner ear
    pocket.  It is not directly bound to LM_ear_l_inner_bottom, but it sits on
    the local fan three topology steps away from that anchor.  The requested
    motion is the vertex-slide direction opposite the inner-bottom side, so the
    target edge is chosen among adjacent same-side vertices that move farther
    from LM_ear_*_inner_bottom in both topology distance and world distance.

    Anchors are used only as references; no landmark-bound vertex is edited.
    """
    if out_obj is None or out_obj.type != 'MESH' or not records:
        return 0
    mesh = out_obj.data
    verts = mesh.vertices
    if not verts:
        return 0
    vert_count = len(verts)
    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    min_bottom_steps = max(1, int(min_bottom_steps))
    max_bottom_steps = max(min_bottom_steps, int(max_bottom_steps))
    min_front_steps = max(0, int(min_front_steps))
    if slide_strength <= 0.0:
        return 0

    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    adj = build_mesh_adjacency(out_obj)
    current_snapshot = [v.co.copy() for v in verts]

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def _members(ids):
        result = set()
        for lm_id in ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            result.update(_record_member_indices(rec, vert_count))
        return {i for i in result if 0 <= i < vert_count}

    def _same_side(side_sign, idx):
        if idx < 0 or idx >= vert_count:
            return False
        try:
            return side_sign * _hfr_world_x(out_obj, current_snapshot[idx]) > -0.002
        except Exception:
            return True

    def _world_vec(a_idx, b_idx):
        try:
            return ((out_obj.matrix_world @ current_snapshot[b_idx]) -
                    (out_obj.matrix_world @ current_snapshot[a_idx]))
        except Exception:
            return current_snapshot[b_idx] - current_snapshot[a_idx]

    def _world_distance(a_idx, b_idx):
        try:
            return ((out_obj.matrix_world @ current_snapshot[a_idx]) -
                    (out_obj.matrix_world @ current_snapshot[b_idx])).length
        except Exception:
            return (current_snapshot[a_idx] - current_snapshot[b_idx]).length

    def _topo_dist(starts, max_steps, side_sign, blocked=None):
        blocked = set(blocked or [])
        dist = {}
        queue = []
        for sidx in starts:
            if 0 <= sidx < vert_count and _same_side(side_sign, sidx):
                dist[sidx] = 0
                queue.append(sidx)
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            cur_d = dist[cur]
            if cur_d >= max_steps:
                continue
            for nb in adj[cur]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in dist or nb in blocked:
                    continue
                if not _same_side(side_sign, nb):
                    continue
                dist[nb] = cur_d + 1
                queue.append(nb)
        return dist

    changed_indices = set()

    for side in ("l", "r"):
        side_sign = -1.0 if side == "l" else 1.0
        inner_bottom_members = _members([f"ear_{side}_inner_bottom"])
        inner_front_members = _members([f"ear_{side}_inner_front_middle"])
        if not inner_bottom_members or not inner_front_members:
            continue

        hard_blockers = _members([
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_back_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_front_lower",
            f"ear_{side}_back_middle",
            f"ear_{side}_back_lower",
            f"ear_{side}_lobe",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
            f"temple_{side}_center",
            f"face_{side}_edge",
            f"jaw_{side}_edge",
            f"outer_face_{side}_upper",
            f"outer_face_{side}_lower",
            f"cheek_{side}_center",
            f"nape_{side}_outer",
            f"neck_top_{side}_side",
            f"neck_top_{side}_back",
        ])
        protected = set(all_anchor_members) | set(hard_blockers)

        bottom_dist = _topo_dist(inner_bottom_members, max_bottom_steps + 1, side_sign)
        front_dist = _topo_dist(inner_front_members, max_bottom_steps + 2, side_sign)

        target_map = {}
        for idx, bdist in sorted(bottom_dist.items()):
            if idx in protected:
                continue
            if bdist < min_bottom_steps or bdist > max_bottom_steps:
                continue
            if front_dist.get(idx, 999) < min_front_steps:
                continue
            if len(adj[idx]) < 5:
                continue
            if not _same_side(side_sign, idx):
                continue
            # The 1475/766 cap is supported by more than one inner-bottom-side
            # neighbor.  This excludes the long outer rows while keeping the
            # local non-anchor pentavalent cap vertex.
            near_bottom_links = sum(1 for nb in adj[idx]
                                    if bottom_dist.get(nb, 999) < bdist)
            if near_bottom_links < 2:
                continue

            cur_bottom_dist = min(_world_distance(idx, bidx) for bidx in inner_bottom_members)
            best = None
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in protected:
                    continue
                if not _same_side(side_sign, nb):
                    continue
                nb_bdist = bottom_dist.get(nb, 999)
                if nb_bdist <= bdist:
                    continue
                nb_bottom_dist = min(_world_distance(nb, bidx) for bidx in inner_bottom_members)
                dist_gain = nb_bottom_dist - cur_bottom_dist
                if dist_gain <= 1.0e-7:
                    continue
                edge_len = _world_vec(idx, nb).length
                if edge_len <= 1.0e-12:
                    continue
                score = dist_gain * 100.0 + nb_bdist * 4.0 + edge_len
                if best is None or score > best[0]:
                    best = (score, nb)
            if best is not None:
                target_map[idx] = best[1]

        for idx, target_idx in sorted(target_map.items()):
            if idx in protected or target_idx < 0 or target_idx >= vert_count:
                continue
            cur = current_snapshot[idx]
            target = current_snapshot[target_idx]
            new_co = cur.lerp(target, slide_strength)
            # Safety: the result must be farther from inner_bottom than before.
            old_d = min(_world_distance(idx, bidx) for bidx in inner_bottom_members)
            try:
                saved = verts[idx].co.copy()
                verts[idx].co = new_co
                new_snapshot = verts[idx].co.copy()
                new_d = min(((out_obj.matrix_world @ new_snapshot) -
                             (out_obj.matrix_world @ current_snapshot[bidx])).length
                            for bidx in inner_bottom_members)
                verts[idx].co = saved
            except Exception:
                new_d = min((new_co - current_snapshot[bidx]).length
                            for bidx in inner_bottom_members)
            if new_d <= old_d + 1.0e-7:
                continue
            verts[idx].co = new_co
            changed_indices.add(idx)

    changed = len(changed_indices)
    if changed:
        mesh.update()
    try:
        out_obj["HFR_eibacg"] = int(changed)
    except Exception:
        pass
    return changed

def deform_template_output_to_landmarks(out_obj, power=2.0, nearest_count=12, anchor_lock=1.0, anchor_iters=2,
                                       topo_propagate=True, topo_iters=36, topo_strength=0.65,
                                       guide_rails=True, guide_rail_strength=1.0, guide_rail_max_len=80,
                                       guide_rail_spread=True, guide_rail_spread_steps=1, guide_rail_spread_strength=0.65,
                                       mls_field=True, mls_strength=0.75, mls_nearest=18,
                                       guide_follow=True, guide_strength=0.55, guide_radius=1.10,
                                       nose_web_fit=True, nose_strength=1.0, nose_radius=2.0, nose_samples=24,
                                       nose_alar_fit=True, alar_strength=0.85, alar_radius=1.0, alar_samples=12,
                                       brow_ridge_fit=True, brow_strength=0.80, brow_radius=1.15, brow_samples=20, brow_smooth=0.22,
                                       brow_inner_support=True, brow_inner_strength=0.70, brow_inner_steps=2, brow_inner_radius=1.10,
                                       eye_loop_fit=True, eye_loop_strength=1.0, eye_loop_max_len=48, eye_loop_steps=96,
                                       eye_direct_fit=True, eye_direct_radius=0.90,
                                       eye_band_steps=3, eye_band_radius=1.45,
                                       feature_loops=True, loop_strength=0.85, loop_radius=1.15,
                                       ear_lobe_fit=False, ear_strength=0.75, ear_radius=1.25,
                                       ear_lobe_y_guard=True, ear_lobe_y_strength=0.85,
                                       ear_lobe_relative=True, ear_lobe_relative_strength=1.0, ear_lobe_xy_strength=1.0,
                                       ear_lower_rail=False, ear_lower_rail_strength=0.90, ear_lower_rail_radius=0.90,
                                       ear_lobe_patch=False, ear_lobe_patch_strength=0.85, ear_lobe_patch_steps=4,
                                       ear_strip_fit=False, ear_strip_strength=0.85, ear_strip_y_lock=1.0,
                                       sparse_ear_safe=True, sparse_ear_y_strength=1.0, sparse_ear_neighbor_blend=0.35,
                                       lobe_directional_stretch=True, lobe_directional_strength=1.0, lobe_directional_steps=2, lobe_directional_falloff=0.65,
                                       head_round_fit=True, head_round_strength=0.70, head_round_steps=7, head_round_iters=2, head_round_z_margin=0.30,
                                       neck_fit=True, neck_strength=0.85, neck_radius=1.20,
                                       ear_local_fit=True, ear_local_strength=0.82, ear_local_steps=4, ear_local_nearest=0,
                                       ear_lower_fit=True, ear_lower_strength=0.70, ear_lower_steps=3, ear_lower_nearest=0,
                                       output_mirror_finish=False, output_mirror_direction='L2R', output_mirror_epsilon=0.0005):
    records = anchor_records_for_template(out_obj)
    if not records:
        raise ValueError("Template has no bound HFR_A_* anchor groups")
    if bool(ear_lobe_y_guard) and not bool(lobe_directional_stretch):
        stabilize_ear_lobe_records(records, y_strength=ear_lobe_y_strength)
    if bool(ear_lobe_relative) and not bool(lobe_directional_stretch):
        solve_ear_lobe_relative_records(
            records,
            solve_strength=ear_lobe_relative_strength,
            xy_strength=ear_lobe_xy_strength,
        )
    verts = out_obj.data.vertices
    original = [v.co.copy() for v in verts]

    rail_constraints = {}
    if bool(guide_rails):
        rail_constraints = build_guide_rail_constraints(
            out_obj,
            original,
            records,
            guide_pairs=solver_guide_rail_pairs(),
            rail_strength=guide_rail_strength,
            max_path_len=guide_rail_max_len,
        )

    eye_loop_constraints = {}
    eye_member_constraints = {}
    if bool(eye_loop_fit):
        eye_loop_constraints = build_eye_boundary_path_constraints(
            out_obj,
            original,
            records,
            path_strength=eye_loop_strength,
            max_path_len=eye_loop_max_len,
        )
        eye_member_constraints = build_eye_member_path_constraints(
            out_obj,
            original,
            records,
            path_strength=eye_loop_strength,
            max_path_len=max(8, int(eye_loop_max_len)),
        )
        try:
            out_obj["HFR_eyefit"] = int(len(eye_loop_constraints))
        except Exception:
            pass
        try:
            out_obj["HFR_eyepth"] = int(len(eye_member_constraints))
        except Exception:
            pass

    fixed_constraints = {}
    if rail_constraints:
        fixed_constraints.update(rail_constraints)
    if eye_loop_constraints:
        fixed_constraints.update(eye_loop_constraints)
    if eye_member_constraints:
        fixed_constraints.update(eye_member_constraints)

    if bool(topo_propagate):
        displacements, _fixed_count = topology_propagate_displacements(
            out_obj,
            original,
            records,
            power=power,
            nearest_count=nearest_count,
            topo_iters=topo_iters,
            topo_strength=topo_strength,
            extra_fixed=fixed_constraints,
        )
    else:
        displacements = [
            _idw_delta_for_point(co, records, power=power, nearest_count=nearest_count)
            for co in original
        ]
        for idx, delta in fixed_constraints.items():
            if 0 <= idx < len(displacements):
                displacements[idx] = delta.copy()

    if bool(eye_loop_fit):
        eye_band_count = eye_topology_band_refine_displacements(
            out_obj,
            original,
            displacements,
            records,
            eye_strength=eye_loop_strength,
            band_steps=eye_band_steps,
            band_radius=eye_band_radius,
        )
        try:
            out_obj["HFR_eyeband"] = int(eye_band_count)
        except Exception:
            pass

    if fixed_constraints:
        apply_fixed_displacement_constraints(displacements, fixed_constraints)
        if bool(guide_rail_spread):
            spread_fixed_displacement_constraints(
                out_obj,
                displacements,
                fixed_constraints,
                spread_steps=guide_rail_spread_steps,
                spread_strength=guide_rail_spread_strength,
            )
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(mls_field):
        mls_refine_displacements(
            original,
            displacements,
            records,
            power=power,
            nearest_count=mls_nearest,
            mls_strength=mls_strength,
        )
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(guide_follow):
        guide_follow_refine_displacements(
            original,
            displacements,
            records,
            guide_pairs=solver_soft_guide_pairs(),
            guide_strength=guide_strength,
            guide_radius=guide_radius,
        )
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(nose_web_fit):
        nose_web_count = nose_web_refine_displacements(
            original,
            displacements,
            records,
            nose_strength=nose_strength,
            nose_radius=nose_radius,
            nose_samples=nose_samples,
        )
        try:
            out_obj["HFR_nweb"] = int(nose_web_count)
        except Exception:
            pass
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(nose_alar_fit):
        nose_alar_count = nose_alar_refine_displacements(
            original,
            displacements,
            records,
            alar_strength=alar_strength,
            alar_radius=alar_radius,
            alar_samples=alar_samples,
        )
        try:
            out_obj["HFR_nalar"] = int(nose_alar_count)
        except Exception:
            pass
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(brow_ridge_fit):
        brow_count = brow_ridge_refine_displacements(
            original,
            displacements,
            records,
            brow_strength=brow_strength,
            brow_radius=brow_radius,
            brow_samples=brow_samples,
        )
        try:
            out_obj["HFR_brow"] = int(brow_count)
        except Exception:
            pass
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(feature_loops):
        feature_loop_refine_displacements(
            original,
            displacements,
            records,
            loop_strength=loop_strength,
            loop_radius=loop_radius,
            loops=FEATURE_LOOPS,
        )
        apply_fixed_displacement_constraints(displacements, fixed_constraints)

    if bool(ear_lobe_fit):
        feature_loop_refine_displacements(
            original,
            displacements,
            records,
            loop_strength=ear_strength,
            loop_radius=ear_radius,
            loops=EAR_FEATURE_LOOPS,
        )

    if bool(ear_lower_rail):
        feature_line_refine_displacements(
            original,
            displacements,
            records,
            lines=EAR_LOWER_RAILS,
            line_strength=ear_lower_rail_strength,
            line_radius=ear_lower_rail_radius,
        )

    # v0.2.10: do not use the old lobe patch by default or from stored scene
    # settings. It can pull unrelated lower-ear vertices into spikes when the
    # template has only a small number of lobe vertices.

    if bool(neck_fit):
        feature_loop_refine_displacements(
            original,
            displacements,
            records,
            loop_strength=neck_strength,
            loop_radius=neck_radius,
            loops=NECK_FEATURE_LOOPS,
        )

    apply_fixed_displacement_constraints(displacements, fixed_constraints)
    for v in verts:
        v.co = original[v.index] + displacements[v.index]
    out_obj.data.update()

    # Optional residual correction: keeps anchor centroids on their landmarks
    # after the broad deformation.  It moves only the explicitly bound anchor
    # vertices, so topology is preserved and no faces/edges are rebuilt.
    lock = max(0.0, min(float(anchor_lock), 1.0))
    iters = max(0, int(anchor_iters))
    if lock > 0.0 and iters > 0:
        for _i in range(iters):
            for rec in records:
                members = [(idx, weight) for idx, weight in rec["members"] if 0 <= idx < len(verts)]
                if not members:
                    continue
                acc = Vector((0.0, 0.0, 0.0))
                total = 0.0
                for idx, weight in members:
                    w = max(float(weight), 0.0001)
                    acc += verts[idx].co * w
                    total += w
                if total <= 0.0:
                    continue
                centroid = acc / total
                residual = (rec["target"] - centroid)
                if rec.get("lm_id") in {"ear_l_lobe", "ear_r_lobe"} and not bool(lobe_directional_stretch):
                    # In conservative mode, prevent free forward/back lobe spikes.
                    residual.y = 0.0
                residual *= lock
                for idx, _weight in members:
                    verts[idx].co += residual
            out_obj.data.update()
    if bool(nose_web_fit):
        nose_post_count = apply_nose_web_surface_fit(
            out_obj,
            original,
            records,
            nose_strength=nose_strength,
            nose_radius=max(float(nose_radius), 2.0),
            nose_samples=max(int(nose_samples), 24),
        )
        try:
            out_obj["HFR_nweb"] = int(out_obj.get("HFR_nweb", 0)) + int(nose_post_count)
        except Exception:
            pass
    if bool(nose_alar_fit):
        nose_alar_post_count = apply_nose_alar_surface_fit(
            out_obj,
            original,
            records,
            alar_strength=alar_strength,
            alar_radius=alar_radius,
            alar_samples=alar_samples,
        )
        try:
            out_obj["HFR_nalar"] = int(out_obj.get("HFR_nalar", 0)) + int(nose_alar_post_count)
        except Exception:
            pass
    if bool(brow_ridge_fit):
        brow_post_count = apply_brow_ridge_surface_fit(
            out_obj,
            original,
            records,
            brow_strength=brow_strength,
            brow_radius=brow_radius,
            brow_samples=brow_samples,
            brow_smooth=brow_smooth,
        )
        try:
            out_obj["HFR_brow"] = int(out_obj.get("HFR_brow", 0)) + int(brow_post_count)
        except Exception:
            pass
    if bool(brow_inner_support):
        apply_brow_inner_support_fit(
            out_obj,
            original,
            records,
            support_strength=brow_inner_strength,
            support_steps=brow_inner_steps,
            support_radius=brow_inner_radius,
        )
    if bool(eye_loop_fit):
        if bool(eye_direct_fit):
            eye_direct_count = apply_eye_direct_loop_fit(
                out_obj,
                original,
                records,
                eye_strength=eye_loop_strength,
                eye_radius=eye_direct_radius,
            )
            try:
                out_obj["HFR_eyedir"] = int(out_obj.get("HFR_eyedir", 0)) + int(eye_direct_count)
            except Exception:
                pass
        eye_member_post_count = apply_eye_member_path_fit(
            out_obj,
            original,
            records,
            eye_strength=eye_loop_strength,
            max_path_len=max(8, int(eye_loop_max_len)),
        )
        try:
            out_obj["HFR_eyepth"] = int(out_obj.get("HFR_eyepth", 0)) + int(eye_member_post_count)
        except Exception:
            pass
        eye_band_post_count = apply_eye_topology_band_fit(
            out_obj,
            original,
            records,
            eye_strength=eye_loop_strength,
            band_steps=eye_band_steps,
            band_radius=eye_band_radius,
        )
        try:
            out_obj["HFR_eyeband"] = int(out_obj.get("HFR_eyeband", 0)) + int(eye_band_post_count)
        except Exception:
            pass
        eye_boundary_post_count = apply_eye_boundary_loop_fit(
            out_obj,
            original,
            records,
            eye_strength=eye_loop_strength,
            eye_steps=eye_loop_steps,
        )
        try:
            out_obj["HFR_eyebnd"] = int(out_obj.get("HFR_eyebnd", 0)) + int(eye_boundary_post_count)
        except Exception:
            pass
    # v0.2.24: MLS Field is now the primary broad between-landmark solver.
    # The older Head Dome Fit is only applied when MLS Field is off.
    if bool(head_round_fit) and not bool(mls_field):
        apply_head_round_fit(
            out_obj,
            records,
            original_positions=original,
            region_steps=head_round_steps,
            smooth_strength=head_round_strength,
            smooth_iters=head_round_iters,
            z_margin=head_round_z_margin,
        )
    if bool(lobe_directional_stretch):
        apply_directional_lobe_stretch(
            out_obj,
            records,
            lm_ids=("ear_l_lobe", "ear_r_lobe"),
            steps=lobe_directional_steps,
            strength=lobe_directional_strength,
            falloff=lobe_directional_falloff,
        )
    elif bool(sparse_ear_safe):
        # Conservative fallback for sparse ear templates when directional stretch is off.
        enforce_sparse_ear_lobe_plane(
            out_obj,
            records,
            y_strength=sparse_ear_y_strength,
            neighbor_blend=sparse_ear_neighbor_blend,
        )
    if bool(ear_local_fit):
        try:
            guard_count = len(ear_attachment_guard_vertex_indices(out_obj, records, original, "l", expand_steps=1, include_lower=True))
            guard_count += len(ear_attachment_guard_vertex_indices(out_obj, records, original, "r", expand_steps=1, include_lower=True))
            lower_guard_count = len(ear_attachment_guard_vertex_indices(out_obj, records, original, "l", expand_steps=1, include_lower=False))
            lower_guard_count += len(ear_attachment_guard_vertex_indices(out_obj, records, original, "r", expand_steps=1, include_lower=False))
            out_obj["HFR_eagrd"] = int(guard_count)
            out_obj["HFR_eagru"] = int(lower_guard_count)
        except Exception:
            pass
        apply_ear_local_frame_fit(
            out_obj,
            records,
            original_positions=original,
            strength=ear_local_strength,
            steps=ear_local_steps,
            nearest_count=ear_local_nearest,
        )
        ear_upper_count = ear_upper_inner_support_fit(
            out_obj,
            records,
            original_positions=original,
            strength=0.36,
            steps=1,
            radius_scale=0.68,
        )
        try:
            out_obj["HFR_earup"] = int(ear_upper_count)
        except Exception:
            pass
        ear_fan_count = apply_ear_inner_lower_fan_fit(
            out_obj,
            records,
            original_positions=original,
            strength=0.78,
            steps=2,
            radius_scale=1.05,
        )
        try:
            out_obj["HFR_earfn"] = int(ear_fan_count)
        except Exception:
            pass
    if bool(ear_lower_fit):
        apply_ear_lower_transition_fit(
            out_obj,
            records,
            original_positions=original,
            strength=ear_lower_strength,
            steps=ear_lower_steps,
            nearest_count=ear_lower_nearest,
        )
        ear_height_guard_count = apply_ear_lower_attachment_height_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.82,
            steps=max(2, int(ear_lower_steps)),
            radius_scale=1.20,
            z_pad_ratio=0.10,
        )
        try:
            out_obj["HFR_elhgt"] = int(ear_height_guard_count)
        except Exception:
            pass
        ear_lower_world_clamp_count = apply_ear_lower_selected_vertex_clamp(
            out_obj,
            records,
            original_positions=original,
            strength=0.94,
            world_z_pad=0.0020,
        )
        try:
            out_obj["HFR_ellwc"] = int(ear_lower_world_clamp_count)
        except Exception:
            pass
        ear_lower_front_height_count = apply_ear_lower_front_connector_height_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.90,
            steps=max(2, min(3, int(ear_lower_steps))),
            world_z_pad=0.00035,
            depth_z_bias=0.00135,
            max_world_z_drop=0.0085,
        )
        try:
            out_obj["HFR_elfhg"] = int(ear_lower_front_height_count)
        except Exception:
            pass
        ear_inner_inward_count = apply_ear_inner_lower_inward_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.88,
            steps=max(2, int(ear_lower_steps)),
            world_x_pad=0.0008,
        )
        try:
            out_obj["HFR_eilig"] = int(ear_inner_inward_count)
        except Exception:
            pass
        ear_inner_depth_count = apply_ear_inner_pocket_depth_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.90,
            steps=max(3, int(ear_lower_steps)),
            world_y_pad=0.0010,
            world_x_pad=0.0010,
        )
        try:
            out_obj["HFR_eipdg"] = int(ear_inner_depth_count)
        except Exception:
            pass
        ear_inner_sheet_count = apply_ear_inner_sheet_outward_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.86,
            steps=max(3, int(ear_lower_steps)),
            world_x_pad=0.00065,
            max_world_x_push=0.0028,
        )
        try:
            out_obj["HFR_eisog"] = int(ear_inner_sheet_count)
        except Exception:
            pass
        ear_lower_front_inset_count = apply_ear_lower_front_connector_inset_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.58,
            steps=max(2, min(3, int(ear_lower_steps))),
            world_x_inset=0.00075,
            second_ring_boost=1.35,
            max_world_x_inset=0.00145,
        )
        try:
            out_obj["HFR_elfig"] = int(ear_lower_front_inset_count)
        except Exception:
            pass
        ear_lower_nape_blend_count = apply_ear_lower_back_nape_direction_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.42,
            dot_threshold=0.72,
            strong_dot=0.90,
        )
        try:
            out_obj["HFR_elnbg"] = int(ear_lower_nape_blend_count)
        except Exception:
            pass
        ear_lobe_upper_lift_count = apply_ear_lobe_upper_connector_lift_guard(
            out_obj,
            records,
            original_positions=original,
            strength=0.96,
            steps=max(3, int(ear_lower_steps)),
            world_z_pad=0.00004,
            upper_blend_bias=0.24,
            max_world_z_lift=0.0105,
        )
        try:
            out_obj["HFR_elulg"] = int(ear_lobe_upper_lift_count)
        except Exception:
            pass
        ear_inner_lower_z_slide_count = apply_ear_inner_lower_negative_z_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.60,
            steps=2,
            z_drop_threshold=0.0030,
            lower_height_pad=0.0022,
            max_slide_strength=0.60,
        )
        try:
            out_obj["HFR_eilzsg"] = int(ear_inner_lower_z_slide_count)
        except Exception:
            pass
        ear_lower_first_z_slide_count = apply_ear_lower_front_first_ring_z_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.35,
            z_drop_threshold=0.0018,
            min_lower_anchor_links=2,
        )
        try:
            out_obj["HFR_elfzsg"] = int(ear_lower_first_z_slide_count)
        except Exception:
            pass
        ear_lobe_upper_z_slide_count = apply_ear_lobe_upper_z_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.50,
            z_drop_threshold=0.0011,
        )
        try:
            out_obj["HFR_eluzsg"] = int(ear_lobe_upper_z_slide_count)
        except Exception:
            pass
        apply_ear_lobe_upper_to_lobe_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.46,
            z_rise_threshold=0.0025,
        )
        apply_ear_front_inner_bridge_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.53,
            max_front_dist=3,
            outward_x_pad=0.00035,
        )
        apply_ear_inner_bottom_opposite_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.42,
            z_drop_threshold=0.0012,
            alignment_threshold=0.45,
        )
        apply_ear_inner_pocket_face_inward_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.22,
            bottom_fan_steps=1,
            min_anchor_links=2,
        )
    if bool(output_mirror_finish):
        apply_output_mirror_finish(
            out_obj,
            original,
            direction=output_mirror_direction,
            center_epsilon=output_mirror_epsilon,
        )
    if bool(ear_lower_fit):
        apply_ear_inner_pocket_outward_relief_guard(
            out_obj,
            records,
            original_positions=original,
            outward_strength=0.34,
            pocket_steps=2,
            include_inner_front_anchor=True,
        )
        apply_ear_back_middle_spoke_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.60,
        )
        apply_ear_inner_bottom_away_cap_slide_guard(
            out_obj,
            records,
            original_positions=original,
            slide_strength=0.63,
            min_bottom_steps=2,
            max_bottom_steps=4,
            min_front_steps=3,
        )
    try:
        create_eye_brow_debug_groups(
            out_obj,
            original,
            records,
            eye_radius=eye_direct_radius if 'eye_direct_radius' in locals() else 0.90,
        )
    except Exception:
        pass
    return len(records)


def build_world_bvh_from_object(context, obj):
    if obj is None or obj.type != 'MESH':
        return None
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        verts = [eval_obj.matrix_world @ v.co for v in mesh.vertices]
        polys = [tuple(tri.vertices) for tri in mesh.loop_triangles]
        if not verts or not polys:
            return None
        return BVHTree.FromPolygons(verts, polys, all_triangles=True)
    finally:
        eval_obj.to_mesh_clear()


def anchor_vertex_indices_for_object(obj):
    indices = set()
    if obj is None or obj.type != 'MESH':
        return indices
    for lm in LANDMARKS:
        group = obj.vertex_groups.get(anchor_group_name(lm["id"]))
        if group is None:
            continue
        for idx, _weight in vertex_indices_in_group(obj, group):
            if 0 <= idx < len(obj.data.vertices):
                indices.add(idx)
    return indices


def enforce_anchor_targets(out_obj, lock=1.0, iters=1):
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    lock = max(0.0, min(float(lock), 1.0))
    iters = max(0, int(iters))
    if lock <= 0.0 or iters <= 0:
        return 0
    verts = out_obj.data.vertices
    inv = out_obj.matrix_world.inverted()
    touched = 0
    for _i in range(iters):
        for lm in LANDMARKS:
            lm_id = lm["id"]
            group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
            members = [(idx, weight) for idx, weight in vertex_indices_in_group(out_obj, group) if 0 <= idx < len(verts)]
            if not members:
                continue
            target = inv @ landmark_location(lm_id)
            acc = Vector((0.0, 0.0, 0.0))
            total = 0.0
            for idx, weight in members:
                w = max(float(weight), 0.0001)
                acc += verts[idx].co * w
                total += w
            if total <= 0.0:
                continue
            centroid = acc / total
            residual = (target - centroid) * lock
            if residual.length <= 1.0e-12:
                continue
            for idx, _weight in members:
                verts[idx].co += residual
                touched += 1
        out_obj.data.update()
    return touched


def _auto_snap_max_distance(out_obj, target_obj, max_dist):
    max_dist = max(float(max_dist), 0.0)
    if max_dist > 0.0:
        return max_dist
    # A zero value used to mean unlimited snapping.  For template fitting that is
    # too destructive while testing, so zero now means an automatic conservative
    # limit based on the generated mesh size.
    try:
        dims = out_obj.dimensions
        diag = (dims.x * dims.x + dims.y * dims.y + dims.z * dims.z) ** 0.5
        if diag > 0.0:
            return max(diag * 0.025, 0.003)
    except Exception:
        pass
    return 0.03



def ear_landmark_ids():
    return {lm["id"] for lm in LANDMARKS if lm.get("grp") == "ear"}


def ear_region_vertex_indices(obj, steps=3):
    """Return the ear-local vertex region used by the snap guard.

    The ear is a thin folded structure.  A generic nearest-surface snap can move
    non-anchor ear vertices to the wrong side of the target ear or even to the
    adjacent head surface, which flips faces.  This region starts from bound ear
    anchors and expands a few topological rings, while stopping at non-ear anchor
    vertices so the head/neck attachment area is not frozen too broadly.
    """
    if obj is None or obj.type != 'MESH':
        return set()
    ear_ids = ear_landmark_ids()
    seeds = set()
    other_anchors = set()
    for lm in LANDMARKS:
        group = obj.vertex_groups.get(anchor_group_name(lm["id"]))
        if group is None:
            continue
        group_indices = {idx for idx, _w in vertex_indices_in_group(obj, group) if 0 <= idx < len(obj.data.vertices)}
        if lm["id"] in ear_ids:
            seeds.update(group_indices)
        else:
            other_anchors.update(group_indices)
    if not seeds:
        return set()
    adj = build_mesh_adjacency(obj)
    region = set(seeds)
    frontier = set(seeds)
    for _ in range(max(0, int(steps))):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in region or nb in other_anchors:
                    continue
                region.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        frontier = nxt
    return region



def side_face_transition_landmark_ids(side):
    return [
        f"face_{side}_edge",
        f"outer_face_{side}_lower",
        f"outer_face_{side}_upper",
        f"jaw_{side}_edge",
        f"chin_{side}_outer",
        f"cheek_{side}_center",
    ]


def side_face_transition_region_vertex_indices(obj, steps=4):
    """Return the side face strip between face_edge and outer_face_lower.

    This region is intentionally separate from the ear snap guard.  It targets
    the small rear/side transition strip that can remain slightly proud of the
    target surface when the conservative global snap distance rejects it.
    """
    if obj is None or obj.type != 'MESH':
        return set()
    try:
        records = anchor_records_for_template(obj)
    except Exception:
        return set()
    if not records:
        return set()
    vert_count = len(obj.data.vertices)
    current = [v.co.copy() for v in obj.data.vertices]
    adj = build_mesh_adjacency(obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}
    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))
    result = set()

    for side in ("l", "r"):
        ids = [lm_id for lm_id in side_face_transition_landmark_ids(side) if lm_id in rec_by_id]
        side_records = [rec_by_id[lm_id] for lm_id in ids]
        if len(side_records) < 4:
            continue
        seeds = set()
        # Main endpoints: this is the strip the user pointed out.
        for lm_id in (f"face_{side}_edge", f"outer_face_{side}_lower", f"jaw_{side}_edge"):
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                seeds.update(_record_member_indices(rec, vert_count))
        if not seeds:
            continue

        blocker_ids = {
            f"ear_{side}_top",
            f"ear_{side}_front_upper",
            f"ear_{side}_front_middle",
            f"ear_{side}_front_lower",
            f"ear_{side}_lobe",
            f"ear_{side}_back_upper",
            f"ear_{side}_back_middle",
            f"ear_{side}_back_lower",
            f"ear_{side}_inner_front_middle",
            f"ear_{side}_inner_bottom",
            f"eye_{side}_outer",
            f"eye_{side}_lower_outer",
            f"mouth_{side}_corner",
            f"neck_top_{side}_side",
            f"neck_{side}_side",
            f"nape_{side}_outer",
            f"head_{side}_side_back",
        }
        blockers = set()
        blocker_points = []
        for lm_id in blocker_ids:
            rec = rec_by_id.get(lm_id)
            if rec is None:
                continue
            blockers.update(_record_member_indices(rec, vert_count))
            blocker_points.append(rec["source"])
        blockers.update(_expanded_vertex_set(blockers, adj, steps=1))

        points = [rec["source"] for rec in side_records]
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
        max_span = max(span.x, span.y, span.z, 1.0e-6)
        margin = max(max_span * 0.46, 0.006)
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        min_z, max_z = min(zs) - margin * 0.85, max(zs) + margin * 0.85
        radial_limit = max(max_span * 1.20, 0.010)

        side_set = set(seeds)
        frontier = set(seeds)
        max_steps = max(1, min(int(steps), 6))
        for _i in range(max_steps):
            nxt = set()
            for vidx in frontier:
                for nb in adj[vidx]:
                    if nb in side_set or nb in blockers:
                        continue
                    if nb < 0 or nb >= vert_count:
                        continue
                    co = current[nb]
                    if side == "l" and co.x > 0.003:
                        continue
                    if side == "r" and co.x < -0.003:
                        continue
                    if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                        continue
                    side_d = _min_distance_to_points(co, points)
                    if side_d > radial_limit:
                        continue
                    if blocker_points:
                        block_d = _min_distance_to_points(co, blocker_points)
                        if block_d <= side_d * 0.96:
                            continue
                    side_set.add(nb)
                    nxt.add(nb)
            if not nxt:
                break
            frontier = nxt
        # Snap only in-between/support vertices; keep explicit anchors controlled
        # by landmarks and post-anchor lock.
        side_set.difference_update(all_anchor_members)
        result.update(side_set)
    return result


def snap_side_face_transition_to_target(context, out_obj, target_obj, bvh=None,
                                        strength=0.90, base_max_dist=0.0,
                                        steps=4):
    if out_obj is None or out_obj.type != 'MESH' or target_obj is None or target_obj.type != 'MESH':
        return 0
    strength = max(0.0, min(float(strength), 1.0))
    if strength <= 0.0:
        return 0
    if bvh is None:
        bvh = build_world_bvh_from_object(context, target_obj)
    if bvh is None:
        return 0
    region = side_face_transition_region_vertex_indices(out_obj, steps=steps)
    if not region:
        try:
            out_obj["HFR_sfsn"] = 0
        except Exception:
            pass
        return 0
    try:
        dims = out_obj.dimensions
        diag = (dims.x * dims.x + dims.y * dims.y + dims.z * dims.z) ** 0.5
    except Exception:
        diag = 0.0
    base = max(float(base_max_dist), 0.0)
    local_max = max(base * 3.0, diag * 0.055 if diag > 0.0 else 0.0, 0.012)
    inv = out_obj.matrix_world.inverted()
    moved = 0
    for idx in sorted(region):
        if idx < 0 or idx >= len(out_obj.data.vertices):
            continue
        v = out_obj.data.vertices[idx]
        world_co = out_obj.matrix_world @ v.co
        hit = bvh.find_nearest(world_co)
        if not hit:
            continue
        nearest_co, _normal, _hit_index, dist = hit
        if nearest_co is None:
            continue
        if local_max > 0.0 and dist > local_max:
            continue
        prox = max(0.0, 1.0 - min(1.0, dist / max(local_max, 1.0e-6)))
        w = strength * (0.45 + 0.55 * _smoothstep01(prox))
        if w <= 0.0:
            continue
        v.co = inv @ world_co.lerp(nearest_co, min(1.0, w))
        moved += 1
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_sfsn"] = int(moved)
    except Exception:
        pass
    return moved


def head_back_snap_landmark_ids(side=None):
    ids = [
        "scalp_front_center", "scalp_top_center", "scalp_back_center",
        "forehead_upper_center", "forehead_center",
        "head_l_side_upper", "head_r_side_upper",
        "head_l_side_back", "head_r_side_back",
        "scalp_l_front", "scalp_r_front", "scalp_l_top", "scalp_r_top",
        "nape_center", "nape_l_outer", "nape_r_outer",
        "neck_back_center", "neck_top_l_back", "neck_top_r_back",
        "neck_top_l_side", "neck_top_r_side",
    ]
    return ids


def head_back_snap_region_vertex_indices(obj, steps=10):
    """Return a broad scalp/back-head/upper-neck exterior region for final snap.

    v0.5.6 still relied on the global conservative Snap Max Distance.  On the
    top/back of the head the landmark-deformed template can be farther from the
    target than front-face vertices, so those exterior vertices are skipped by
    the global snap.  This region gives only the scalp/back/side-head shell a
    second, larger-distance snap pass while blocking eyes, nose, mouth, ears,
    jaw/cheek detail, and explicit anchors.
    """
    if obj is None or obj.type != 'MESH':
        return set()
    try:
        records = anchor_records_for_template(obj)
    except Exception:
        return set()
    if not records:
        return set()
    vert_count = len(obj.data.vertices)
    current = [v.co.copy() for v in obj.data.vertices]
    adj = build_mesh_adjacency(obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records}

    snap_ids = [lm_id for lm_id in head_back_snap_landmark_ids() if lm_id in rec_by_id]
    snap_records = [rec_by_id[lm_id] for lm_id in snap_ids]
    if len(snap_records) < 6:
        return set()

    all_anchor_members = set()
    for rec in records:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    seeds = set()
    for rec in snap_records:
        seeds.update(_record_member_indices(rec, vert_count))
    if not seeds:
        return set()

    blocked_prefixes = ("eye_", "mouth_", "nose_", "brow_", "ear_")
    blocked_ids = set()
    for rec in records:
        lm_id = rec.get("lm_id", "")
        if lm_id.startswith(blocked_prefixes):
            blocked_ids.add(lm_id)
    blocked_ids.update({
        "cheek_l_center", "cheek_r_center",
        "face_l_edge", "face_r_edge",
        "outer_face_l_upper", "outer_face_r_upper",
        "outer_face_l_lower", "outer_face_r_lower",
        "jaw_l_edge", "jaw_r_edge",
        "chin_l_outer", "chin_r_outer", "chin_l_lower_outer", "chin_r_lower_outer",
        "chin_l_lower", "chin_r_lower", "chin_center",
        "neck_front_center", "neck_top_l_front", "neck_top_r_front",
    })
    blockers = set()
    for lm_id in blocked_ids:
        rec = rec_by_id.get(lm_id)
        if rec is not None:
            blockers.update(_record_member_indices(rec, vert_count))
    blockers.update(_expanded_vertex_set(blockers, adj, steps=1))

    points = [rec["source"] for rec in snap_records]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    span = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    max_span = max(span.x, span.y, span.z, 1.0e-6)
    margin_x = max(span.x * 0.30, max_span * 0.18, 0.008)
    margin_y = max(span.y * 0.34, max_span * 0.20, 0.010)
    margin_z = max(span.z * 0.34, max_span * 0.20, 0.010)
    min_x, max_x = min(xs) - margin_x, max(xs) + margin_x
    min_y, max_y = min(ys) - margin_y, max(ys) + margin_y
    min_z, max_z = min(zs) - margin_z, max(zs) + margin_z
    radial_limit = max(max_span * 0.90, 0.035)

    region = set(idx for idx in seeds if 0 <= idx < vert_count)
    frontier = set(region)
    max_steps = max(1, min(int(steps), 16))
    for _i in range(max_steps):
        nxt = set()
        for vidx in frontier:
            for nb in adj[vidx]:
                if nb in region or nb in blockers:
                    continue
                if nb < 0 or nb >= vert_count:
                    continue
                co = current[nb]
                if co.x < min_x or co.x > max_x or co.y < min_y or co.y > max_y or co.z < min_z or co.z > max_z:
                    continue
                if _min_distance_to_points(co, points) > radial_limit:
                    continue
                region.add(nb)
                nxt.add(nb)
        if not nxt:
            break
        frontier = nxt

    # Explicit anchors are landmark-controlled and are optionally corrected by
    # post-anchor lock.  Snap only the in-between exterior surface vertices.
    region.difference_update(all_anchor_members)
    return region


def snap_head_back_to_target(context, out_obj, target_obj, bvh=None,
                             strength=0.96, base_max_dist=0.0, steps=10):
    if out_obj is None or out_obj.type != 'MESH' or target_obj is None or target_obj.type != 'MESH':
        return 0
    strength = max(0.0, min(float(strength), 1.0))
    if strength <= 0.0:
        return 0
    if bvh is None:
        bvh = build_world_bvh_from_object(context, target_obj)
    if bvh is None:
        return 0
    region = head_back_snap_region_vertex_indices(out_obj, steps=steps)
    if not region:
        try:
            out_obj["HFR_hbsn"] = 0
        except Exception:
            pass
        return 0
    try:
        dims = out_obj.dimensions
        diag = (dims.x * dims.x + dims.y * dims.y + dims.z * dims.z) ** 0.5
    except Exception:
        diag = 0.0
    base = max(float(base_max_dist), 0.0)
    local_max = max(base * 4.0, diag * 0.14 if diag > 0.0 else 0.0, 0.045)
    inv = out_obj.matrix_world.inverted()
    moved = 0
    for idx in sorted(region):
        if idx < 0 or idx >= len(out_obj.data.vertices):
            continue
        v = out_obj.data.vertices[idx]
        world_co = out_obj.matrix_world @ v.co
        hit = bvh.find_nearest(world_co)
        if not hit:
            continue
        nearest_co, _normal, _hit_index, dist = hit
        if nearest_co is None:
            continue
        if local_max > 0.0 and dist > local_max:
            continue
        prox = max(0.0, 1.0 - min(1.0, dist / max(local_max, 1.0e-6)))
        w = strength * (0.55 + 0.45 * _smoothstep01(prox))
        if w <= 0.0:
            continue
        v.co = inv @ world_co.lerp(nearest_co, min(1.0, w))
        moved += 1
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_hbsn"] = int(moved)
    except Exception:
        pass
    return moved

def snap_output_to_target(context, out_obj, target_obj, strength=0.75, max_dist=0.0,
                          protect_anchor=True, anchor_strength=0.25,
                          ear_snap_guard=True, ear_snap_strength=0.0, ear_snap_steps=3,
                          eye_snap_guard=True, eye_snap_strength=0.0, eye_snap_steps=96):
    if out_obj is None or out_obj.type != 'MESH':
        raise ValueError("Output mesh is invalid")
    if target_obj is None or target_obj.type != 'MESH':
        raise ValueError("Target Mesh is not assigned")
    strength = max(0.0, min(float(strength), 1.0))
    if strength <= 0.0:
        return 0
    bvh = build_world_bvh_from_object(context, target_obj)
    if bvh is None:
        raise ValueError("Could not build BVH from Target Mesh")
    max_dist = _auto_snap_max_distance(out_obj, target_obj, max_dist)
    anchor_indices = anchor_vertex_indices_for_object(out_obj) if protect_anchor else set()
    anchor_strength = max(0.0, min(float(anchor_strength), 1.0))
    ear_indices = ear_region_vertex_indices(out_obj, steps=ear_snap_steps) if bool(ear_snap_guard) else set()
    ear_snap_strength = max(0.0, min(float(ear_snap_strength), 1.0))
    eye_indices = eye_support_region_vertex_indices(
        out_obj,
        steps=eye_snap_steps,
        band_steps=3,
        band_radius=1.45,
        max_path_len=max(12, int(eye_snap_steps // 4) if eye_snap_steps else 24),
    ) if bool(eye_snap_guard) else set()
    eye_snap_strength = max(0.0, min(float(eye_snap_strength), 1.0))
    inv = out_obj.matrix_world.inverted()
    moved = 0
    guarded = 0
    eye_guarded = 0
    for v in out_obj.data.vertices:
        world_co = out_obj.matrix_world @ v.co
        hit = bvh.find_nearest(world_co)
        if not hit:
            continue
        nearest_co, _normal, _index, dist = hit
        if nearest_co is None:
            continue
        if max_dist > 0.0 and dist > max_dist:
            continue
        local_strength = min(strength, anchor_strength) if v.index in anchor_indices else strength
        if v.index in ear_indices:
            guarded += 1
            local_strength = min(local_strength, ear_snap_strength)
        if v.index in eye_indices:
            eye_guarded += 1
            local_strength = min(local_strength, eye_snap_strength)
        if local_strength <= 0.0:
            continue
        new_world = world_co.lerp(nearest_co, local_strength)
        v.co = inv @ new_world
        moved += 1
    out_obj.data.update()
    head_back_moved = snap_head_back_to_target(
        context,
        out_obj,
        target_obj,
        bvh=bvh,
        strength=max(strength, 0.90),
        base_max_dist=max_dist,
        steps=10,
    )
    moved += int(head_back_moved)
    side_moved = snap_side_face_transition_to_target(
        context,
        out_obj,
        target_obj,
        bvh=bvh,
        strength=max(strength, 0.82),
        base_max_dist=max_dist,
        steps=4,
    )
    moved += int(side_moved)
    try:
        out_obj["HFR_earsg"] = int(guarded)
        out_obj["HFR_eyesg"] = int(eye_guarded)
        out_obj["HFR_sfsn"] = int(side_moved)
        out_obj["HFR_hbsn"] = int(head_back_moved)
    except Exception:
        pass
    return moved





def apply_back_center_column_inward_slide_guard(out_obj, records=None, slide_strength=0.30):
    """Slide posterior first side-column vertices inward symmetrically.

    v0.5.50 could move only one side because candidate detection was evaluated
    per side.  This version is centered on the posterior seam itself: for every
    center seam vertex, it finds the directly connected left/right first-column
    vertices and applies the same relative slide toward that shared seam point.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_centroid(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            if 0 <= idx < vert_count:
                w = max(float(weight), 0.0001)
                acc += current[idx] * w
                total += w
        if total <= 0.0:
            return None
        return acc / total

    nape = group_centroid("nape_center")
    scalp_back = group_centroid("scalp_back_center")
    neck_back = group_centroid("neck_back_center")
    if nape is None or scalp_back is None:
        try:
            out_obj["HFR_bcisg"] = 0
        except Exception:
            pass
        return 0

    side_refs = []
    for lm_id in ("head_l_side_back", "head_r_side_back", "head_l_side_upper", "head_r_side_upper"):
        p = group_centroid(lm_id)
        if p is not None:
            side_refs.append(p)
    side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    center_tol = max(side_width * 0.06, 0.040)
    first_col_min = max(side_width * 0.15, 0.07)
    first_col_max = side_width * 0.58
    outward_eps = max(side_width * 0.018, 0.050)

    y_refs = [nape.y, scalp_back.y]
    z_refs = [nape.z, scalp_back.z]
    if neck_back is not None:
        y_refs.append(neck_back.y)
        z_refs.append(neck_back.z)
    for p in side_refs:
        y_refs.append(p.y)
        z_refs.append(p.z)

    y_span = max(max(y_refs) - min(y_refs), side_width * 0.15, 1.0e-6)
    z_span = max(max(z_refs) - min(z_refs), side_width * 0.15, 1.0e-6)
    y_min = min(y_refs) - y_span * 0.18
    y_max = max(y_refs) + y_span * 0.22
    z_min = min(z_refs) - z_span * 0.24
    z_max = max(z_refs) + z_span * 0.24

    def has_outward_neighbor(idx, sign):
        co = current[idx]
        signed_x = co.x * sign
        for nb in adj[idx]:
            if nb < 0 or nb >= vert_count:
                continue
            nb_co = current[nb]
            if nb_co.x * sign > signed_x + outward_eps:
                return True
        return False

    moved_indices = set()

    # Work from the seam outward.  If a seam vertex has both left and right
    # first-column neighbors, they are moved together; if only one is present it
    # can still move, but the pair test prevents one-sided misses when both sides
    # are topologically available.
    for center_idx in range(vert_count):
        center_co = current[center_idx]
        if abs(center_co.x) > center_tol:
            continue
        if center_co.y < y_min or center_co.y > y_max or center_co.z < z_min or center_co.z > z_max:
            continue

        side_candidates = {"l": [], "r": []}
        for nb in adj[center_idx]:
            if nb < 0 or nb >= vert_count or nb in all_anchor_members:
                continue
            nb_co = current[nb]
            if nb_co.y < y_min or nb_co.y > y_max or nb_co.z < z_min or nb_co.z > z_max:
                continue
            abs_x = abs(nb_co.x)
            if abs_x < first_col_min or abs_x > first_col_max:
                continue
            side = "r" if nb_co.x > 0.0 else "l"
            sign = 1.0 if side == "r" else -1.0
            if not has_outward_neighbor(nb, sign):
                continue
            # Prefer the closest first-column neighbor for each side of this seam
            # vertex.  This matches the 682/1391-style pair sharing seam 489.
            side_candidates[side].append((abs_x, nb))

        targets = []
        for side in ("l", "r"):
            if side_candidates[side]:
                side_candidates[side].sort(key=lambda item: (item[0], item[1]))
                targets.append(side_candidates[side][0][1])

        for idx in targets:
            if idx in moved_indices:
                continue
            verts[idx].co = current[idx].lerp(center_co, slide_strength)
            moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_bcisg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_back_outer_column_inward_slide_guard(out_obj, records=None, slide_strength=0.30):
    """Slide posterior outer-middle column vertices inward by a relative percentage.

    This targets the 905/906/1191-style strip and the mirrored counterpart.  It
    does not use fixed coordinates: candidates are non-anchor posterior side
    vertices in the second side column, between the inner first column and the
    outer side column.  Each candidate slides toward the adjacent same-side
    inward neighbor by the given percentage.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_centroid(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            if 0 <= idx < vert_count:
                w = max(float(weight), 0.0001)
                acc += current[idx] * w
                total += w
        if total <= 0.0:
            return None
        return acc / total

    nape = group_centroid("nape_center")
    scalp_back = group_centroid("scalp_back_center")
    if nape is None or scalp_back is None:
        try:
            out_obj["HFR_boisg"] = 0
        except Exception:
            pass
        return 0

    side_refs = []
    for lm_id in ("head_l_side_back", "head_r_side_back", "head_l_side_upper", "head_r_side_upper"):
        p = group_centroid(lm_id)
        if p is not None:
            side_refs.append(p)
    side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    # Relative posterior band.  This excludes the lower nape/neck rows and the
    # top scalp transition, while keeping the row containing 905/906/1191 and
    # the mirrored 153/154/472-style vertices.
    y_low = min(nape.y, scalp_back.y)
    y_high = max(nape.y, scalp_back.y)
    y_span = max(y_high - y_low, 1.0e-6)
    y_min = y_low + y_span * 0.34
    y_max = y_low + y_span * 0.90

    z_mid = (nape.z + scalp_back.z) * 0.5
    z_margin = max(side_width * 0.18, abs(nape.z - scalp_back.z) * 3.0, 0.10)
    z_min = z_mid - z_margin
    z_max = z_mid + z_margin

    # Second side column: farther from the center than the first posterior seam
    # neighbor column, but not yet the outer silhouette column.
    x_min = side_width * 0.50
    x_max = side_width * 0.68
    inward_gap = max(side_width * 0.18, 0.18)
    outward_gap = max(side_width * 0.12, 0.12)

    moved_indices = set()
    targets = {}

    for idx in range(vert_count):
        if idx in all_anchor_members:
            continue
        co = current[idx]
        abs_x = abs(co.x)
        if abs_x < x_min or abs_x > x_max:
            continue
        if co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
            continue
        sign = 1.0 if co.x >= 0.0 else -1.0

        inward_candidates = []
        outward_found = False
        vertical_selected_links = 0
        for nb in adj[idx]:
            if nb < 0 or nb >= vert_count:
                continue
            nb_co = current[nb]
            nb_abs_x = abs(nb_co.x)
            same_side = (nb_co.x * sign) > 0.0
            if not same_side:
                continue
            if nb_abs_x < abs_x - inward_gap:
                yz_dist = ((nb_co.y - co.y) * (nb_co.y - co.y) +
                           (nb_co.z - co.z) * (nb_co.z - co.z)) ** 0.5
                inward_candidates.append((nb_abs_x, yz_dist, nb, nb_co))
            if nb_abs_x > abs_x + outward_gap:
                outward_found = True
            if abs(nb_abs_x - abs_x) <= side_width * 0.08:
                vertical_selected_links += 1

        # Require an outside rail as well as an inward rail so the guard does not
        # catch the inner column already handled by the center-column pass.
        if not inward_candidates or not outward_found:
            continue

        # Pick the most inward same-side adjacent rail.  This makes 905->1187,
        # 906->1186, 1191->1378 and mirrors to 153->466, 154->465, 472->669.
        inward_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        targets[idx] = inward_candidates[0][3]

    for idx, target_co in targets.items():
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_boisg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_side_head_ear_opposite_slide_guard(out_obj, records=None, slide_strength=0.50):
    """Slide the side-head vertex away from the ear by a relative percentage.

    Targets the 1127-style vertex and its mirrored counterpart without fixed
    coordinates.  The candidate is found by topology: a non-anchor side vertex
    just behind the ear/head-side junction whose two adjacent rail vertices are
    both directly connected to the head_*_side_back anchor.  The target edge is
    selected as the same-side neighbor that is farther from the local ear anchor
    centroid and closer to the head centerline, then the candidate is blended
    0.50 toward that neighbor, matching a vertex-slide-like relative operation.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def ear_centroid(side):
        pts = []
        for rec in records or []:
            lm_id = rec.get("lm_id", "")
            if not lm_id.startswith(f"ear_{side}_"):
                continue
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    pts.append(current[idx])
        if not pts:
            for vg in out_obj.vertex_groups:
                if not vg.name.startswith(f"HFR_A_ear_{side}_"):
                    continue
                for idx, _w in vertex_indices_in_group(out_obj, vg):
                    if 0 <= idx < vert_count:
                        pts.append(current[idx])
        if not pts:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        for p in pts:
            acc += p
        return acc / len(pts)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved = 0
    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        head_back = group_centroid(f"head_{side}_side_back")
        head_upper = group_centroid(f"head_{side}_side_upper")
        ecent = ear_centroid(side)
        if head_back is None or ecent is None:
            continue
        side_width = max(abs(head_back.x), abs(head_upper.x) if head_upper is not None else 0.0, 1.0e-6)
        anchor_indices = set(idx for idx, _w in group_members(f"head_{side}_side_back"))
        if not anchor_indices:
            continue

        # Local, relative band around the head-side-back / ear attachment.  This
        # catches the 1127/402-style point but avoids the ear mesh and neck rows.
        x_min = side_width * 0.86
        x_max = side_width * 1.08
        y_margin = max(side_width * 0.18, 0.20)
        z_margin = max(side_width * 0.18, 0.20)
        y_min = min(head_back.y, ecent.y) - y_margin
        y_max = max(head_back.y, ecent.y) + y_margin
        z_min = min(head_back.z, ecent.z) - z_margin
        z_max = max(head_back.z, ecent.z) + z_margin

        candidates = []
        for idx in range(vert_count):
            if idx in all_anchor_members:
                continue
            co = current[idx]
            if co.x * sign <= 0.0:
                continue
            abs_x = abs(co.x)
            if abs_x < x_min or abs_x > x_max:
                continue
            if co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
                continue
            # This mirror pass is for the lower side-head / ear-opposite support
            # vertex.  Upper side-back candidates satisfy the same broad rail
            # test, so require the candidate to sit below head_side_back along
            # the local template Y band.
            if co.y > head_back.y - side_width * 0.025:
                continue

            rail_neighbors = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if any(anchor_idx in adj[nb] for anchor_idx in anchor_indices):
                    nb_co = current[nb]
                    if nb_co.x * sign > 0.0:
                        rail_neighbors.append(nb)
            if len(set(rail_neighbors)) < 2:
                continue

            target_options = []
            cur_dist = (co - ecent).length
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                nb_co = current[nb]
                if nb_co.x * sign <= 0.0:
                    continue
                # Opposite from the ear on this side means farther from the ear
                # anchor cluster and also less lateral, i.e. closer to the head
                # centerline.  This chooses 1127->1389 and 402->680.
                if abs(nb_co.x) >= abs_x:
                    continue
                dist_gain = (nb_co - ecent).length - cur_dist
                if dist_gain <= 0.0:
                    continue
                yz_dist = ((nb_co.y - co.y) * (nb_co.y - co.y) +
                           (nb_co.z - co.z) * (nb_co.z - co.z)) ** 0.5
                center_gain = abs_x - abs(nb_co.x)
                target_options.append((dist_gain * 10.0 + center_gain - yz_dist * 0.12,
                                       dist_gain, center_gain, -yz_dist, nb, nb_co))
            if not target_options:
                continue
            target_options.sort(reverse=True)
            # Prefer candidates nearest to head_side_back and then most lateral;
            # this avoids moving lower nape rows that merely satisfy the same
            # broad region test.
            score = -(co - head_back).length + abs_x * 0.02 + len(set(rail_neighbors)) * 0.05
            candidates.append((score, idx, target_options[0][5]))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        idx = candidates[0][1]
        if idx in moved_indices:
            continue
        target_co = candidates[0][2]
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)
        moved += 1

    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_eopsg"] = int(moved)
    except Exception:
        pass
    return moved




def apply_side_head_ear_opposite_mirror_match_guard(out_obj, records=None, match_strength=1.0):
    """Mirror-match the side-head ear-opposite support pair.

    The earlier ear-opposite guard applied the same relative slide independently
    per side.  If the two sides had already drifted differently after snapping,
    the counterpart could still remain visually mismatched.  This pass finds the
    same 1127/402-style topology pair on both sides, keeps the side that is
    already farther along its ear-opposite rail, and places the other side at
    the X-mirrored counterpart position.  No Blender Vertex Slide operator or
    fixed world coordinate is used.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def ear_centroid(side):
        pts = []
        for rec in records or []:
            lm_id = rec.get("lm_id", "")
            if not lm_id.startswith(f"ear_{side}_"):
                continue
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    pts.append(current[idx])
        if not pts:
            for vg in out_obj.vertex_groups:
                if not vg.name.startswith(f"HFR_A_ear_{side}_"):
                    continue
                for idx, _w in vertex_indices_in_group(out_obj, vg):
                    if 0 <= idx < vert_count:
                        pts.append(current[idx])
        if not pts:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        for p in pts:
            acc += p
        return acc / len(pts)

    match_strength = max(0.0, min(float(match_strength), 1.0))
    if match_strength <= 0.0:
        return 0

    def find_side_candidate(side, sign):
        head_back = group_centroid(f"head_{side}_side_back")
        head_upper = group_centroid(f"head_{side}_side_upper")
        ecent = ear_centroid(side)
        if head_back is None or ecent is None:
            return None
        side_width = max(abs(head_back.x), abs(head_upper.x) if head_upper is not None else 0.0, 1.0e-6)
        anchor_indices = set(idx for idx, _w in group_members(f"head_{side}_side_back"))
        if not anchor_indices:
            return None

        x_min = side_width * 0.84
        x_max = side_width * 1.10
        y_margin = max(side_width * 0.20, 0.20)
        z_margin = max(side_width * 0.22, 0.20)
        y_min = min(head_back.y, ecent.y) - y_margin
        y_max = max(head_back.y, ecent.y) + y_margin
        z_min = min(head_back.z, ecent.z) - z_margin
        z_max = max(head_back.z, ecent.z) + z_margin

        candidates = []
        for idx in range(vert_count):
            if idx in all_anchor_members:
                continue
            co = current[idx]
            if co.x * sign <= 0.0:
                continue
            abs_x = abs(co.x)
            if abs_x < x_min or abs_x > x_max:
                continue
            if co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
                continue

            rail_neighbors = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if any(anchor_idx in adj[nb] for anchor_idx in anchor_indices):
                    nb_co = current[nb]
                    if nb_co.x * sign > 0.0:
                        rail_neighbors.append(nb)
            if len(set(rail_neighbors)) < 2:
                continue

            cur_dist = (co - ecent).length
            target_options = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                nb_co = current[nb]
                if nb_co.x * sign <= 0.0:
                    continue
                if abs(nb_co.x) >= abs_x:
                    continue
                dist_gain = (nb_co - ecent).length - cur_dist
                if dist_gain <= 0.0:
                    continue
                yz_dist = ((nb_co.y - co.y) * (nb_co.y - co.y) +
                           (nb_co.z - co.z) * (nb_co.z - co.z)) ** 0.5
                center_gain = abs_x - abs(nb_co.x)
                target_options.append((dist_gain * 10.0 + center_gain - yz_dist * 0.12,
                                       dist_gain, center_gain, -yz_dist, nb, nb_co))
            if not target_options:
                continue
            target_options.sort(reverse=True)
            target_co = target_options[0][5]
            # Keep the same topology intent as the ear-opposite slide guard, but
            # also favor the support point whose direct rail neighbors straddle
            # head_side_back.  This isolates the 1127/402 pair.
            score = -(co - head_back).length + abs_x * 0.02 + len(set(rail_neighbors)) * 0.05
            dist_to_target = (co - target_co).length
            candidates.append((score, idx, target_options[0][4], target_co, dist_to_target, side_width))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0]

    left = find_side_candidate("l", -1.0)
    right = find_side_candidate("r", 1.0)
    if left is None or right is None:
        try:
            out_obj["HFR_eomsg"] = 0
        except Exception:
            pass
        return 0

    _ls, l_idx, _lt_idx, l_target_co, l_dist, l_width = left
    _rs, r_idx, _rt_idx, r_target_co, r_dist, r_width = right
    if l_idx in all_anchor_members or r_idx in all_anchor_members:
        return 0

    l_norm = l_dist / max(float(l_width), 1.0e-6)
    r_norm = r_dist / max(float(r_width), 1.0e-6)

    moved = 0
    eps = 1.0e-7
    if l_norm <= r_norm:
        # Left side is already farther along the intended opposite-from-ear rail;
        # mirror-match the right counterpart to it.
        target = Vector((-current[l_idx].x, current[l_idx].y, current[l_idx].z))
        if (current[r_idx] - target).length > eps:
            verts[r_idx].co = current[r_idx].lerp(target, match_strength)
            moved += 1
    else:
        target = Vector((-current[r_idx].x, current[r_idx].y, current[r_idx].z))
        if (current[l_idx] - target).length > eps:
            verts[l_idx].co = current[l_idx].lerp(target, match_strength)
            moved += 1

    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_eomsg"] = int(moved)
    except Exception:
        pass
    return moved


def apply_side_head_ear_toward_strip_slide_guard(out_obj, records=None, slide_strength=0.38):
    """Slide the side-head lower strip toward the ear by a relative percentage.

    Targets the 1202/1394-style edge and its mirrored counterpart without fixed
    coordinates.  The selected edge is detected as the best same-side inner strip
    pair near the ear/back-head attachment: each endpoint must have an adjacent
    same-side neighbor that is farther lateral and closer to the local ear anchor
    centroid.  Each endpoint is then blended toward that outward/earward rail by
    0.38, matching a vertex-slide-like relative operation.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def ear_centroid(side):
        pts = []
        for rec in records or []:
            lm_id = rec.get("lm_id", "")
            if not lm_id.startswith(f"ear_{side}_"):
                continue
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    pts.append(current[idx])
        if not pts:
            for vg in out_obj.vertex_groups:
                if not vg.name.startswith(f"HFR_A_ear_{side}_"):
                    continue
                for idx, _w in vertex_indices_in_group(out_obj, vg):
                    if 0 <= idx < vert_count:
                        pts.append(current[idx])
        if not pts:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        for p in pts:
            acc += p
        return acc / len(pts)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        head_back = group_centroid(f"head_{side}_side_back")
        head_upper = group_centroid(f"head_{side}_side_upper")
        lobe = group_centroid(f"ear_{side}_lobe")
        ecent = ear_centroid(side)
        if head_back is None or ecent is None:
            continue
        side_width = max(abs(head_back.x), abs(head_upper.x) if head_upper is not None else 0.0, 1.0e-6)

        # Relative lower side-head strip band.  This catches the 1202/1394 and
        # 484/685 style edge while avoiding the ear mesh itself, upper scalp rows,
        # and lower nape/neck rows.
        x_min = side_width * 0.72
        x_max = side_width * 0.90
        y_refs = [head_back.y, ecent.y]
        if lobe is not None:
            y_refs.append(lobe.y)
        y_min = min(y_refs) - side_width * 0.10
        y_max = max(y_refs) + side_width * 0.18
        z_min = head_back.z - side_width * 0.13
        z_max = head_back.z - side_width * 0.005

        candidates = {}
        for idx in range(vert_count):
            if idx in all_anchor_members:
                continue
            co = current[idx]
            if co.x * sign <= 0.0:
                continue
            abs_x = abs(co.x)
            if abs_x < x_min or abs_x > x_max:
                continue
            if co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
                continue

            cur_dist = (co - ecent).length
            target_options = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                nb_co = current[nb]
                if nb_co.x * sign <= 0.0:
                    continue
                nb_abs_x = abs(nb_co.x)
                # Toward the ear means farther lateral on the same side and
                # closer to the local ear anchor cluster.
                if nb_abs_x <= abs_x + side_width * 0.035:
                    continue
                dist_gain = cur_dist - (nb_co - ecent).length
                if dist_gain <= 0.0:
                    continue
                yz_dist = ((nb_co.y - co.y) * (nb_co.y - co.y) +
                           (nb_co.z - co.z) * (nb_co.z - co.z)) ** 0.5
                lateral_gain = nb_abs_x - abs_x
                score = dist_gain * 10.0 + lateral_gain * 0.5 - yz_dist * 0.10
                target_options.append((score, dist_gain, lateral_gain, -yz_dist, nb, nb_co))

            if not target_options:
                continue
            target_options.sort(reverse=True)
            candidates[idx] = (co, target_options[0])

        # Move the best connected pair, not isolated vertices.  This makes the
        # edge slide act on 1202/1394 -> 877/1212 and the mirrored 484/685 ->
        # 118/496 as one strip-level operation.
        pair_options = []
        for a, (a_co, a_target) in candidates.items():
            for b in adj[a]:
                if b not in candidates or a >= b:
                    continue
                b_co, b_target = candidates[b]
                # Prefer a vertical-ish strip edge, but keep it relative to the
                # detected topology rather than any fixed vertex index.
                same_side_gap = abs(abs(a_co.x) - abs(b_co.x))
                edge_len = (b_co - a_co).length
                target_link = 0.0
                if b_target[4] in adj[a_target[4]]:
                    target_link = 0.35
                pair_score = a_target[0] + b_target[0] + target_link - same_side_gap * 0.15 - edge_len * 0.02
                pair_options.append((pair_score, a, b, a_target[5], b_target[5]))

        if not pair_options:
            continue
        pair_options.sort(reverse=True)
        _score, a, b, a_target_co, b_target_co = pair_options[0]
        if a not in moved_indices:
            verts[a].co = current[a].lerp(a_target_co, slide_strength)
            moved_indices.add(a)
        if b not in moved_indices:
            verts[b].co = current[b].lerp(b_target_co, slide_strength)
            moved_indices.add(b)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_etsg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_side_head_ear_toward_inner_support_slide_guard(out_obj, records=None, slide_strength=0.26):
    """Slide the inner side-head ear-support vertex toward the ear.

    This targets the 1211-style vertex and its mirrored counterpart without using
    fixed coordinates.  It is the support vertex just medial to the lower
    side-head/ear strip adjusted by the previous guard.  The candidate is found
    by same-side topology: it must have an earward adjacent rail that is farther
    lateral and closer to the local ear anchor centroid, plus a more medial
    neighbor.  The best lower ear-side support candidate per side is blended
    toward that earward rail by 0.26, matching a vertex-slide-like relative
    operation.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def ear_centroid(side):
        pts = []
        for rec in records or []:
            lm_id = rec.get("lm_id", "")
            if not lm_id.startswith(f"ear_{side}_"):
                continue
            for idx in _record_member_indices(rec, vert_count):
                if 0 <= idx < vert_count:
                    pts.append(current[idx])
        if not pts:
            for vg in out_obj.vertex_groups:
                if not vg.name.startswith(f"HFR_A_ear_{side}_"):
                    continue
                for idx, _w in vertex_indices_in_group(out_obj, vg):
                    if 0 <= idx < vert_count:
                        pts.append(current[idx])
        if not pts:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        for p in pts:
            acc += p
        return acc / len(pts)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        head_back = group_centroid(f"head_{side}_side_back")
        head_upper = group_centroid(f"head_{side}_side_upper")
        lobe = group_centroid(f"ear_{side}_lobe")
        ecent = ear_centroid(side)
        if head_back is None or ecent is None or lobe is None:
            continue
        side_width = max(abs(head_back.x), abs(head_upper.x) if head_upper is not None else 0.0, 1.0e-6)

        # Inner support band just medial to the 1202/1394-style strip.  The band
        # is relative to the side-head anchor width and ear-lobe level, so it
        # follows different landmark placements instead of using absolute offsets.
        x_min = side_width * 0.70
        x_max = side_width * 0.84
        y_min = lobe.y - side_width * 0.18
        y_max = lobe.y + side_width * 0.22
        z_min = head_back.z - side_width * 0.18
        z_max = head_back.z - side_width * 0.04
        earward_gap = side_width * 0.07
        inward_gap = side_width * 0.09

        candidates = []
        for idx in range(vert_count):
            if idx in all_anchor_members:
                continue
            co = current[idx]
            if co.x * sign <= 0.0:
                continue
            abs_x = abs(co.x)
            if abs_x < x_min or abs_x > x_max:
                continue
            if co.y < y_min or co.y > y_max or co.z < z_min or co.z > z_max:
                continue

            cur_dist = (co - ecent).length
            has_inward_neighbor = False
            target_options = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                nb_co = current[nb]
                if nb_co.x * sign <= 0.0:
                    continue
                nb_abs_x = abs(nb_co.x)
                if nb_abs_x < abs_x - inward_gap:
                    has_inward_neighbor = True
                if nb_abs_x <= abs_x + earward_gap:
                    continue
                dist_gain = cur_dist - (nb_co - ecent).length
                if dist_gain <= 0.0:
                    continue
                yz_dist = ((nb_co.y - co.y) * (nb_co.y - co.y) +
                           (nb_co.z - co.z) * (nb_co.z - co.z)) ** 0.5
                lateral_gain = nb_abs_x - abs_x
                # Select the rail most like 1211->1394 / 495->685: clearly
                # earward, closer to the ear cluster, and still part of the
                # lower ear-side support band rather than the upper scalp rows.
                target_score = dist_gain * 10.0 + lateral_gain * 0.45 - yz_dist * 0.10
                target_options.append((target_score, dist_gain, lateral_gain, -yz_dist, nb, nb_co))

            if not has_inward_neighbor or not target_options:
                continue
            target_options.sort(reverse=True)
            target_co = target_options[0][5]
            y_lobe = abs(co.y - lobe.y)
            z_target = abs(co.z - (head_back.z - side_width * 0.115))
            # One support vertex per side.  Penalize rows away from the lobe band
            # so upper neighboring rows are not chosen over the selected row.
            score = target_options[0][0] - y_lobe * 2.20 - z_target * 0.45 + len(adj[idx]) * 0.03
            candidates.append((score, idx, target_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, idx, target_co = candidates[0]
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_eitsg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_jaw_ear_lower_down_slide_guard(out_obj, records=None, slide_strength=0.44):
    """Slide the jaw/ear lower transition support vertex downward.

    This targets the 1156-style vertex and its mirrored counterpart without
    fixed coordinates.  The candidate is the non-anchor side vertex directly
    connected to the same-side jaw_edge anchor whose adjacent non-anchor edge
    provides the strongest World-Z downward slide.  The movement is a relative
    vertex-slide-like blend toward that lower adjacent vertex.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        jaw_members = [idx for idx, _w in group_members(f"jaw_{side}_edge")]
        jaw_center = group_centroid(f"jaw_{side}_edge")
        if not jaw_members or jaw_center is None:
            continue
        head_back = group_centroid(f"head_{side}_side_back")
        head_upper = group_centroid(f"head_{side}_side_upper")
        side_width = max(
            abs(jaw_center.x),
            abs(head_back.x) if head_back is not None else 0.0,
            abs(head_upper.x) if head_upper is not None else 0.0,
            1.0e-6,
        )
        down_eps = max(side_width * 0.0007, 1.0e-6)
        max_y_span = side_width * 0.42
        max_z_span = side_width * 0.34

        candidate_indices = set()
        for anchor_idx in jaw_members:
            for nb in adj[anchor_idx]:
                if nb < 0 or nb >= vert_count:
                    continue
                if nb in all_anchor_members:
                    continue
                co = current[nb]
                if co.x * sign <= 0.0:
                    continue
                # Keep this local to the jaw/ear transition, not the chin row.
                if abs(co.y - jaw_center.y) > max_y_span or abs(co.z - jaw_center.z) > max_z_span:
                    continue
                candidate_indices.add(nb)

        candidates = []
        for idx in candidate_indices:
            co = current[idx]
            co_wz = world_z(co)
            target_options = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count or nb in all_anchor_members:
                    continue
                nb_co = current[nb]
                if nb_co.x * sign <= 0.0:
                    continue
                dz_down = co_wz - world_z(nb_co)
                if dz_down <= down_eps:
                    continue
                # Prefer true vertex-slide neighbors in the same lower side/head
                # strip: mostly the same local band around jaw_edge, not a jump
                # into chin anchors or ear anchors.
                band_penalty = abs(nb_co.y - co.y) * 0.10 + abs(nb_co.x - co.x) * 0.05
                target_options.append((dz_down * 100.0 - band_penalty, dz_down, nb, nb_co))
            if not target_options:
                continue
            target_options.sort(reverse=True)
            target_score, dz_down, target_idx, target_co = target_options[0]
            # The selected 1156/432-style vertex is slightly above jaw_edge and
            # has the strongest non-anchor downward slide edge among the
            # jaw_edge neighbors.  Vertices already below the jaw rail receive a
            # penalty so lower chin-row supports are not chosen.
            above_jaw = co_wz - world_z(jaw_center)
            side_band = abs(abs(co.x) - abs(jaw_center.x))
            score = target_score + max(0.0, above_jaw) * 22.0 - max(0.0, -above_jaw) * 16.0 - side_band * 0.08
            candidates.append((score, idx, target_idx, target_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, idx, _target_idx, target_co = candidates[0]
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_jedsg"] = int(moved)
    except Exception:
        pass
    return moved




def apply_ear_front_lower_support_down_slide_guard(out_obj, records=None, slide_strength=0.41):
    """Slide the ear-front-lower support vertex downward by a relative amount.

    Targets the 1147-style vertex and its mirrored counterpart.  The candidate
    is detected from the same-side ear_front_lower anchor: a non-anchor neighbor
    that sits just above that anchor in the local ear-front strip, then slides
    toward the adjacent non-anchor vertex with the strongest World-Z downward
    direction.  This preserves vertex-slide behavior without fixed coordinates.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        ear_members = [idx for idx, _w in group_members(f"ear_{side}_front_lower")]
        ear_center = group_centroid(f"ear_{side}_front_lower")
        if not ear_members or ear_center is None:
            continue

        side_refs = [ear_center]
        for lm_id in (
            f"ear_{side}_front_middle",
            f"ear_{side}_front_upper",
            f"ear_{side}_lobe",
            f"head_{side}_side_back",
            f"head_{side}_side_upper",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

        # 1147/423-style vertices are direct neighbors of ear_front_lower,
        # almost level in local-Y, and just above the anchor in local-Z.  This
        # excludes the lobe/downward neighbor and the upper ear-front supports.
        max_y_delta = max(side_width * 0.022, 0.060)
        min_z_delta = max(side_width * 0.0035, 0.010)
        max_z_delta = max(side_width * 0.055, 0.140)
        down_eps = max(side_width * 0.0007, 1.0e-6)

        candidates = []
        for anchor_idx in ear_members:
            for idx in adj[anchor_idx]:
                if idx < 0 or idx >= vert_count:
                    continue
                if idx in all_anchor_members:
                    continue
                co = current[idx]
                if co.x * sign <= 0.0:
                    continue
                dy = abs(co.y - ear_center.y)
                dz_local = co.z - ear_center.z
                if dy > max_y_delta:
                    continue
                if dz_local < min_z_delta or dz_local > max_z_delta:
                    continue

                co_wz = world_z(co)
                target_options = []
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count:
                        continue
                    if nb in all_anchor_members:
                        continue
                    nb_co = current[nb]
                    if nb_co.x * sign <= 0.0:
                        continue
                    dz_down = co_wz - world_z(nb_co)
                    if dz_down <= down_eps:
                        continue
                    # Prefer a real local slide edge in the same ear/lower-jaw
                    # support strip, not a long diagonal jump.
                    local_penalty = abs(nb_co.y - co.y) * 0.10 + abs(nb_co.x - co.x) * 0.05
                    target_options.append((dz_down * 100.0 - local_penalty, dz_down, nb, nb_co))

                if not target_options:
                    continue
                target_options.sort(reverse=True)
                target_score, dz_down, target_idx, target_co = target_options[0]

                # Prefer the direct strip vertex closest to ear_front_lower in
                # local-Y and just above it in local-Z.
                score = target_score - dy * 2.5 - abs(dz_local - side_width * 0.013) * 0.40
                candidates.append((score, idx, target_idx, target_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, idx, _target_idx, target_co = candidates[0]
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_efldsg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_ear_front_lower_upper_support_down_slide_guard(out_obj, records=None, slide_strength=0.65):
    """Slide the upper support next to ear_front_lower downward.

    This targets the 1149-style vertex and its mirrored counterpart without
    fixed coordinates.  The candidate is the non-anchor direct neighbor of
    ear_front_lower that sits in the upper/front-lower support wedge.  It is
    moved by a relative vertex-slide-like blend toward the directly connected
    same-side lower non-anchor rail vertex.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        ear_members = [idx for idx, _w in group_members(f"ear_{side}_front_lower")]
        ear_center = group_centroid(f"ear_{side}_front_lower")
        if not ear_members or ear_center is None:
            continue

        side_refs = [ear_center]
        for lm_id in (
            f"ear_{side}_front_middle",
            f"ear_{side}_front_upper",
            f"ear_{side}_lobe",
            f"head_{side}_side_back",
            f"head_{side}_side_upper",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

        # 1149/425-style vertices are direct neighbors of ear_front_lower but
        # sit clearly above the lower support vertex handled by v0.5.57.  The
        # local-Z band keeps this pass away from 1147/423-style lower supports,
        # lobe anchors, and the upper ear-front rim.
        min_dz_local = max(side_width * 0.040, 0.18)
        max_dz_local = max(side_width * 0.095, 0.55)
        max_y_delta = max(side_width * 0.052, 0.24)
        max_x_delta = max(side_width * 0.030, 0.18)
        down_eps = max(side_width * 0.0007, 1.0e-6)

        candidates = []
        for anchor_idx in ear_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            anchor_co = current[anchor_idx]
            for idx in adj[anchor_idx]:
                if idx < 0 or idx >= vert_count:
                    continue
                if idx in all_anchor_members:
                    continue
                co = current[idx]
                if co.x * sign <= 0.0:
                    continue
                dz_local = co.z - ear_center.z
                dy = abs(co.y - ear_center.y)
                dx_from_anchor = abs(abs(co.x) - abs(anchor_co.x))
                if dz_local < min_dz_local or dz_local > max_dz_local:
                    continue
                if dy > max_y_delta or dx_from_anchor > max_x_delta:
                    continue

                co_wz = world_z(co)
                target_options = []
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count:
                        continue
                    if nb in all_anchor_members:
                        continue
                    nb_co = current[nb]
                    if nb_co.x * sign <= 0.0:
                        continue
                    dz_down = co_wz - world_z(nb_co)
                    if dz_down <= down_eps:
                        continue
                    # Prefer the same vertical/lateral rail below this vertex,
                    # not the diagonal inward cheek/ear-sheet neighbor.  This
                    # produces the intended 1149->1361 and 425->652 slide path.
                    rail_penalty = abs(abs(nb_co.x) - abs(co.x)) * 2.40 + abs(nb_co.y - co.y) * 0.12
                    target_score = dz_down * 100.0 - rail_penalty
                    target_options.append((target_score, dz_down, nb, nb_co))

                if not target_options:
                    continue
                target_options.sort(reverse=True)
                target_score, dz_down, target_idx, target_co = target_options[0]
                # Prefer the upper support wedge just above ear_front_lower.
                wedge_z = abs(dz_local - side_width * 0.058)
                score = target_score + dz_local * 0.60 - dy * 0.70 - wedge_z * 0.35
                candidates.append((score, idx, target_idx, target_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, idx, _target_idx, target_co = candidates[0]
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_eflusg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_ear_face_edge_upper_support_down_slide_guard(out_obj, records=None, slide_strength=0.47):
    """Slide the face-edge/ear-front upper support downward.

    This targets the 1148-style vertex and its mirrored counterpart without
    fixed vertex indices or absolute coordinates.  The candidate is the
    non-anchor side vertex directly connected to face_l/r_edge, positioned just
    below that face-edge anchor in local topology but above it in World-Z after
    deformation.  It is moved by a relative vertex-slide-like blend toward the
    directly connected lower same-side rail vertex.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        face_members = [idx for idx, _w in group_members(f"face_{side}_edge")]
        face_center = group_centroid(f"face_{side}_edge")
        if not face_members or face_center is None:
            continue

        side_refs = [face_center]
        for lm_id in (
            f"ear_{side}_front_lower",
            f"ear_{side}_front_middle",
            f"outer_face_{side}_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

        min_local_below = max(side_width * 0.012, 0.070)
        max_local_below = max(side_width * 0.050, 0.300)
        min_world_above = max(side_width * 0.00035, 0.0010)
        max_y_delta = max(side_width * 0.085, 0.42)
        max_x_delta = max(side_width * 0.100, 0.56)
        down_eps = max(side_width * 0.0007, 1.0e-6)

        candidates = []
        for anchor_idx in face_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            anchor_co = current[anchor_idx]
            anchor_wz = world_z(anchor_co)
            for idx in adj[anchor_idx]:
                if idx < 0 or idx >= vert_count:
                    continue
                if idx in all_anchor_members:
                    continue
                co = current[idx]
                if co.x * sign <= 0.0:
                    continue

                local_below = anchor_co.z - co.z
                if local_below < min_local_below or local_below > max_local_below:
                    continue
                world_above = world_z(co) - anchor_wz
                if world_above < min_world_above:
                    continue
                if abs(co.y - anchor_co.y) > max_y_delta:
                    continue
                if abs(abs(co.x) - abs(anchor_co.x)) > max_x_delta:
                    continue

                co_wz = world_z(co)
                target_options = []
                for nb in adj[idx]:
                    if nb < 0 or nb >= vert_count:
                        continue
                    if nb in all_anchor_members:
                        continue
                    nb_co = current[nb]
                    if nb_co.x * sign <= 0.0:
                        continue
                    dz_down = co_wz - world_z(nb_co)
                    if dz_down <= down_eps:
                        continue
                    # Choose the lower same-side rail, not the shallow ear-side
                    # diagonal.  This produces the intended 1148->881 and
                    # 424->122 style slide path.
                    x_penalty = abs(abs(nb_co.x) - abs(co.x)) * 0.60
                    y_penalty = abs(nb_co.y - co.y) * 0.10
                    target_score = dz_down * 100.0 - x_penalty - y_penalty
                    target_options.append((target_score, dz_down, nb, nb_co))

                if not target_options:
                    continue
                target_options.sort(reverse=True)
                target_score, dz_down, target_idx, target_co = target_options[0]

                # Prefer the side support just outside face_edge and below it in
                # source/local topology while still sitting high in World-Z.
                score = target_score + world_above * 60.0 - abs(local_below - side_width * 0.024) * 0.70
                candidates.append((score, idx, target_idx, target_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, idx, _target_idx, target_co = candidates[0]
        if idx in moved_indices:
            continue
        verts[idx].co = current[idx].lerp(target_co, slide_strength)
        moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_efedsg"] = int(moved)
    except Exception:
        pass
    return moved




def apply_ear_face_edge_pair_toward_ear_slide_guard(out_obj, records=None, slide_strength=0.27):
    """Slide the face-edge support edge toward the ear-side adjacent edge.

    This targets the 881/1148-style selected edge and the mirrored 122/424-style
    edge without fixed indices.  Starting from the face_l/r_edge anchor, it finds
    the directly connected outer support vertex, its lower rail vertex, and the
    ear-side adjacent edge across the quad.  The two detected vertices are then
    blended toward that adjacent ear-side edge.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        face_members = [idx for idx, _w in group_members(f"face_{side}_edge")]
        face_center = group_centroid(f"face_{side}_edge")
        if not face_members or face_center is None:
            continue

        side_refs = [face_center]
        for lm_id in (
            f"ear_{side}_front_lower",
            f"ear_{side}_front_middle",
            f"ear_{side}_front_upper",
            f"outer_face_{side}_upper",
            f"head_{side}_side_upper",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

        signed_eps = max(side_width * 0.010, 0.040)
        toward_eps = max(side_width * 0.014, 0.055)
        max_anchor_y_delta = max(side_width * 0.090, 0.55)
        max_anchor_z_delta = max(side_width * 0.090, 0.55)
        min_lower_world_drop = max(side_width * 0.00035, 0.0010)

        candidates = []
        for anchor_idx in face_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            anchor_co = current[anchor_idx]
            anchor_signed_x = anchor_co.x * sign

            for upper_idx in adj[anchor_idx]:
                if upper_idx < 0 or upper_idx >= vert_count:
                    continue
                if upper_idx in all_anchor_members:
                    continue
                upper_co = current[upper_idx]
                upper_signed_x = upper_co.x * sign
                if upper_signed_x <= anchor_signed_x + signed_eps:
                    continue
                if upper_signed_x <= 0.0:
                    continue
                if abs(upper_co.y - anchor_co.y) > max_anchor_y_delta:
                    continue
                if abs(upper_co.z - anchor_co.z) > max_anchor_z_delta:
                    continue

                upper_wz = world_z(upper_co)

                lower_options = []
                for lower_idx in adj[upper_idx]:
                    if lower_idx < 0 or lower_idx >= vert_count:
                        continue
                    if lower_idx in all_anchor_members or lower_idx == anchor_idx:
                        continue
                    lower_co = current[lower_idx]
                    if lower_co.x * sign <= 0.0:
                        continue
                    world_drop = upper_wz - world_z(lower_co)
                    if world_drop <= min_lower_world_drop:
                        continue

                    lower_targets = []
                    for lt_idx in adj[lower_idx]:
                        if lt_idx < 0 or lt_idx >= vert_count:
                            continue
                        if lt_idx in all_anchor_members or lt_idx == upper_idx:
                            continue
                        lt_co = current[lt_idx]
                        if lt_co.x * sign <= 0.0:
                            continue
                        lt_signed_x = lt_co.x * sign
                        if lt_signed_x <= (lower_co.x * sign) + toward_eps:
                            continue
                        lower_targets.append((lt_signed_x, lt_idx, lt_co))

                    if not lower_targets:
                        continue
                    lower_targets.sort(reverse=True)
                    lt_signed_x, lt_idx, lt_co = lower_targets[0]
                    lower_options.append((world_drop, lower_idx, lower_co, lt_idx, lt_co))

                if not lower_options:
                    continue
                lower_options.sort(reverse=True)
                world_drop, lower_idx, lower_co, lt_idx, lt_co = lower_options[0]

                upper_targets = []
                for ut_idx in adj[upper_idx]:
                    if ut_idx < 0 or ut_idx >= vert_count:
                        continue
                    if ut_idx in all_anchor_members or ut_idx in (anchor_idx, lower_idx):
                        continue
                    ut_co = current[ut_idx]
                    if ut_co.x * sign <= 0.0:
                        continue
                    ut_signed_x = ut_co.x * sign
                    if ut_signed_x <= upper_signed_x + toward_eps:
                        continue
                    # The ear-side target edge should be the adjacent edge across
                    # the same quad: upper target connects to the lower target.
                    if lt_idx not in adj[ut_idx]:
                        continue
                    upper_targets.append((ut_signed_x, ut_idx, ut_co))

                if not upper_targets:
                    continue
                upper_targets.sort(reverse=True)
                ut_signed_x, ut_idx, ut_co = upper_targets[0]

                pair_width_gain = (ut_signed_x - upper_signed_x) + ((lt_co.x * sign) - (lower_co.x * sign))
                edge_parallel = abs((upper_co - lower_co).length - (ut_co - lt_co).length)
                score = pair_width_gain * 10.0 + world_drop * 100.0 - edge_parallel * 0.25
                candidates.append((score, upper_idx, ut_idx, lower_idx, lt_idx, upper_co, ut_co, lower_co, lt_co))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        _score, upper_idx, ut_idx, lower_idx, lt_idx, upper_co, ut_co, lower_co, lt_co = candidates[0]

        if upper_idx not in moved_indices:
            verts[upper_idx].co = current[upper_idx].lerp(ut_co, slide_strength)
            moved_indices.add(upper_idx)
        if lower_idx not in moved_indices:
            verts[lower_idx].co = current[lower_idx].lerp(lt_co, slide_strength)
            moved_indices.add(lower_idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_efetsg"] = int(moved)
    except Exception:
        pass
    return moved



def apply_ear_lower_wedge_down_slide_guard(out_obj, records=None, slide_strength=0.32):
    """Slide the lower ear-front wedge vertices downward.

    This targets the 887/1137/1151-style local wedge and the mirrored
    129/412/427-style wedge.  It starts from the same-side ear_front_lower and
    jaw_edge anchors, detects the non-anchor center wedge vertex two topology
    steps between them, then moves that center vertex and its two branch vertices
    toward the directly connected lower rail vertices.  The movement is a
    relative vertex-slide-like blend along existing edges, not an absolute offset.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        ear_members = [idx for idx, _w in group_members(f"ear_{side}_front_lower")]
        jaw_members = [idx for idx, _w in group_members(f"jaw_{side}_edge")]
        ear_center = group_centroid(f"ear_{side}_front_lower")
        jaw_center = group_centroid(f"jaw_{side}_edge")
        if not ear_members or not jaw_members or ear_center is None or jaw_center is None:
            continue

        side_refs = [ear_center, jaw_center]
        for lm_id in (
            f"face_{side}_edge",
            f"ear_{side}_front_middle",
            f"ear_{side}_lobe",
            f"head_{side}_side_back",
            f"outer_face_{side}_lower",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])
        down_eps = max(side_width * 0.00035, 0.0010)
        max_y_delta = max(side_width * 0.14, 0.75)
        max_z_delta = max(side_width * 0.14, 0.75)

        ear_support = set()
        for anchor_idx in ear_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            for nb in adj[anchor_idx]:
                if 0 <= nb < vert_count and nb not in all_anchor_members:
                    if current[nb].x * sign > 0.0:
                        ear_support.add(nb)

        jaw_support = set()
        for anchor_idx in jaw_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            for nb in adj[anchor_idx]:
                if 0 <= nb < vert_count and nb not in all_anchor_members:
                    if current[nb].x * sign > 0.0:
                        jaw_support.add(nb)

        if not ear_support or not jaw_support:
            continue

        def same_side(idx):
            return 0 <= idx < vert_count and current[idx].x * sign > 0.0

        def best_down_target(idx, exclude=None, require_subset=None, allow_anchor=True):
            exclude = set(exclude or ())
            co = current[idx]
            co_wz = world_z(co)
            options = []
            for nb in adj[idx]:
                if nb < 0 or nb >= vert_count or nb in exclude:
                    continue
                if not allow_anchor and nb in all_anchor_members:
                    continue
                if require_subset is not None and nb not in require_subset:
                    continue
                if not same_side(nb):
                    continue
                nb_co = current[nb]
                dz_down = co_wz - world_z(nb_co)
                if dz_down <= down_eps:
                    continue
                # Keep the slide local to the existing wedge/rail.  The strongest
                # World-Z drop wins, with a small penalty for diagonal jumps.
                diag_penalty = abs(nb_co.y - co.y) * 0.08 + abs(nb_co.x - co.x) * 0.04
                options.append((dz_down * 100.0 - diag_penalty, dz_down, nb, nb_co))
            if not options:
                return None
            options.sort(reverse=True)
            return options[0]

        center_candidates = []
        for idx in range(vert_count):
            if idx in all_anchor_members or not same_side(idx):
                continue
            co = current[idx]
            if abs(co.y - ear_center.y) > max_y_delta and abs(co.y - jaw_center.y) > max_y_delta:
                continue
            if abs(co.z - ear_center.z) > max_z_delta and abs(co.z - jaw_center.z) > max_z_delta:
                continue
            # The selected center vertex is not itself adjacent to either anchor;
            # it bridges an ear-front support neighbor and a jaw-side support
            # neighbor.
            if any(nb in ear_members for nb in adj[idx]):
                continue
            if any(nb in jaw_members for nb in adj[idx]):
                continue
            ear_links = [nb for nb in adj[idx] if nb in ear_support]
            jaw_links = [nb for nb in adj[idx] if nb in jaw_support]
            if not ear_links or not jaw_links:
                continue
            down = best_down_target(idx, require_subset=jaw_support, allow_anchor=False)
            if down is None:
                continue
            target_score, dz_down, target_idx, target_co = down
            # Prefer the 887/129-style center: it has two jaw-side support links
            # and one ear-front support link, and it sits between ear_front_lower
            # and jaw_edge in world Z.
            wz = world_z(co)
            wz_low = min(world_z(ear_center), world_z(jaw_center))
            wz_high = max(world_z(ear_center), world_z(jaw_center))
            band_bonus = 1.0 if (wz_low - side_width * 0.003 <= wz <= wz_high + side_width * 0.003) else 0.0
            score = target_score + len(jaw_links) * 5.0 + len(ear_links) * 2.0 + band_bonus
            center_candidates.append((score, idx, target_idx, target_co, set(ear_links), set(jaw_links)))

        if not center_candidates:
            continue
        center_candidates.sort(reverse=True)
        _score, center_idx, center_target_idx, center_target_co, center_ear_links, center_jaw_links = center_candidates[0]

        if center_idx not in moved_indices:
            verts[center_idx].co = current[center_idx].lerp(center_target_co, slide_strength)
            moved_indices.add(center_idx)

        branch_candidates = []
        skip_branch = set(center_ear_links)
        skip_branch.add(center_target_idx)
        for nb in adj[center_idx]:
            if nb < 0 or nb >= vert_count:
                continue
            if nb in all_anchor_members or nb in skip_branch:
                continue
            if not same_side(nb):
                continue
            down = best_down_target(nb, exclude={center_idx}, allow_anchor=True)
            if down is None:
                continue
            target_score, dz_down, target_idx, target_co = down
            # Avoid sliding already-lower rail vertices; keep branches close to
            # the center wedge and require a real downward edge.
            if world_z(current[nb]) < world_z(current[center_target_idx]) - down_eps:
                continue
            score = target_score + dz_down * 20.0
            if nb in center_jaw_links:
                score += 3.0
            branch_candidates.append((score, nb, target_idx, target_co))

        # Move the two side branches, matching the selected 1137/1151 and
        # mirrored 412/427 vertices.  The center target/lower rail and the
        # ear-front support neighbor are intentionally not moved here.
        branch_candidates.sort(reverse=True)
        for _score, idx, _target_idx, target_co in branch_candidates[:2]:
            if idx in moved_indices:
                continue
            verts[idx].co = current[idx].lerp(target_co, slide_strength)
            moved_indices.add(idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_elwdsg"] = int(moved)
    except Exception:
        pass
    return moved




def apply_ear_upper_wedge_edge_down_slide_guard(out_obj, records=None, slide_strength=0.44):
    """Slide the upper ear-front wedge support edge downward.

    This targets the 1140/1361-style selected edge and the mirrored
    415/652-style edge without fixed vertex indices.  It finds the non-anchor
    edge sitting above the lower ear-front wedge edge by topology: one endpoint
    has a jaw-edge support target below it, while the other endpoint has a
    directly connected lower partner that also connects to that jaw-side target.
    The two detected vertices are blended toward that adjacent lower edge, which
    is a vertex-slide-like relative movement along existing mesh edges.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        ear_members = [idx for idx, _w in group_members(f"ear_{side}_front_lower")]
        jaw_members = [idx for idx, _w in group_members(f"jaw_{side}_edge")]
        face_center = group_centroid(f"face_{side}_edge")
        ear_center = group_centroid(f"ear_{side}_front_lower")
        jaw_center = group_centroid(f"jaw_{side}_edge")
        if not ear_members or not jaw_members or ear_center is None or jaw_center is None:
            continue

        side_refs = [ear_center, jaw_center]
        if face_center is not None:
            side_refs.append(face_center)
        for lm_id in (
            f"ear_{side}_front_middle",
            f"outer_face_{side}_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])
        down_eps = max(side_width * 0.00035, 0.0010)
        max_y_delta = max(side_width * 0.16, 0.85)
        max_z_delta = max(side_width * 0.16, 0.85)

        def same_side(idx):
            return 0 <= idx < vert_count and current[idx].x * sign > 0.0

        ear_support = set()
        for anchor_idx in ear_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            for nb in adj[anchor_idx]:
                if 0 <= nb < vert_count and nb not in all_anchor_members and same_side(nb):
                    ear_support.add(nb)

        jaw_support = set()
        for anchor_idx in jaw_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            for nb in adj[anchor_idx]:
                if 0 <= nb < vert_count and nb not in all_anchor_members and same_side(nb):
                    jaw_support.add(nb)

        if not ear_support or not jaw_support:
            continue

        def edge_key(a, b):
            return (a, b) if a < b else (b, a)

        def lower_targets_for_oriented_edge(upper_inner, upper_outer):
            """Return target pair for upper_inner/upper_outer orientation.

            upper_outer is the ear/jaw side endpoint.  It must have a directly
            connected jaw support vertex below it.  upper_inner then slides to a
            lower same-side partner connected to that jaw-side target.
            """
            if upper_inner in all_anchor_members or upper_outer in all_anchor_members:
                return []
            if upper_inner in ear_support or upper_inner in jaw_support:
                return []
            if upper_outer in ear_support or upper_outer in jaw_support:
                return []
            if not same_side(upper_inner) or not same_side(upper_outer):
                return []

            uo_co = current[upper_outer]
            ui_co = current[upper_inner]
            uo_wz = world_z(uo_co)
            ui_wz = world_z(ui_co)
            if abs(uo_co.y - ear_center.y) > max_y_delta and abs(uo_co.y - jaw_center.y) > max_y_delta:
                return []
            if abs(ui_co.y - ear_center.y) > max_y_delta and abs(ui_co.y - jaw_center.y) > max_y_delta:
                return []
            if abs(uo_co.z - ear_center.z) > max_z_delta and abs(uo_co.z - jaw_center.z) > max_z_delta:
                return []
            if abs(ui_co.z - ear_center.z) > max_z_delta and abs(ui_co.z - jaw_center.z) > max_z_delta:
                return []

            # The outer endpoint belongs to the upper wedge just behind the ear
            # lower anchor: it must bridge at least one ear-front support and one
            # jaw-edge support through direct topology.
            if not any(nb in ear_support for nb in adj[upper_outer]):
                return []

            options = []
            for outer_target in adj[upper_outer]:
                if outer_target not in jaw_support:
                    continue
                ot_co = current[outer_target]
                ot_down = uo_wz - world_z(ot_co)
                if ot_down <= down_eps:
                    continue
                for inner_target in adj[upper_inner]:
                    if inner_target < 0 or inner_target >= vert_count:
                        continue
                    if inner_target in all_anchor_members or inner_target in (upper_inner, upper_outer):
                        continue
                    if inner_target in ear_support or inner_target in jaw_support:
                        continue
                    if not same_side(inner_target):
                        continue
                    if outer_target not in adj[inner_target]:
                        continue
                    it_co = current[inner_target]
                    it_down = ui_wz - world_z(it_co)
                    if it_down <= down_eps:
                        continue
                    # Prefer a true adjacent lower edge rather than a diagonal
                    # fan jump: both slide targets should be close to their
                    # corresponding upper endpoint and connected to each other.
                    inner_diag = abs(it_co.y - ui_co.y) * 0.08 + abs(it_co.x - ui_co.x) * 0.035
                    outer_diag = abs(ot_co.y - uo_co.y) * 0.08 + abs(ot_co.x - uo_co.x) * 0.035
                    target_score = (it_down + ot_down) * 100.0 - inner_diag - outer_diag
                    options.append((target_score, inner_target, outer_target, it_co, ot_co, it_down, ot_down))
            return options

        candidates = []
        seen_edges = set()
        for a in range(vert_count):
            if a in all_anchor_members or not same_side(a):
                continue
            for b in adj[a]:
                if b < 0 or b >= vert_count or b <= a:
                    continue
                if b in all_anchor_members or not same_side(b):
                    continue
                ek = edge_key(a, b)
                if ek in seen_edges:
                    continue
                seen_edges.add(ek)

                oriented_options = []
                oriented_options.extend((a, b, opt) for opt in lower_targets_for_oriented_edge(a, b))
                oriented_options.extend((b, a, opt) for opt in lower_targets_for_oriented_edge(b, a))
                if not oriented_options:
                    continue

                for upper_inner, upper_outer, opt in oriented_options:
                    target_score, inner_target, outer_target, it_co, ot_co, it_down, ot_down = opt
                    ui_co = current[upper_inner]
                    uo_co = current[upper_outer]
                    avg_wz = (world_z(ui_co) + world_z(uo_co)) * 0.5
                    upper_ref = avg_wz
                    if face_center is not None:
                        upper_ref = max(upper_ref, world_z(face_center))
                    upper_band = abs(avg_wz - upper_ref)
                    # Choose the upper support edge, not the already-lower wedge
                    # handled by v0.5.61.  The selected 1140/1361 and mirrored
                    # 415/652 edge sits closest to face/ear-front-lower height.
                    face_bonus = 0.0
                    if face_center is not None:
                        dist_to_face = (abs(ui_co.x - face_center.x) + abs(ui_co.y - face_center.y) + abs(ui_co.z - face_center.z))
                        face_bonus = -dist_to_face * 0.12
                    score = target_score + avg_wz * 25.0 - upper_band * 60.0 + face_bonus
                    candidates.append((score, upper_inner, upper_outer, inner_target, outer_target, it_co, ot_co))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        _score, upper_inner, upper_outer, inner_target, outer_target, it_co, ot_co = candidates[0]

        if upper_inner not in moved_indices:
            verts[upper_inner].co = current[upper_inner].lerp(it_co, slide_strength)
            moved_indices.add(upper_inner)
        if upper_outer not in moved_indices:
            verts[upper_outer].co = current[upper_outer].lerp(ot_co, slide_strength)
            moved_indices.add(upper_outer)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_euwdsg"] = int(moved)
    except Exception:
        pass
    return moved


def apply_ear_front_face_edge_down_slide_guard(out_obj, records=None, slide_strength=0.32):
    """Slide the upper face/ear-front support edge downward.

    This targets the 881/1149-style selected edge and the mirrored 122/425-style
    edge without fixed vertex indices.  From the ear_front_lower anchor it finds
    the directly connected ear-front support endpoint, then detects the adjacent
    same-side face support endpoint whose lower partner forms a connected lower
    edge.  The selected-style edge is blended toward that lower adjacent edge,
    which is a vertex-slide-like relative movement along existing topology.
    """
    if out_obj is None or out_obj.type != 'MESH':
        return 0
    verts = out_obj.data.vertices
    vert_count = len(verts)
    if vert_count <= 0:
        return 0
    if records is None:
        try:
            records = anchor_records_for_template(out_obj)
        except Exception:
            records = []

    current = [v.co.copy() for v in verts]
    adj = build_mesh_adjacency(out_obj)
    rec_by_id = {rec.get("lm_id"): rec for rec in records or []}
    mw = out_obj.matrix_world.copy()

    all_anchor_members = set()
    for rec in records or []:
        all_anchor_members.update(_record_member_indices(rec, vert_count))

    def group_members(lm_id):
        group = out_obj.vertex_groups.get(anchor_group_name(lm_id))
        members = vertex_indices_in_group(out_obj, group)
        if not members:
            rec = rec_by_id.get(lm_id)
            if rec is not None:
                members = [(idx, 1.0) for idx in _record_member_indices(rec, vert_count)]
        return [(idx, weight) for idx, weight in members if 0 <= idx < vert_count]

    def group_centroid(lm_id):
        members = group_members(lm_id)
        if not members:
            return None
        acc = Vector((0.0, 0.0, 0.0))
        total = 0.0
        for idx, weight in members:
            w = max(float(weight), 0.0001)
            acc += current[idx] * w
            total += w
        if total <= 0.0:
            return None
        return acc / total

    def world_z(local_co):
        return float((mw @ local_co).z)

    slide_strength = max(0.0, min(float(slide_strength), 1.0))
    if slide_strength <= 0.0:
        return 0

    moved_indices = set()

    for side, sign in (("l", -1.0), ("r", 1.0)):
        ear_members = [idx for idx, _w in group_members(f"ear_{side}_front_lower")]
        ear_center = group_centroid(f"ear_{side}_front_lower")
        face_center = group_centroid(f"face_{side}_edge")
        jaw_center = group_centroid(f"jaw_{side}_edge")
        if not ear_members or ear_center is None:
            continue

        side_refs = [ear_center]
        if face_center is not None:
            side_refs.append(face_center)
        if jaw_center is not None:
            side_refs.append(jaw_center)
        for lm_id in (
            f"ear_{side}_front_middle",
            f"outer_face_{side}_upper",
            f"head_{side}_side_upper",
            f"head_{side}_side_back",
        ):
            p = group_centroid(lm_id)
            if p is not None:
                side_refs.append(p)
        side_width = max([abs(p.x) for p in side_refs] + [1.0e-6])

        down_eps = max(side_width * 0.00035, 0.0010)
        max_y_delta = max(side_width * 0.18, 0.95)
        max_z_delta = max(side_width * 0.18, 0.95)

        def same_side(idx):
            return 0 <= idx < vert_count and current[idx].x * sign > 0.0

        ear_support = set()
        for anchor_idx in ear_members:
            if anchor_idx < 0 or anchor_idx >= vert_count:
                continue
            for nb in adj[anchor_idx]:
                if 0 <= nb < vert_count and nb not in all_anchor_members and same_side(nb):
                    ear_support.add(nb)

        if not ear_support:
            continue

        candidates = []
        for outer_idx in ear_support:
            if outer_idx in all_anchor_members or not same_side(outer_idx):
                continue
            outer_co = current[outer_idx]
            outer_wz = world_z(outer_co)
            for inner_idx in adj[outer_idx]:
                if inner_idx < 0 or inner_idx >= vert_count:
                    continue
                if inner_idx in all_anchor_members or inner_idx in ear_support:
                    continue
                if not same_side(inner_idx):
                    continue
                inner_co = current[inner_idx]
                inner_wz = world_z(inner_co)

                if abs(outer_co.y - ear_center.y) > max_y_delta:
                    continue
                if abs(outer_co.z - ear_center.z) > max_z_delta:
                    continue
                if face_center is not None:
                    if abs(inner_co.y - face_center.y) > max_y_delta and abs(inner_co.y - ear_center.y) > max_y_delta:
                        continue
                    if abs(inner_co.z - face_center.z) > max_z_delta and abs(inner_co.z - ear_center.z) > max_z_delta:
                        continue

                for outer_target in adj[outer_idx]:
                    if outer_target < 0 or outer_target >= vert_count:
                        continue
                    if outer_target in all_anchor_members or outer_target in ear_support:
                        continue
                    if outer_target == inner_idx or not same_side(outer_target):
                        continue
                    outer_target_co = current[outer_target]
                    outer_down = outer_wz - world_z(outer_target_co)
                    if outer_down <= down_eps:
                        continue

                    for inner_target in adj[inner_idx]:
                        if inner_target < 0 or inner_target >= vert_count:
                            continue
                        if inner_target in all_anchor_members or inner_target in ear_support:
                            continue
                        if inner_target in (outer_idx, outer_target) or not same_side(inner_target):
                            continue
                        if outer_target not in adj[inner_target]:
                            continue
                        inner_target_co = current[inner_target]
                        inner_down = inner_wz - world_z(inner_target_co)
                        if inner_down <= down_eps:
                            continue

                        # Prefer the true lower adjacent quad edge rather than
                        # a diagonal fan jump or an already-lower ear wedge.
                        edge_parallel = abs((inner_co - outer_co).length - (inner_target_co - outer_target_co).length)
                        pair_drop = inner_down + outer_down
                        avg_wz = (inner_wz + outer_wz) * 0.5
                        face_bonus = 0.0
                        if face_center is not None:
                            face_dist = (
                                abs(inner_co.x - face_center.x) +
                                abs(inner_co.y - face_center.y) +
                                abs(inner_co.z - face_center.z)
                            )
                            face_bonus = -face_dist * 0.12
                        ear_bonus = -((outer_co - ear_center).length) * 0.08
                        score = pair_drop * 100.0 + avg_wz * 25.0 - edge_parallel * 0.45 + face_bonus + ear_bonus
                        candidates.append((
                            score,
                            inner_idx,
                            outer_idx,
                            inner_target,
                            outer_target,
                            inner_target_co,
                            outer_target_co,
                        ))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        _score, inner_idx, outer_idx, inner_target, outer_target, inner_target_co, outer_target_co = candidates[0]

        if inner_idx not in moved_indices:
            verts[inner_idx].co = current[inner_idx].lerp(inner_target_co, slide_strength)
            moved_indices.add(inner_idx)
        if outer_idx not in moved_indices:
            verts[outer_idx].co = current[outer_idx].lerp(outer_target_co, slide_strength)
            moved_indices.add(outer_idx)

    moved = len(moved_indices)
    if moved:
        out_obj.data.update()
    try:
        out_obj["HFR_effdsg"] = int(moved)
    except Exception:
        pass
    return moved

def write_generate_report(template, out_obj, target, anchors_used, snapped, missing=None, empty=None, quality_warnings=None, side_warnings=None, mirror_sync_count=0, final_force_snap=False, context=None, copy_to_clipboard=True):
    text = bpy.data.texts.get("HFR_Generate_Report")
    if text is None:
        text = bpy.data.texts.new("HFR_Generate_Report")
    text.clear()
    text.write("HFR Generate Retopology Report\n")
    text.write("Template Mesh: %s\n" % (template.name if template else "None"))
    text.write("Output Mesh: %s\n" % (out_obj.name if out_obj else "None"))
    text.write("Target Mesh: %s\n" % (target.name if target else "None"))
    text.write("Anchors Used: %d\n" % int(anchors_used))
    text.write("Snapped Vertices: %d\n" % int(snapped))
    text.write("Mirror Sync Before Generate: %d\n" % int(mirror_sync_count))
    text.write("Final Button Force Snap: %s\n" % ("ON" if final_force_snap else "OFF"))
    if out_obj is not None:
        text.write("Output Mirror Finish Vertices: %d\n" % int(out_obj.get("HFR_mfix", 0)))
        text.write("Output Mirror Direction: %s\n" % str(out_obj.get("HFR_mdir", "OFF")))
        text.write("Nose Web Fit Vertices: %d\n" % int(out_obj.get("HFR_nweb", 0)))
        text.write("Nose Web Surface Vertices: %d\n" % int(out_obj.get("HFR_npost", 0)))
        text.write("Nose Alar Fit Vertices: %d\n" % int(out_obj.get("HFR_nalar", 0)))
        text.write("Nose Alar Surface Vertices: %d\n" % int(out_obj.get("HFR_nalar_post", 0)))
        text.write("Brow Ridge Fit Vertices: %d\n" % int(out_obj.get("HFR_brow", 0)))
        text.write("Brow Ridge Smooth Vertices: %d\n" % int(out_obj.get("HFR_brow_sm", 0)))
        text.write("Brow Inner Support Vertices: %d\n" % int(out_obj.get("HFR_brinn", 0)))
        text.write("Eye Boundary Path Vertices: %d\n" % int(out_obj.get("HFR_eyefit", 0)))
        text.write("Eye Member Path Vertices: %d\n" % int(out_obj.get("HFR_eyepth", 0)))
        text.write("Eye Direct Loop Vertices: %d\n" % int(out_obj.get("HFR_eyedir", 0)))
        text.write("Debug Group Vertices: %d\n" % int(out_obj.get("HFR_dbg", 0)))
        text.write("Eye Topology Band Vertices: %d\n" % int(out_obj.get("HFR_eyeband", 0)))
        text.write("Eye Boundary Fit Vertices: %d\n" % int(out_obj.get("HFR_eyebnd", 0)))
        text.write("Eye Snap Guard Vertices: %d\n" % int(out_obj.get("HFR_eyesg", 0)))
        text.write("Side Face Snap Vertices: %d\n" % int(out_obj.get("HFR_sfsn", 0)))
        text.write("Head Back Snap Vertices: %d\n" % int(out_obj.get("HFR_hbsn", 0)))
        text.write("Ear Local Fit Vertices: %d\n" % int(out_obj.get("HFR_earlf", 0)))
        text.write("Ear Upper Support Vertices: %d\n" % int(out_obj.get("HFR_earup", 0)))
        text.write("Ear Inner-Lower Fan Vertices: %d\n" % int(out_obj.get("HFR_earfn", 0)))
        text.write("Ear Attachment Guard Vertices: %d\n" % int(out_obj.get("HFR_eagrd", 0)))
        text.write("Ear Upper Attachment Guard Vertices: %d\n" % int(out_obj.get("HFR_eagru", 0)))
        text.write("Ear Lower Fit Vertices: %d\n" % int(out_obj.get("HFR_earlo", 0)))
        text.write("Ear Lower Height Guard Vertices: %d\n" % int(out_obj.get("HFR_elhgt", 0)))
        text.write("Ear Lower World Clamp Vertices: %d\n" % int(out_obj.get("HFR_ellwc", 0)))
        text.write("Ear Inner Inward Guard Vertices: %d\n" % int(out_obj.get("HFR_eilig", 0)))
        text.write("Ear Inner Pocket Depth Guard Vertices: %d\n" % int(out_obj.get("HFR_eipdg", 0)))
        text.write("Ear Inner Sheet Outward Guard Vertices: %d\n" % int(out_obj.get("HFR_eisog", 0)))
        text.write("Ear Lower Front Height Guard Vertices: %d\n" % int(out_obj.get("HFR_elfhg", 0)))
        text.write("Ear Lower Front Inset Guard Vertices: %d\n" % int(out_obj.get("HFR_elfig", 0)))
        text.write("Ear Lower Nape Blend Guard Vertices: %d\n" % int(out_obj.get("HFR_elnbg", 0)))
        text.write("Ear Lobe Upper Lift Guard Vertices: %d\n" % int(out_obj.get("HFR_elulg", 0)))
        text.write("Ear Inner Lower Z Slide Guard Vertices: %d\n" % int(out_obj.get("HFR_eilzsg", 0)))
        text.write("Ear Lower Front Z Slide Guard Vertices: %d\n" % int(out_obj.get("HFR_elfzsg", 0)))
        text.write("Ear Lobe Upper Z Slide Guard Vertices: %d\n" % int(out_obj.get("HFR_eluzsg", 0)))
        text.write("Back Center Inward Slide Vertices: %d\n" % int(out_obj.get("HFR_bcisg", 0)))
        text.write("Back Outer Inward Slide Vertices: %d\n" % int(out_obj.get("HFR_boisg", 0)))
        text.write("Ear Opposite Slide Vertices: %d\n" % int(out_obj.get("HFR_eopsg", 0)))
        text.write("Ear Opposite Mirror Align Vertices: %d\n" % int(out_obj.get("HFR_eomsg", 0)))
        text.write("Ear Toward Strip Slide Vertices: %d\n" % int(out_obj.get("HFR_etsg", 0)))
        text.write("Ear Inner Toward Slide Vertices: %d\n" % int(out_obj.get("HFR_eitsg", 0)))
        text.write("Jaw Ear Down Slide Vertices: %d\n" % int(out_obj.get("HFR_jedsg", 0)))
        text.write("Ear Front Lower Down Slide Vertices: %d\n" % int(out_obj.get("HFR_efldsg", 0)))
        text.write("Ear Front Lower Upper Down Slide Vertices: %d\n" % int(out_obj.get("HFR_eflusg", 0)))
        text.write("Ear Face Edge Upper Down Slide Vertices: %d\n" % int(out_obj.get("HFR_efedsg", 0)))
        text.write("Ear Face Edge Toward Slide Vertices: %d\n" % int(out_obj.get("HFR_efetsg", 0)))
        text.write("Ear Lower Wedge Down Slide Vertices: %d\n" % int(out_obj.get("HFR_elwdsg", 0)))
        text.write("Ear Upper Wedge Down Slide Vertices: %d\n" % int(out_obj.get("HFR_euwdsg", 0)))
        text.write("Ear Front Face Edge Down Slide Vertices: %d\n" % int(out_obj.get("HFR_effdsg", 0)))
    if quality_warnings:
        text.write("\nQuality Warnings\n")
        for warning in quality_warnings:
            text.write("- %s\n" % warning)
    if side_warnings:
        text.write("\nSide Binding Warnings\n")
        for warning in side_warnings:
            text.write("- %s\n" % warning)
    if missing:
        text.write("\nMissing Groups\n")
        for lm_id in missing:
            text.write("- %s\n" % anchor_group_name(lm_id))
    if empty:
        text.write("\nEmpty Groups\n")
        for lm_id in empty:
            text.write("- %s\n" % anchor_group_name(lm_id))
    if context is not None and copy_to_clipboard:
        try:
            context.window_manager.clipboard = text.as_string()
        except Exception:
            pass
    return text

# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class HFR_OT_AddLandmarks(bpy.types.Operator):
    bl_idname = "hfr.add_landmarks"
    bl_label = "Add HFR Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    group: EnumProperty(
        name="Group",
        items=[
            ('ALL', "All", "Add all template landmarks"),
            ('FACE', "Face", "Add all non-ear landmarks"),
            ('EYE', "Eyes", "Add eye landmarks"),
            ('MOUTH', "Mouth", "Add mouth landmarks"),
            ('NOSE', "Nose", "Add nose landmarks"),
            ('SCALP', "Scalp", "Add forehead/scalp landmarks"),
            ('EAR', "Ears", "Add ear landmarks"),
            ('NECK', "Neck", "Add neck/nape landmarks"),
        ],
        default='ALL',
    )

    reset_existing: BoolProperty(
        name="Reset Existing",
        default=False,
        description="Move existing landmarks back to their default positions",
    )

    def execute(self, context):
        ensure_base_collections()
        cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
        count = 0
        # Add Landmark buttons now always use the built-in default landmark
        # coordinates. Target fitting remains available only through the
        # Initial Placement > Fit ... operators, so Add All Landmarks returns to
        # the exported default placement instead of the older full-size layout.
        for lm in matching_landmarks_for_group(self.group):
            create_or_update_landmark(context.scene, lm, reset=self.reset_existing, context=context, fit_to_target=False)
            count += 1
        refresh_all_guides(recreate=False, scene=context.scene, context=context)
        self.report({'INFO'}, f"Added/updated {count} HFR landmarks")
        return {'FINISHED'}


class HFR_OT_ResetLandmarks(bpy.types.Operator):
    bl_idname = "hfr.reset_landmarks"
    bl_label = "Reset HFR Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    group: EnumProperty(
        name="Group",
        items=[
            ('ALL', "All", "Reset all landmarks"),
            ('FACE', "Face", "Reset all non-ear landmarks"),
            ('EAR', "Ears", "Reset ear landmarks"),
        ],
        default='ALL',
    )

    use_target_fit: BoolProperty(
        name="Use Target Fit",
        description="Use Initial Placement > Fit On Add/Reset when resetting landmarks",
        default=True,
        options={'HIDDEN'},
    )

    def execute(self, context):
        cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
        count = 0
        fit_to_target = bool(self.use_target_fit and context.scene.hfr_lm_use_target_fit)
        for lm in matching_landmarks_for_group(self.group):
            create_or_update_landmark(context.scene, lm, reset=True, context=context, fit_to_target=fit_to_target)
            count += 1
        refresh_all_guides(recreate=False, scene=context.scene, context=context)
        self.report({'INFO'}, f"Reset {count} HFR landmarks")
        return {'FINISHED'}


class HFR_OT_DeleteLandmarks(bpy.types.Operator):
    bl_idname = "hfr.delete_landmarks"
    bl_label = "Delete HFR Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    delete_guides: BoolProperty(name="Delete Guides", default=True)

    def execute(self, context):
        count = 0
        for obj in list(bpy.data.objects):
            if obj.get(PID_LM) or (self.delete_guides and (obj.get(PID_GUIDE) or obj.get(PID_BIND_GUIDE))):
                bpy.data.objects.remove(obj, do_unlink=True)
                count += 1
        self.report({'INFO'}, f"Deleted {count} HFR objects")
        return {'FINISHED'}


class HFR_OT_RefreshGuides(bpy.types.Operator):
    bl_idname = "hfr.refresh_guides"
    bl_label = "Refresh HFR Guides"
    bl_options = {'REGISTER', 'UNDO'}

    recreate: BoolProperty(name="Recreate", default=False)

    def execute(self, context):
        count = refresh_all_guides(recreate=self.recreate, scene=context.scene, context=context)
        self.report({'INFO'}, f"Refreshed {count} HFR guide lines")
        return {'FINISHED'}


class HFR_OT_MirrorLandmarks(bpy.types.Operator):
    bl_idname = "hfr.mirror_landmarks"
    bl_label = "Mirror HFR Landmarks"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        name="Direction",
        items=[
            ('L2R', "L -> R", "Mirror left landmarks to right landmarks"),
            ('R2L', "R -> L", "Mirror right landmarks to left landmarks"),
        ],
        default='L2R',
    )

    def execute(self, context):
        count = apply_mirror(self.direction)
        refresh_all_guides(recreate=False, scene=context.scene, context=context)
        self.report({'INFO'}, f"Mirrored {count} HFR landmarks")
        return {'FINISHED'}


class HFR_OT_SaveLandmarkDefaults(bpy.types.Operator):
    bl_idname = "hfr.save_lm_defaults"
    bl_label = "Save Landmark Position"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = save_landmark_defaults_to_scene(context.scene)
        self.report({'INFO'}, f"Saved {count} landmark positions to scene")
        return {'FINISHED'}


class HFR_OT_LoadLandmarkDefaults(bpy.types.Operator):
    bl_idname = "hfr.load_lm_defaults"
    bl_label = "Load Landmark Position"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = load_landmark_defaults_from_scene(context.scene)
        refresh_all_guides(recreate=False, scene=context.scene, context=context)
        self.report({'INFO'}, f"Loaded {count} landmark positions from scene")
        return {'FINISHED'}


class HFR_OT_ExportLandmarkPositions(bpy.types.Operator):
    bl_idname = "hfr.export_lm_positions"
    bl_label = "Export Landmark Position"
    bl_options = {'REGISTER'}

    def execute(self, context):
        text, payload = write_landmark_position_export_text(context)
        missing = len(payload.get("missing_landmarks", []))
        count = int(payload.get("landmark_count", 0))
        if missing:
            self.report({'WARNING'}, f"Exported {count} landmark positions to {text.name} and clipboard; {missing} fallback defaults used")
        else:
            self.report({'INFO'}, f"Exported {count} landmark positions to {text.name} and clipboard")
        return {'FINISHED'}


class HFR_OT_ExportSelectedVertices(bpy.types.Operator):
    bl_idname = "hfr.export_selected_vertices"
    bl_label = "Export Selected Vertices"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        selected = []
        mode = obj.mode
        try:
            if mode == 'EDIT':
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                selected = [v.index for v in bm.verts if v.select]
            else:
                selected = [v.index for v in obj.data.vertices if v.select]
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        coords = []
        for idx in selected:
            if 0 <= idx < len(obj.data.vertices):
                co = obj.data.vertices[idx].co
                coords.append([idx, float(co.x), float(co.y), float(co.z)])
        payload = {
            "object": obj.name,
            "mode": mode,
            "selected_count": len(selected),
            "indices": selected,
            "local_coords": coords,
        }
        text = bpy.data.texts.get("HFR_Selected_Vertex_Export")
        if text is None:
            text = bpy.data.texts.new("HFR_Selected_Vertex_Export")
        text.clear()
        blob = json.dumps(payload, indent=2, ensure_ascii=False)
        text.write(blob)
        try:
            context.window_manager.clipboard = blob
        except Exception:
            pass
        self.report({'INFO'}, f"Exported {len(selected)} selected vertices to {text.name} and clipboard")
        return {'FINISHED'}




class HFR_OT_ExportMeshVertexDiagnostic(bpy.types.Operator):
    bl_idname = "hfr.export_mesh_vertex_diagnostic"
    bl_label = "Export Mesh Vertex Diagnostic"
    bl_options = {'REGISTER'}

    def _round_vec(self, vec, digits=6):
        return [round(float(vec.x), digits), round(float(vec.y), digits), round(float(vec.z), digits)]

    def _vertex_group_items(self, obj, idx):
        if idx < 0 or idx >= len(obj.data.vertices):
            return []
        items = []
        for item in obj.data.vertices[idx].groups:
            try:
                vg = obj.vertex_groups[item.group]
                name = vg.name
            except Exception:
                name = str(item.group)
            items.append({"name": name, "weight": round(float(item.weight), 6)})
        return items

    def _active_group_name(self, obj):
        try:
            group = obj.vertex_groups.active
            return group.name if group else ""
        except Exception:
            return ""

    def _collect_mesh_snapshot(self, context, obj):
        mode = obj.mode
        mw = obj.matrix_world.copy()
        normal_mtx = mw.to_3x3()
        vertices = []
        edges = []
        faces = []
        selected_vertices = set()
        selected_edges = set()
        selected_faces = set()

        if mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            try:
                bm.verts.index_update()
                bm.edges.index_update()
                bm.faces.index_update()
            except Exception:
                pass
            for v in bm.verts:
                idx = int(v.index)
                if v.select:
                    selected_vertices.add(idx)
                wco = mw @ v.co
                try:
                    wn = (normal_mtx @ v.normal).normalized()
                except Exception:
                    wn = v.normal.copy()
                vertices.append({
                    "i": idx,
                    "selected": bool(v.select),
                    "co": self._round_vec(v.co),
                    "world_co": self._round_vec(wco),
                    "normal": self._round_vec(v.normal),
                    "world_normal": self._round_vec(wn),
                    "groups": self._vertex_group_items(obj, idx),
                })
            for e in bm.edges:
                idx = int(e.index)
                if e.select:
                    selected_edges.add(idx)
                edges.append({
                    "i": idx,
                    "verts": [int(e.verts[0].index), int(e.verts[1].index)],
                    "selected": bool(e.select),
                })
            for f in bm.faces:
                idx = int(f.index)
                if f.select:
                    selected_faces.add(idx)
                faces.append({
                    "i": idx,
                    "verts": [int(v.index) for v in f.verts],
                    "selected": bool(f.select),
                    "normal": self._round_vec(f.normal),
                })
        else:
            mesh = obj.data
            for v in mesh.vertices:
                idx = int(v.index)
                if v.select:
                    selected_vertices.add(idx)
                wco = mw @ v.co
                try:
                    wn = (normal_mtx @ v.normal).normalized()
                except Exception:
                    wn = v.normal.copy()
                vertices.append({
                    "i": idx,
                    "selected": bool(v.select),
                    "co": self._round_vec(v.co),
                    "world_co": self._round_vec(wco),
                    "normal": self._round_vec(v.normal),
                    "world_normal": self._round_vec(wn),
                    "groups": self._vertex_group_items(obj, idx),
                })
            for e in mesh.edges:
                idx = int(e.index)
                if e.select:
                    selected_edges.add(idx)
                edges.append({
                    "i": idx,
                    "verts": [int(e.vertices[0]), int(e.vertices[1])],
                    "selected": bool(e.select),
                })
            for p in mesh.polygons:
                idx = int(p.index)
                if p.select:
                    selected_faces.add(idx)
                faces.append({
                    "i": idx,
                    "verts": [int(v) for v in p.vertices],
                    "selected": bool(p.select),
                    "normal": self._round_vec(p.normal),
                })

        vertices.sort(key=lambda item: item["i"])
        edges.sort(key=lambda item: item["i"])
        faces.sort(key=lambda item: item["i"])
        return vertices, edges, faces, selected_vertices, selected_edges, selected_faces

    def execute(self, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            obj = getattr(context.scene, "hfr_template_obj", None)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first or assign Template Mesh")
            return {'CANCELLED'}

        try:
            vertices, edges, faces, selected_vertices, selected_edges, selected_faces = self._collect_mesh_snapshot(context, obj)
        except Exception as exc:
            self.report({'ERROR'}, "Mesh diagnostic export failed: %s" % exc)
            return {'CANCELLED'}

        neighbor_map = {item["i"]: set() for item in vertices}
        face_map = {item["i"]: [] for item in vertices}
        for e in edges:
            a, b = e["verts"]
            if a in neighbor_map:
                neighbor_map[a].add(b)
            if b in neighbor_map:
                neighbor_map[b].add(a)
        for f in faces:
            for idx in f["verts"]:
                if idx in face_map:
                    face_map[idx].append(f["i"])
        for item in vertices:
            idx = item["i"]
            item["neighbors"] = sorted(neighbor_map.get(idx, []))
            item["faces"] = sorted(face_map.get(idx, []))

        selected_one_ring = set(selected_vertices)
        for idx in list(selected_vertices):
            selected_one_ring.update(neighbor_map.get(idx, set()))

        anchor_groups = {}
        for group in obj.vertex_groups:
            name = group.name
            if not name.startswith(ANCHOR_GROUP_PREFIX):
                continue
            members = []
            for v in obj.data.vertices:
                for item in v.groups:
                    if item.group == group.index:
                        members.append({"i": int(v.index), "weight": round(float(item.weight), 6)})
                        break
            if members:
                anchor_groups[name] = members

        payload = {
            "hfr_export_type": "mesh_vertex_diagnostic",
            "addon_version": [1, 0, 0],
            "object": obj.name,
            "object_type": obj.type,
            "mode": obj.mode,
            "is_hfr_output": bool(obj.get(PID_OUTPUT)),
            "is_hfr_template": bool(obj.get(PID_TEMPLATE)),
            "active_vertex_group": self._active_group_name(obj),
            "vertex_count": len(vertices),
            "edge_count": len(edges),
            "face_count": len(faces),
            "selected": {
                "vertex_count": len(selected_vertices),
                "vertices": sorted(selected_vertices),
                "edge_count": len(selected_edges),
                "edges": sorted(selected_edges),
                "face_count": len(selected_faces),
                "faces": sorted(selected_faces),
                "one_ring_vertices": sorted(selected_one_ring),
            },
            "selected_vertex_details": [item for item in vertices if item["i"] in selected_vertices],
            "one_ring_vertex_details": [item for item in vertices if item["i"] in selected_one_ring],
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "hfr_anchor_groups": anchor_groups,
        }

        blob = json.dumps(payload, indent=2, ensure_ascii=False)
        text = bpy.data.texts.get("HFR_Mesh_Vertex_Diagnostic")
        if text is None:
            text = bpy.data.texts.new("HFR_Mesh_Vertex_Diagnostic")
        text.clear()
        text.write(blob)
        try:
            context.window_manager.clipboard = blob
            clip_msg = " and clipboard"
        except Exception:
            clip_msg = ""
        self.report({'INFO'}, "Exported %d vertices / %d selected to %s%s" % (len(vertices), len(selected_vertices), text.name, clip_msg))
        return {'FINISHED'}




class HFR_OT_ExportGenerateReport(bpy.types.Operator):
    bl_idname = "hfr.export_generate_report"
    bl_label = "Export Generate Report"
    bl_options = {'REGISTER'}

    def execute(self, context):
        text = bpy.data.texts.get("HFR_Generate_Report")
        if text is None or not text.as_string().strip():
            self.report({'ERROR'}, "Run Generate Retopology first; HFR_Generate_Report is empty")
            return {'CANCELLED'}
        blob = text.as_string()
        clip_msg = ""
        try:
            context.window_manager.clipboard = blob
            clip_msg = " and clipboard"
        except Exception:
            pass
        self.report({'INFO'}, f"Exported Generate Report from {text.name}{clip_msg}")
        return {'FINISHED'}


class HFR_OT_FitLandmarksToTarget(bpy.types.Operator):
    bl_idname = "hfr.fit_landmarks_to_target"
    bl_label = "Fit HFR Landmarks To Target"
    bl_options = {'REGISTER', 'UNDO'}

    group: EnumProperty(
        name="Group",
        items=[
            ('ALL', "All", "Fit all landmarks to the target mesh bounds"),
            ('FACE', "Face", "Fit all non-ear landmarks"),
            ('EAR', "Ears", "Fit ear landmarks"),
            ('EYE', "Eyes", "Fit eye landmarks"),
            ('MOUTH', "Mouth", "Fit mouth landmarks"),
            ('NOSE', "Nose", "Fit nose landmarks"),
            ('SCALP', "Scalp", "Fit forehead/scalp landmarks"),
            ('NECK', "Neck", "Fit neck/nape landmarks"),
        ],
        default='ALL',
    )

    def execute(self, context):
        target = fit_target_object(context)
        if not target:
            self.report({'ERROR'}, "Assign Target Mesh or select a mesh object first")
            return {'CANCELLED'}
        ensure_base_collections()
        cleanup_removed_landmarks_and_guides(remove_unused_guides=True)
        count = 0
        for lm in matching_landmarks_for_group(self.group):
            create_or_update_landmark(
                context.scene,
                lm,
                reset=True,
                context=context,
                fit_to_target=True,
            )
            count += 1
        refresh_all_guides(recreate=False, scene=context.scene, context=context)
        self.report({'INFO'}, f"Fitted {count} HFR landmarks to {target.name}")
        return {'FINISHED'}


class HFR_OT_LoadDefaultTemplate(bpy.types.Operator):
    bl_idname = "hfr.load_default_template"
    bl_label = "Reload Default Template"
    bl_options = {'REGISTER', 'UNDO'}

    replace_existing: BoolProperty(
        name="Replace Existing",
        description="Remove already loaded default template objects before appending the bundled template",
        default=False,
        options={'HIDDEN'},
    )

    def execute(self, context):
        try:
            obj, state, removed = load_default_template_asset(context, replace=self.replace_existing)
            missing, empty, bound = binding_status_summary(obj)
            if missing or empty:
                self.report(
                    {'WARNING'},
                    "Default template %s: %s, %d/%d anchors bound, %d missing, %d empty"
                    % (state, obj.name, len(bound), len(LANDMARKS), len(missing), len(empty))
                )
            else:
                msg = "Default template %s: %s, %d anchors ready" % (state, obj.name, len(bound))
                if removed:
                    msg += "; removed %d old template(s)" % removed
                self.report({'INFO'}, msg)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class HFR_OT_InitializeWorkspace(bpy.types.Operator):
    bl_idname = "hfr.initialize_workspace"
    bl_label = "Initialize HFR Workspace"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        try:
            ensure_base_collections()
            removed, moved, fixed = cleanup_hfr_collections(remove_unknown_hfr_named=True)
            obj, state, tpl_removed = load_default_template_asset(context, replace=False, select_loaded=True)
            styled = apply_landmark_style_and_guides(scene, context)
            missing, empty, bound = binding_status_summary(obj)
            if missing or empty:
                self.report(
                    {'WARNING'},
                    "Workspace ready; template binding incomplete: %d missing, %d empty. Enable DevOption > Template Binding to inspect."
                    % (len(missing), len(empty))
                )
            else:
                self.report(
                    {'INFO'},
                    "Workspace ready: template %s, %d anchors bound, cleanup %d/%d/%d, styled %d"
                    % (state, len(bound), removed + tpl_removed, moved, fixed, styled)
                )
        except Exception as exc:
            self.report({'ERROR'}, "Initialize HFR Workspace failed: %s" % exc)
            return {'CANCELLED'}
        return {'FINISHED'}


class HFR_OT_CreateAnchorGroups(bpy.types.Operator):
    bl_idname = "hfr.create_anchor_groups"
    bl_label = "Create Template Anchor Groups"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Assign Template Mesh or select the template mesh object first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        removed = 0
        for lm_id in sorted(OBSOLETE_ANCHOR_IDS):
            group = obj.vertex_groups.get(anchor_group_name(lm_id))
            if group is not None:
                obj.vertex_groups.remove(group)
                removed += 1
        count = 0
        for lm in LANDMARKS:
            name = anchor_group_name(lm["id"])
            if obj.vertex_groups.get(name) is None:
                obj.vertex_groups.new(name=name)
                count += 1
        if removed:
            self.report({'INFO'}, f"Anchor groups ready on {obj.name}; created {count}, removed {removed} obsolete")
        else:
            self.report({'INFO'}, f"Anchor groups ready on {obj.name}; created {count}")
        return {'FINISHED'}


class HFR_OT_SetBindingLandmarkFromSelection(bpy.types.Operator):
    bl_idname = "hfr.set_binding_lm_from_selection"
    bl_label = "Use Selected Landmark"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        lm_id = active_or_selected_landmark_id(context)
        if lm_id not in LM_BY_ID:
            self.report({'ERROR'}, "Select one HFR landmark object first")
            return {'CANCELLED'}
        context.scene.hfr_bind_lm_id = lm_id
        self.report({'INFO'}, f"Active binding landmark: LM_{lm_id}")
        return {'FINISHED'}


class HFR_OT_SetActiveBindingLandmark(bpy.types.Operator):
    bl_idname = "hfr.set_active_binding_landmark"
    bl_label = "Set Active Binding Landmark"
    bl_options = {'REGISTER', 'UNDO'}

    lm_id: StringProperty(name="Landmark ID", default="")

    def execute(self, context):
        if self.lm_id not in LM_BY_ID:
            self.report({'ERROR'}, "Invalid HFR landmark ID")
            return {'CANCELLED'}
        context.scene.hfr_bind_lm_id = self.lm_id
        self.report({'INFO'}, f"Active binding landmark: LM_{self.lm_id}")
        return {'FINISHED'}


class HFR_OT_NextUnboundLandmark(bpy.types.Operator):
    bl_idname = "hfr.next_unbound_landmark"
    bl_label = "Next Unbound"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        missing, empty, _bound = binding_status_summary(obj)
        unbound = set(missing) | set(empty)
        if not unbound:
            self.report({'INFO'}, "All HFR landmarks are bound")
            return {'FINISHED'}
        order = [lm["id"] for lm in LANDMARKS]
        current = getattr(context.scene, "hfr_bind_lm_id", "")
        start = order.index(current) if current in order else -1
        for step in range(1, len(order) + 1):
            lm_id = order[(start + step) % len(order)]
            if lm_id in unbound:
                context.scene.hfr_bind_lm_id = lm_id
                self.report({'INFO'}, f"Next unbound landmark: LM_{lm_id}")
                return {'FINISHED'}
        return {'FINISHED'}


class HFR_OT_BindSelectedVertices(bpy.types.Operator):
    bl_idname = "hfr.bind_selected_vertices"
    bl_label = "Bind Selected Vertices"
    bl_options = {'REGISTER', 'UNDO'}

    replace_existing: BoolProperty(
        name="Replace Existing",
        description="Clear this landmark's current anchor group before assigning selected vertices",
        default=True,
    )

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        lm_id = getattr(context.scene, "hfr_bind_lm_id", "")
        if lm_id not in LM_BY_ID:
            lm_id = active_or_selected_landmark_id(context)
            context.scene.hfr_bind_lm_id = lm_id
        indices = selected_template_vertex_indices(context, obj)
        if not indices:
            self.report({'ERROR'}, "Select one or more vertices on the Template Mesh first")
            return {'CANCELLED'}
        try:
            count = bind_vertices_to_landmark(context, obj, lm_id, indices, replace=self.replace_existing)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        refresh_binding_guides(scene=context.scene, context=context)
        self.report({'INFO'}, f"Bound {count} vertices to {anchor_group_name(lm_id)}")
        return {'FINISHED'}


class HFR_OT_ClearAnchorBinding(bpy.types.Operator):
    bl_idname = "hfr.clear_anchor_binding"
    bl_label = "Clear Active Binding"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        lm_id = getattr(context.scene, "hfr_bind_lm_id", "")
        if lm_id not in LM_BY_ID:
            self.report({'ERROR'}, "Choose Active Landmark first")
            return {'CANCELLED'}
        count = clear_anchor_binding(context, obj, lm_id)
        refresh_binding_guides(scene=context.scene, context=context)
        self.report({'INFO'}, f"Cleared {count} vertices from {anchor_group_name(lm_id)}")
        return {'FINISHED'}


class HFR_OT_SelectAnchorVertices(bpy.types.Operator):
    bl_idname = "hfr.select_anchor_vertices"
    bl_label = "Select Bound Vertices"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        lm_id = getattr(context.scene, "hfr_bind_lm_id", "")
        if lm_id not in LM_BY_ID:
            self.report({'ERROR'}, "Choose Active Landmark first")
            return {'CANCELLED'}
        count = select_anchor_vertices(context, obj, lm_id)
        self.report({'INFO'}, f"Selected {count} vertices in {anchor_group_name(lm_id)}")
        return {'FINISHED'}


class HFR_OT_MirrorAnchorGroups(bpy.types.Operator):
    bl_idname = "hfr.mirror_anchor_groups"
    bl_label = "Mirror Anchor Groups"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        name="Direction",
        items=[
            ('L2R', "L -> R", "Mirror left anchor groups to right anchor groups"),
            ('R2L', "R -> L", "Mirror right anchor groups to left anchor groups"),
        ],
        default='L2R',
    )

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        try:
            groups, verts, misses = mirror_anchor_groups(
                context,
                obj,
                direction=self.direction,
                tolerance=float(getattr(context.scene, "hfr_bind_mirror_tol", 0.01)),
            )
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Mirrored {groups} groups / {verts} vertices; misses {misses}")
        return {'FINISHED'}


class HFR_OT_ExportTemplateBinding(bpy.types.Operator):
    bl_idname = "hfr.export_template_binding"
    bl_label = "Export Template Binding"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            obj = getattr(context.scene, "hfr_template_obj", None)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        try:
            payload, text = write_template_binding_export(context, obj)
        except Exception as exc:
            self.report({'ERROR'}, "Template binding export failed: %s" % exc)
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported %d bound groups to %s and clipboard" % (int(payload.get("bound_count", 0)), text.name))
        return {'FINISHED'}


class HFR_OT_ImportTemplateBinding(bpy.types.Operator):
    bl_idname = "hfr.import_template_binding"
    bl_label = "Import Template Binding"
    bl_options = {'REGISTER', 'UNDO'}

    replace_existing: BoolProperty(
        name="Replace Existing",
        default=True,
        description="Clear each imported HFR_A_* group before applying the imported vertex weights",
    )

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            obj = getattr(context.scene, "hfr_template_obj", None)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        try:
            payload = _binding_payload_from_context(context)
            imported, skipped = import_template_binding_payload(context, obj, payload, replace_existing=self.replace_existing)
        except Exception as exc:
            self.report({'ERROR'}, "Template binding import failed: %s" % exc)
            return {'CANCELLED'}
        self.report({'INFO'}, "Imported %d binding groups; skipped %d invalid item(s)" % (imported, skipped))
        return {'FINISHED'}


class HFR_OT_ValidateTemplateBinding(bpy.types.Operator):
    bl_idname = "hfr.validate_template_binding"
    bl_label = "Validate Template Binding"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = template_object(context)
        if obj is None:
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        context.scene.hfr_template_obj = obj
        try:
            missing, empty, bound = validate_template_binding(context, obj)
            quality_warnings = binding_quality_warnings(obj)
            side_warnings = binding_side_warnings(obj)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if quality_warnings or side_warnings:
            self.report({'WARNING'}, f"Binding report: {len(quality_warnings)} quality / {len(side_warnings)} side warning(s). See HFR_Template_Binding_Report")
        else:
            self.report({'INFO'}, f"Binding report: bound {len(bound)}, missing {len(missing)}, empty {len(empty)}")
        return {'FINISHED'}


class HFR_OT_RefreshBindingGuides(bpy.types.Operator):
    bl_idname = "hfr.refresh_binding_guides"
    bl_label = "Refresh Binding Guides"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not binding_mode_enabled(context.scene):
            self.report({'INFO'}, "Binding Mode is off; enable it to show binding guides")
            return {'FINISHED'}
        count = refresh_binding_guides(scene=context.scene, context=context)
        self.report({'INFO'}, f"Refreshed {count} template binding guides")
        return {'FINISHED'}


class HFR_OT_UpdateStyle(bpy.types.Operator):
    bl_idname = "hfr.update_lm_style"
    bl_label = "Update HFR Landmark Style"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = apply_landmark_style_and_guides(context.scene, context)
        self.report({'INFO'}, f"Updated style for {count} landmarks")
        return {'FINISHED'}



class HFR_OT_GenerateRetopology(bpy.types.Operator):
    bl_idname = "hfr.generate_retopology"
    bl_label = "Generate Retopology"
    bl_options = {'REGISTER', 'UNDO'}

    force_no_snap: BoolProperty(
        name="Force No Snap",
        description="Generate only the landmark deformation preview, without surface snapping",
        default=False,
        options={'HIDDEN'},
    )

    force_snap_to_target: BoolProperty(
        name="Force Snap To Target",
        description="For the final Generate Retopology button, snap to Target Mesh even when the developer Snap option is off",
        default=False,
        options={'HIDDEN'},
    )

    def execute(self, context):
        scene = context.scene
        template = template_object(context)
        if template is None or template.type != 'MESH':
            self.report({'ERROR'}, "Assign Template Mesh first")
            return {'CANCELLED'}
        scene.hfr_template_obj = template
        mirror_sync_count = force_landmark_mirror_sync(scene, context)

        missing, empty, bound = binding_status_summary(template)
        if missing or empty:
            write_generate_report(template, None, generate_target_object(context), 0, 0, missing=missing, empty=empty, mirror_sync_count=mirror_sync_count, context=context)
            self.report({'ERROR'}, f"Template binding incomplete: {len(missing)} missing, {len(empty)} empty. See HFR_Generate_Report")
            return {'CANCELLED'}

        target = generate_target_object(context)
        force_snap = bool(getattr(self, "force_snap_to_target", False))
        snap_enabled = (bool(getattr(scene, "hfr_gen_snap_to_target", False)) or force_snap) and not bool(getattr(self, "force_no_snap", False))
        if snap_enabled and (target is None or target.type != 'MESH'):
            self.report({'ERROR'}, "Snap To Target is enabled for final generation, but Target Mesh is not assigned")
            return {'CANCELLED'}
        if target == template:
            self.report({'ERROR'}, "Target Mesh and Template Mesh must be different objects")
            return {'CANCELLED'}

        quality_warnings = binding_quality_warnings(template)
        side_warnings = binding_side_warnings(template)
        if quality_warnings:
            # Do not block generation, but write the warning into the generate report later.
            pass

        try:
            if template.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        try:
            out_obj = make_retopo_output_from_template(
                context,
                template,
                output_name=getattr(scene, "hfr_gen_output_name", "HFR_Retopo"),
                replace=bool(getattr(scene, "hfr_gen_replace_output", True)),
                display_wire=bool(getattr(scene, "hfr_gen_output_wire", True)),
                show_in_front=bool(getattr(scene, "hfr_gen_output_in_front", False)),
            )
            hfr_source_positions = [v.co.copy() for v in out_obj.data.vertices]
            anchors_used = deform_template_output_to_landmarks(
                out_obj,
                power=float(getattr(scene, "hfr_gen_power", 2.0)),
                nearest_count=int(getattr(scene, "hfr_gen_nearest", 12)),
                anchor_lock=float(getattr(scene, "hfr_gen_anchor_lock", 1.0)),
                anchor_iters=int(getattr(scene, "hfr_gen_anchor_iters", 2)),
                topo_propagate=bool(getattr(scene, "hfr_gen_topo_propagate", True)),
                topo_iters=int(getattr(scene, "hfr_gen_topo_iters", 36)),
                topo_strength=float(getattr(scene, "hfr_gen_topo_strength", 0.65)),
                guide_rails=bool(getattr(scene, "hfr_gen_guide_rails", True)),
                guide_rail_strength=float(getattr(scene, "hfr_gen_guide_rail_strength", 1.0)),
                guide_rail_max_len=int(getattr(scene, "hfr_gen_guide_rail_max_len", 80)),
                guide_rail_spread=bool(getattr(scene, "hfr_gen_guide_rail_spread", True)),
                guide_rail_spread_steps=int(getattr(scene, "hfr_gen_guide_rail_spread_steps", 1)),
                guide_rail_spread_strength=float(getattr(scene, "hfr_gen_guide_rail_spread_strength", 0.65)),
                mls_field=bool(getattr(scene, "hfr_gen_mls_field", True)),
                mls_strength=float(getattr(scene, "hfr_gen_mls_strength", 0.75)),
                mls_nearest=int(getattr(scene, "hfr_gen_mls_nearest", 18)),
                guide_follow=bool(getattr(scene, "hfr_gen_guide_follow", True)),
                guide_strength=float(getattr(scene, "hfr_gen_guide_strength", 0.55)),
                guide_radius=float(getattr(scene, "hfr_gen_guide_radius", 1.10)),
                nose_web_fit=bool(getattr(scene, "hfr_gen_nose_web_fit", True)),
                nose_strength=float(getattr(scene, "hfr_gen_nose_web_strength", 1.0)),
                nose_radius=float(getattr(scene, "hfr_gen_nose_web_radius", 0.60)),
                nose_samples=int(getattr(scene, "hfr_gen_nose_web_samples", 24)),
                nose_alar_fit=bool(getattr(scene, "hfr_gen_nose_alar_fit", True)),
                alar_strength=float(getattr(scene, "hfr_gen_nose_alar_strength", 0.85)),
                alar_radius=float(getattr(scene, "hfr_gen_nose_alar_radius", 1.0)),
                alar_samples=int(getattr(scene, "hfr_gen_nose_alar_samples", 12)),
                brow_ridge_fit=bool(getattr(scene, "hfr_gen_brow_ridge_fit", True)),
                brow_strength=float(getattr(scene, "hfr_gen_brow_ridge_strength", 0.80)),
                brow_radius=float(getattr(scene, "hfr_gen_brow_ridge_radius", 1.15)),
                brow_samples=int(getattr(scene, "hfr_gen_brow_ridge_samples", 20)),
                brow_smooth=float(getattr(scene, "hfr_gen_brow_ridge_smooth", 0.22)),
                brow_inner_support=bool(getattr(scene, "hfr_gen_brow_inner_support", True)),
                brow_inner_strength=float(getattr(scene, "hfr_gen_brow_inner_strength", 0.70)),
                brow_inner_steps=int(getattr(scene, "hfr_gen_brow_inner_steps", 2)),
                brow_inner_radius=float(getattr(scene, "hfr_gen_brow_inner_radius", 1.10)),
                eye_loop_fit=bool(getattr(scene, "hfr_gen_eye_loop_fit", True)),
                eye_loop_strength=float(getattr(scene, "hfr_gen_eye_loop_strength", 1.0)),
                eye_loop_max_len=int(getattr(scene, "hfr_gen_eye_loop_max_len", 48)),
                eye_loop_steps=int(getattr(scene, "hfr_gen_eye_snap_steps", 96)),
                eye_direct_fit=bool(getattr(scene, "hfr_gen_eye_direct_fit", True)),
                eye_direct_radius=float(getattr(scene, "hfr_gen_eye_direct_radius", 0.90)),
                eye_band_steps=int(getattr(scene, "hfr_gen_eye_band_steps", 3)),
                eye_band_radius=float(getattr(scene, "hfr_gen_eye_band_radius", 1.45)),
                feature_loops=bool(getattr(scene, "hfr_gen_feature_loops", True)),
                loop_strength=float(getattr(scene, "hfr_gen_loop_strength", 0.85)),
                loop_radius=float(getattr(scene, "hfr_gen_loop_radius", 1.15)),
                ear_lobe_fit=bool(getattr(scene, "hfr_gen_ear_lobe_fit", True)),
                ear_strength=float(getattr(scene, "hfr_gen_ear_strength", 0.75)),
                ear_radius=float(getattr(scene, "hfr_gen_ear_radius", 1.25)),
                ear_lobe_y_guard=bool(getattr(scene, "hfr_gen_ear_lobe_y_guard", True)),
                ear_lobe_y_strength=float(getattr(scene, "hfr_gen_ear_lobe_y_strength", 0.85)),
                ear_lobe_relative=bool(getattr(scene, "hfr_gen_ear_lobe_relative", True)),
                ear_lobe_relative_strength=float(getattr(scene, "hfr_gen_ear_lobe_relative_strength", 1.0)),
                ear_lobe_xy_strength=float(getattr(scene, "hfr_gen_ear_lobe_xy_strength", 1.0)),
                ear_lower_rail=bool(getattr(scene, "hfr_gen_ear_lower_rail", True)),
                ear_lower_rail_strength=float(getattr(scene, "hfr_gen_ear_lower_rail_strength", 0.90)),
                ear_lower_rail_radius=float(getattr(scene, "hfr_gen_ear_lower_rail_radius", 0.90)),
                ear_lobe_patch=bool(getattr(scene, "hfr_gen_ear_lobe_patch", True)),
                ear_lobe_patch_strength=float(getattr(scene, "hfr_gen_ear_lobe_patch_strength", 0.85)),
                ear_lobe_patch_steps=int(getattr(scene, "hfr_gen_ear_lobe_patch_steps", 4)),
                ear_strip_fit=bool(getattr(scene, "hfr_gen_ear_strip_fit", True)),
                ear_strip_strength=float(getattr(scene, "hfr_gen_ear_strip_strength", 0.85)),
                ear_strip_y_lock=float(getattr(scene, "hfr_gen_ear_strip_y_lock", 1.0)),
                sparse_ear_safe=bool(getattr(scene, "hfr_gen_sparse_ear_safe", True)),
                sparse_ear_y_strength=float(getattr(scene, "hfr_gen_sparse_ear_y_strength", 1.0)),
                sparse_ear_neighbor_blend=float(getattr(scene, "hfr_gen_sparse_ear_neighbor_blend", 0.35)),
                lobe_directional_stretch=bool(getattr(scene, "hfr_gen_lobe_directional_stretch", True)),
                lobe_directional_strength=float(getattr(scene, "hfr_gen_lobe_directional_strength", 1.0)),
                lobe_directional_steps=int(getattr(scene, "hfr_gen_lobe_directional_steps", 2)),
                lobe_directional_falloff=float(getattr(scene, "hfr_gen_lobe_directional_falloff", 0.65)),
                head_round_fit=bool(getattr(scene, "hfr_gen_head_round_fit", True)),
                head_round_strength=float(getattr(scene, "hfr_gen_head_round_strength", 0.80)),
                head_round_steps=int(getattr(scene, "hfr_gen_head_round_steps", 8)),
                head_round_iters=int(getattr(scene, "hfr_gen_head_round_iters", 2)),
                head_round_z_margin=float(getattr(scene, "hfr_gen_head_round_z_margin", 0.30)),
                neck_fit=bool(getattr(scene, "hfr_gen_neck_fit", True)),
                neck_strength=float(getattr(scene, "hfr_gen_neck_strength", 0.85)),
                neck_radius=float(getattr(scene, "hfr_gen_neck_radius", 1.20)),
                ear_local_fit=bool(getattr(scene, "hfr_gen_ear_local_fit", True)),
                ear_local_strength=float(getattr(scene, "hfr_gen_ear_local_strength", 0.82)),
                ear_local_steps=int(getattr(scene, "hfr_gen_ear_local_steps", 4)),
                ear_local_nearest=int(getattr(scene, "hfr_gen_ear_local_nearest", 0)),
                output_mirror_finish=(
                    bool(getattr(scene, "hfr_gen_output_mirror_finish", True))
                    and bool(getattr(scene, "hfr_lm_mirror_x", False))
                ),
                output_mirror_direction=getattr(scene, "hfr_lm_mirror_dir", 'L2R'),
                output_mirror_epsilon=float(getattr(scene, "hfr_gen_output_mirror_epsilon", 0.0005)),
            )
            snapped = 0
            if snap_enabled:
                snapped = snap_output_to_target(
                    context,
                    out_obj,
                    target,
                    strength=float(getattr(scene, "hfr_gen_snap_strength", 0.60)),
                    max_dist=float(getattr(scene, "hfr_gen_snap_max_dist", 0.0)),
                    protect_anchor=bool(getattr(scene, "hfr_gen_protect_anchors", True)),
                    anchor_strength=float(getattr(scene, "hfr_gen_anchor_snap_strength", 0.20)),
                    ear_snap_guard=bool(getattr(scene, "hfr_gen_ear_snap_guard", True)),
                    ear_snap_strength=float(getattr(scene, "hfr_gen_ear_snap_strength", 0.0)),
                    ear_snap_steps=int(getattr(scene, "hfr_gen_ear_snap_steps", 3)),
                    eye_snap_guard=bool(getattr(scene, "hfr_gen_eye_snap_guard", True)),
                    eye_snap_strength=float(getattr(scene, "hfr_gen_eye_snap_strength", 0.0)),
                    eye_snap_steps=int(getattr(scene, "hfr_gen_eye_snap_steps", 96)),
                )
                if bool(getattr(scene, "hfr_gen_post_anchor_lock", True)):
                    enforce_anchor_targets(
                        out_obj,
                        lock=float(getattr(scene, "hfr_gen_post_anchor_lock_strength", 0.50)),
                        iters=int(getattr(scene, "hfr_gen_post_anchor_iters", 1)),
                    )
                try:
                    post_records = anchor_records_for_template_with_source_positions(out_obj, hfr_source_positions)
                    if bool(getattr(scene, "hfr_gen_eye_loop_fit", True)) and bool(getattr(scene, "hfr_gen_eye_direct_fit", True)):
                        apply_eye_direct_loop_fit(
                            out_obj,
                            hfr_source_positions,
                            post_records,
                            eye_strength=float(getattr(scene, "hfr_gen_eye_loop_strength", 1.0)),
                            eye_radius=float(getattr(scene, "hfr_gen_eye_direct_radius", 0.90)),
                        )
                except Exception as _post_exc:
                    try:
                        out_obj["HFR_postfit_err"] = str(_post_exc)[:180]
                    except Exception:
                        pass
            try:
                post_records = anchor_records_for_template_with_source_positions(out_obj, hfr_source_positions)
                apply_back_center_column_inward_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.30,
                )
                apply_back_outer_column_inward_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.30,
                )
                apply_side_head_ear_opposite_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.50,
                )
                apply_side_head_ear_toward_strip_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.38,
                )
                apply_side_head_ear_toward_inner_support_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.26,
                )
                apply_jaw_ear_lower_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.44,
                )
                apply_ear_front_lower_support_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.41,
                )
                apply_ear_front_lower_upper_support_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.65,
                )
                apply_ear_face_edge_upper_support_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.47,
                )
                apply_ear_face_edge_pair_toward_ear_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.27,
                )
                apply_ear_lower_wedge_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.32,
                )
                apply_ear_upper_wedge_edge_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.44,
                )
                apply_ear_front_face_edge_down_slide_guard(
                    out_obj,
                    records=post_records,
                    slide_strength=0.32,
                )
                apply_side_head_ear_opposite_mirror_match_guard(
                    out_obj,
                    records=post_records,
                    match_strength=1.0,
                )
            except Exception as _bcis_exc:
                try:
                    out_obj["HFR_bcis_err"] = str(_bcis_exc)[:180]
                except Exception:
                    pass
            write_generate_report(template, out_obj, target, anchors_used, snapped, quality_warnings=quality_warnings, side_warnings=side_warnings, mirror_sync_count=mirror_sync_count, final_force_snap=force_snap, context=context)
            try:
                bpy.ops.object.select_all(action='DESELECT')
                out_obj.select_set(True)
                context.view_layer.objects.active = out_obj
            except Exception:
                pass
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        self.report({'INFO'}, f"Generated {out_obj.name}: {anchors_used} anchors, {snapped} snapped vertices; report copied to clipboard")
        return {'FINISHED'}


class HFR_OT_CleanupCollections(bpy.types.Operator):
    bl_idname = "hfr.cleanup_collections"
    bl_label = "Clean Unneeded HFR Objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        removed, moved, fixed = cleanup_hfr_collections(remove_unknown_hfr_named=True)
        styled = apply_landmark_style_and_guides(context.scene, context)
        self.report({'INFO'}, f"Cleaned HFR collections: removed {removed}, moved {moved}, styled {styled}")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def _draw_hfr_section(layout, scene, prop_name, title):
    box = layout.box()
    is_open = bool(getattr(scene, prop_name, True))
    icon = 'TRIA_DOWN' if is_open else 'TRIA_RIGHT'
    row = box.row(align=True)
    row.prop(scene, prop_name, text=title, icon=icon, emboss=False)
    return box if is_open else None


class HFR_PT_TemplateLandmarks(bpy.types.Panel):
    bl_label = "Humanoid Face Retopology(HFR)"
    bl_idname = "HFR_PT_template_landmarks"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "HFR"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Registration-time timers can run before a newly opened scene is fully
        # ready.  Re-check from the panel draw and schedule a deferred load so a
        # scene that already contains only the target mesh still receives the
        # bundled template automatically.
        try:
            if should_auto_load_default_template(context):
                schedule_auto_load_default_template(delay=0.05)
        except Exception:
            pass

        box = _draw_hfr_section(layout, scene, "hfr_ui_style_open", "1. Setup")
        if box:
            box.prop(scene, "hfr_lm_target_obj", text="Target Mesh")
            box.prop(scene, "hfr_template_obj", text="Template Mesh")
            row = box.row(align=True)
            op = row.operator("hfr.load_default_template", text="Reload Default Template")
            op.replace_existing = True
            row.operator("hfr.initialize_workspace", text="Initialize HFR Workspace")

            target = getattr(scene, "hfr_lm_target_obj", None)
            target_ok = bool(target is not None and target.type == 'MESH' and not target.get(PID_LM) and not target.get(PID_OUTPUT) and not target.get(PID_TEMPLATE))
            template = template_object(context)
            template_ok = bool(template is not None and template.type == 'MESH')
            lm_count = sum(1 for lm in LANDMARKS if find_landmark_obj(lm["id"]) is not None)
            if template_ok:
                missing, empty, bound = binding_status_summary(template)
                bind_ok = not missing and not empty
                bind_text = "OK (%d/%d)" % (len(bound), len(LANDMARKS)) if bind_ok else "Check (%d missing, %d empty)" % (len(missing), len(empty))
            else:
                bind_ok = False
                bind_text = "Missing"

            status_col = box.column(align=True)
            status_col.label(text="Status", icon='INFO')
            status_col.label(text="Target Mesh: " + ("OK" if target_ok else "Missing"), icon='CHECKMARK' if target_ok else 'ERROR')
            status_col.label(text="Template Mesh: " + ("OK" if template_ok else "Missing"), icon='CHECKMARK' if template_ok else 'ERROR')
            status_col.label(text="Template Binding: " + bind_text, icon='CHECKMARK' if bind_ok else 'ERROR')
            status_col.label(text="Landmarks: " + ("Created (%d/%d)" % (lm_count, len(LANDMARKS)) if lm_count else "Not Created"), icon='CHECKMARK' if lm_count else 'ERROR')

            row = box.row(align=True)
            row.prop(scene, "hfr_adv_options", text="Advanced")
            if HFR_SHOW_DEV_OPTIONS:
                row.prop(scene, "hfr_dev_options", text="DevOption")
            else:
                row.label(text="Release build: DevOption hidden", icon='LOCKED')

        adv_options = bool(getattr(scene, "hfr_adv_options", False))
        dev_options = bool(HFR_SHOW_DEV_OPTIONS and getattr(scene, "hfr_dev_options", False))

        box = _draw_hfr_section(layout, scene, "hfr_ui_groups_open", "2. Landmarks")
        if box:
            op = box.operator("hfr.add_landmarks", text="Add All Landmarks")
            op.group = 'ALL'
            op.reset_existing = False
            row = box.row(align=True)
            op = row.operator("hfr.reset_landmarks", text="Reset All Landmarks")
            op.group = 'ALL'
            op.use_target_fit = False
            op = row.operator("hfr.delete_landmarks", text="Delete Landmarks / Guides")
            op.delete_guides = True

            row = box.row(align=True)
            row.prop(scene, "hfr_lm_mirror_x", text="Landmark Mirror X")
            row.prop(scene, "hfr_lm_mirror_dir", text="")
            row = box.row(align=True)
            op = row.operator("hfr.mirror_landmarks", text="Set Position L -> R")
            op.direction = 'L2R'
            op = row.operator("hfr.mirror_landmarks", text="Set Position R -> L")
            op.direction = 'R2L'
            box.prop(scene, "hfr_lm_show_front", text="Landmark to Front")

            row = box.row(align=True)
            for label, group in [("Eyes", 'EYE'), ("Nose", 'NOSE'), ("Mouth", 'MOUTH')]:
                op = row.operator("hfr.add_landmarks", text=label)
                op.group = group
                op.reset_existing = False

            row = box.row(align=True)
            op = row.operator("hfr.add_landmarks", text="Scalp")
            op.group = 'SCALP'
            op.reset_existing = False
            op = row.operator("hfr.add_landmarks", text="Ears")
            op.group = 'EAR'
            op.reset_existing = False
            op = row.operator("hfr.add_landmarks", text="Neck")
            op.group = 'NECK'
            op.reset_existing = False

        box = _draw_hfr_section(layout, scene, "hfr_ui_final_generate_open", "3. Generate")
        if box:
            box.prop(scene, "hfr_gen_output_name", text="Output Name")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_replace_output", text="Replace Existing")
            row.prop(scene, "hfr_gen_output_wire", text="Wire Output")
            op = box.operator("hfr.generate_retopology", text="Generate Retopology")
            op.force_snap_to_target = True
            box.label(text="Adjust landmarks and run Generate again if the result needs correction.", icon='INFO')

        box = _draw_hfr_section(layout, scene, "hfr_ui_cleanup_open", "4. Cleanup")
        if box:
            box.operator("hfr.cleanup_collections", text="Clean Unneeded Objects")
            op = box.operator("hfr.delete_landmarks", text="Delete HFR Landmarks / Guides")
            op.delete_guides = True

        if adv_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_edit_open", "Advanced: Landmark Tools")
        else:
            box = None
        if box:
            row = box.row(align=True)
            row.operator("hfr.save_lm_defaults", text="Save Landmark Position")
            row.operator("hfr.load_lm_defaults", text="Load Landmark Position")
            box.operator("hfr.export_lm_positions", text="Export Landmark Position")
            box.separator()
            row = box.row(align=True)
            row.prop(scene, "hfr_lm_use_target_fit", text="Fit On Add/Reset")
            row = box.row(align=True)
            row.prop(scene, "hfr_lm_fit_region", text="Region")
            row.prop(scene, "hfr_lm_fit_margin", text="Margin")
            row = box.row(align=True)
            op = row.operator("hfr.fit_landmarks_to_target", text="Fit All To Target")
            op.group = 'ALL'
            op = row.operator("hfr.fit_landmarks_to_target", text="Fit Face")
            op.group = 'FACE'

        if adv_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_advanced_output_open", "Advanced: Output / Snap")
        else:
            box = None
        if box:
            box.prop(scene, "hfr_gen_output_name", text="Output Name")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_replace_output", text="Replace Existing")
            row.prop(scene, "hfr_gen_output_wire", text="Wire Output")
            row.prop(scene, "hfr_gen_output_in_front", text="In Front")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_output_mirror_finish", text="Mirror Finish")
            row.prop(scene, "hfr_gen_output_mirror_epsilon", text="Mirror Eps")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_snap_to_target", text="Snap To Target")
            if bool(getattr(scene, "hfr_gen_snap_to_target", False)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_snap_strength", text="Snap Strength")
                row.prop(scene, "hfr_gen_snap_max_dist", text="Max Dist")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_protect_anchors", text="Protect Anchors")
                row.prop(scene, "hfr_gen_anchor_snap_strength", text="Anchor Snap")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_snap_guard", text="Ear Guard")
                row.prop(scene, "hfr_gen_eye_snap_guard", text="Eye Guard")
            box.label(text="User-facing final-output controls. Detailed solver numbers are under DevOption.", icon='INFO')

        if adv_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_advanced_feature_open", "Advanced: Feature Controls")
        else:
            box = None
        if box:
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_mls_field", text="MLS Field")
            row.prop(scene, "hfr_gen_guide_rails", text="Guide Rails")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_guide_follow", text="Guide Follow")
            row.prop(scene, "hfr_gen_feature_loops", text="Feature Loops")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_eye_loop_fit", text="Eye Boundary")
            row.prop(scene, "hfr_gen_brow_ridge_fit", text="Brow Ridge")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_nose_web_fit", text="Nose Web")
            row.prop(scene, "hfr_gen_nose_alar_fit", text="Nose Alar")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_ear_local_fit", text="Ear Local")
            row.prop(scene, "hfr_gen_ear_lobe_fit", text="Ear Lobe")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_lobe_directional_stretch", text="Lobe Dir Stretch")
            row.prop(scene, "hfr_gen_neck_fit", text="Neck Fit")
            box.label(text="Use these only when a generated area needs a broad on/off adjustment.", icon='INFO')

        if dev_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_dev_diagnostics_open", "DevOption: Diagnostics")
        else:
            box = None
        if box:
            box.operator("hfr.export_selected_vertices", text="Export Selected Vertices")
            box.operator("hfr.export_mesh_vertex_diagnostic", text="Export Mesh Vertex Diagnostic")
            box.operator("hfr.export_generate_report", text="Export Generate Report")
            box.operator("hfr.refresh_guides", text="Refresh HFR Guides")
            box.operator("hfr.update_style", text="Update HFR Landmark Style")
            box.label(text="Developer diagnostics. Hide DevOption for release builds.", icon='INFO')

        if dev_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_binding_open", "DevOption: Template Binding")
        else:
            box = None
        if box:
            box.prop(scene, "hfr_template_obj", text="Template Mesh")
            box.prop(scene, "hfr_tpl_obj_name", text="Default Template Object")
            row = box.row(align=True)
            op = row.operator("hfr.load_default_template", text="Reload Default Template")
            op.replace_existing = True
            row = box.row(align=True)
            row.prop(scene, "hfr_bind_mode_enabled", text="Binding Mode")
            box.operator("hfr.create_anchor_groups", text="Create Anchor Groups On Template")
            row = box.row(align=True)
            row.operator("hfr.export_template_binding", text="Export Template Binding")
            box.label(text="Bundled binding is applied automatically with the default template.", icon='INFO')

            if not binding_mode_enabled(scene):
                box.label(text="Binding Mode is off: live binding status/guides are suspended.", icon='INFO')
                box.label(text="Turn it on only while assigning or checking anchors.", icon='BLANK1')
            else:
                template = template_object(context)
                if template is not None:
                    missing, empty, bound = binding_status_summary(template)
                    total = len(LANDMARKS)
                    unbound_count = len(missing) + len(empty)
                    box.label(text=f"Binding Status: {len(bound)} / {total} bound | {unbound_count} unbound", icon='INFO')
                    quality_warnings = binding_quality_warnings(template)
                    side_warnings = binding_side_warnings(template)
                    if side_warnings:
                        warn_col = box.column(align=True)
                        warn_col.label(text=f"Side Binding Warnings: {len(side_warnings)}", icon='ERROR')
                        for warning in side_warnings[:3]:
                            warn_col.label(text=warning[:120], icon='BLANK1')
                        if len(side_warnings) > 3:
                            warn_col.label(text="Use Validate Template Binding for full report.", icon='BLANK1')
                    if quality_warnings:
                        warn_col = box.column(align=True)
                        warn_col.label(text=f"Binding Quality Warnings: {len(quality_warnings)}", icon='ERROR')
                        for warning in quality_warnings[:3]:
                            warn_col.label(text=warning[:120], icon='BLANK1')
                        if len(quality_warnings) > 3:
                            warn_col.label(text="Use Validate Template Binding for full report.", icon='BLANK1')
                    active_lm = getattr(scene, "hfr_bind_lm_id", "")
                    status, vcount = _binding_status_for_lm(template, active_lm)
                    if status == 'BOUND':
                        box.label(text=f"Active Status: Bound ({vcount} vertices)", icon='CHECKMARK')
                    elif status == 'EMPTY':
                        box.label(text="Active Status: Unbound / empty group", icon='ERROR')
                    elif status == 'MISSING':
                        box.label(text="Active Status: Unbound / missing group", icon='ERROR')
                    else:
                        box.label(text="Active Status: Template not ready", icon='QUESTION')
                    if unbound_count:
                        row = box.row(align=True)
                        row.operator("hfr.next_unbound_landmark", text="Next Unbound")
                        row.operator("hfr.validate_template_binding", text="Write Full Report")
                        col = box.column(align=True)
                        col.label(text="Unbound landmarks:")
                        for lm_id in (missing + empty)[:12]:
                            op = col.operator("hfr.set_active_binding_landmark", text="[ ] LM_" + lm_id)
                            op.lm_id = lm_id
                        if unbound_count > 12:
                            col.label(text=f"... plus {unbound_count - 12} more. Use Write Full Report.")
                else:
                    box.label(text="Binding Status: assign Template Mesh first", icon='ERROR')

                row = box.row(align=True)
                row.prop(scene, "hfr_bind_lm_id", text="Active Landmark")
                row.operator("hfr.set_binding_lm_from_selection", text="Use Selected")
                row = box.row(align=True)
                op = row.operator("hfr.bind_selected_vertices", text="Bind Selected Vertices")
                op.replace_existing = True
                row.operator("hfr.select_anchor_vertices", text="Select Bound")
                row = box.row(align=True)
                row.operator("hfr.clear_anchor_binding", text="Clear Active Binding")
                row.operator("hfr.refresh_binding_guides", text="Refresh Bind Guides")
                row = box.row(align=True)
                op = row.operator("hfr.mirror_anchor_groups", text="Mirror Anchors L -> R")
                op.direction = 'L2R'
                op = row.operator("hfr.mirror_anchor_groups", text="R -> L")
                op.direction = 'R2L'
                row = box.row(align=True)
                row.prop(scene, "hfr_bind_mirror_tol", text="Mirror Tol")
                row.prop(scene, "hfr_bind_show_guides", text="Bind Guides")
                box.operator("hfr.validate_template_binding", text="Validate Template Binding")
                box.label(text="Anchor group names: HFR_A_*", icon='INFO')

        if dev_options:
            box = _draw_hfr_section(layout, scene, "hfr_ui_generate_open", "DevOption: Solver Parameters")
        else:
            box = None
        if box:
            box.prop(scene, "hfr_template_obj", text="Template Mesh")
            box.prop(scene, "hfr_gen_output_name", text="Output Name")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_replace_output", text="Replace Existing")
            row.prop(scene, "hfr_gen_output_wire", text="Wire Output")
            row.prop(scene, "hfr_gen_output_in_front", text="In Front")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_power", text="Deform Power")
            row.prop(scene, "hfr_gen_nearest", text="Nearest Anchors")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_anchor_lock", text="Anchor Lock")
            row.prop(scene, "hfr_gen_anchor_iters", text="Lock Iter")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_topo_propagate", text="Topo Propagate")
            if bool(getattr(scene, "hfr_gen_topo_propagate", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_topo_iters", text="Topo Iter")
                row.prop(scene, "hfr_gen_topo_strength", text="Topo Str")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_guide_rails", text="Guide Rails")
            if bool(getattr(scene, "hfr_gen_guide_rails", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_guide_rail_strength", text="Rail Str")
                row.prop(scene, "hfr_gen_guide_rail_max_len", text="Rail Max")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_guide_rail_spread", text="Rail Spread")
                row.prop(scene, "hfr_gen_guide_rail_spread_steps", text="Spread Steps")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_guide_rail_spread_strength", text="Spread Str")
                box.label(text="Re-locks guide paths after MLS/Guide/Loop refiners and spreads rail motion to nearby surface.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_mls_field", text="MLS Field")
            if bool(getattr(scene, "hfr_gen_mls_field", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_mls_strength", text="MLS Str")
                row.prop(scene, "hfr_gen_mls_nearest", text="MLS Near")
                box.label(text="Primary broad solver: local affine field makes areas between landmarks follow together.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_guide_follow", text="Guide Follow")
            if bool(getattr(scene, "hfr_gen_guide_follow", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_guide_strength", text="Guide Str")
                row.prop(scene, "hfr_gen_guide_radius", text="Guide Radius")
                box.label(text="Interpolates movement along landmark guides so vertices between landmarks follow together.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_nose_web_fit", text="Nose Web Fit")
            if bool(getattr(scene, "hfr_gen_nose_web_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_nose_web_strength", text="Nose Str")
                row.prop(scene, "hfr_gen_nose_web_radius", text="Nose Radius")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_nose_web_samples", text="Nose Samples")
                box.label(text="Moves the web between nose side rails and the bridge-tip rail.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_nose_alar_fit", text="Nose Alar Fit")
            if bool(getattr(scene, "hfr_gen_nose_alar_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_nose_alar_strength", text="Alar Str")
                row.prop(scene, "hfr_gen_nose_alar_radius", text="Alar Radius")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_nose_alar_samples", text="Alar Samples")
                box.label(text="Local nostril-wing follow for nose side_lower / alar / nostril / base.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_brow_ridge_fit", text="Brow Ridge Fit")
            if bool(getattr(scene, "hfr_gen_brow_ridge_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_brow_ridge_strength", text="Brow Str")
                row.prop(scene, "hfr_gen_brow_ridge_radius", text="Brow Radius")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_brow_ridge_samples", text="Brow Samples")
                row.prop(scene, "hfr_gen_brow_ridge_smooth", text="Brow Smooth")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_brow_inner_support", text="Inner Support")
                if bool(getattr(scene, "hfr_gen_brow_inner_support", True)):
                    row = box.row(align=True)
                    row.prop(scene, "hfr_gen_brow_inner_strength", text="Inner Str")
                    row.prop(scene, "hfr_gen_brow_inner_steps", text="Inner Steps")
                    row = box.row(align=True)
                    row.prop(scene, "hfr_gen_brow_inner_radius", text="Inner Radius")
                box.label(text="Keeps the eye-upper loop and brow ridge separated while the inner support fan follows brow_inner.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_eye_loop_fit", text="Eye Boundary Fit")
            if bool(getattr(scene, "hfr_gen_eye_loop_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_eye_loop_strength", text="Eye Str")
                row.prop(scene, "hfr_gen_eye_loop_max_len", text="Eye Max")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_eye_direct_fit", text="Direct Fit")
                row.prop(scene, "hfr_gen_eye_direct_radius", text="Direct Radius")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_eye_band_steps", text="Band Steps")
                row.prop(scene, "hfr_gen_eye_band_radius", text="Band Radius")
                box.label(text="Fits the local eyelid topology band, not only shortest paths or the open boundary.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_feature_loops", text="Feature Loops")
            if bool(getattr(scene, "hfr_gen_feature_loops", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_loop_strength", text="Loop Str")
                row.prop(scene, "hfr_gen_loop_radius", text="Loop Radius")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_output_mirror_finish", text="Output Mirror Finish")
            row.prop(scene, "hfr_gen_output_mirror_epsilon", text="Mirror Eps")
            box.label(text="When Landmark Mirror X is ON, final output is copied from the source side to the followed side.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_head_round_fit", text="Head Dome Fit")
            if bool(getattr(scene, "hfr_gen_head_round_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_head_round_strength", text="Dome Str")
                row.prop(scene, "hfr_gen_head_round_steps", text="Dome Steps")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_head_round_iters", text="Smooth Iter")
                row.prop(scene, "hfr_gen_head_round_z_margin", text="Base Z Marg")
                if bool(getattr(scene, "hfr_gen_mls_field", True)):
                    box.label(text="Head Dome Fit is bypassed while MLS Field is ON.", icon='INFO')
                else:
                    box.label(text="Legacy scalp post-fix. Use only when MLS Field is OFF.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_ear_lobe_fit", text="Ear Lobe Fit")
            row.prop(scene, "hfr_gen_neck_fit", text="Neck Fit")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_lobe_directional_stretch", text="Lobe Dir Stretch")
            if bool(getattr(scene, "hfr_gen_lobe_directional_stretch", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_lobe_directional_strength", text="Dir Str")
                row.prop(scene, "hfr_gen_lobe_directional_steps", text="Dir Steps")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_lobe_directional_falloff", text="Dir Falloff")
                box.label(text="Moves lower-ear neighbors in the dragged direction; naturalness is not enforced.", icon='INFO')
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_ear_local_fit", text="Ear Local Fit")
            if bool(getattr(scene, "hfr_gen_ear_local_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_local_strength", text="Ear Local Str")
                row.prop(scene, "hfr_gen_ear_local_steps", text="Ear Steps")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_local_nearest", text="Ear Near")
                box.label(text="Re-fits the ear patch from same-side ear anchors to prevent folded ear faces.", icon='INFO')
            if bool(getattr(scene, "hfr_gen_ear_lobe_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_strength", text="Ear Str")
                row.prop(scene, "hfr_gen_ear_radius", text="Ear Radius")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_lobe_y_guard", text="Lobe Y Guard")
                row.prop(scene, "hfr_gen_ear_lobe_relative", text="Lobe Relative")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_lobe_y_strength", text="Y Guard Str")
                row.prop(scene, "hfr_gen_ear_lobe_relative_strength", text="Rel Str")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_lobe_xy_strength", text="XY Lock Str")
                row.prop(scene, "hfr_gen_ear_lower_rail", text="Lower Rail")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_lower_rail_strength", text="Rail Str")
                row.prop(scene, "hfr_gen_ear_lower_rail_radius", text="Rail Radius")
                box.label(text="When Lobe Dir Stretch is ON, guard/relative/sparse-safe corrections are bypassed for the lobe.", icon='INFO')
            if bool(getattr(scene, "hfr_gen_neck_fit", True)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_neck_strength", text="Neck Str")
                row.prop(scene, "hfr_gen_neck_radius", text="Neck Radius")
            row = box.row(align=True)
            row.prop(scene, "hfr_gen_snap_to_target", text="Snap To Target")
            if bool(getattr(scene, "hfr_gen_snap_to_target", False)):
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_snap_strength", text="Snap Strength")
                row.prop(scene, "hfr_gen_snap_max_dist", text="Max Dist")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_protect_anchors", text="Protect Anchors")
                row.prop(scene, "hfr_gen_anchor_snap_strength", text="Anchor Snap")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_post_anchor_lock", text="Post Lock")
                row.prop(scene, "hfr_gen_post_anchor_lock_strength", text="Post Lock Str")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_snap_guard", text="Ear Snap Guard")
                row.prop(scene, "hfr_gen_ear_snap_strength", text="Ear Snap")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_ear_snap_steps", text="Ear Steps")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_eye_snap_guard", text="Eye Snap Guard")
                row.prop(scene, "hfr_gen_eye_snap_strength", text="Eye Snap")
                row = box.row(align=True)
                row.prop(scene, "hfr_gen_eye_snap_steps", text="Eye Steps")
                box.label(text="Protects thin ear geometry and eye-hole boundaries from nearest-surface snap distortion.", icon='INFO')
            box.operator("hfr.export_generate_report", text="Export Generate Report")
            box.label(text="Generate Retopology copies HFR_Generate_Report to clipboard.", icon='INFO')



# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    HFR_OT_AddLandmarks,
    HFR_OT_ResetLandmarks,
    HFR_OT_DeleteLandmarks,
    HFR_OT_RefreshGuides,
    HFR_OT_MirrorLandmarks,
    HFR_OT_SaveLandmarkDefaults,
    HFR_OT_LoadLandmarkDefaults,
    HFR_OT_ExportLandmarkPositions,
    HFR_OT_ExportSelectedVertices,
    HFR_OT_ExportMeshVertexDiagnostic,
    HFR_OT_ExportGenerateReport,
    HFR_OT_FitLandmarksToTarget,
    HFR_OT_LoadDefaultTemplate,
    HFR_OT_InitializeWorkspace,
    HFR_OT_CreateAnchorGroups,
    HFR_OT_SetBindingLandmarkFromSelection,
    HFR_OT_SetActiveBindingLandmark,
    HFR_OT_NextUnboundLandmark,
    HFR_OT_BindSelectedVertices,
    HFR_OT_ClearAnchorBinding,
    HFR_OT_SelectAnchorVertices,
    HFR_OT_MirrorAnchorGroups,
    HFR_OT_ExportTemplateBinding,
    HFR_OT_ImportTemplateBinding,
    HFR_OT_ValidateTemplateBinding,
    HFR_OT_RefreshBindingGuides,
    HFR_OT_UpdateStyle,
    HFR_OT_GenerateRetopology,
    HFR_OT_CleanupCollections,
    HFR_PT_TemplateLandmarks,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.hfr_lm_scale = FloatProperty(
        name="Object Size",
        description="Landmark object dimensions in meters. Mesh data is resized so object scale remains 1/1/1.",
        default=0.003,
        min=0.001,
        max=0.03,
    )
    bpy.types.Scene.hfr_lm_show_names = BoolProperty(
        name="Names",
        description="Legacy option. Landmark names are hidden by default in this branch.",
        default=False,
    )
    bpy.types.Scene.hfr_lm_show_front = BoolProperty(
        name="Landmark to Front",
        description="Show front-facing landmarks in front of other geometry; scalp/back landmarks remain normal",
        default=True,
        update=hfr_lm_front_update,
    )
    bpy.types.Scene.hfr_lm_auto_guides = BoolProperty(
        name="Guides",
        description="Legacy hidden option. Guides are always enabled in this branch.",
        default=True,
    )
    bpy.types.Scene.hfr_lm_live_guides = BoolProperty(
        name="Live Guides",
        description="Legacy hidden option. Live guide updates are always enabled in this branch.",
        default=True,
    )
    bpy.types.Scene.hfr_lm_mirror_x = BoolProperty(
        name="Landmark Mirror X",
        description="Automatically mirror edited landmarks across world X=0",
        default=False,
    )
    bpy.types.Scene.hfr_lm_mirror_dir = EnumProperty(
        name="Mirror Direction",
        items=[
            ('L2R', "L -> R", "Move left landmarks and update matching right landmarks"),
            ('R2L', "R -> L", "Move right landmarks and update matching left landmarks"),
        ],
        default='L2R',
    )
    bpy.types.Scene.hfr_lm_target_obj = PointerProperty(
        name="Target Mesh",
        type=bpy.types.Object,
        description="Mesh used to place initial landmarks around the head/face",
    )
    bpy.types.Scene.hfr_lm_use_target_fit = BoolProperty(
        name="Fit On Add/Reset",
        description="Place new or reset landmarks by fitting the default layout to the target mesh bounds",
        default=True,
    )
    bpy.types.Scene.hfr_lm_auto_scale = BoolProperty(
        name="Auto Scale",
        description="Legacy option kept for compatibility. Landmark object size is fixed by Object Size.",
        default=False,
    )
    bpy.types.Scene.hfr_lm_scale_ratio = FloatProperty(
        name="Auto Scale Ratio",
        description="Landmark radius as a fraction of the fitted target size",
        default=0.018,
        min=0.002,
        max=0.08,
    )
    bpy.types.Scene.hfr_lm_fit_margin = FloatProperty(
        name="Fit Margin",
        description="Extra margin applied to the fitted landmark layout",
        default=0.02,
        min=-0.30,
        max=0.50,
    )
    bpy.types.Scene.hfr_lm_fit_region = EnumProperty(
        name="Fit Region",
        items=[
            ('AUTO', "Auto", "Use head-only fitting when a full body-like mesh is detected"),
            ('BOUNDS', "Bounds", "Fit to the full selected mesh bounds"),
            ('HEAD', "Head", "Use the upper head-like part of the selected mesh bounds"),
        ],
        default='AUTO',
    )
    bpy.types.Scene.hfr_template_obj = PointerProperty(
        name="Template Mesh",
        type=bpy.types.Object,
        description="Retopology template mesh whose vertices will be bound to HFR landmarks",
    )
    bpy.types.Scene.hfr_auto_load_tpl = BoolProperty(
        name="Auto Load Bundled Template",
        description="Automatically append templates/HFRTemplate.blend when HFR is enabled, a target mesh exists, and no Template Mesh is assigned",
        default=True,
    )
    bpy.types.Scene.hfr_tpl_obj_name = StringProperty(
        name="Default Template Object",
        description="Object name to append from templates/HFRTemplate.blend",
        default=DEFAULT_TEMPLATE_OBJECT,
    )
    bpy.types.Scene.hfr_bind_lm_id = EnumProperty(
        name="Active Landmark",
        description="Landmark used by Bind Selected Vertices. Labels show [OK] for already-bound landmarks.",
        items=binding_landmark_items,
    )
    bpy.types.Scene.hfr_bind_mode_enabled = BoolProperty(
        name="Binding Mode",
        description="Enable live template binding status and binding guide refresh. Turn off when not binding to avoid viewport/UI lag.",
        default=False,
        update=hfr_bind_mode_update,
    )
    bpy.types.Scene.hfr_bind_show_guides = BoolProperty(
        name="Bind Guides",
        description="Show guide lines from landmarks to their bound template anchor centroids",
        default=True,
    )
    bpy.types.Scene.hfr_bind_mirror_tol = FloatProperty(
        name="Mirror Tolerance",
        description="Maximum local-space distance used when mirroring anchor vertex groups",
        default=0.01,
        min=0.0,
        max=1.0,
        precision=4,
    )
    bpy.types.Scene.hfr_gen_output_name = StringProperty(
        name="Output Name",
        description="Name of the generated retopology object",
        default="HFR_Retopo",
    )
    bpy.types.Scene.hfr_gen_replace_output = BoolProperty(
        name="Replace Existing Output",
        description="Remove existing HFR generated output objects before generating a new one",
        default=True,
    )
    bpy.types.Scene.hfr_gen_output_wire = BoolProperty(
        name="Wire Output",
        description="Show generated retopology object as wire so it can be inspected over the target mesh",
        default=False,
    )
    bpy.types.Scene.hfr_gen_output_in_front = BoolProperty(
        name="Output In Front",
        description="Draw the generated retopology object in front of the target mesh. Default is off so back-side vertices do not show through in Material Preview",
        default=False,
    )
    bpy.types.Scene.hfr_gen_power = FloatProperty(
        name="Deform Power",
        description="IDW power used when propagating landmark anchor deltas through the template",
        default=2.0,
        min=0.25,
        max=8.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nearest = bpy.props.IntProperty(
        name="Nearest Anchors",
        description="Number of nearest anchors used for each vertex. 0 uses all anchors",
        default=12,
        min=0,
        max=128,
    )
    bpy.types.Scene.hfr_gen_anchor_lock = FloatProperty(
        name="Anchor Lock",
        description="How strongly bound anchor centroids are corrected to their landmarks after deformation",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_anchor_iters = bpy.props.IntProperty(
        name="Anchor Iterations",
        description="Number of anchor centroid correction passes after broad deformation",
        default=2,
        min=0,
        max=8,
    )
    bpy.types.Scene.hfr_gen_topo_propagate = BoolProperty(
        name="Topology Propagation",
        description="Propagate anchor movement through the template edge graph so vertices between anchors move with eye/mouth loops",
        default=True,
    )
    bpy.types.Scene.hfr_gen_topo_iters = bpy.props.IntProperty(
        name="Topology Iterations",
        description="Relaxation passes used for topology-based displacement propagation",
        default=36,
        min=0,
        max=160,
    )
    bpy.types.Scene.hfr_gen_topo_strength = FloatProperty(
        name="Topology Strength",
        description="How strongly each pass blends a non-anchor vertex displacement toward neighboring displacements",
        default=0.65,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_guide_rails = BoolProperty(
        name="Guide Rails",
        description="Constrain actual template edge paths between bound landmark pairs so intermediate vertices follow the landmarks directly",
        default=True,
    )
    bpy.types.Scene.hfr_gen_guide_rail_strength = FloatProperty(
        name="Guide Rail Strength",
        description="How strongly mesh edge paths between landmarks are constrained to interpolated endpoint movement",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_guide_rail_max_len = IntProperty(
        name="Guide Rail Max Path",
        description="Skip guide rails whose shortest mesh path is longer than this vertex count. 0 disables the limit",
        default=80,
        min=0,
        max=512,
    )
    bpy.types.Scene.hfr_gen_guide_rail_spread = BoolProperty(
        name="Guide Rail Spread",
        description="Spread fixed guide-rail displacement to nearby surface rings so only the exact edge path does not move alone",
        default=True,
    )
    bpy.types.Scene.hfr_gen_guide_rail_spread_steps = IntProperty(
        name="Guide Rail Spread Steps",
        description="How many topological rings around each guide rail also follow the rail motion",
        default=1,
        min=0,
        max=8,
    )
    bpy.types.Scene.hfr_gen_guide_rail_spread_strength = FloatProperty(
        name="Guide Rail Spread Strength",
        description="How strongly neighboring vertices follow the fixed guide rails",
        default=0.65,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_mls_field = BoolProperty(
        name="MLS Field",
        description="Use weighted moving least-squares local affine deformation so regions between landmarks follow the edited landmark cage",
        default=True,
    )
    bpy.types.Scene.hfr_gen_mls_strength = FloatProperty(
        name="MLS Strength",
        description="How strongly the broad field follows the local affine transform solved from nearby landmark anchors",
        default=0.75,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_mls_nearest = IntProperty(
        name="MLS Nearest Anchors",
        description="Number of nearby landmark anchors used to solve each local affine deformation. 0 uses all anchors",
        default=18,
        min=0,
        max=128,
    )
    bpy.types.Scene.hfr_gen_guide_follow = BoolProperty(
        name="Guide Follow",
        description="Use the landmark guide network as soft deformation rails so regions between landmarks follow the edited landmarks",
        default=True,
    )
    bpy.types.Scene.hfr_gen_guide_strength = FloatProperty(
        name="Guide Strength",
        description="How strongly vertices near landmark guides follow the linearly interpolated movement between the two guide landmarks",
        default=0.55,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_guide_radius = FloatProperty(
        name="Guide Radius",
        description="Auto radius multiplier for the soft guide-follow correction zone around each landmark guide segment",
        default=1.10,
        min=0.05,
        max=5.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nose_web_fit = BoolProperty(
        name="Nose Web Fit",
        description="Locally fit vertices between nose side rails and the nose_bridge-to-nose_tip rail",
        default=True,
    )
    bpy.types.Scene.hfr_gen_nose_web_strength = FloatProperty(
        name="Nose Web Strength",
        description="How strongly the nose side web follows bilinear landmark movement",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nose_web_radius = FloatProperty(
        name="Nose Web Radius",
        description="Auto radius multiplier for the nose side web correction band",
        default=0.60,
        min=0.05,
        max=2.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nose_web_samples = IntProperty(
        name="Nose Web Samples",
        description="Sampling density along the nose bridge-to-tip web strip",
        default=24,
        min=4,
        max=64,
    )
    bpy.types.Scene.hfr_gen_nose_alar_fit = BoolProperty(
        name="Nose Alar Fit",
        description="Locally fit the nostril wing around side_lower, alar, nostril, and nose_base",
        default=True,
    )
    bpy.types.Scene.hfr_gen_nose_alar_strength = FloatProperty(
        name="Nose Alar Strength",
        description="How strongly the nostril wing follows LM_nose_l/r_alar movement",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nose_alar_radius = FloatProperty(
        name="Nose Alar Radius",
        description="Auto radius multiplier for the local alar / nostril-wing correction band",
        default=1.0,
        min=0.05,
        max=2.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_nose_alar_samples = IntProperty(
        name="Nose Alar Samples",
        description="Reserved sampling density for the nostril-wing correction",
        default=12,
        min=4,
        max=64,
    )
    bpy.types.Scene.hfr_gen_brow_ridge_fit = BoolProperty(
        name="Brow Ridge Fit",
        description="Locally fit the band between the eye-upper rail and the brow rail",
        default=True,
    )
    bpy.types.Scene.hfr_gen_brow_ridge_strength = FloatProperty(
        name="Brow Ridge Strength",
        description="How strongly the brow band follows interpolated brow / eye landmark movement",
        default=0.80,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_brow_ridge_radius = FloatProperty(
        name="Brow Ridge Radius",
        description="Auto radius multiplier for the local brow ridge correction band",
        default=1.15,
        min=0.05,
        max=2.5,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_brow_ridge_samples = IntProperty(
        name="Brow Ridge Samples",
        description="Sampling density along the brow and eye-upper rails",
        default=20,
        min=4,
        max=64,
    )
    bpy.types.Scene.hfr_gen_brow_ridge_smooth = FloatProperty(
        name="Brow Ridge Smooth",
        description="Light smoothing for the middle of the brow band to avoid a sharp crease",
        default=0.22,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_brow_inner_support = BoolProperty(
        name="Brow Inner Support",
        description="Move the small under-brow support fan with LM_brow_l/r_inner so connected vertices do not stay high",
        default=True,
    )
    bpy.types.Scene.hfr_gen_brow_inner_strength = FloatProperty(
        name="Brow Inner Strength",
        description="How strongly the under-brow support fan follows brow_inner / eye_upper_inner / nose_root deltas",
        default=0.70,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_brow_inner_steps = IntProperty(
        name="Brow Inner Steps",
        description="Topological radius from brow_inner anchor vertices for the inner support fan",
        default=2,
        min=1,
        max=5,
    )
    bpy.types.Scene.hfr_gen_brow_inner_radius = FloatProperty(
        name="Brow Inner Radius",
        description="Source-space radius multiplier limiting the under-brow support fan",
        default=1.10,
        min=0.05,
        max=2.5,
        precision=2,
    )

    bpy.types.Scene.hfr_gen_eye_loop_fit = BoolProperty(
        name="Eye Boundary Fit",
        description="Fit actual eye-hole boundary vertices between eye landmarks",
        default=True,
    )
    bpy.types.Scene.hfr_gen_eye_loop_strength = FloatProperty(
        name="Eye Boundary Strength",
        description="How strongly eye-hole boundary vertices follow interpolated eye landmark movement",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_eye_loop_max_len = IntProperty(
        name="Eye Boundary Max Path",
        description="Maximum boundary path length used between neighboring eye landmarks. 0 uses no limit",
        default=48,
        min=0,
        max=160,
    )
    bpy.types.Scene.hfr_gen_eye_direct_fit = BoolProperty(
        name="Eye Direct Fit",
        description="Directly move source-space vertices close to the eye landmark loop so between-landmark eyelid vertices follow",
        default=True,
    )
    bpy.types.Scene.hfr_gen_eye_direct_radius = FloatProperty(
        name="Eye Direct Radius",
        description="Source-space radius multiplier for the direct eye loop fit",
        default=0.90,
        min=0.05,
        max=2.50,
        precision=2,
    )

    bpy.types.Scene.hfr_gen_eye_band_steps = IntProperty(
        name="Eye Band Steps",
        description="Topological expansion steps from eye anchors for the local eyelid support-band fit",
        default=3,
        min=1,
        max=8,
    )
    bpy.types.Scene.hfr_gen_eye_band_radius = FloatProperty(
        name="Eye Band Radius",
        description="Source-space radius multiplier used to trim the local eyelid support-band fit",
        default=1.45,
        min=0.30,
        max=3.00,
        precision=2,
    )

    bpy.types.Scene.hfr_gen_feature_loops = BoolProperty(
        name="Feature Loops",
        description="Locally refine eye and mouth loop deformation by interpolating between ordered loop anchors",
        default=True,
    )
    bpy.types.Scene.hfr_gen_loop_strength = FloatProperty(
        name="Loop Strength",
        description="How strongly eye/mouth vertices close to feature loops follow segment-interpolated anchor movement",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_loop_radius = FloatProperty(
        name="Loop Radius",
        description="Auto radius multiplier for the local eye/mouth feature-loop deformation zone",
        default=1.15,
        min=0.05,
        max=4.0,
        precision=2,
    )

    bpy.types.Scene.hfr_gen_ear_lobe_fit = BoolProperty(
        name="Ear Lobe Fit",
        description="Apply an additional local loop correction around the ear outer ring, including the ear_lobe landmarks",
        default=False,
    )
    bpy.types.Scene.hfr_gen_ear_strength = FloatProperty(
        name="Ear Strength",
        description="How strongly the ear outer ring follows ear/lobe anchor movement",
        default=0.75,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_radius = FloatProperty(
        name="Ear Radius",
        description="Auto radius multiplier for the local ear outer-ring deformation zone",
        default=1.25,
        min=0.05,
        max=5.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_y_guard = BoolProperty(
        name="Ear Lobe Y Guard",
        description="Keep ear_lobe from drifting forward/back when it is pulled mainly downward/upward",
        default=True,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_y_strength = FloatProperty(
        name="Ear Lobe Y Strength",
        description="How strongly ear_lobe Y is stabilized against neighboring lower-ear anchors",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_relative = BoolProperty(
        name="Ear Lobe Relative Solve",
        description="Solve ear_lobe relative to lower-ear anchors so downward edits do not turn into forward/up spikes",
        default=True,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_relative_strength = FloatProperty(
        name="Ear Lobe Relative Strength",
        description="How strongly the ear_lobe uses the relative lower-ear solve",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_xy_strength = FloatProperty(
        name="Ear Lobe XY Lock Strength",
        description="How strongly ear_lobe X/Y follow neighboring lower-ear anchors",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lower_rail = BoolProperty(
        name="Ear Lower Rail",
        description="Use open lower-ear rails through the lobe instead of relying only on the closed ear outer loop",
        default=False,
    )
    bpy.types.Scene.hfr_gen_ear_lower_rail_strength = FloatProperty(
        name="Ear Lower Rail Strength",
        description="How strongly lower-ear vertices follow the open front-lobe-back rail",
        default=0.90,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lower_rail_radius = FloatProperty(
        name="Ear Lower Rail Radius",
        description="Search radius multiplier for the open lower-ear rail correction",
        default=0.90,
        min=0.05,
        max=5.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_patch = BoolProperty(
        name="Ear Lobe Patch",
        description="Disabled in v0.2.10; retained only for old scene compatibility",
        default=False,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_patch_strength = FloatProperty(
        name="Ear Lobe Patch Strength",
        description="How strongly nearby lower-ear vertices follow the ear_lobe anchor delta",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_lobe_patch_steps = FloatProperty(
        name="Ear Lobe Patch Steps",
        description="Topological neighborhood depth around each ear_lobe anchor",
        default=4,
        min=1,
        max=12,
        precision=0,
    )
    bpy.types.Scene.hfr_gen_ear_strip_fit = BoolProperty(
        name="Ear Strip Fit",
        description="Disabled in v0.2.10; retained only for old scene compatibility",
        default=False,
    )
    bpy.types.Scene.hfr_gen_ear_strip_strength = FloatProperty(
        name="Ear Strip Strength",
        description="How strongly the post-lock lower-ear strip fit reshapes the lobe area",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_strip_y_lock = FloatProperty(
        name="Ear Strip Y Lock",
        description="How strongly the post-lock lower-ear strip is kept on the lower-ear Y frame",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_sparse_ear_safe = BoolProperty(
        name="Sparse Ear Safe",
        description="Conservative post-lock ear-lobe stabilization for sparse lower-ear topology",
        default=True,
    )
    bpy.types.Scene.hfr_gen_sparse_ear_y_strength = FloatProperty(
        name="Sparse Ear Y Strength",
        description="How strongly the sparse ear-lobe solve clamps the lobe to the local lower-ear Y plane",
        default=1.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_sparse_ear_neighbor_blend = FloatProperty(
        name="Sparse Ear Neighbor Blend",
        description="Soft Y blending amount for one-ring neighbors around sparse ear-lobe anchors",
        default=0.35,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_lobe_directional_stretch = BoolProperty(
        name="Lobe Directional Stretch",
        description="Stretch the lower ear in the exact dragged direction of the ear_lobe landmark instead of trying to keep it natural",
        default=True,
    )
    bpy.types.Scene.hfr_gen_lobe_directional_strength = FloatProperty(
        name="Lobe Directional Strength",
        description="How strongly neighboring lower-ear vertices follow the full ear_lobe drag vector",
        default=1.0,
        min=0.0,
        max=2.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_lobe_directional_steps = IntProperty(
        name="Lobe Directional Steps",
        description="Topological ring depth that follows the ear_lobe drag vector",
        default=2,
        min=1,
        max=8,
    )
    bpy.types.Scene.hfr_gen_lobe_directional_falloff = FloatProperty(
        name="Lobe Directional Falloff",
        description="Per-ring falloff for directional lower-ear stretching",
        default=0.65,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_local_fit = BoolProperty(
        name="Ear Local Fit",
        description="Re-fit the ear patch using same-side ear anchors only, preserving the local ear loop order",
        default=True,
    )
    bpy.types.Scene.hfr_gen_ear_local_strength = FloatProperty(
        name="Ear Local Strength",
        description="How strongly the ear-local patch follows the same-side ear anchor frame",
        default=0.82,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_local_steps = IntProperty(
        name="Ear Local Steps",
        description="Topological ring depth for shell-only Ear Local Fit. Internally clamped to avoid leaking into nape/head attachment strips",
        default=4,
        min=1,
        max=8,
    )
    bpy.types.Scene.hfr_gen_ear_local_nearest = IntProperty(
        name="Ear Local Nearest",
        description="Nearest same-side ear anchors used for the ear MLS fit. 0 uses all ear anchors",
        default=0,
        min=0,
        max=16,
    )
    bpy.types.Scene.hfr_gen_head_round_fit = BoolProperty(
        name="Head Dome Fit",
        description="Remap the scalp from the original dome to a new dome estimated from scalp/head landmarks",
        default=True,
    )
    bpy.types.Scene.hfr_gen_head_round_strength = FloatProperty(
        name="Head Dome Strength",
        description="How strongly the scalp follows the dome remap after the main deformation",
        default=0.80,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_head_round_steps = IntProperty(
        name="Head Dome Steps",
        description="Topological depth used to collect the scalp region for the dome solve",
        default=8,
        min=1,
        max=20,
    )
    bpy.types.Scene.hfr_gen_head_round_iters = IntProperty(
        name="Head Dome Smooth Iterations",
        description="How many light scalp-only smoothing passes run after the dome remap",
        default=2,
        min=0,
        max=8,
    )
    bpy.types.Scene.hfr_gen_head_round_z_margin = FloatProperty(
        name="Head Dome Base Z Margin",
        description="Extra downward Z allowance when collecting the scalp region below top scalp landmarks",
        default=0.30,
        min=0.0,
        max=2.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_neck_fit = BoolProperty(
        name="Neck Fit",
        description="Apply stronger local fitting to neck top/base loops so neck length can be shortened by moving neck landmarks",
        default=True,
    )
    bpy.types.Scene.hfr_gen_neck_strength = FloatProperty(
        name="Neck Strength",
        description="How strongly neck loop vertices follow neck top/base landmark movement",
        default=0.85,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_neck_radius = FloatProperty(
        name="Neck Radius",
        description="Auto radius multiplier for the local neck loop deformation zone",
        default=1.20,
        min=0.05,
        max=5.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_output_mirror_finish = BoolProperty(
        name="Output Mirror Finish",
        description="When Landmark Mirror X is enabled, mirror the final generated output from the source side to the followed side",
        default=True,
    )
    bpy.types.Scene.hfr_gen_output_mirror_epsilon = FloatProperty(
        name="Output Mirror Epsilon",
        description="Local X tolerance for treating template vertices as center-seam vertices during final output mirroring",
        default=0.0005,
        min=0.0,
        max=0.05,
        precision=5,
    )
    bpy.types.Scene.hfr_gen_snap_to_target = BoolProperty(
        name="Snap To Target",
        description="After landmark deformation, snap generated vertices toward the Target Mesh surface. Keep this off for first preview tests",
        default=False,
    )
    bpy.types.Scene.hfr_gen_snap_strength = FloatProperty(
        name="Snap Strength",
        description="Percentage move toward the nearest Target Mesh surface point",
        default=0.60,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_snap_max_dist = FloatProperty(
        name="Snap Max Distance",
        description="Maximum world-space snap distance. 0 uses an automatic conservative limit",
        default=0.0,
        min=0.0,
        max=10.0,
        precision=4,
    )
    bpy.types.Scene.hfr_gen_protect_anchors = BoolProperty(
        name="Protect Anchors",
        description="Use a lower snap strength on explicitly bound anchor vertices",
        default=True,
    )
    bpy.types.Scene.hfr_gen_anchor_snap_strength = FloatProperty(
        name="Anchor Snap Strength",
        description="Snap strength used for explicitly bound anchor vertices when Protect Anchors is enabled",
        default=0.20,
        min=0.0,
        max=1.0,
        precision=2,
    )

    bpy.types.Scene.hfr_gen_ear_snap_guard = BoolProperty(
        name="Ear Snap Guard",
        description="Use reduced snap strength on the ear-local vertex region to prevent thin ear faces from flipping to the wrong target surface",
        default=True,
    )
    bpy.types.Scene.hfr_gen_ear_snap_strength = FloatProperty(
        name="Ear Snap Strength",
        description="Maximum snap strength for ear-local vertices while Ear Snap Guard is enabled. 0 keeps the landmark-fitted ear shape",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_ear_snap_steps = IntProperty(
        name="Ear Snap Guard Steps",
        description="Topological ring depth around bound ear anchors protected by Ear Snap Guard",
        default=3,
        min=0,
        max=8,
    )
    bpy.types.Scene.hfr_gen_post_anchor_lock = BoolProperty(
        name="Post Snap Anchor Lock",
        description="After snapping, pull anchor centroids back toward their landmarks",
        default=True,
    )
    bpy.types.Scene.hfr_gen_post_anchor_lock_strength = FloatProperty(
        name="Post Snap Lock Strength",
        description="Strength used when pulling anchor centroids back to landmarks after snap",
        default=0.50,
        min=0.0,
        max=1.0,
        precision=2,
    )
    bpy.types.Scene.hfr_gen_post_anchor_iters = bpy.props.IntProperty(
        name="Post Snap Lock Iterations",
        description="Number of anchor correction passes after snap",
        default=1,
        min=0,
        max=4,
    )
    bpy.types.Scene.hfr_ui_style_open = BoolProperty(name="1. Setup", default=True)
    bpy.types.Scene.hfr_adv_options = BoolProperty(
        name="Advanced",
        description="Show user-facing advanced landmark, output, snap, and feature controls",
        default=False,
    )
    bpy.types.Scene.hfr_dev_options = BoolProperty(
        name="DevOption",
        description="Show developer-only binding, diagnostics, and solver-parameter sections. Hide this for release builds",
        default=False,
    )
    bpy.types.Scene.hfr_ui_initial_open = BoolProperty(name="Advanced: Initial Placement", default=True)
    bpy.types.Scene.hfr_ui_groups_open = BoolProperty(name="2. Landmarks", default=True)
    bpy.types.Scene.hfr_ui_final_generate_open = BoolProperty(name="3. Generate", default=True)
    bpy.types.Scene.hfr_ui_cleanup_open = BoolProperty(name="4. Cleanup", default=False)
    bpy.types.Scene.hfr_ui_edit_open = BoolProperty(name="Advanced: Landmark Tools", default=True)
    bpy.types.Scene.hfr_ui_advanced_output_open = BoolProperty(name="Advanced: Output / Snap", default=False)
    bpy.types.Scene.hfr_ui_advanced_feature_open = BoolProperty(name="Advanced: Feature Controls", default=False)
    bpy.types.Scene.hfr_ui_dev_diagnostics_open = BoolProperty(name="DevOption: Diagnostics", default=False)
    bpy.types.Scene.hfr_ui_binding_open = BoolProperty(name="DevOption: Template Binding", default=False)
    bpy.types.Scene.hfr_ui_generate_open = BoolProperty(name="DevOption: Solver Parameters", default=False)
    # Do not touch bpy.data collections during add-on registration.
    # Blender Preferences runs register() with restricted data access, so
    # collections are created/migrated lazily from operators/panel usage.
    try:
        bpy.app.timers.register(_hfr_deferred_landmark_smooth, first_interval=0.10)
    except Exception:
        pass
    schedule_auto_load_default_template(delay=0.25)
    ensure_live_update_handler()


def unregister():
    remove_live_update_handler()
    for attr in (
        "hfr_lm_scale",
        "hfr_lm_show_names",
        "hfr_lm_show_front",
        "hfr_lm_auto_guides",
        "hfr_lm_live_guides",
        "hfr_lm_mirror_x",
        "hfr_lm_mirror_dir",
        "hfr_lm_target_obj",
        "hfr_lm_use_target_fit",
        "hfr_lm_auto_scale",
        "hfr_lm_scale_ratio",
        "hfr_lm_fit_margin",
        "hfr_lm_fit_region",
        "hfr_template_obj",
        "hfr_auto_load_tpl",
        "hfr_tpl_obj_name",
        "hfr_bind_lm_id",
        "hfr_bind_mode_enabled",
        "hfr_bind_show_guides",
        "hfr_bind_mirror_tol",
        "hfr_gen_output_name",
        "hfr_gen_replace_output",
        "hfr_gen_output_wire",
        "hfr_gen_output_in_front",
        "hfr_gen_power",
        "hfr_gen_nearest",
        "hfr_gen_anchor_lock",
        "hfr_gen_anchor_iters",
        "hfr_gen_topo_propagate",
        "hfr_gen_topo_iters",
        "hfr_gen_topo_strength",
        "hfr_gen_guide_rails",
        "hfr_gen_guide_rail_strength",
        "hfr_gen_guide_rail_max_len",
        "hfr_gen_guide_rail_spread",
        "hfr_gen_guide_rail_spread_steps",
        "hfr_gen_guide_rail_spread_strength",
        "hfr_gen_mls_field",
        "hfr_gen_mls_strength",
        "hfr_gen_mls_nearest",
        "hfr_gen_guide_follow",
        "hfr_gen_guide_strength",
        "hfr_gen_guide_radius",
        "hfr_gen_nose_web_fit",
        "hfr_gen_nose_web_strength",
        "hfr_gen_nose_web_radius",
        "hfr_gen_nose_web_samples",
        "hfr_gen_nose_alar_fit",
        "hfr_gen_nose_alar_strength",
        "hfr_gen_nose_alar_radius",
        "hfr_gen_nose_alar_samples",
        "hfr_gen_brow_ridge_fit",
        "hfr_gen_brow_ridge_strength",
        "hfr_gen_brow_ridge_radius",
        "hfr_gen_brow_ridge_samples",
        "hfr_gen_brow_ridge_smooth",
        "hfr_gen_brow_inner_support",
        "hfr_gen_brow_inner_strength",
        "hfr_gen_brow_inner_steps",
        "hfr_gen_brow_inner_radius",
        "hfr_gen_eye_loop_fit",
        "hfr_gen_eye_loop_strength",
        "hfr_gen_eye_loop_max_len",
        "hfr_gen_eye_direct_fit",
        "hfr_gen_eye_direct_radius",
        "hfr_gen_eye_band_steps",
        "hfr_gen_eye_band_radius",
        "hfr_gen_feature_loops",
        "hfr_gen_loop_strength",
        "hfr_gen_loop_radius",
        "hfr_gen_ear_lobe_fit",
        "hfr_gen_ear_strength",
        "hfr_gen_ear_radius",
        "hfr_gen_ear_lobe_y_guard",
        "hfr_gen_ear_lobe_y_strength",
        "hfr_gen_ear_lobe_relative",
        "hfr_gen_ear_lobe_relative_strength",
        "hfr_gen_ear_lobe_xy_strength",
        "hfr_gen_ear_lower_rail",
        "hfr_gen_ear_lower_rail_strength",
        "hfr_gen_ear_lower_rail_radius",
        "hfr_gen_ear_lobe_patch",
        "hfr_gen_ear_lobe_patch_strength",
        "hfr_gen_ear_lobe_patch_steps",
        "hfr_gen_sparse_ear_safe",
        "hfr_gen_sparse_ear_y_strength",
        "hfr_gen_sparse_ear_neighbor_blend",
        "hfr_gen_lobe_directional_stretch",
        "hfr_gen_lobe_directional_strength",
        "hfr_gen_lobe_directional_steps",
        "hfr_gen_lobe_directional_falloff",
        "hfr_gen_ear_local_fit",
        "hfr_gen_ear_local_strength",
        "hfr_gen_ear_local_steps",
        "hfr_gen_ear_local_nearest",
        "hfr_gen_ear_lower_fit",
        "hfr_gen_ear_lower_strength",
        "hfr_gen_ear_lower_steps",
        "hfr_gen_ear_lower_nearest",
        "hfr_gen_head_round_fit",
        "hfr_gen_head_round_strength",
        "hfr_gen_head_round_steps",
        "hfr_gen_head_round_iters",
        "hfr_gen_head_round_z_margin",
        "hfr_gen_neck_fit",
        "hfr_gen_neck_strength",
        "hfr_gen_neck_radius",
        "hfr_gen_output_mirror_finish",
        "hfr_gen_output_mirror_epsilon",
        "hfr_gen_snap_to_target",
        "hfr_gen_snap_strength",
        "hfr_gen_snap_max_dist",
        "hfr_gen_protect_anchors",
        "hfr_gen_anchor_snap_strength",
        "hfr_gen_ear_snap_guard",
        "hfr_gen_ear_snap_strength",
        "hfr_gen_ear_snap_steps",
        "hfr_gen_eye_snap_guard",
        "hfr_gen_eye_snap_strength",
        "hfr_gen_eye_snap_steps",
        "hfr_gen_post_anchor_lock",
        "hfr_gen_post_anchor_lock_strength",
        "hfr_gen_post_anchor_iters",
        "hfr_ui_style_open",
        "hfr_adv_options",
        "hfr_dev_options",
        "hfr_ui_initial_open",
        "hfr_ui_groups_open",
        "hfr_ui_final_generate_open",
        "hfr_ui_cleanup_open",
        "hfr_ui_edit_open",
        "hfr_ui_advanced_output_open",
        "hfr_ui_advanced_feature_open",
        "hfr_ui_dev_diagnostics_open",
        "hfr_ui_binding_open",
        "hfr_ui_generate_open",
    ):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
