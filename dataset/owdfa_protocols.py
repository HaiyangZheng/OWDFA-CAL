# -------------------------------------------------
# 1. Class code → folder index mapping (0–40)
# -------------------------------------------------
# Notes:
#   - Folder indices 0–20 are inherited from OWDFA (ICCV 2023) and reproduced by us.
#   - Folder indices 21–40 correspond to newly added classes in OWDFA-40.
#   - We map folder indices to semantic manipulation codes for easier management
#     and extensibility.
#
# Extension guideline:
#   To add a new manipulation type, simply create a new folder following the
#   naming convention, e.g., "41_new_deepfake_method", and add a corresponding
#   entry such as "S11": 41 to the dictionary below.
#
# Folder naming convention:
#   <index>_<manipulation_type>_<method_name>

CODE2IDX = {
    # ===== Real Face =====
    "REAL": 0,   # 0_Real_CelebDF & 0_Real_FF++

    # ===== Face Swapping =====
    # From OWDFA（ICCV23）
    "S1": 1,    # 1_Swap_Deepfakes
    "S2": 2,    # 2_Swap_FaceSwap
    "S3": 3,    # 3_Swap_DeepFaceLab
    "S4": 4,    # 4_Swap_FaceShifter
    "S5": 5,    # 5_Swap_FSGAN

    # Newly added in OW-DFA40
    "S6": 21,   # 21_Swap_BlendFace
    "S7": 22,   # 22_Swap_UniFace
    "S8": 23,   # 23_Swap_MobileSwap
    "S9": 24,   # 24_Swap_e4s
    "S10": 25,  # 25_Swap_FaceDancer

    # ===== Face Reenactment =====
    # From OWDFA（ICCV23）
    "R1": 6,  # 6_Reenact_Face2Face
    "R2": 7,  # 7_Reenact_NeuralTextures
    "R3": 8,  # 8_Reenact_FOMM
    "R4": 9,  # 9_Reenact_ATVG-Net
    "R5": 10,  # 10_Reenact_TalkingHead

    # Newly added in OW-DFA40
    "R6": 26,  # 26_Reenact_TPSMM
    "R7": 27,  # 27_Reenact_LIA
    "R8": 28,  # 28_Reenact_DaGAN
    "R9": 29,  # 29_Reenact_SadTalker
    "R10": 30, # 30_Reenact_MCNet
    "R11": 31, # 31_Reenact_HyperReenact
    "R12": 40, # 40_Reenact_OneShot

    # ===== Face Editing =====
    # From OWDFA（ICCV23）
    "E1": 11,  # 11_Edit_MaskGAN
    "E2": 12,  # 12_Edit_StarGAN2
    "E3": 13,  # 13_Edit_SC-FEGAN
    "E4": 14,  # 14_Edit_FaceAPP
    "E5": 15,  # 15_Edit_StarGAN

    # Newly added in OW-DFA40
    "E6": 38,  # 38_Edit_e4e

    # ===== Entire Face Synthesis + Diffusion-based =====
    # From OWDFA（ICCV23）
    "G1": 16,  # 16_Generate_StyleGAN2_Net
    "G2": 17,  # 17_Generate_StyleGAN
    "G3": 18,  # 18_Generate_PGGAN
    "G4": 19,  # 19_Generate_CycleGAN
    "G5": 20,  # 20_Generate_StyleGAN2_NIR

    # Newly added in OW-DFA40
    "G6": 32,  # 32_Generate_StyleGAN-XL
    "G7": 33,  # 33_Generate_SD2.1
    "G8": 34,  # 34_Generate_RDDM
    "G9":  35, # 35_Generate_PixArt
    "G10": 36, # 36_Generate_DiT-XL
    "G11": 37, # 37_Generate_SiT-XL
    "G12": 39, # 39_Generate_StyleGAN3
}

# -------------------------------------------------
# 2. Training protocols
# -------------------------------------------------
# Each protocol defines:
#   - known_cls_codes: manipulation types treated as known classes during training
#   - train_cls_codes: classes included in the training split ("ALL" means all classes)
#
# These protocols are designed to simulate different open-world settings
# with varying degrees of known/novel class exposure.
PROTOCOLS = {
    1: {
        "known_cls_codes": [
            "REAL",
            "S1", "S3", "S6", "S8", "S10",
            "R1", "R3", "R7", "R9", "R11",
            "E1", "E4",
            "G2", "G4", "G7", "G9", "G11", "G12",  
        ],
        "train_cls_codes": "ALL",  # use all available classes for training
    },

    2: {
        "known_cls_codes": [
            "REAL",
            "S1", "S3", "S6", "S8", "S10",
            "R1", "R3", "R7", "R9", "R11",
            "E1", "E4",
        ],
        "train_cls_codes": "ALL", # use all available classes for training
    },

    3: {
        "known_cls_codes": [
            "REAL",
            "S1", "S3", "S5", "S6", "S8", "S9", "S10",
            "R1", "R2", "R3", "R5", "R6", "R7", "R9", "R11",
            "E1", "E3", "E4", "E6",
            "G2", "G3", "G4", "G5", "G7", "G9", "G10", "G11", "G12",
        ],
        "train_cls_codes": "ALL", # use all available classes for training
    },
}


# -------------------------------------------------
# 3. Unified interface: protocol → class indices
# -------------------------------------------------
def get_classes_from_protocol(protocol_id):
    """
    Given a protocol ID, return the corresponding class indices.

    Args:
        protocol_id (int): Protocol identifier (e.g., 1, 2, or 3).

    Returns:
        known_classes (List[int]):
            Indices of known classes under the selected protocol.
        train_classes (List[int]):
            Indices of all classes used during training.

    Raises:
        ValueError: If the protocol ID is not defined.
    """
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Protocol {protocol_id} is not defined!")

    cfg = PROTOCOLS[protocol_id]

    # ---- Known classes ----
    known_codes = cfg["known_cls_codes"]
    known_classes = sorted([CODE2IDX[c] for c in known_codes])

    # ---- Training classes ----
    if cfg["train_cls_codes"] == "ALL":
        train_classes = sorted(CODE2IDX.values())
    else:
        train_codes = cfg["train_cls_codes"]
        train_classes = sorted([CODE2IDX[c] for c in train_codes])

    return known_classes, train_classes