"""
Protocol Map Builder for Stage 1.

Generates training and testing protocol maps using odor schedules.
Maps trial_label to odor_name, reward, and cs_type.

Security: Does NOT print raw data rows. Only shapes and counts.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# ===== MAPPING CONSTANTS (from experiment design) =====

HEXANOL = "Hexanol"

ODOR_CANON = {
    "acv": "ACV",
    "apple cider vinegar": "ACV",
    "apple-cider-vinegar": "ACV",
    "3-octonol": "3-octonol",
    "3 octonol": "3-octonol",
    "3-octanol": "3-octonol",
    "3 octanol": "3-octonol",
    "benz": "Benz",
    "benzaldehyde": "Benz",
    "benz-ald": "Benz",
    "benzadhyde": "Benz",
    "ethyl butyrate": "EB",
    "eb_control": "EB_control",
    "eb control": "EB_control",
    "hex_control": "hex_control",
    "hex control": "hex_control",
    "benz_control": "benz_control",
    "benz control": "benz_control",
    "optogenetics benzaldehyde": "opto_benz",
    "optogenetics benzaldehyde 1": "opto_benz_1",
    "optogenetics ethyl butyrate": "opto_EB",
    "opto_eb(6-training)": "opto_EB_6_training",
    "10s_odor_benz": "10s_Odor_Benz",
    "optogenetics apple cider vinegar": "opto_ACV",
    "optogenetics acv": "opto_ACV",
    "optogenetics hexanol": "opto_hex",
    "optogenetics hex": "opto_hex",
    "hexanol": "opto_hex",
    "opto_hex": "opto_hex",
    "opto_air": "opto_AIR",
    "opto_acv": "opto_ACV",
    "optogenetics 3-octanol": "opto_3-oct",
    "opto_3-oct": "opto_3-oct",
}

DISPLAY_LABEL = {
    "ACV": "Apple Cider Vinegar",
    "3-octonol": "3-Octonol",
    "Benz": "Benzaldehyde",
    "10s_Odor_Benz": "Benzaldehyde",
    "EB": "Ethyl Butyrate",
    "EB_control": "Ethyl Butyrate",
    "hex_control": "Hexanol",
    "benz_control": "Benzaldehyde",
    "opto_benz": "Benzaldehyde",
    "opto_benz_1": "Benzaldehyde",
    "opto_EB": "Ethyl Butyrate",
    "opto_EB_6_training": "Ethyl Butyrate (6-Training)",
    "opto_ACV": "Apple Cider Vinegar",
    "opto_hex": "Hexanol",
    "opto_AIR": "AIR",
    "opto_3-oct": "3-Octonol",
}

PRIMARY_ODOR_LABEL = {
    "EB_control": "Ethyl Butyrate",
    "hex_control": HEXANOL,
    "benz_control": "Benzaldehyde",
}

TRAINING_ODOR_SCHEDULE_DEFAULT = {
    1: "Benzaldehyde",
    2: "Benzaldehyde",
    3: "Benzaldehyde",
    4: "Benzaldehyde",
    5: HEXANOL,
    6: "Benzaldehyde",
    7: HEXANOL,
    8: "Benzaldehyde",
}

TRAINING_ODOR_SCHEDULE_OVERRIDES = {
    "hex_control": {
        1: HEXANOL,
        2: HEXANOL,
        3: HEXANOL,
        4: HEXANOL,
        5: "Apple Cider Vinegar",
        6: HEXANOL,
        7: "Apple Cider Vinegar",
        8: HEXANOL,
    },
    "opto_hex": {
        1: HEXANOL,
        2: HEXANOL,
        3: HEXANOL,
        4: HEXANOL,
        5: "Apple Cider Vinegar",
        6: HEXANOL,
        7: "Apple Cider Vinegar",
        8: HEXANOL,
    },
    "EB_control": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: HEXANOL,
        6: "Ethyl Butyrate",
        7: HEXANOL,
        8: "Ethyl Butyrate",
    },
    "opto_EB": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: HEXANOL,
        6: "Ethyl Butyrate",
        7: HEXANOL,
        8: "Ethyl Butyrate",
    },
    "opto_EB_6_training": {
        1: "Ethyl Butyrate",
        2: "Ethyl Butyrate",
        3: "Ethyl Butyrate",
        4: "Ethyl Butyrate",
        5: HEXANOL,
        6: "Ethyl Butyrate",
        7: HEXANOL,
        8: "Ethyl Butyrate",
    },
    "opto_AIR": {
        1: "AIR",
        2: "AIR",
        3: "AIR",
        4: "AIR",
        5: HEXANOL,
        6: "AIR",
        7: HEXANOL,
        8: "AIR",
    },
    "opto_3-oct": {
        1: "3-Octonol",
        2: "3-Octonol",
        3: "3-Octonol",
        4: "3-Octonol",
        5: HEXANOL,
        6: "3-Octonol",
        7: HEXANOL,
        8: "3-Octonol",
    },
    "ACV": {
        1: "Apple Cider Vinegar",
        2: "Apple Cider Vinegar",
        3: "Apple Cider Vinegar",
        4: "Apple Cider Vinegar",
        5: HEXANOL,
        6: "Apple Cider Vinegar",
        7: HEXANOL,
        8: "Apple Cider Vinegar",
    },
    "opto_ACV": {
        1: "Apple Cider Vinegar",
        2: "Apple Cider Vinegar",
        3: "Apple Cider Vinegar",
        4: "Apple Cider Vinegar",
        5: HEXANOL,
        6: "Apple Cider Vinegar",
        7: HEXANOL,
        8: "Apple Cider Vinegar",
    },
}

TESTING_DATASET_ALIAS = {
    "opto_hex": "hex_control",
    "opto_EB": "EB_control",
    "opto_EB_6_training": "EB_control",
    "opto_benz": "benz_control",
    "opto_benz_1": "benz_control",
    "opto_AIR": "opto_AIR",
    "opto_ACV": "ACV",
    "opto_3-oct": "opto_3-oct",
}

# Testing pulse -> odor mapping
TESTING_PULSE_ODOR_MAPPING = {
    "ACV": {
        6: "3-Octonol",
        7: "Ethyl Butyrate",
        8: "Benzaldehyde",
        9: "Citral",
        10: "Linalool",
    },
    "3-octonol": {6: "Benzaldehyde", 7: "Citral", 8: "Linalool"},
    "Benz": {6: "Citral", 7: "Linalool"},
    "benz_control": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
    "EB": {6: "Apple Cider Vinegar", 7: "3-Octonol", 8: "Benzaldehyde", 9: "Citral", 10: "Linalool"},
    "EB_control": {
        6: "Apple Cider Vinegar",
        7: "3-Octonol",
        8: "Benzaldehyde",
        9: "Citral",
        10: "Linalool",
    },
    "hex_control": {6: "Benzaldehyde", 7: "3-Octonol", 8: "Ethyl Butyrate", 9: "Citral", 10: "Linalool"},
    "10s_Odor_Benz": {6: "Benzaldehyde", 7: "Benzaldehyde"},
    "opto_AIR": {
        1: "Hexanol",
        2: "AIR",
        3: "Hexanol",
        4: "AIR",
        5: "AIR",
        6: "Apple Cider Vinegar",
        7: "Ethyl Butyrate",
        8: "Benzaldehyde",
        9: "Citral",
        10: "3-Octonol",
    },
    "opto_ACV": {
        6: "3-Octonol",
        7: "Ethyl Butyrate",
        8: "Benzaldehyde",
        9: "Citral",
        10: "Linalool",
    },
    "opto_3-oct": {
        6: "Apple Cider Vinegar",
        7: "Ethyl Butyrate",
        8: "Benzaldehyde",
        9: "Citral",
        10: "Linalool",
    },
}


def canonicalize_odor(name: str) -> str:
    """Map odor name to canonical form using ODOR_CANON."""
    if not name or pd.isna(name):
        return "UNKNOWN"
    name_lower = str(name).lower().strip()
    return ODOR_CANON.get(name_lower, name)


def display_label(canon: str) -> str:
    """Get display label for canonical odor name."""
    if not canon or canon == "UNKNOWN":
        return "UNKNOWN"
    return DISPLAY_LABEL.get(canon, canon)


def extract_pulse_idx(trial_label: str) -> Optional[int]:
    """Extract pulse index from trial_label (e.g., 'trial_5' -> 5)."""
    if not trial_label or pd.isna(trial_label):
        return None
    match = re.search(r'_(\d+)$', str(trial_label).strip())
    if match:
        return int(match.group(1))
    return None


def build_protocol_map_for_training(training_csv_path: str, out_csv_path: str) -> None:
    """
    Generate protocol map for training data.

    Reads training wide CSV, determines odor schedule based on dataset condition,
    extracts pulse_idx, and maps to odor_name using schedules.

    Rules:
    - Pulses 1-4, 6, 8: Primary odor (CS+), reward=1
    - Pulses 5, 7: Interleaved odor (CS-), reward=0 (no reward given during training)
    - Use TRAINING_ODOR_SCHEDULE_OVERRIDES if dataset in overrides, else DEFAULT
    """
    print(f"[Training] Loading from {training_csv_path}")
    train_df = pd.read_csv(training_csv_path)
    print(f"  Shape: {train_df.shape}")

    records = []

    for _, row in train_df.iterrows():
        dataset = row.get("dataset", "UNKNOWN")
        trial_label = row.get("trial_label", "UNKNOWN")
        
        pulse_idx = extract_pulse_idx(trial_label)
        if pulse_idx is None:
            print(f"  Warning: Could not extract pulse_idx from trial_label={trial_label}, skipping")
            continue

        # Select schedule based on dataset condition
        if dataset in TRAINING_ODOR_SCHEDULE_OVERRIDES:
            schedule = TRAINING_ODOR_SCHEDULE_OVERRIDES[dataset]
        else:
            schedule = TRAINING_ODOR_SCHEDULE_DEFAULT

        if pulse_idx not in schedule:
            print(f"  Warning: pulse_idx {pulse_idx} not in schedule for dataset={dataset}, skipping")
            continue

        odor_name = schedule[pulse_idx]
        
        # Determine reward and cs_type
        # Pulses 5, 7: interleaved odor (no reward during training = control odor)
        # Pulses 1-4, 6, 8: primary odor (CS+)
        if pulse_idx in [5, 7]:
            # Interleaved odor = no reward (CS-)
            reward = 0
            cs_type = "CS-"
        else:
            # Primary odor = reward given (CS+)
            reward = 1
            cs_type = "CS+"

        records.append({
            "dataset": dataset,
            "trial_label": trial_label,
            "pulse_idx": pulse_idx,
            "odor_name": odor_name,
            "odor_display": display_label(canonicalize_odor(odor_name)),
            "reward": reward,
            "cs_type": cs_type,
            "phase": "training",
        })

    out_df = pd.DataFrame(records)
    print(f"  Generated {len(out_df)} rows")
    
    # Check for duplicates on (dataset, trial_label) - must be unique per row since fly varies
    dup_count = out_df.duplicated(subset=["dataset", "trial_label"], keep=False).sum()
    if dup_count > 0:
        # This is expected - same trial_label appears for multiple flies
        # We keep all since protocol_map is used in a join on (dataset, trial_label)
        pass

    out_df.to_csv(out_csv_path, index=False)
    print(f"  Written to {out_csv_path}")


def build_protocol_map_for_testing(
    testing_csv_path: str,
    out_csv_path: str,
    model_predictions_csv_path: Optional[str] = None
) -> None:
    """
    Generate protocol map for testing data.

    Reads testing wide CSV, applies dataset alias mapping, uses TESTING_PULSE_ODOR_MAPPING
    to assign odor_name based on pulse_idx.

    For reward/cs_type: marked as UNKNOWN unless model_predictions_csv provides PER labels.
    """
    print(f"[Testing] Loading from {testing_csv_path}")
    test_df = pd.read_csv(testing_csv_path)
    print(f"  Shape: {test_df.shape}")

    # Load model predictions if available (for PER labels)
    per_dict = {}  # {(dataset, fly, trial_label): per_prediction}
    if model_predictions_csv_path:
        print(f"[Testing] Loading model predictions from {model_predictions_csv_path}")
        pred_df = pd.read_csv(model_predictions_csv_path)
        for _, row in pred_df.iterrows():
            key = (row.get("dataset"), row.get("fly"), row.get("trial_label"))
            per_dict[key] = row.get("prediction", 0)
        print(f"  Loaded {len(per_dict)} predictions")

    records = []

    for _, row in test_df.iterrows():
        dataset = row.get("dataset", "UNKNOWN")
        fly = row.get("fly", "UNKNOWN")
        trial_label = row.get("trial_label", "UNKNOWN")
        
        pulse_idx = extract_pulse_idx(trial_label)
        if pulse_idx is None:
            print(f"  Warning: Could not extract pulse_idx from trial_label={trial_label}, skipping")
            continue

        # Apply dataset alias if needed
        alias_dataset = TESTING_DATASET_ALIAS.get(dataset, dataset)

        # Get odor mapping for this dataset+pulse
        odor_name = "UNKNOWN"
        if alias_dataset in TESTING_PULSE_ODOR_MAPPING:
            pulse_map = TESTING_PULSE_ODOR_MAPPING[alias_dataset]
            odor_name = pulse_map.get(pulse_idx, "UNKNOWN")

        # For testing, reward is UNKNOWN unless we have PER labels
        reward = -1
        cs_type = "UNKNOWN"
        per_prediction = per_dict.get((dataset, fly, trial_label), None)
        if per_prediction is not None:
            # If model predicted PER (1), we can infer some reward context,
            # but for now just keep it UNKNOWN for testing phase
            pass

        records.append({
            "dataset": dataset,
            "trial_label": trial_label,
            "pulse_idx": pulse_idx,
            "odor_name": odor_name,
            "odor_display": display_label(canonicalize_odor(odor_name)),
            "reward": reward,
            "cs_type": cs_type,
            "phase": "testing",
        })

    out_df = pd.DataFrame(records)
    print(f"  Generated {len(out_df)} rows")

    out_df.to_csv(out_csv_path, index=False)
    print(f"  Written to {out_csv_path}")


def merge_protocol_maps(training_map_path: str, testing_map_path: str, out_csv_path: str) -> None:
    """
    Merge training and testing protocol maps into a single file for Stage 1 to use.
    """
    print(f"[Merge] Combining training and testing protocol maps...")
    train_map = pd.read_csv(training_map_path)
    test_map = pd.read_csv(testing_map_path)
    
    # Ensure required columns
    required = ["dataset", "trial_label", "odor_name", "reward", "cs_type"]
    for col in required:
        if col not in train_map.columns or col not in test_map.columns:
            raise ValueError(f"Missing required column: {col}")
    
    merged = pd.concat([train_map, test_map], ignore_index=True)
    print(f"  Training rows: {len(train_map)}, Testing rows: {len(test_map)}")
    print(f"  Merged rows: {len(merged)}")
    
    merged.to_csv(out_csv_path, index=False)
    print(f"  Written to {out_csv_path}")
