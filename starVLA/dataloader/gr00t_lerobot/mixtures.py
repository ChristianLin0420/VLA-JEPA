"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],

    "droid": [
        ("", 1.0, "libero_franka"),
    ],

    "fr3_realworld": [
        ("", 1.0, "fr3_real_world"),
    ],

    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
        # ("libero_90_no_noops_lerobot", 1.0, "libero_ur5"),
    ],

    # Full cross-embodiment robot mixture: LIBERO x4 (libero_franka) + Droid
    # (droid_libero schema, oxe_droid tag) + Bridge (oxe_bridge) + Fractal (oxe_rt1).
    # All video payloads
    # are resumed in full from IPEC-COMMUNITY/{droid,bridge_orig,fractal20220817_data}
    # _lerobot. Droid uses the LIBERO-compatible config (action-7 / state-8, same
    # video.primary_image/video.wrist_image keys) but keeps its own normalization tag.
    # Use with_state=false so the shared
    # 7-dim EEF-delta action space works across heterogeneous embodiments.
    "all_robot": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("droid_lerobot", 1.0, "droid_libero"),
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


}
