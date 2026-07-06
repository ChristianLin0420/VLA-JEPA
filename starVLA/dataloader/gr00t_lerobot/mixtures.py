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

    # MIKASA-Robo-VLA anchor subset (21 of 90 tasks; converted LeRobot v3->v2
    # by scripts/data/mikasa_v3_to_v2.py; 128x128 top+wrist, 7-D proprio/action,
    # actions pre-normalized [-1,1] over +-0.1 m/step -> own 'mikasa_robo'
    # stats/unnorm block, never merged with LIBERO franka).
    #
    # Selection is dictated by dataloader mechanics: contiguous_segment mode
    # requires episode_length >= 1 + max_delta(7) + (K-1)*stride(3*7) = 29
    # frames or a task contributes zero segments and is skipped
    # (datasets.py:_build_segment_catalog). The short variants of
    # RememberColor/Shape, FindImposter, and no-shuffle ShellGame are 10-30
    # frames -> unusable; anchors therefore use the *_long / shuffle variants.
    # Coverage across MIKASA memory families: spatial (ShellGame*), object
    # (RememberColor{3,5,9}/Shape/ShapeAndColor), object permanence
    # (TakeItBack), sequential (ChainOfColors*/SeqOfColors), capacity
    # (BunchOfColors/GatherAndRecall), counting (BatteriesChecker/BlinkCount),
    # duration (TimedTransfer), spatiotemporal (TraceShape), velocity
    # (InterceptGrab).
    "mikasa_vla": [
        ("mikasa_shell_game_shuffle_touch_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_touch_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_color_lamp_touch_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_remember_color_3_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_remember_color_5_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_remember_color_9_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_take_it_back_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_chain_of_colors_3_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_chain_of_colors_5_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_chain_of_colors_7_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_seq_of_colors_5_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_bunch_of_colors_5_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_gather_and_recall_3_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_gather_and_recall_5_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_remember_shape_5_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_remember_shape_and_color_3x3_long_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_batteries_checker_easy_3_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_blink_count_button_press_medium_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_timed_transfer_medium_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_trace_shape_medium_vla_v0", 1.0, "mikasa_robo"),
        ("mikasa_intercept_grab_fast_vla_v0", 1.0, "mikasa_robo"),
    ],

    # all_robot + MIKASA at ~1:4 effective weight. Mechanics: the mixture is
    # built with balance_dataset_weights=False (get_vla_dataset defaults), so
    # the per-draw probability of a dataset is weight/sum(weights), independent
    # of dataset size. all_robot = 7 datasets x 1.0 = 7.0 units; MIKASA total
    # must be 7/4 = 1.75 units -> 1.75/21 = 1/12 per task. Mikasa fraction of
    # robot draws = 1.75/8.75 = 20% (1:4). Weight 1.0 also marks the existing
    # corpora as 'primary' for epoch-length accounting, which mikasa's 1/12
    # deliberately does not disturb. (Dataset multiplicity cannot express this:
    # duplicate (name, robot_type) entries are dropped by get_vla_dataset.)
    "all_robot_mikasa": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("droid_lerobot", 1.0, "droid_libero"),
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
        ("mikasa_shell_game_shuffle_touch_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_touch_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_color_lamp_touch_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_remember_color_3_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_remember_color_5_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_remember_color_9_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_take_it_back_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_chain_of_colors_3_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_chain_of_colors_5_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_chain_of_colors_7_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_seq_of_colors_5_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_bunch_of_colors_5_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_gather_and_recall_3_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_gather_and_recall_5_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_remember_shape_5_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_remember_shape_and_color_3x3_long_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_batteries_checker_easy_3_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_blink_count_button_press_medium_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_timed_transfer_medium_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_trace_shape_medium_vla_v0", 0.0833333, "mikasa_robo"),
        ("mikasa_intercept_grab_fast_vla_v0", 0.0833333, "mikasa_robo"),
    ],

    # memv2 stage-1 mixture, per the G0 demand-audit verdict: vanilla LIBERO
    # carries ~zero manufacturable episode-specific demand (R2 gap <= 0 on all
    # four suites), so half the draw mass is content-demanding data.
    # LIBERO 4x1.0 = 4.0; LIBERO-Mem 2.0; MIKASA anchors 21 x 2/21 = 2.0
    # -> demand fraction (libero_mem + mikasa) = 4/8 = 50%.
    "memv2_stage1_mix": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_mem_1.0.0_lerobot", 2.0, "libero_franka"),
        ("mikasa_shell_game_shuffle_touch_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_touch_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_shell_game_shuffle_color_lamp_touch_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_remember_color_3_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_remember_color_5_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_remember_color_9_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_take_it_back_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_chain_of_colors_3_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_chain_of_colors_5_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_chain_of_colors_7_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_seq_of_colors_5_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_bunch_of_colors_5_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_gather_and_recall_3_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_gather_and_recall_5_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_remember_shape_5_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_remember_shape_and_color_3x3_long_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_batteries_checker_easy_3_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_blink_count_button_press_medium_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_timed_transfer_medium_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_trace_shape_medium_vla_v0", 0.0952381, "mikasa_robo"),
        ("mikasa_intercept_grab_fast_vla_v0", 0.0952381, "mikasa_robo"),
    ],

}
