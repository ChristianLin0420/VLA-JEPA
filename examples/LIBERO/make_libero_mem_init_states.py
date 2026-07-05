"""Generate LIBERO-style .pruned_init files for the LIBERO-Mem suite.

The libero-mem fork (github.com/libero-mem/libero-mem) registers the
`libero_mem` benchmark but ships NO `init_files/libero_mem/` -- their own
evaluator (scripts/mem_5_run_evaluation_env_gt.py) instead reads each demo's
`init_state` (or `states[0]`) straight from the demonstration HDF5s.  Our
eval_libero.py, however, uses the standard
`task_suite.get_task_init_states(task_id)` path, which torch.loads
`<LIBERO_HOME>/libero/libero/init_files/<suite>/<task>.pruned_init`.

This script bridges the two: for every libero_mem task it stacks the per-demo
initial sim states from the LIBERO-Mem-Raw HDF5s (120 demos/task; natural
sort order demo_1..demo_120, identical to their evaluator's
`natural_sort_key` ordering, so 'seen' = indices [0,100) and 'unseen' =
[100,120)) and torch.saves the (N, 58) float64 array as a .pruned_init.

By default it emits BOTH orderings' worth of states (all 120), so
NUM_TRIALS<=20 with --split unseen replicates their held-out protocol and
--split all gives the full pool.  eval_libero.py indexes
initial_states[episode_idx], so which states an eval consumes is set purely
by what this script writes; we default to `unseen-first` = the 20 held-out
init states followed by the 100 seen ones, making trials 0..19 the
paper-comparable protocol.

Usage (login node, vlajepa_eval env):
  /lustre/fsw/portfolios/edgeai/users/chrislin/miniconda3/envs/vlajepa_eval/bin/python \
      examples/LIBERO/make_libero_mem_init_states.py \
      --raw-dir /lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/libero-mem-data/LIBERO-Mem-Raw \
      --libero-home /lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/libero-mem-home
"""

import argparse
import pathlib
import re

import h5py
import numpy as np
import torch

LIBERO_MEM_TASKS = [
    "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
    "KITCHEN_SCENE1_2_lift_the_bottle_and_put_it_down_on_the_plate",
    "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times",
    "KITCHEN_SCENE1_4_pick_up_the_bottle_and_put_it_down_the_plate_3_times",
    "KITCHEN_SCENE1_5_lift_the_bowl_and_place_it_back_on_the_plate_5_times",
    "KITCHEN_SCENE1_6_pick_up_the_bowl_and_place_it_on_the_plate_7_times",
    "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate",
    "KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate",
    "KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center",
    "KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center",
]

SEEN_COUNT = 100  # their evaluator: seen = first 100 (natural sort), unseen = last 20


def natural_sort_key(s):
    # mirrors scripts/mem_5_run_evaluation_env_gt.py
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="LIBERO-Mem-Raw snapshot dir")
    ap.add_argument("--libero-home", required=True, help="libero-mem-home tree")
    ap.add_argument(
        "--order",
        choices=["unseen-first", "natural"],
        default="unseen-first",
        help="unseen-first: 20 held-out inits first (trials 0..19 = their "
        "unseen protocol); natural: demo_1..demo_120 as-is",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.libero_home) / "libero/libero/init_files/libero_mem"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in LIBERO_MEM_TASKS:
        h5_path = pathlib.Path(args.raw_dir) / f"{task}_demo.hdf5"
        with h5py.File(h5_path, "r") as f:
            data = f["data"]
            demos = sorted(data.keys(), key=natural_sort_key)
            states = []
            for d in demos:
                g = data[d]
                s = g["init_state"][()] if "init_state" in g else g["states"][()][0]
                states.append(np.asarray(s, dtype=np.float64))
        arr = np.stack(states, axis=0)
        if args.order == "unseen-first":
            arr = np.concatenate([arr[SEEN_COUNT:], arr[:SEEN_COUNT]], axis=0)
        out_path = out_dir / f"{task}.pruned_init"
        # Save as a torch Tensor (not numpy): torch>=2.6 torch.load defaults to
        # weights_only=True, which rejects pickled numpy arrays; the stock
        # LIBERO .pruned_init files load under that default, so match them.
        torch.save(torch.from_numpy(arr), out_path)
        print(f"{task}: {tuple(arr.shape)} -> {out_path}")


if __name__ == "__main__":
    main()
