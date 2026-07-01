from pathlib import Path
from typing import Sequence
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag

def collate_fn(batch):
    return batch

def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,  # 新增参数
    delete_pause_frame: bool = False,
    sample_mode: str = "single_step",
    action_horizon: int = 7,
    video_horizon: int = 16,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    data_config_cls = ROBOT_TYPE_CONFIG_MAP[robot_type]
    data_config = data_config_cls(
        observation_indices=list(range(video_horizon)),
        action_indices=list(range(action_horizon))
    )
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="torchvision_av",
        delete_pause_frame=delete_pause_frame,
        sample_mode=sample_mode,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    delete_pause_frame: bool | None = None,
    sample_mode: str | None = None,
    segment_length: int | None = None,
    burn_in_max_decisions: int | None = None,
    segment_stride: int | None = None,
    action_horizon: int = 7,
    video_horizon: int = 16,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = (
        bool(data_cfg.get("delete_pause_frame", True))
        if delete_pause_frame is None
        else bool(delete_pause_frame)
    )
    sample_mode = (
        str(data_cfg.get("sample_mode", "single_step"))
        if sample_mode is None
        else str(sample_mode)
    )
    segment_length = (
        int(data_cfg.get("segment_length", 4))
        if segment_length is None
        else int(segment_length)
    )
    burn_in_max_decisions = (
        int(data_cfg.get("burn_in_max_decisions", 8))
        if burn_in_max_decisions is None
        else int(burn_in_max_decisions)
    )
    segment_stride = (
        int(data_cfg.get("segment_stride", action_horizon))
        if segment_stride is None
        else int(segment_stride)
    )
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((make_LeRobotSingleDataset(Path(data_root_dir), 
                                                          d_name, 
                                                          robot_type, 
                                                          delete_pause_frame=delete_pause_frame,
                                                          sample_mode=sample_mode,
                                                          action_horizon=action_horizon,
                                                          video_horizon=video_horizon), d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        with_state=data_cfg.get("with_state", False),
        resolution_size=data_cfg.get("resolution_size", 224),
        video_resolution_size=data_cfg.get("video_resolution_size", 256),
        seed=seed,
        sample_mode=sample_mode,
        segment_length=segment_length,
        burn_in_max_decisions=burn_in_max_decisions,
        segment_stride=segment_stride,
        **kwargs,
    )

if __name__ == "__main__":
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=1, # For Debug
        collate_fn=collate_fn,
    )

    from tqdm import tqdm
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        print(batch)
        pass
