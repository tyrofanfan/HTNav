import argparse
from typing import Literal, Optional
from dataclasses import dataclass, asdict

# from gsamllavanav.maps.gsam_map import GSamParams


@dataclass
class ExperimentArgs:

    seed: int
    mode: Literal['train', 'eval']
    
    model: Literal['mgp', 'seq2seq_with_map', 'cma_with_map']

    # logger
    log: bool
    silent: bool
    resume_log_id: str

    # observation
    map_size: int
    map_meters: float
    map_update_interval: int
    max_depth: float
    altitude: float
    ablate: Literal['rgb', 'depth', 'tracking', 'landmark', 'gsam', '']
    alt_env: Literal['flood', 'ground_fissure', '']

    # gsam
    gsam_rgb_shape: tuple[int, int]
    gsam_use_segmentation_mask: bool
    gsam_use_bbox_confidence: bool
    gsam_use_map_cache: bool
    gsam_box_threshold: float
    gsam_text_threshold: float
    gsam_max_box_size: float
    gsam_max_box_area: float

    # training params
    learning_rate: float
    train_batch_size: int
    epochs: int
    checkpoint: Optional[str]
    save_every: int
    train_trajectory_type: Literal['sp', 'mturk', 'both']
    train_episode_sample_size: int

    # grpo
    grpo_pretrain_epochs: int
    grpo_start_epoch: int
    grpo_group_size: int
    grpo_rollout_steps: int
    grpo_temperature: float
    grpo_loss_weight: float
    grpo_kl_weight: float
    
    use_grpo: bool

    use_rl: bool 
    rl_start_epoch: int 
    rl_update_frequency: int 
    rl_loss_weight: float 
    ppo_timesteps: int 
    ppo_batch_size: int
    ppo_learning_rate: float 
    ppo_gamma: float 
    ppo_gae_lambda: float
    
    
    # eval params
    eval_every: int
    eval_batch_size: int
    eval_at_start: bool
    eval_max_timestep: int
    eval_client: Literal['crop', 'airsim']
    success_dist: float
    success_iou: float
    move_iteration: int
    progress_stop_val: float
    eval_goal_selector: Literal['gdino', 'llava']
    gps_noise_scale: float

    # airsim
    sim_ip: str
    sim_port: int

    def to_dict(self):
        return asdict(self)
    
    @property
    def map_shape(self):
        return self.map_size, self.map_size
    
    @property
    def map_pixels_per_meter(self):
        return self.map_size / self.map_meters
    
    # @property
    # def gsam_params(self):
    #     return GSamParams(
    #         self.gsam_use_segmentation_mask,
    #         self.gsam_use_bbox_confidence,
    #         self.gsam_box_threshold, self.gsam_text_threshold,
    #         self.gsam_max_box_size, self.gsam_max_box_area
    #     )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')

    parser.add_argument('--model', type=str, choices=['mgp', 'seq2seq_with_map', 'cma_with_map'], default='mgp')

    # logger
    parser.add_argument('--log', action='store_true', default=False, help="log results to wandb")
    parser.add_argument('--silent', action='store_true', default=False, help="disable printing log info to stdout")
    parser.add_argument('--resume_log_id', type=str, default='')

    # observation
    parser.add_argument('--map_size', type=int, default=240)
    parser.add_argument('--map_meters', type=float, default=410.)
    parser.add_argument('--map_update_interval', type=int, default=5)
    parser.add_argument('--max_depth', type=float, default=200.)
    parser.add_argument('--altitude', type=float, default=50)
    parser.add_argument('--ablate', type=str, choices=['rgb', 'depth', 'tracking', 'landmark', 'gsam', ''], default='')
    parser.add_argument('--alt_env', type=str, choices=['', 'flood', 'ground_fissure'], default='')

    # gsam
    parser.add_argument('--gsam_rgb_shape', type=int, default=500)
    parser.add_argument('--gsam_use_segmentation_mask', action='store_true', default=False)
    parser.add_argument('--gsam_use_bbox_confidence', action='store_true', default=False)
    parser.add_argument('--gsam_use_map_cache', action='store_true', default=False)
    parser.add_argument('--gsam_box_threshold', type=float, default=0.35)
    parser.add_argument('--gsam_text_threshold', type=float, default=0.25)
    parser.add_argument('--gsam_max_box_size', type=float, default=50.)
    parser.add_argument('--gsam_max_box_area', type=float, default=3000.)
    
    # training params
    parser.add_argument('--learning_rate', type=float, default=1.0e-03)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--train_trajectory_type', type=str, choices=['sp', 'mturk', 'both'], default='sp')
    parser.add_argument('--train_episode_sample_size', type=int, default=-1)

    # grpo
    parser.add_argument('--grpo_pretrain_epochs', type=int, default=1, help='Supervised pretraining epochs before GRPO starts')
    parser.add_argument('--grpo_start_epoch', type=int, default=-1, help='Epoch to start GRPO; -1 means use grpo_pretrain_epochs')
    parser.add_argument('--grpo_group_size', type=int, default=4, help='Number of sampled candidates per state for group rollout')
    parser.add_argument('--grpo_rollout_steps', type=int, default=4, help='Number of action samples in each rollout sequence')
    parser.add_argument('--grpo_temperature', type=float, default=1.0, help='Sampling temperature for GRPO rollout')
    parser.add_argument('--grpo_loss_weight', type=float, default=0.20, help='Weight of GRPO loss relative to supervised loss')
    parser.add_argument('--grpo_kl_weight', type=float, default=0.01, help='KL penalty weight against the frozen reference policy')
    parser.add_argument('--use_grpo', action='store_true', default=False, help='Enable GRPO training (only when training)')
    
    parser.add_argument('--use_rl', action='store_true',default=True, help='Enable reinforcement learning')
    parser.add_argument('--rl_start_epoch', type=int, default=2, help='Epoch to start RL (-1 means half of total epochs)')
    parser.add_argument('--rl_update_frequency', type=int, default=400, help='RL update frequency (batches)')
    parser.add_argument('--rl_loss_weight', type=float, default=0.20, help='Weight for RL loss')
    parser.add_argument('--ppo_timesteps', type=int, default=1024, help='每次PPO更新的时间步数')
    parser.add_argument('--ppo_batch_size', type=int, default=64, help='PPO批次大小')
    parser.add_argument('--ppo_learning_rate', type=float, default=3e-5, help='PPO学习率')
    parser.add_argument('--ppo_gamma', type=float, default=0.99, help='PPO折扣因子')
    parser.add_argument('--ppo_gae_lambda', type=float, default=0.95, help='GAE lambda参数')
    
    # eval params
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--eval_at_start', action='store_true', default=False)
    parser.add_argument('--eval_max_timestep', type=int, default=20)
    parser.add_argument('--eval_client', type=str, choices=['crop', 'airsim'], default='crop')
    parser.add_argument('--success_dist', type=float, default=20.)
    parser.add_argument('--success_iou', type=float, default=0.4)
    parser.add_argument('--move_iteration', type=int, default=5)
    parser.add_argument('--progress_stop_val', type=float, default=0.98)
    parser.add_argument('--eval_goal_selector', type=str, choices=['gdino', 'llava'], default='gdino')
    parser.add_argument('--gps_noise_scale', type=float, default=0.)

    # airsim
    parser.add_argument('--sim_ip', type=str, default="172.23.96.1")
    parser.add_argument('--sim_port', type=int, default=41451)

    args = parser.parse_args()
    args.gsam_rgb_shape = args.gsam_rgb_shape, args.gsam_rgb_shape

    return ExperimentArgs(**vars(args))