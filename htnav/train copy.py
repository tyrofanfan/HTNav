import random
import os
import numpy as np
import torch
import gc
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter 
from gsamllavanav.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from gsamllavanav.cityreferobject import get_city_refer_objects, MultiMapObjects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import convert_trajectory_to_shortest_path, generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.observation import cropclient
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.space import Pose4D, Point3D, Point2D
from gsamllavanav.evaluate import eval_goal_predictor, run_episodes_batch, GoalPredictorMetrics
from gsamllavanav import logger
import matplotlib.pyplot as plt
from gsamllavanav.models.rl import MacroPlanner, MicroActor
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action, lookahead_continuous_action, LookaheadTeacherParams
from gsamllavanav.teacher.trajectory import _moved_pose
from PIL import Image
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gsamllavanav.actions import DiscreteAction
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn


class PPOLossCallback(BaseCallback):
    """收集PPO训练过程中的损失信息"""
    
    def __init__(self, verbose=0):
        super(PPOLossCallback, self).__init__(verbose)
        self.losses = []
        
    def _on_step(self) -> bool:
        """每步调用，记录损失"""
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            print(f"可用日志项: {list(self.model.logger.name_to_value.keys())}")
        
        # 确认是否存在这些键
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            if "train/policy_loss" in self.model.logger.name_to_value or "train/value_loss" in self.model.logger.name_to_value:
                policy_loss = self.model.logger.name_to_value.get("train/policy_loss", 0)
                value_loss = self.model.logger.name_to_value.get("train/value_loss", 0)
                approx_kl = self.model.logger.name_to_value.get("train/approx_kl", 0)
                
                # 组合损失
                combined_loss = policy_loss + 0.5 * value_loss + 0.01 * approx_kl
                self.losses.append(combined_loss)
                print(f"记录真实损失: {combined_loss:.6f} (策略损失: {policy_loss:.6f}, 价值损失: {value_loss:.6f})")
        
        return True
    
    def get_mean_loss(self):
        print("sumloss",sum(self.losses))
        print("lenloss",len(self.losses))
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

class NavigationFeaturesExtractor(BaseFeaturesExtractor):
    """为PPO提取导航相关特征的网络"""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 定义各个特征提取器
        rgb_shape = observation_space.spaces['rgb'].shape
        depth_shape = observation_space.spaces['depth'].shape
        map_shape = observation_space.spaces['map'].shape
        
        # RGB图像特征提取
        self.rgb_extractor = nn.Sequential(
            nn.Conv2d(rgb_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 深度图特征提取
        self.depth_extractor = nn.Sequential(
            nn.Conv2d(depth_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 地图特征提取 - 修改为使用map_shape[0]作为输入通道数
        self.map_extractor = nn.Sequential(
            nn.Conv2d(map_shape[0], 32, kernel_size=3, stride=2), # 使用map_shape[0]而不是map_shape[2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 位姿特征
        self.pose_extractor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU()
        )
        with torch.no_grad():
            rgb_sample = torch.as_tensor(observation_space.spaces['rgb'].sample()[None]).float()
            depth_sample = torch.as_tensor(observation_space.spaces['depth'].sample()[None]).float()
            map_sample = torch.as_tensor(observation_space.spaces['map'].sample()[None]).float()
            
            rgb_out = self.rgb_extractor(rgb_sample)
            depth_out = self.depth_extractor(depth_sample)
            map_out = self.map_extractor(map_sample)  # 移除permute操作
            
            # 调试信息
            print(f"Feature shapes - RGB: {rgb_out.shape}, Depth: {depth_out.shape}, Map: {map_out.shape}")
            
            total_concat_size = rgb_out.shape[1] + depth_out.shape[1] + map_out.shape[1] + 32
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations) -> torch.Tensor:
        rgb_features = self.rgb_extractor(observations["rgb"].float() / 255.0)
        depth_features = self.depth_extractor(observations["depth"].float())
        # 确保地图数据直接传递，不做permute
        map_features = self.map_extractor(observations["map"].float())
        pose_features = self.pose_extractor(observations["pose"].float())
        
        # 拼接所有特征
        combined = torch.cat([rgb_features, depth_features, map_features, pose_features], dim=1)
        
        # 融合特征
        return self.fusion(combined)

class PPONavigationAgent:
    """使用PPO算法的导航代理，与GoalPredictor集成"""
    def __init__(self, env, goal_predictor, device="cuda:0", 
                 learning_rate=1e-5, n_steps=512, batch_size=32,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2):
        self.env = DummyVecEnv([lambda: PPOEnvWrapper(env)])
        self.device = device
        self.goal_predictor = goal_predictor
        self.learning_rate = learning_rate
        
        # 自定义策略参数
        policy_kwargs = dict(
            features_extractor_class=NavigationFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(
                pi=[128, 64],  # 策略网络架构
                vf=[256, 1]    # 价值网络架构
            )
        )
        # 创建PPO模型
        self.model = PPO(
            policy="MultiInputPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            tensorboard_log=f"./ppo_tensorboard/",
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            ent_coef=0.005,  # 略微降低熵系数
            vf_coef=1.0,     # 增加价值函数系数，帮助稳定价值学习
            max_grad_norm=0.5,  # 添加梯度裁剪
        )
        
        # 初始状态
        self.last_loss = 0.0
        self.total_timesteps = 0
        self.trained_once = False
    
    def collect_and_update(self, timesteps=1024, macro_loss=None, micro_loss=None):
        was_training = self.goal_predictor.training
        self.goal_predictor.eval()

        if hasattr(self.model, 'rollout_buffer'):
            self.model.rollout_buffer.reset()
        self.model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False,
            log_interval=1  # 每一步都记录日志
        )
        
        # 从模型的logger中提取最后记录的损失
        policy_loss = 0.0
        value_loss = 0.0
        
        
        combined_loss = policy_loss + 0.5 * value_loss
        # 确保损失不为零
        self.last_loss = max(combined_loss, 0.01)
        self.trained_once = True
        
        extra_loss = torch.tensor(0.0, device=self.device)
        if macro_loss is not None and micro_loss is not None:
            with torch.no_grad():
                extra_loss = 0.1 * macro_loss + 0.01 * micro_loss  # 拟定权重=0.1
        
        if extra_loss.requires_grad:
            # 获取policy网络所有可训练参数
            policy_params = list(self.model.policy.parameters())
            self.model.policy.optimizer.zero_grad()
            extra_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(policy_params, max_norm=0.5)
            self.model.policy.optimizer.step()

        self.update_predictor()
        
        if was_training:
            self.goal_predictor.train()
        return torch.tensor(self.last_loss, device=self.device), extra_loss
    
    def update_predictor(self):
        """使用PPO学习的策略来更新MicroActor和MacroPlanner"""
        if not self.trained_once:
            return False
            
        update_success = False
            
        try:
            policy = self.model.policy
            
            # 1. 更新MicroActor (微观动作决策)
            try:
                # 获取PPO的动作网络
                ppo_action_net = self.model.policy.action_net
                # 获取MicroActor的动作输出层
                micro_actor = self.goal_predictor.micro_actor
                micro_action_layers = [m for m in micro_actor.modules() if isinstance(m, nn.Linear)]
                if micro_action_layers:
                    # 获取MicroActor的最后一层
                    micro_last_layer = micro_action_layers[-1]  # 最后一层通常是动作输出层

                    # 检查形状是否匹配
                    if micro_last_layer.weight.shape == ppo_action_net.weight.shape:
                        with torch.no_grad():
                            # 部分更新权重(alpha混合)
                            alpha = 0.2  # 混合比例
                            micro_last_layer.weight.data = (1-alpha) * micro_last_layer.weight.data + alpha * ppo_action_net.weight.data
                            if hasattr(micro_last_layer, 'bias') and hasattr(ppo_action_net, 'bias'):
                                micro_last_layer.bias.data = (1-alpha) * micro_last_layer.bias.data + alpha * ppo_action_net.bias.data
                        
                        print("成功更新MicroActor")
                        update_success = True
            except Exception as e:
                print(f"更新MicroActor时出错: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. 更新MacroPlanner (宏观规划)
            try:
                # 从PPO提取价值网络参数，用于宏观规划(长期价值判断)
                value_net_params = list(self.model.policy.mlp_extractor.value_net.parameters())
                
                # 获取MacroPlanner的参数
                macro_planner = self.goal_predictor.macro_planner
                macro_layers = [m for m in macro_planner.modules() if isinstance(m, nn.Linear)]
                
                if macro_layers and value_net_params:
                    # 对MacroPlanner的各层进行更新
                    for i, layer in enumerate(macro_layers):
                        if i < len(value_net_params) // 2:  # 确保不越界
                            # 获取对应的PPO价值网络参数
                            ppo_weight = value_net_params[i*2]  # 权重
                            ppo_bias = value_net_params[i*2+1]  # 偏置
                            
                            # 如果形状匹配，则进行更新
                            if layer.weight.shape == ppo_weight.shape:
                                with torch.no_grad():
                                    beta = 0.1 # 宏观规划使用更保守的更新率
                                    layer.weight.data = (1-beta) * layer.weight.data + beta * ppo_weight
                                    if hasattr(layer, 'bias') and ppo_bias is not None:
                                        if layer.bias.shape == ppo_bias.shape:
                                            layer.bias.data = (1-beta) * layer.bias.data + beta * ppo_bias
                    
                    print("成功更新MacroPlanner")
                    update_success = True
            except Exception as e:
                print(f"更新MacroPlanner时出错: {e}")
                import traceback
                traceback.print_exc()
                
            return update_success
                    
        except Exception as e:
            print(f"更新策略时出错: {e}")
            import traceback
            traceback.print_exc()
            return False


class PPOEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 调整观测空间以兼容PPO
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(0, 255, shape=(3, 224, 224), dtype=np.float32),
            'depth': spaces.Box(0, 1, shape=(1, 256, 256), dtype=np.float32),
            'map': spaces.Box(0, 1, shape=(3, 240, 240), dtype=np.float32),  # 修改这里的通道数为3
            'pose': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        })
    def observation(self, observation):
        """处理观测以适配PPO"""
        # 确保所有观测都是类型兼容的numpy数组，并维持通道优先(C,H,W)格式
        result = {}
        
        # 处理RGB图像: 保持(C,H,W)格式
        if isinstance(observation['rgb'], torch.Tensor):
            rgb = observation['rgb'].squeeze(0).cpu().numpy()
            # 确保格式是(C,H,W)
            if rgb.shape[0] != 3 and rgb.shape[-1] == 3:
                rgb = np.transpose(rgb, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        else:
            rgb = observation['rgb']
            # 确保格式是(C,H,W)
            if rgb.shape[0] != 3 and rgb.shape[-1] == 3:  
                rgb = np.transpose(rgb, (2, 0, 1))
        result['rgb'] = rgb
        
        # 处理深度图: 保持(C,H,W)格式
        if isinstance(observation['depth'], torch.Tensor):
            depth = observation['depth'].squeeze(0).cpu().numpy()
            if depth.shape[0] != 1 and len(depth.shape) == 3:
                depth = np.transpose(depth, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            elif len(depth.shape) == 2:
                depth = depth[np.newaxis, ...]  # 添加通道维度在前面(C,H,W)
        else:
            depth = observation['depth']
            if depth.shape[0] != 1 and len(depth.shape) == 3:
                depth = np.transpose(depth, (2, 0, 1))
            elif len(depth.shape) == 2:
                depth = depth[np.newaxis, ...]
        result['depth'] = depth
        
        # 处理地图: 保持(C,H,W)格式
        if isinstance(observation['map'], torch.Tensor):
            map_tensor = observation['map'].squeeze(0).cpu().numpy()
            if map_tensor.shape[0] != 4 and map_tensor.shape[-1] == 4:
                map_tensor = np.transpose(map_tensor, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        else:
            map_tensor = observation['map']
            if map_tensor.shape[0] != 4 and map_tensor.shape[-1] == 4:
                map_tensor = np.transpose(map_tensor, (2, 0, 1))
        result['map'] = map_tensor
        
        # 处理位姿向量
        if isinstance(observation['pose'], torch.Tensor):
            result['pose'] = observation['pose'].squeeze(0).cpu().numpy()
        else:
            result['pose'] = observation['pose']
            
        return result
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 统一接口：将terminated和truncated合并为done，保持与传统gym兼容
        done = terminated or truncated
        # 将处理后的obs传回
        return self.observation(obs), reward, terminated, truncated, info

class DroneNavigationEnv(gym.Env):
    """无人机导航环境"""
    
    def __init__(self, episodes, args, device='cuda:7'):
        super().__init__()
        self.episodes = episodes
        self.args = args
        self.device = device
        
        # 离散动作空间：0=STOP, 1=MOVE_FORWARD, 2=TURN_RIGHT, 3=TURN_LEFT, 4=GO_UP, 5=GO_DOWN
        self.action_space = spaces.Discrete(6)
        
        # 观测空间
        # 修改为通道在前(C,H,W)格式
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(0, 255, shape=(3, 224, 224), dtype=np.float32),
            'depth': spaces.Box(0, 1, shape=(1, 256, 256), dtype=np.float32),
            'map': spaces.Box(0, 1, shape=(4, args.map_size, args.map_size), dtype=np.float32),  # 注意这里也修改为CHW格式
            'pose': spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        })
        
        self.current_episode = None
        self.current_step = 0
        self.current_pose = None
        self.max_steps = 100  
        self.trajectory = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # 随机选择一个episode
        self.current_episode_idx = random.randrange(len(self.episodes))
        self.current_episode = self.episodes[self.current_episode_idx]
        
        # 初始化位置
        self.current_pose = Pose4D(
        x=self.current_episode.start_pose.x,
        y=self.current_episode.start_pose.y,
        z=self.current_episode.start_pose.z,
        yaw=self.current_episode.start_pose.yaw
        )
        
        # 设置目标位置
        self.target_position = self.current_episode.target_position  # 正确的属性名
        
        # 计算初始距离
        self.initial_distance = np.sqrt(
            (self.current_pose.x - self.target_position.x)**2 + 
            (self.current_pose.y - self.target_position.y)**2
        )
        self.prev_distance = self.initial_distance
        
        self.current_step = 0
        
        observation = self._get_observation()
        return observation, {}
        
    def step(self, action):
        """执行一个动作，返回新状态、奖励等"""
        self.current_step += 1
        
        # 保存前一个位置，用于计算奖励
        prev_pose = self.current_pose
        
        # 根据选择的action更新无人机位置
        action_mapping = [
            DiscreteAction.STOP.value,
            DiscreteAction.MOVE_FORWARD.value,
            DiscreteAction.TURN_RIGHT.value,
            DiscreteAction.TURN_LEFT.value,
            DiscreteAction.GO_UP.value,
            DiscreteAction.GO_DOWN.value
        ]
        # 确保action在有效范围内
        if 0 <= action < len(action_mapping):
            action_value = action_mapping[action]
        else:
            # 默认不移动
            action_value = DiscreteAction.STOP.value
        self.current_pose = _moved_pose(self.current_pose, *action_value)
        
        # 获取新的观测
        obs = self._get_observation()
        
        # 计算与目标的距离
        prev_dist = self.target_position.xy.dist_to(prev_pose.xy)
        curr_dist = self.target_position.xy.dist_to(self.current_pose.xy)
        
        # 计算奖励
        reward = self._compute_reward(prev_pose, self.current_pose, prev_dist, curr_dist)
        
            # 检查是否到达目标或超过最大步数
        terminated = curr_dist < 20.0  
        truncated = self.current_step >= self.max_steps  # 任务提前终止（超时）
        # 额外信息
        info = {
            'distance': curr_dist,
            'goal': self.target_position.xy,
            'is_success': curr_dist < 5.0
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """获取当前观测"""
        # 获取RGB和深度图
        rgb = cropclient.crop_image(
            self.current_episode.map_name, 
            self.current_pose, 
            (224, 224), 
            'rgb'
            )
            
        depth = cropclient.crop_image(
            self.current_episode.map_name, 
            self.current_pose, 
            (256, 256), 
            'depth'
            ) / self.args.max_depth
        
        # 获取地图
        maps = LandmarkNavMap.generate_maps_for_an_episode(
            self.current_episode, 
            self.args.map_shape, 
            self.args.map_pixels_per_meter,
            self.args.map_update_interval, 
            self.args.gsam_rgb_shape, 
            self.args.gsam_use_map_cache
        )
        
        # 转换为tensor
        rgb_tensor = torch.tensor(rgb.transpose(2, 0, 1), device=self.device, dtype=torch.float32)
        depth_tensor = torch.tensor(depth.reshape(1, *depth.shape[:2]), device=self.device, dtype=torch.float32)
        if maps[0].shape[-1] == 4:  # 如果通道在最后
            map_tensor = torch.tensor(maps[0].transpose(2, 0, 1), device=self.device, dtype=torch.float32)
        else:
            map_tensor = torch.tensor(maps[0], device=self.device, dtype=torch.float32)
        pose_tensor = torch.tensor([
            self.current_pose.x, 
            self.current_pose.y, 
            self.current_pose.z, 
            self.current_pose.yaw
        ], device=self.device, dtype=torch.float32)
        
        return {
            'rgb': rgb_tensor.unsqueeze(0),  # 添加batch维度
            'depth': depth_tensor.unsqueeze(0),
            'map': map_tensor.unsqueeze(0),
            'pose': pose_tensor.unsqueeze(0)
        }
    
    def _compute_reward(self, prev_pose, curr_pose, prev_dist, curr_dist):
        # 1. 标准化距离变化奖励
        distance_delta = prev_dist - curr_dist
        distance_reward = np.clip(5.0 * distance_delta, -2.0, 2.0)  # 限制奖励范围
        
        # 2. 简化的方向奖励
        target_dir = np.arctan2(self.target_position.y - curr_pose.y, self.target_position.x - curr_pose.x)
        angle_diff = abs((curr_pose.yaw - target_dir + np.pi) % (2 * np.pi) - np.pi)
        direction_reward = 0.5 * (1 - angle_diff / np.pi)
        
        # 3. 成功奖励（稀疏）
        goal_reward = 10.0 if curr_dist < 5.0 else 0.0
        
        # 4. 步骤惩罚（鼓励效率）
        step_penalty = -0.01
        
        # 组合奖励并限制范围
        total_reward = distance_reward + direction_reward + goal_reward + step_penalty
        total_reward = np.clip(total_reward, -5.0, 15.0)  # 限制总奖励范围
        
        return total_reward
    
    def _is_spinning(self, prev_pose, curr_pose):
        """检测是否原地打转"""
        # 简单实现：位置变化很小但朝向变化较大
        pos_change = np.sqrt((curr_pose.x - prev_pose.x)**2 + (curr_pose.y - prev_pose.y)**2)
        return pos_change < 0.1 and abs(curr_pose.yaw - prev_pose.yaw) > 0.1

def compute_returns(rewards, dones, gamma=0.99):
    """计算折扣回报"""
    returns = []
    R = 0
    
    for r, done in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - done)
        returns.insert(0, R)
    
    # 归一化回报
    if len(returns) > 1:
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns.tolist()
    return returns

def initialize_ppo_value_network(ppo_agent, goal_predictor):
    print("开始将预训练值函数参数迁移到PPO...")
    
    # 获取GoalPredictor的值函数预测头参数
    value_head_params = list(goal_predictor.value_prediction_head.parameters())
    
    # 找到PPO模型中的值函数网络
    value_net = None
    for name, module in ppo_agent.model.policy.named_modules():
        if 'value_net' in name:  # 通常是 "mlp_extractor.value_net"
            value_net = module
            print(f"找到PPO值函数网络: {name}")
            break
    # 获取PPO值函数网络的参数
    ppo_value_params = list(value_net.parameters())
    
    # 尝试参数迁移（从后向前，通常后面的层更特定于任务）
    layers_transferred = 0
    
    # 逆序遍历参数以优先迁移输出层
    for i in range(min(len(ppo_value_params), len(value_head_params))):
        ppo_param = ppo_value_params[-(i+1)]  # 从后往前
        pretrain_param = value_head_params[-(i+1)]
        
        if ppo_param.shape == pretrain_param.shape:
            with torch.no_grad():
                ppo_param.data.copy_(pretrain_param.data)
                layers_transferred += 1
    return layers_transferred > 0

def debug_save_episode_maps(episodes, args, device, out_dir="debug_maps"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    crop_margin = 100  # 可调节边界留白
    for idx, episode in enumerate(episodes[:2]):
        trajectory = list(episode.sample_trajectory(args.map_update_interval))
        with open(f"{out_dir}/ep{idx}_descriptions.txt", "w") as f:
            for desc in episode.target_object.descriptions:
                f.write(desc + "\n")
        with open(f"{out_dir}/ep{idx}_description.txt", "w") as f:
            f.write(episode.target_description + "\n")
        with open(f"{out_dir}/ep{idx}_map_name.txt", "w") as f:
            f.write(episode.map_name + "\n")
        traj_xy = np.array([[p.x, p.y] for p in trajectory])
        min_x, min_y = traj_xy.min(axis=0) - crop_margin
        max_x, max_y = traj_xy.max(axis=0) + crop_margin
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        width, height = max_x - min_x, max_y - min_y
        crop_size = int(max(width, height)*1.2)
        center_pose = Pose4D(center_x, center_y, 0, 0)
        output_size = (crop_size, crop_size)
        full_rgb = cropclient.crop_image(episode.map_name, center_pose, output_size, 'rgb')
        plt.figure(figsize=(8,8))
        plt.imshow(
            full_rgb, 
            extent=[center_x-crop_size/2, center_x+crop_size/2, center_y+crop_size/2, center_y-crop_size/2]
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/ep{idx}_full_map_rgb.png", dpi=150)
        plt.close()
        from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
        maps = LandmarkNavMap.generate_maps_for_an_episode(
            episode, args.map_shape, args.map_pixels_per_meter, args.map_update_interval, args.gsam_rgb_shape, args.gsam_use_map_cache
        )  # shape: (T, 3, H, W)
        for t, pose in enumerate(trajectory):
            # RGB
            rgb = cropclient.crop_image(episode.map_name, pose, (512, 512), 'rgb')
            Image.fromarray(rgb.astype(np.uint8)).save(f"{out_dir}/ep{idx}_step{t}_rgb.png")
            # 深度（黑白）
            depth = cropclient.crop_image(episode.map_name, pose, (512, 512), 'depth')
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            plt.imsave(f"{out_dir}/ep{idx}_step{t}_depth.png", depth, cmap='gray')
            # 地图
            from matplotlib.colors import ListedColormap
            view_map = maps[t, 0]
            explored_map = maps[t, 1]
            landmark_map = maps[t, 2]

            view_cmap = ListedColormap([[0.97, 0.97, 0.92], [0, 0, 0.8]])
            explored_cmap = ListedColormap([[0.97, 0.97, 0.92], [0, 0.8, 0]])
            landmark_cmap = ListedColormap([[0.97, 0.97, 0.92], [0.9, 0.5, 0]])

            plt.imsave(f"{out_dir}/ep{idx}_step{t}_view_map.png", view_map, cmap=view_cmap, vmin=0, vmax=1)
            plt.imsave(f"{out_dir}/ep{idx}_step{t}_explored_map.png", explored_map, cmap=explored_cmap, vmin=0, vmax=1)
            plt.imsave(f"{out_dir}/ep{idx}_step{t}_landmark_map.png", landmark_map, cmap=landmark_cmap, vmin=0, vmax=1)
            # 三图重合
            h, w = view_map.shape
            bg_color = np.array([0.97, 0.97, 0.92], dtype=np.float32)
            view_color = np.array([0, 0, 0.8], dtype=np.float32)
            explored_color = np.array([0, 0.8, 0], dtype=np.float32)
            landmark_color = np.array([0.9, 0.5, 0], dtype=np.float32)

            composite = np.ones((h, w, 3), dtype=np.float32) * bg_color

            # 只对有标记的像素做混合
            mask = (view_map + explored_map + landmark_map) > 0
            composite[mask] = (
                view_map[mask, None] * view_color +
                explored_map[mask, None] * explored_color +
                landmark_map[mask, None] * landmark_color
            )
            composite = np.clip(composite, 0, 1)
            plt.imsave(f"{out_dir}/ep{idx}_step{t}_composite_map.png", composite)
        print(f"Episode {idx} 全过程debug图片已保存到 {out_dir}")

def train(args: ExperimentArgs, device='cuda:7'):
    print(f"Training on device: {device}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # setup logger
    logger.init(args)
    for metric in GoalPredictorMetrics.names():
        logger.define_metric('val_seen_' + metric, 'epoch')
        logger.define_metric('val_unseen_' + metric, 'epoch')

    # load data
    start_epoch = 0
    objects = get_city_refer_objects()
    train_episodes = _load_train_episodes(objects, args)
  

    if args.train_episode_sample_size > 0:
        train_episodes = random.sample(train_episodes, args.train_episode_sample_size)
    train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_seen', 'all', args.altitude))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_unseen', 'all', args.altitude))
    cropclient.load_image_cache()
    debug_save_episode_maps(train_episodes, args, device)

    # init model & optim
    goal_predictor = GoalPredictor(args.map_size).to(device)
    optimizer = AdamW(goal_predictor.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, goal_predictor, optimizer = _load_checkpoint(goal_predictor, optimizer, args)

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

    writer = SummaryWriter(log_dir=f"runs/{args.experiment_name if hasattr(args, 'experiment_name') else 'default'}")

    # 新增：混合训练参数
    use_rl = hasattr(args, 'use_rl') and args.use_rl
    rl_start_epoch = getattr(args, 'rl_start_epoch', 0)  # 更早开始RL
    rl_update_frequency = getattr(args, 'rl_update_frequency',400)  # 减少频率增加每次质量
    rl_loss_weight = getattr(args, 'rl_loss_weight', 0.15)  # 增加RL权重


    ppo_agent = None
    if use_rl:
        env = DroneNavigationEnv(train_episodes, args, device)
        ppo_agent = PPONavigationAgent(
            env=env,
            goal_predictor=goal_predictor,
            device=device,
            learning_rate=getattr(args, 'ppo_learning_rate', args.learning_rate / 10),
            n_steps=getattr(args, 'ppo_timesteps', 1024),
            batch_size=getattr(args, 'ppo_batch_size', 64),
            gamma=getattr(args, 'ppo_gamma', 0.99),
            gae_lambda=getattr(args, 'ppo_gae_lambda', 0.95)
        )

    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        total_loss_sum = 0
        goal_pred_loss_sum = 0
        progress_loss_sum = 0
        macro_loss_sum = 0
        micro_loss_sum = 0
        rl_loss_sum = 0
        value_loss_sum = 0
        batch_idx = 0
        epoch_use_rl = use_rl and epoch >= rl_start_epoch
        if epoch_use_rl and epoch == rl_start_epoch:
            initialize_ppo_value_network(ppo_agent, goal_predictor)

        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            maps, rgbs, normalized_depths = prepare_inputs(episodes_batch, args, device)
            normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensor, value_targets = prepare_labels(episodes_batch, args, device)
            # normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensor = prepare_labels(episodes_batch, args, device)
            outputs = goal_predictor(
            maps, rgbs, normalized_depths,
            pose=pose_tensor,
            goal_desc=normalized_goal_xys,
            flip_depth=True
            )
            pred_normalized_goal_xys = outputs["pred_xy"]
            pred_progresses = outputs["pred_progress"]
            macro_goal = outputs["macro_goal"]
            micro_action_logits = outputs["micro_action_logits"]
            pred_value = outputs["pred_value"] 
            
            goal_prediction_loss = F.mse_loss(pred_normalized_goal_xys, normalized_goal_xys)
            progress_loss = F.mse_loss(pred_progresses, progresses)
            value_loss = F.mse_loss(pred_value, value_targets)
            sup_loss = goal_prediction_loss + progress_loss  

            class_weights = torch.ones(6, device=device)
            class_weights[4] = 0.1  # GO_UP 权重降低
            class_weights[5] = 0.1  # GO_DOWN 权重降低
            # macro_loss = torch.tensor(0.0, device=device)  # 改为tensor
            # micro_loss = torch.tensor(0.0, device=device) 
            macro_loss = F.mse_loss(macro_goal, macro_goal_teachers)
            micro_loss = F.cross_entropy(micro_action_logits, micro_action_teachers)
            if epoch_use_rl and batch_idx % rl_update_frequency == 0:
                rl_loss, extra_loss = ppo_agent.collect_and_update(timesteps=2*len(episodes_batch)*40,
                macro_loss=macro_loss,
                micro_loss=micro_loss
            )
                rl_loss_sum += rl_loss.item()
                
                if batch_idx % (rl_update_frequency * 5) == 0:
                    success = ppo_agent.update_predictor()
                    if success:
                        print(f"已将PPO策略更新到GoalPredictor (epoch {epoch}, batch {batch_idx})")
            else:
                rl_loss = torch.tensor(0.0, device=device)
            
            if epoch_use_rl:
                total_loss = sup_loss + rl_loss_weight * rl_loss 
            else:
                total_loss = sup_loss

            # === 记录损失 ===
            total_loss_sum += total_loss.item()
            goal_pred_loss_sum += goal_prediction_loss.item()
            progress_loss_sum += progress_loss.item()
            value_loss_sum += value_loss.item()
            macro_loss_sum += macro_loss.item()
            micro_loss_sum += micro_loss.item()
            batch_idx += 1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(goal_predictor.parameters(), max_norm=0.5)  # 可选梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            
        logger.log({
        'epoch': epoch,
        'loss': total_loss_sum / batch_idx,
        'goal_pred_loss': goal_pred_loss_sum / batch_idx,
        'progress_loss': progress_loss_sum / batch_idx,
        "value_loss:": value_loss_sum / batch_idx, 
        })

      
        gc.collect()
        torch.cuda.empty_cache()
        
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                epoch, goal_predictor, optimizer, args
            )
            
        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(
                goal_predictor, val_seen_episodes, val_unseen_episodes, 
                args, device
            )

    writer.close()


def normalize_position(pos: Point2D, map_name: str, map_meters: float):
    return (pos.x - MAP_BOUNDS[map_name].x_min) / map_meters, (MAP_BOUNDS[map_name].y_max - pos.y) / map_meters


def prepare_inputs(episodes_batch: list[Episode], args: ExperimentArgs, device: str):

    maps = np.concatenate([
        LandmarkNavMap.generate_maps_for_an_episode(
            episode, args.map_shape, args.map_pixels_per_meter, args.map_update_interval, args.gsam_rgb_shape, args.gsam_use_map_cache
        )
        for episode in episodes_batch
    ])

    rgbs = np.stack([
        cropclient.crop_image(episode.map_name, pose, (224, 224), 'rgb')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2)

    normalized_depths = np.stack([
        cropclient.crop_image(episode.map_name, pose, (256, 256), 'depth')
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2) / args.max_depth

    if args.ablate == 'rgb':
        rgbs = np.zeros_like(rgbs)
    if args.ablate == 'depth':
        normalized_depths = np.zeros_like(normalized_depths)
    if args.ablate == 'tracking':
        maps[:, :2] = 0
    if args.ablate == 'landmark':
        maps[:, 2] = 0
    if args.ablate == 'gsam':
        maps[:, 3:] = 0

    maps = torch.tensor(maps, device=device).to(device)
    rgbs = torch.tensor(rgbs, device=device).to(device)
    normalized_depths = torch.tensor(normalized_depths, device=device, dtype=torch.float32)
    
    return maps, rgbs, normalized_depths


def prepare_labels(episodes_batch: list[Episode], args: ExperimentArgs, device: str):
    normalized_goal_xys = []
    progresses = []
    macro_goal_teachers = []
    micro_action_teachers = []
    pose_tensors = []
    value_targets = []

    lookahead = 3  
    teacher_params = LookaheadTeacherParams(lookahead=lookahead)
    
    # 一次遍历完成所有数据收集
    for episode in episodes_batch:
        # 获取轨迹 - 只调用一次sample_trajectory
        trajectory_pose4d = list(episode.sample_trajectory(args.map_update_interval))
        trajectory_point3d = [Point3D(pose.x, pose.y, pose.z) for pose in trajectory_pose4d]
        
        # 循环每个位置，同时提取所有需要的信息
        for i, pose in enumerate(trajectory_pose4d):
            # 归一化目标位置
            normalized_goal_xys.append(
                normalize_position(episode.target_position, episode.map_name, args.map_meters)
            )
            
            # 进度值
            progress = np.clip(
                1 - episode.target_position.xy.dist_to(pose.xy) / 
                episode.target_position.xy.dist_to(episode.start_pose.xy), 
                0, 1
            )
            progresses.append(progress)
            
            value_targets.append(progress)
            # 宏观目标
            macro_goal = lookahead_continuous_action(pose, trajectory_point3d, lookahead)
            macro_goal_teachers.append(macro_goal)
            
            # 微观动作
            action = lookahead_discrete_action(pose, trajectory_point3d, teacher_params)
            micro_action_teachers.append(action.index)
            
            # 位姿张量
            pose_tensor = torch.tensor(
                [pose.x, pose.y, pose.z, pose.yaw], 
                device=device, 
                dtype=torch.float32
            )
            pose_tensors.append(pose_tensor)
    
    # 转换为张量
    normalized_goal_xys = torch.tensor(normalized_goal_xys, device=device, dtype=torch.float32)
    progresses = torch.tensor(progresses, device=device, dtype=torch.float32).reshape(-1, 1)
    value_targets = torch.tensor(value_targets, device=device, dtype=torch.float32).reshape(-1, 1)
    micro_action_teachers = torch.tensor(micro_action_teachers, device=device, dtype=torch.long)
    pose_tensors = torch.stack(pose_tensors)        
    # 如果是Action对象，转换为二维数组
    if hasattr(macro_goal_teachers[0], 'forward_stride'):
        macro_goal_teachers = [
            [goal.forward_stride, goal.d_yaw] for goal in macro_goal_teachers
        ]
    else:
        # 确保每个目标只有两个值
        macro_goal_teachers = [goal[:2] if len(goal) > 2 else goal for goal in macro_goal_teachers]

    if len(macro_goal_teachers) > 0:
        # 如果是Action对象，提取所需的属性
        if hasattr(macro_goal_teachers[0], 'forward_stride'):
            # 提取前向移动和角度变化
            values = [[goal.forward_stride, goal.d_yaw] for goal in macro_goal_teachers]
            # 找到最大绝对值
            max_abs_value = max(max(abs(x) for x in pair) for pair in values)
            if max_abs_value > 0:
                macro_goal_teachers = [[v/max_abs_value for v in pair] for pair in values]
            else:
                macro_goal_teachers = values
        else:
            # 如果已经是列表或数组，同样归一化
            max_abs = max(max(abs(x) for x in goal) for goal in macro_goal_teachers)
            if max_abs > 0:
                macro_goal_teachers = [[v/max_abs for v in goal] for goal in macro_goal_teachers]    
    macro_goal_teachers = torch.tensor(macro_goal_teachers, device=device, dtype=torch.float32)

    return normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensors, value_targets
    # return normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensors


def _load_train_episodes(objects: MultiMapObjects, args: ExperimentArgs) -> list[Episode]:
    mturk_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('train_seen', 'all', args.altitude))
    if args.train_trajectory_type == 'mturk':
        return mturk_episodes
    if args.train_trajectory_type == 'sp':
        return [convert_trajectory_to_shortest_path(eps, 'linear_xy') for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]
    if args.train_trajectory_type == 'both':
        return mturk_episodes + [convert_trajectory_to_shortest_path(eps, 'linear_xy') for eps in tqdm(mturk_episodes, desc='converting to shortest path episode')]


def _eval_predictor_and_log_metrics(
    goal_predictor: GoalPredictor,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_goal_predictor(args, val_seen_episodes, *run_episodes_batch(args, goal_predictor, val_seen_episodes, device))
    val_unseen_metrics = eval_goal_predictor(args, val_unseen_episodes,  *run_episodes_batch(args, goal_predictor, val_unseen_episodes, device))
    logger.log({'val_seen_' + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({'val_unseen_' + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(
    goal_predictor: GoalPredictor,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint['epoch'] + 1
    goal_predictor.load_state_dict(checkpoint['predictor_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return start_epoch, goal_predictor, optimizer


def _save_checkpoint(
    epoch: int,
    goal_predictor: GoalPredictor,
    optimizer: torch.optim.Optimizer,
    args: ExperimentArgs,
):
    ablation = f"-{args.ablate}" if args.ablate else ''
    train_size = '' if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    checkpoint_dir = GOAL_PREDICTOR_CHECKPOINT_DIR/f"{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存模型检查点
    torch.save(
        {
            'epoch': epoch,
            'predictor_state_dict': goal_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )
  
        
    