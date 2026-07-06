from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from collections import deque
from collections.abc import Callable
import torch.nn.functional as F
from gsamllavanav.models.rl import MacroPlanner, MicroActor

from .ddppo.resenet_encoders import TorchVisionResNet50, ResnetDepthEncoder

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class MapEncoder(nn.Module):
    #引用map_encoder
    def __init__(self, map_size: int, in_channels=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResidualBlock(32),
            nn.Flatten()
        )
        self.out_features = (map_size // 2**4)**2 * 32
    
    def forward(self, maps):
        x = self.main(maps)
        return x


# class GoalPredictionHead(nn.Module):
#     def __init__(self, n_map_features):
#         super().__init__()
#         self.se = SEBlock(n_map_features)
#         self.prediction_head = nn.Sequential(
#             nn.Linear(n_map_features, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 2),
#             nn.Sigmoid(),
#         )
#     def forward(self, map_features):
#         map_features = self.se(map_features)
#         return self.prediction_head(map_features)

# class GoalPredictionHead(nn.Module):

#     def __init__(self, n_map_features: int):
#         super(GoalPredictionHead, self).__init__()
        
#         self.prediction_head = nn.Sequential(
#             nn.Linear(n_map_features, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 2),
#             nn.Sigmoid(),
#         )

#     def forward(self, map_features):
#         return self.prediction_head(map_features)

class SCConv(nn.Module):
    def __init__(self, in_channels, k=3, s=1, pooling_r=4):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r)
        self.conv_kxk = nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=s, padding=k//2, groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
    def forward(self, x):
        u = self.conv_kxk(x)
        x_pool = self.pooling(x)
        c = self.conv_1x1(x_pool)
        c = nn.functional.interpolate(c, size=x.size()[2:], mode='bilinear', align_corners=False)
        out = self.bn(u * c)
        out = self.act(out)
        return out

class GoalPredictionHead(nn.Module):
    def __init__(self, n_map_features: int, scconv_shape=(32, 4, 4)):
        super().__init__()
        self.scconv_shape = scconv_shape
        in_channels = scconv_shape[0]
        self.linear1 = nn.Linear(n_map_features, in_channels * scconv_shape[1] * scconv_shape[2])
        self.scconv = SCConv(in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.norm_fusion = nn.LayerNorm(in_channels * scconv_shape[1] * scconv_shape[2])
        self.dropout = nn.Dropout(p=0.3)   # SCConv后Dropout
        self.prediction_head = nn.Sequential(
            nn.Linear(in_channels * scconv_shape[1] * scconv_shape[2], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )
    def forward(self, map_features):
        x = self.linear1(map_features)
        B = x.size(0)
        x = x.view(B, *self.scconv_shape)
        shortcut = x
        out = self.scconv(x)
        out = out + shortcut
        out = self.flatten(out)
        out = self.norm_fusion(out)
        out = self.dropout(out)  # Dropout加在这里
        out = self.prediction_head(out)
        return out


class AdaptiveFusion(nn.Module):
    def __init__(self, map_dim, rgb_dim, depth_dim, fusion_dim=32):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(3, fusion_dim),  # 注意这里应该是3
            nn.ReLU(),
            nn.Linear(fusion_dim, 3)
        )

    def forward(self, map_f, rgb_f, depth_f):
        pooled_map = map_f.mean(dim=1, keepdim=True)
        pooled_rgb = rgb_f.mean(dim=1, keepdim=True)
        pooled_depth = depth_f.mean(dim=1, keepdim=True)
        pooled = torch.cat([pooled_map, pooled_rgb, pooled_depth], dim=1)
        weights = torch.softmax(self.weight_net(pooled), dim=1)
        map_f = map_f * weights[:, 0:1]
        rgb_f = rgb_f * weights[:, 1:2]
        depth_f = depth_f * weights[:, 2:3]
        return torch.cat([map_f, rgb_f, depth_f], dim=1)


class ProgressPredictionHead(nn.Module):
    def __init__(self, n_map_features, n_rgb_features, n_depth_features):
        super().__init__()
        self.adaptive_fusion = AdaptiveFusion(n_map_features, n_rgb_features, n_depth_features)
        self.norm_fusion = nn.LayerNorm(n_map_features + n_rgb_features + n_depth_features)
        self.dropout = nn.Dropout(0.3)  
        self.prediction_head = nn.Sequential(
            nn.Linear(n_map_features + n_rgb_features + n_depth_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, map_features, rgb_features, depth_features):
        fused = self.adaptive_fusion(map_features, rgb_features, depth_features)
        fused = self.norm_fusion(fused)
        fused = self.dropout(fused)
        return self.prediction_head(fused)


# class ProgressPredictionHead(nn.Module):

#     def __init__(self, n_map_features: int, n_rgb_features: int, n_depth_features):
#         super(ProgressPredictionHead, self).__init__()

#         self.prediction_head = nn.Sequential(
#             nn.Linear(n_map_features + n_rgb_features + n_depth_features, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, map_features, rgb_features, depth_features):
#         return self.prediction_head(torch.cat((map_features, rgb_features, depth_features), dim=1))
class ValuePredictionHead(nn.Module):
    def __init__(self, n_map_features: int, n_rgb_features: int, n_depth_features: int, use_sigmoid: bool = False):
        super().__init__()
        self.adaptive_fusion = AdaptiveFusion(n_map_features, n_rgb_features, n_depth_features)
        self.norm_fusion = nn.LayerNorm(n_map_features + n_rgb_features + n_depth_features)
        self.dropout = nn.Dropout(0.3) 
        self.value_prediction_head = nn.Sequential(
            nn.Linear(n_map_features + n_rgb_features + n_depth_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ) 
    def forward(self, map_features: Tensor, rgb_features: Tensor, depth_features: Tensor) -> Tensor:
        fused = self.adaptive_fusion(map_features, rgb_features, depth_features)
        fused = self.norm_fusion(fused)
        fused = self.dropout(fused)
        pred_value = self.value_prediction_head(fused)
        return pred_value





class GoalPredictor(nn.Module):
    def __init__(self, map_size: int):
        super(GoalPredictor, self).__init__()
 
        self.goal_map_encoder = MapEncoder(map_size)
        self.goal_rgb_encoder = TorchVisionResNet50().eval()
        self.goal_depth_encoder = ResnetDepthEncoder().eval()

        self.progress_map_encoder = MapEncoder(map_size)
        self.progress_rgb_encoder = TorchVisionResNet50().eval()
        self.progress_depth_encoder = ResnetDepthEncoder().eval()

        self.value_map_encoder = MapEncoder(map_size)
        self.value_rgb_encoder = TorchVisionResNet50().eval()
        self.value_depth_encoder = ResnetDepthEncoder().eval()

        self.goal_prediction_head = GoalPredictionHead(self.goal_map_encoder.out_features)
        self.progress_prediction_head = ProgressPredictionHead(
            self.progress_map_encoder.out_features, 
            self.progress_rgb_encoder.out_features, 
            self.progress_depth_encoder.out_features
        )
        self.value_prediction_head = ValuePredictionHead(
            n_map_features=self.value_map_encoder.out_features,
            n_rgb_features=self.value_rgb_encoder.out_features,
            n_depth_features=self.value_depth_encoder.out_features,
        )
        self.macro_planner = MacroPlanner(
            map_features_dim=self.progress_map_encoder.out_features,  
            pose_dim=4,                                            
            goal_desc_dim=2                                   
        ) 
        self.micro_actor = MicroActor(
            obs_dim=self.progress_rgb_encoder.out_features + self.progress_depth_encoder.out_features,  # RGB+Depth特征拼接
            pose_dim=4,            
            macro_goal_dim=2,      
            num_actions=6          
        )

    def forward(self, maps: Tensor, rgbs: Tensor, depths: Tensor, pose: Tensor = None, goal_desc: Tensor = None, flip_depth=True):
        """
        参数说明：
        maps:      (B, C, H, W) 地图数据
        rgbs:      (B, C, H, W) RGB图像
        depths:    (B, C, H, W) 深度图
        """
        batch_size = maps.shape[0]

        if flip_depth:
            depths = depths.flip(-2)  

        goal_map_features = self.goal_map_encoder(maps)
        progress_map_features = self.progress_map_encoder(maps)
        rgb_features = self.progress_rgb_encoder(rgbs)
        depth_features = self.progress_depth_encoder(depths)
        obs_features = torch.cat([rgb_features, depth_features], dim=-1)  
        value_map_features = self.value_map_encoder(maps)
        value_rgb_features = self.value_rgb_encoder(rgbs)
        value_depth_features = self.value_depth_encoder(depths)

        pred_normalized_goal_xys = self.goal_prediction_head(goal_map_features)
        pred_progress = self.progress_prediction_head(progress_map_features, rgb_features, depth_features)
        
        pred_value = self.value_prediction_head(value_map_features, value_rgb_features, value_depth_features)
        macro_goal = None
        micro_action_logits = None
        if pose is not None:
            default_goal_desc = torch.zeros(pose.shape[0], 2, device=pose.device)
            goal_desc_to_use = goal_desc if goal_desc is not None else default_goal_desc
            
            macro_goal = self.macro_planner(progress_map_features, pose, goal_desc_to_use)
            micro_action_logits = self.micro_actor(obs_features, pose, macro_goal)
        return {
            "pred_xy": pred_normalized_goal_xys,
            "pred_progress": pred_progress,
            "goal_map_features":goal_map_features,
            "macro_goal": macro_goal,
            "micro_action_logits": micro_action_logits,
            "pred_value": pred_value
        }


    def predict(
        self, to_world_xy: Callable[[tuple[float, float]], tuple[float, float]],
        maps: Tensor, rgb: Tensor, depth: Tensor, flip_depth = True
    ):
        outputs = self(maps, rgb, depth, flip_depth=flip_depth)
        pred_xy = to_world_xy(outputs["pred_xy"])
        return pred_xy, outputs["pred_progress"]
    
