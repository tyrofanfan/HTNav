import torch
import torch.nn as nn

class MacroPlanner(nn.Module):
    def __init__(self, map_features_dim: int, pose_dim: int, goal_desc_dim: int):
        super().__init__()
        input_dim = map_features_dim + pose_dim + goal_desc_dim
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, map_features, pose, goal_desc):
        x = torch.cat([map_features, pose, goal_desc], dim=-1)
        return self.fc(x)


class MicroActor(nn.Module):
    def __init__(self, obs_dim: int, pose_dim: int, macro_goal_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + pose_dim + macro_goal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 增加dropout防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6种离散动作
        )

    def forward(self, obs, pose, macro_goal):
        x = torch.cat([obs, pose, macro_goal], dim=-1)
        return self.fc(x)

