import random

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
from gsamllavanav.space import Point2D
from gsamllavanav.evaluate import eval_goal_predictor, run_episodes_batch, GoalPredictorMetrics
from gsamllavanav import logger
import matplotlib.pyplot as plt


def train(args: ExperimentArgs, device='cuda:3'):

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
    # train_episodes = train_episodes[:1]

    # # ====== 增强地图可视化调试（只做一次） ======
    # # 取第一个episode做调试
    # episode = train_episodes[0]
    # print(f"[DEBUG] 开始处理episode: {episode.id}")
    # print(f"[DEBUG] 地图名称: {episode.map_name}")
    # print(f"[DEBUG] 目标: {episode.target_processed_description.target}")
    # print(f"[DEBUG] 环境: {episode.target_processed_description.surroundings}")
    # print(f"[DEBUG] 地标: {episode.target_processed_description.landmarks}")
    
    # # 加载图像缓存
    # print("[DEBUG] 正在加载图像缓存...")
    # cropclient.load_image_cache()
    # print("[DEBUG] 图像缓存加载完成")
    
    # # 创建导航地图
    # nav_map = LandmarkNavMap(
    #     episode.map_name, args.map_shape, args.map_pixels_per_meter,
    #     episode.target_processed_description.landmarks,
    #     episode.target_processed_description.target,
    #     episode.target_processed_description.surroundings
    # )
    # print("[DEBUG] LandmarkNavMap创建完成")

    # # 获取导航路径上的所有RGB图像
    # trajectory = episode.trajectory  # 使用Episode类的trajectory属性
    # print(f"[DEBUG] 获取到导航路径，共 {len(trajectory)} 个位置点")
    
    # # 更新观察并生成GIF
    # print("[DEBUG] 开始处理导航过程...")
    # for step_idx, pose in enumerate(trajectory):
    #     # 获取RGB图像
    #     rgb = cropclient.crop_image(episode.map_name, pose, args.gsam_rgb_shape, 'rgb')
        
    #     # 标记最后一步（用于生成GIF）
    #     if step_idx == len(trajectory) - 1:
    #         nav_map.landmark_map.is_last_step = True
            
    #     # 更新观察
    #     nav_map.landmark_map.update_observation(
    #         camera_pose=pose,
    #         image_bgr=rgb,
    #         visualize=True
    #     )
        
    #     if step_idx % 10 == 0:  # 每10步打印一次进度
    #         print(f"[DEBUG] 已处理 {step_idx + 1}/{len(trajectory)} 步")
    
    # print("[DEBUG] 导航过程处理完成")

    # # 使用新的可视化方法
    # print("[DEBUG] 开始生成可视化...")
    
    # base_map = nav_map.landmark_map.to_array(mode='base')[0]  # [H, W]
    # enhanced_map = nav_map.landmark_map.to_array(mode='clip')[0]  # [H, W]
    # semantic_layer = nav_map.landmark_map.semantic_layer  # [H, W]

    # # 使用更合适的colormap
    # plt.imsave('debug_base_map.png', base_map, cmap='gray')
    # plt.imsave('debug_semantic_layer.png', semantic_layer, cmap='viridis')
    # plt.imsave('debug_enhanced_map.png', enhanced_map, cmap='viridis')
    
    # print("[DEBUG] 可视化完成。文件已保存到:")
    # print("- debug_base_map.png (原始地标图)")
    # print("- debug_semantic_layer.png (语义层)")
    # print("- debug_enhanced_map.png (增强后的地图)")
    # # ====== 结束调试代码 ======

    if args.train_episode_sample_size > 0:
        train_episodes = random.sample(train_episodes, args.train_episode_sample_size)
    train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_seen', 'all', args.altitude))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories('val_unseen', 'all', args.altitude))
    cropclient.load_image_cache()

    # init model & optim
    goal_predictor = GoalPredictor(args.map_size).to(device)
    optimizer = AdamW(goal_predictor.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, goal_predictor, optimizer = _load_checkpoint(goal_predictor, optimizer, args)

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

    writer = SummaryWriter(log_dir=f"runs/{args.experiment_name if hasattr(args, 'experiment_name') else 'default'}")

    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc='epochs', unit='epoch', colour='#448844'):
        total_loss_sum = 0
        goal_pred_loss_sum = 0
        progress_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        reward_mean_sum = 0
        r_short_sum = 0
        r_long_sum = 0
        penalty_sum = 0
        batch_count = 0
        
        for episodes_batch in tqdm(train_dataloader, desc='train episodes', unit='batch', colour='#88dd88'):
            
            maps, rgbs, normalized_depths = prepare_inputs(episodes_batch, args, device)
            normalized_goal_xys, progresses = prepare_labels(episodes_batch, args, device)
            
            outputs = goal_predictor(
            maps, rgbs, normalized_depths,
            target_positions=normalized_goal_xys,  # 传入真实目标坐标
            record_history=True,  # 记录轨迹用于RL训练
            flip_depth=True
            )
            pred_normalized_goal_xys = outputs["pred_xy"]
            pred_progresses = outputs["pred_progress"]
            # pred_normalized_goal_xys, pred_progresses = goal_predictor(maps, rgbs, normalized_depths, flip_depth=True)
            
            goal_prediction_loss = F.mse_loss(pred_normalized_goal_xys, normalized_goal_xys)
            progress_loss = F.mse_loss(pred_progresses, progresses)
            sup_loss = goal_prediction_loss + progress_loss

            # log_action_probs = outputs["action_probs"]
            # action_probs = log_action_probs.exp() 
            # action_mask = outputs["action_mask"]
            # tracking_quality = outputs["tracking_quality"]
            # state_values = outputs["state_value"]   # (B,1) 状态价值

            rewards, reward_info = goal_predictor.calculate_rewards(normalized_goal_xys)
            advantages = rewards - state_values.detach()  # 优势函数 (B,1)

            # # 改进的动作选择策略：结合宏观指导和微观感知
            # # 使用epsilon-greedy策略，大部分时间选择最大概率动作
            # epsilon = 0.1  # 探索概率
            # if torch.rand(1).item() < epsilon:
            #     masked_probs = action_probs * action_mask
            #     masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-8)
            #     print("masked_probs:", masked_probs)
            #     print("masked_probs sum:", masked_probs.sum(dim=1))
            #     assert not torch.isnan(masked_probs).any(), "masked_probs中有NaN"
            #     assert not torch.isinf(masked_probs).any(), "masked_probs中有Inf"
            #     assert (masked_probs.sum(dim=1) > 0).all(), "masked_probs某一行全为0"
            #     actions = torch.multinomial(masked_probs, num_samples=1).squeeze(1)
            # else:
            #     actions = torch.argmax(action_probs, dim=1)

            # print("actions:", actions)
            # print("actions dtype:", actions.dtype)
            # print("actions min:", actions.min().item(), "max:", actions.max().item())
            # assert actions.dtype == torch.long, "actions必须是long类型"
            # assert actions.min() >= 0, f"actions最小值越界: {actions.min().item()}"
            # assert actions.max() < 6, f"actions最大值越界: {actions.max().item()}"

            # # one-hot前
            # print("actions for onehot:", actions)
            # print("actions shape:", actions.shape)
            # print("actions unique:", actions.unique())

            # # gather前
            # print("log_action_probs shape:", log_action_probs.shape)
            # print("actions shape:", actions.shape)
            # print("actions:", actions)

            # log_probs = log_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            # policy_loss = -(log_probs * advantages.squeeze()).mean()
            
            # # 添加路径跟踪损失
            # tracking_loss = F.mse_loss(tracking_quality.squeeze(), torch.ones_like(tracking_quality.squeeze()))
            
            # # 添加动作掩码损失（鼓励动作选择符合宏观方向）
            # mask_loss = -torch.log(action_mask + 1e-8).mean()
            
            # value_loss = F.mse_loss(state_values.squeeze(), rewards.squeeze())

            rl_weight = 0.25  # 强化学习权重
            tracking_weight = 0.1  # 路径跟踪权重
            mask_weight = 0.05  # 动作掩码权重
            # total_loss = sup_loss + rl_weight * (policy_loss + 0.25 * value_loss) 
            # + tracking_weight * tracking_loss + mask_weight * mask_loss
            # total_loss = sup_loss + rl_weight * ( 0.25 * value_loss) 
            total_loss = sup_loss
            total_loss_sum += total_loss.item()
            goal_pred_loss_sum += goal_prediction_loss.item()
            progress_loss_sum += progress_loss.item()
            # policy_loss_sum += policy_loss.item()
            # value_loss_sum += value_loss.item()
            reward_mean_sum += sum(ri["reward"] for ri in reward_info) / len(reward_info)
            r_short_sum += sum(ri["r_short"] for ri in reward_info) / len(reward_info)
            r_long_sum += sum(ri["r_long"] for ri in reward_info) / len(reward_info)
            penalty_sum += sum(ri["penalty"] for ri in reward_info) / len(reward_info)
            batch_count += 1
        
            # ================= 反向传播&优化 =================
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(goal_predictor.parameters(), max_norm=0.5)  # 可选梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            
            # ================= 日志记录（新增RL指标） =================
            # logger.log({
            #     'loss': total_loss.item(),
            #     'goal_pred_loss': goal_prediction_loss.item(),
            #     'progress_loss': progress_loss.item(),
            #     'rl/policy_loss': policy_loss.item(),
            #     'rl/value_loss': value_loss.item(),
            #     'rl/reward_mean': rewards.mean().item()
            # })
            # global_step = epoch * len(train_dataloader) + batch_count
            # writer.add_scalar('Loss/total', total_loss.item(), global_step)
            # writer.add_scalar('Loss/goal_pred', goal_prediction_loss.item(), global_step)
            # writer.add_scalar('Loss/progress', progress_loss.item(), global_step)
            # writer.add_scalar('RL/policy_loss', policy_loss.item(), global_step)
            # writer.add_scalar('RL/value_loss', value_loss.item(), global_step)
            # writer.add_scalar('RL/reward_mean', rewards.mean().item(), global_step)
            # writer.add_scalar('RL/r_short', sum(ri["r_short"] for ri in reward_info) / len(reward_info), global_step)
            # writer.add_scalar('RL/r_long', sum(ri["r_long"] for ri in reward_info) / len(reward_info), global_step)
            # writer.add_scalar('RL/penalty', sum(ri["penalty"] for ri in reward_info) / len(reward_info), global_step)


        logger.log({
        'epoch': epoch,
        'loss': total_loss_sum / batch_count,
        'goal_pred_loss': goal_pred_loss_sum / batch_count,
        'progress_loss': progress_loss_sum / batch_count,
        'rl/policy_loss': policy_loss_sum / batch_count,
        'rl/value_loss': value_loss_sum / batch_count,
        'rl/reward_mean': reward_mean_sum / batch_count,
        'rl/r_short_mean': r_short_sum / batch_count,
        'rl/r_long_mean': r_long_sum / batch_count,
        'rl/penalty_mean': penalty_sum / batch_count,
        })

      
        gc.collect()
        torch.cuda.empty_cache()
        
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(epoch, goal_predictor, optimizer, args)
        
        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

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
    normalized_goal_xys = [
        normalize_position(episode.target_position, episode.map_name, args.map_meters)
        for episode in episodes_batch
        for _ in episode.sample_trajectory(args.map_update_interval)
    ]

    progresses = [
        np.clip(1 - episode.target_position.xy.dist_to(pose.xy) / episode.target_position.xy.dist_to(episode.start_pose.xy), 0, 1)
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]
    
    normalized_goal_xys = torch.tensor(normalized_goal_xys, device=device, dtype=torch.float32)
    progresses = torch.tensor(progresses, device=device, dtype=torch.float32).reshape(-1, 1)

    return normalized_goal_xys, progresses


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
    torch.save(
        {
            'epoch': epoch,
            'predictor_state_dict': goal_predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_dir/f"{epoch:03d}.pth"
    )