import gc
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# Run:
# /cver/cjfan/miniconda3/envs/flightgpt/bin/python -m gsamllavanav.train_grpo --mode train --model mgp

from gsamllavanav import logger
from gsamllavanav.cityreferobject import MultiMapObjects, get_city_refer_objects
from gsamllavanav.dataset.episode import Episode
from gsamllavanav.dataset.generate import (
    convert_trajectory_to_shortest_path,
    generate_episodes_from_mturk_trajectories,
)
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.defaultpaths import GOAL_PREDICTOR_CHECKPOINT_DIR
from gsamllavanav.evaluate import GoalPredictorMetrics, eval_goal_predictor, run_episodes_batch
from gsamllavanav.mapdata import MAP_BOUNDS
from gsamllavanav.maps.landmark_nav_map import LandmarkNavMap
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.observation import cropclient
from gsamllavanav.parser import ExperimentArgs
from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import (
    LookaheadTeacherParams,
    lookahead_continuous_action,
    lookahead_discrete_action,
)


class GRPOAgent:
    def __init__(
        self,
        goal_predictor: GoalPredictor,
        map_size: int,
        device: str = "cuda:0",
        supervised_loss_weight: float = 1.0,
        grpo_loss_weight: float = 0.2,
        kl_weight: float = 0.01,
        group_size: int = 4,
        sample_temperature: float = 1.0,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 0.5,
    ):
        self.goal_predictor = goal_predictor
        self.map_size = map_size
        self.device = device
        self.supervised_loss_weight = supervised_loss_weight
        self.grpo_loss_weight = grpo_loss_weight
        self.kl_weight = kl_weight
        self.group_size = group_size
        self.sample_temperature = sample_temperature
        self.max_grad_norm = max_grad_norm
        self.optimizer = AdamW(self.goal_predictor.parameters(), lr=learning_rate)
        self.reference_model: GoalPredictor | None = None
        self.total_updates = 0
        self.total_group_samples = 0

    def attach_reference_model(self):
        self.reference_model = GoalPredictor(self.map_size).to(self.device)
        self.reference_model.load_state_dict(self.goal_predictor.state_dict())
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def copy_value_head_from_supervised_model(self, source_model: GoalPredictor):
        source_params = list(source_model.value_prediction_head.parameters())
        target_params = list(self.goal_predictor.value_prediction_head.parameters())
        for target_param, source_param in zip(target_params, source_params):
            if target_param.shape == source_param.shape:
                with torch.no_grad():
                    target_param.copy_(source_param)

    def supervised_step(
        self,
        maps: torch.Tensor,
        rgbs: torch.Tensor,
        normalized_depths: torch.Tensor,
        normalized_goal_xys: torch.Tensor,
        progresses: torch.Tensor,
        macro_goal_teachers: torch.Tensor,
        micro_action_teachers: torch.Tensor,
        pose_tensor: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.goal_predictor.train()
        outputs = self.goal_predictor(
            maps,
            rgbs,
            normalized_depths,
            pose=pose_tensor,
            goal_desc=normalized_goal_xys,
            flip_depth=True,
        )

        goal_prediction_loss = F.mse_loss(outputs["pred_xy"], normalized_goal_xys)
        progress_loss = F.mse_loss(outputs["pred_progress"], progresses)
        value_loss = F.mse_loss(outputs["pred_value"], value_targets)
        macro_loss = F.mse_loss(outputs["macro_goal"], macro_goal_teachers)
        micro_loss = F.cross_entropy(outputs["micro_action_logits"], micro_action_teachers)

        supervised_loss = goal_prediction_loss + progress_loss + macro_loss + micro_loss + value_loss
        self.optimizer.zero_grad()
        supervised_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_predictor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": supervised_loss.detach(),
            "goal_prediction_loss": goal_prediction_loss.detach(),
            "progress_loss": progress_loss.detach(),
            "macro_loss": macro_loss.detach(),
            "micro_loss": micro_loss.detach(),
            "value_loss": value_loss.detach(),
            "grpo_loss": torch.tensor(0.0, device=self.device),
            "reward": torch.tensor(0.0, device=self.device),
            "advantage": torch.tensor(0.0, device=self.device),
        }

    def grpo_step(
        self,
        maps: torch.Tensor,
        rgbs: torch.Tensor,
        normalized_depths: torch.Tensor,
        normalized_goal_xys: torch.Tensor,
        progresses: torch.Tensor,
        macro_goal_teachers: torch.Tensor,
        micro_action_teachers: torch.Tensor,
        pose_tensor: torch.Tensor,
        value_targets: torch.Tensor,
        rollout_steps: int = 4,
    ) -> dict[str, torch.Tensor]:
        if self.reference_model is None:
            self.attach_reference_model()

    def _kl_divergence(self, current_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        current_log_probs = torch.log_softmax(current_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        return F.kl_div(current_log_probs, ref_probs, reduction="batchmean")

    def grpo_step(
        self,
        maps: torch.Tensor,
        rgbs: torch.Tensor,
        normalized_depths: torch.Tensor,
        normalized_goal_xys: torch.Tensor,
        progresses: torch.Tensor,
        macro_goal_teachers: torch.Tensor,
        micro_action_teachers: torch.Tensor,
        pose_tensor: torch.Tensor,
        value_targets: torch.Tensor,
        rollout_steps: int = 4,
    ) -> dict[str, torch.Tensor]:
        if self.reference_model is None:
            self.attach_reference_model()

        self.goal_predictor.train()
        current_outputs = self.goal_predictor(
            maps,
            rgbs,
            normalized_depths,
            pose=pose_tensor,
            goal_desc=normalized_goal_xys,
            flip_depth=True,
        )
        with torch.no_grad():
            ref_outputs = self.reference_model(
                maps,
                rgbs,
                normalized_depths,
                pose=pose_tensor,
                goal_desc=normalized_goal_xys,
                flip_depth=True,
            )

        supervised_goal_loss = F.mse_loss(current_outputs["pred_xy"], normalized_goal_xys)
        supervised_progress_loss = F.mse_loss(current_outputs["pred_progress"], progresses)
        supervised_value_loss = F.mse_loss(current_outputs["pred_value"], value_targets)
        supervised_macro_loss = F.mse_loss(current_outputs["macro_goal"], macro_goal_teachers)
        supervised_micro_loss = F.cross_entropy(current_outputs["micro_action_logits"], micro_action_teachers)
        supervised_loss = (
            supervised_goal_loss
            + supervised_progress_loss
            + supervised_value_loss
            + supervised_macro_loss
            + supervised_micro_loss
        )

        rollout_rewards = []
        rollout_log_probs = []
        rollout_advantages = []

        for _ in range(self.group_size):
            sampled_actions = self._sample_action_sequences(current_outputs["micro_action_logits"], rollout_steps)
            sampled_log_prob = self._sequence_log_prob(current_outputs["micro_action_logits"], sampled_actions)
            sampled_reward = self._rollout_reward(
                sampled_actions=sampled_actions,
                current_outputs=current_outputs,
                ref_outputs=ref_outputs,
                normalized_goal_xys=normalized_goal_xys,
                progresses=progresses,
                macro_goal_teachers=macro_goal_teachers,
                value_targets=value_targets,
            )
            rollout_log_probs.append(sampled_log_prob)
            rollout_rewards.append(sampled_reward)

        rewards = torch.stack(rollout_rewards, dim=0)
        reward_mean = rewards.mean(dim=0, keepdim=True)
        reward_std = rewards.std(dim=0, keepdim=True).clamp_min(1e-6)
        advantages = (rewards - reward_mean) / reward_std

        for index in range(self.group_size):
            rollout_advantages.append(advantages[index])

        policy_loss = -(
            torch.stack(rollout_log_probs, dim=0) * torch.stack(rollout_advantages, dim=0).detach()
        ).mean()
        kl_loss = self._kl_divergence(current_outputs["micro_action_logits"], ref_outputs["micro_action_logits"])
        grpo_loss = policy_loss + 0.5 * supervised_value_loss + self.kl_weight * kl_loss
        total_loss = self.supervised_loss_weight * supervised_loss + self.grpo_loss_weight * grpo_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_predictor.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.total_updates += 1
        self.total_group_samples += self.group_size

        return {
            "total_loss": total_loss.detach(),
            "goal_prediction_loss": supervised_goal_loss.detach(),
            "progress_loss": supervised_progress_loss.detach(),
            "macro_loss": supervised_macro_loss.detach(),
            "micro_loss": supervised_micro_loss.detach(),
            "value_loss": supervised_value_loss.detach(),
            "grpo_loss": grpo_loss.detach(),
            "reward": rewards.mean().detach(),
            "advantage": advantages.mean().detach(),
        }

    def _sample_action_sequences(self, logits: torch.Tensor, rollout_steps: int) -> torch.Tensor:
        probs = torch.softmax(logits / self.sample_temperature, dim=-1)
        action_seq = []
        for _ in range(rollout_steps):
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
            action_seq.append(action)
        return torch.stack(action_seq, dim=0)

    def _sequence_log_prob(self, logits: torch.Tensor, sampled_actions: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        step_log_probs = []
        for step_actions in sampled_actions:
            step_log_probs.append(log_probs.gather(dim=-1, index=step_actions.unsqueeze(-1)).squeeze(-1))
        return torch.stack(step_log_probs, dim=0).mean(dim=0)

    def _rollout_reward(
        self,
        sampled_actions: torch.Tensor,
        current_outputs: dict[str, torch.Tensor],
        ref_outputs: dict[str, torch.Tensor],
        normalized_goal_xys: torch.Tensor,
        progresses: torch.Tensor,
        macro_goal_teachers: torch.Tensor,
        value_targets: torch.Tensor,
    ) -> torch.Tensor:
        action_reward = sampled_actions.float().mean(dim=0) / 5.0
        goal_reward = 1.0 - F.mse_loss(current_outputs["pred_xy"], normalized_goal_xys, reduction="none").mean(dim=-1)
        progress_reward = 1.0 - F.mse_loss(current_outputs["pred_progress"], progresses, reduction="none").squeeze(-1)
        macro_reward = 1.0 - F.mse_loss(current_outputs["macro_goal"], macro_goal_teachers, reduction="none").mean(dim=-1)
        value_reward = 1.0 - F.mse_loss(current_outputs["pred_value"], value_targets, reduction="none").squeeze(-1)
        ref_kl = self._per_sample_kl(current_outputs["micro_action_logits"], ref_outputs["micro_action_logits"])
        return (
            0.20 * action_reward
            + 0.25 * goal_reward
            + 0.15 * progress_reward
            + 0.20 * macro_reward
            + 0.20 * value_reward
            - 0.10 * ref_kl
        ).detach()

    def _per_sample_kl(self, current_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        current_log_probs = torch.log_softmax(current_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        return torch.sum(ref_probs * (torch.log(ref_probs.clamp_min(1e-8)) - current_log_probs), dim=-1)


def normalize_position(pos: Point2D, map_name: str, map_meters: float):
    return (pos.x - MAP_BOUNDS[map_name].x_min) / map_meters, (MAP_BOUNDS[map_name].y_max - pos.y) / map_meters


def prepare_inputs(episodes_batch: list[Episode], args: ExperimentArgs, device: str):
    maps = np.concatenate([
        LandmarkNavMap.generate_maps_for_an_episode(
            episode,
            args.map_shape,
            args.map_pixels_per_meter,
            args.map_update_interval,
            args.gsam_rgb_shape,
            args.gsam_use_map_cache,
        )
        for episode in episodes_batch
    ])

    rgbs = np.stack([
        cropclient.crop_image(episode.map_name, pose, (224, 224), "rgb")
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2)

    normalized_depths = np.stack([
        cropclient.crop_image(episode.map_name, pose, (256, 256), "depth")
        for episode in episodes_batch
        for pose in episode.sample_trajectory(args.map_update_interval)
    ]).transpose(0, 3, 1, 2) / args.max_depth

    if args.ablate == "rgb":
        rgbs = np.zeros_like(rgbs)
    if args.ablate == "depth":
        normalized_depths = np.zeros_like(normalized_depths)
    if args.ablate == "tracking":
        maps[:, :2] = 0
    if args.ablate == "landmark":
        maps[:, 2] = 0
    if args.ablate == "gsam":
        maps[:, 3:] = 0

    maps = torch.tensor(maps, device=device)
    rgbs = torch.tensor(rgbs, device=device)
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

    for episode in episodes_batch:
        trajectory_pose4d = list(episode.sample_trajectory(args.map_update_interval))
        trajectory_point3d = [Point3D(pose.x, pose.y, pose.z) for pose in trajectory_pose4d]
        for pose in trajectory_pose4d:
            normalized_goal_xys.append(normalize_position(episode.target_position, episode.map_name, args.map_meters))
            progress = np.clip(
                1 - episode.target_position.xy.dist_to(pose.xy) / episode.target_position.xy.dist_to(episode.start_pose.xy),
                0,
                1,
            )
            progresses.append(progress)
            value_targets.append(progress)
            macro_goal = lookahead_continuous_action(pose, trajectory_point3d, lookahead)
            macro_goal_teachers.append(macro_goal)
            action = lookahead_discrete_action(pose, trajectory_point3d, teacher_params)
            micro_action_teachers.append(action.index)
            pose_tensor = torch.tensor([pose.x, pose.y, pose.z, pose.yaw], device=device, dtype=torch.float32)
            pose_tensors.append(pose_tensor)

    normalized_goal_xys = torch.tensor(normalized_goal_xys, device=device, dtype=torch.float32)
    progresses = torch.tensor(progresses, device=device, dtype=torch.float32).reshape(-1, 1)
    value_targets = torch.tensor(value_targets, device=device, dtype=torch.float32).reshape(-1, 1)
    micro_action_teachers = torch.tensor(micro_action_teachers, device=device, dtype=torch.long)
    pose_tensors = torch.stack(pose_tensors)

    if hasattr(macro_goal_teachers[0], "forward_stride"):
        macro_goal_teachers = [[goal.forward_stride, goal.d_yaw] for goal in macro_goal_teachers]
    else:
        macro_goal_teachers = [goal[:2] if len(goal) > 2 else goal for goal in macro_goal_teachers]

    if len(macro_goal_teachers) > 0:
        if hasattr(macro_goal_teachers[0], "forward_stride"):
            values = [[goal.forward_stride, goal.d_yaw] for goal in macro_goal_teachers]
            max_abs_value = max(max(abs(x) for x in pair) for pair in values)
            macro_goal_teachers = [[v / max_abs_value for v in pair] for pair in values] if max_abs_value > 0 else values
        else:
            max_abs = max(max(abs(x) for x in goal) for goal in macro_goal_teachers)
            macro_goal_teachers = [[v / max_abs for v in goal] for goal in macro_goal_teachers] if max_abs > 0 else macro_goal_teachers
    macro_goal_teachers = torch.tensor(macro_goal_teachers, device=device, dtype=torch.float32)

    return normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensors, value_targets


def _load_train_episodes(objects: MultiMapObjects, args: ExperimentArgs) -> list[Episode]:
    mturk_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories("train_seen", "all", args.altitude))
    if args.train_trajectory_type == "mturk":
        return mturk_episodes
    if args.train_trajectory_type == "sp":
        return [convert_trajectory_to_shortest_path(eps, "linear_xy") for eps in tqdm(mturk_episodes, desc="converting to shortest path episode")]
    if args.train_trajectory_type == "both":
        return mturk_episodes + [convert_trajectory_to_shortest_path(eps, "linear_xy") for eps in tqdm(mturk_episodes, desc="converting to shortest path episode")]
    return mturk_episodes


def _eval_predictor_and_log_metrics(
    goal_predictor: GoalPredictor,
    val_seen_episodes: list[Episode],
    val_unseen_episodes: list[Episode],
    args: ExperimentArgs,
    device: str,
):
    val_seen_metrics = eval_goal_predictor(args, val_seen_episodes, *run_episodes_batch(args, goal_predictor, val_seen_episodes, device))
    val_unseen_metrics = eval_goal_predictor(args, val_unseen_episodes, *run_episodes_batch(args, goal_predictor, val_unseen_episodes, device))
    logger.log({"val_seen_" + k: v for k, v in val_seen_metrics.to_dict().items()})
    logger.log({"val_unseen_" + k: v for k, v in val_unseen_metrics.to_dict().items()})


def _load_checkpoint(goal_predictor: GoalPredictor, optimizer: torch.optim.Optimizer, args: ExperimentArgs):
    checkpoint = torch.load(args.checkpoint)
    start_epoch: int = checkpoint["epoch"] + 1
    goal_predictor.load_state_dict(checkpoint["predictor_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return start_epoch, goal_predictor, optimizer


def _save_checkpoint(epoch: int, goal_predictor: GoalPredictor, optimizer: torch.optim.Optimizer, args: ExperimentArgs):
    ablation = f"-{args.ablate}" if args.ablate else ""
    train_size = "" if args.train_episode_sample_size < 0 else f"_{args.train_episode_sample_size}"
    checkpoint_dir = GOAL_PREDICTOR_CHECKPOINT_DIR / f"grpo_{args.train_trajectory_type}_{args.altitude}_{args.gsam_box_threshold}{ablation}{train_size}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "epoch": epoch,
            "predictor_state_dict": goal_predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_dir / f"{epoch:03d}.pth",
    )


def train(args: ExperimentArgs, device: str = "cuda:7"):
    print(f"Training on device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.init(args)
    for metric in GoalPredictorMetrics.names():
        logger.define_metric("val_seen_" + metric, "epoch")
        logger.define_metric("val_unseen_" + metric, "epoch")

    start_epoch = 0
    objects = get_city_refer_objects()
    train_episodes = _load_train_episodes(objects, args)
    if args.train_episode_sample_size > 0:
        train_episodes = random.sample(train_episodes, args.train_episode_sample_size)
    train_dataloader = DataLoader(train_episodes, args.train_batch_size, shuffle=True, collate_fn=lambda x: x)
    val_seen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories("val_seen", "all", args.altitude))
    val_unseen_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories("val_unseen", "all", args.altitude))
    cropclient.load_image_cache()

    goal_predictor = GoalPredictor(args.map_size).to(device)
    optimizer = AdamW(goal_predictor.parameters(), args.learning_rate)
    if args.checkpoint:
        start_epoch, goal_predictor, optimizer = _load_checkpoint(goal_predictor, optimizer, args)

    if args.eval_at_start:
        _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

    writer = SummaryWriter(log_dir=f"runs/{args.experiment_name if hasattr(args, 'experiment_name') else 'grpo'}")

    grpo_agent = GRPOAgent(
        goal_predictor=goal_predictor,
        map_size=args.map_size,
        device=device,
        supervised_loss_weight=1.0,
        grpo_loss_weight=getattr(args, "grpo_loss_weight", 0.2),
        kl_weight=getattr(args, "grpo_kl_weight", 0.01),
        group_size=getattr(args, "grpo_group_size", 4),
        sample_temperature=getattr(args, "grpo_temperature", 1.0),
        learning_rate=args.learning_rate,
        max_grad_norm=0.5,
    )
    grpo_agent.optimizer = optimizer
    grpo_agent.copy_value_head_from_supervised_model(goal_predictor)
    grpo_agent.attach_reference_model()

    supervised_pretrain_epochs = getattr(args, "grpo_pretrain_epochs", 1)
    grpo_start_epoch = supervised_pretrain_epochs if getattr(args, "grpo_start_epoch", -1) < 0 else args.grpo_start_epoch

    episodes_batch: list[Episode]
    for epoch in trange(start_epoch, args.epochs, desc="epochs", unit="epoch", colour="#448844"):
        total_loss_sum = 0.0
        goal_pred_loss_sum = 0.0
        progress_loss_sum = 0.0
        macro_loss_sum = 0.0
        micro_loss_sum = 0.0
        value_loss_sum = 0.0
        grpo_loss_sum = 0.0
        grpo_reward_sum = 0.0
        batch_idx = 0
        use_grpo = epoch >= grpo_start_epoch

        for episodes_batch in tqdm(train_dataloader, desc="train episodes", unit="batch", colour="#88dd88"):
            maps, rgbs, normalized_depths = prepare_inputs(episodes_batch, args, device)
            normalized_goal_xys, progresses, macro_goal_teachers, micro_action_teachers, pose_tensor, value_targets = prepare_labels(episodes_batch, args, device)

            if not use_grpo:
                metrics = grpo_agent.supervised_step(
                    maps,
                    rgbs,
                    normalized_depths,
                    normalized_goal_xys,
                    progresses,
                    macro_goal_teachers,
                    micro_action_teachers,
                    pose_tensor,
                    value_targets,
                )
                total_loss = metrics["total_loss"]
                grpo_loss = torch.tensor(0.0, device=device)
                grpo_reward = torch.tensor(0.0, device=device)
            else:
                metrics = grpo_agent.grpo_step(
                    maps,
                    rgbs,
                    normalized_depths,
                    normalized_goal_xys,
                    progresses,
                    macro_goal_teachers,
                    micro_action_teachers,
                    pose_tensor,
                    value_targets,
                    rollout_steps=getattr(args, "grpo_rollout_steps", 4),
                )
                total_loss = metrics["total_loss"]
                grpo_loss = metrics["grpo_loss"]
                grpo_reward = metrics["reward"]

            with torch.no_grad():
                out = goal_predictor(maps, rgbs, normalized_depths, pose=pose_tensor, goal_desc=normalized_goal_xys, flip_depth=True)
                goal_prediction_loss = F.mse_loss(out["pred_xy"], normalized_goal_xys)
                progress_loss = F.mse_loss(out["pred_progress"], progresses)
                value_loss = F.mse_loss(out["pred_value"], value_targets)
                macro_loss = F.mse_loss(out["macro_goal"], macro_goal_teachers)
                micro_loss = F.cross_entropy(out["micro_action_logits"], micro_action_teachers)

            total_loss_sum += float(metrics["total_loss"].item())
            goal_pred_loss_sum += float(goal_prediction_loss.item())
            progress_loss_sum += float(progress_loss.item())
            macro_loss_sum += float(macro_loss.item())
            micro_loss_sum += float(micro_loss.item())
            value_loss_sum += float(value_loss.item())
            grpo_loss_sum += float(metrics["grpo_loss"].item())
            grpo_reward_sum += float(metrics["reward"].item())
            batch_idx += 1

            writer.add_scalar("train/total_loss", float(metrics["total_loss"].item()), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar("train/grpo_loss", float(metrics["grpo_loss"].item()), epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar("train/grpo_reward", float(metrics["reward"].item()), epoch * len(train_dataloader) + batch_idx)

        logger.log(
            {
                "epoch": epoch,
                "loss": total_loss_sum / max(batch_idx, 1),
                "goal_pred_loss": goal_pred_loss_sum / max(batch_idx, 1),
                "progress_loss": progress_loss_sum / max(batch_idx, 1),
                "macro_loss": macro_loss_sum / max(batch_idx, 1),
                "micro_loss": micro_loss_sum / max(batch_idx, 1),
                "value_loss": value_loss_sum / max(batch_idx, 1),
                "grpo_loss": grpo_loss_sum / max(batch_idx, 1),
                "grpo_reward": grpo_reward_sum / max(batch_idx, 1),
            }
        )

        gc.collect()
        torch.cuda.empty_cache()

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(epoch, goal_predictor, optimizer, args)

        if (epoch + 1) % args.eval_every == 0:
            _eval_predictor_and_log_metrics(goal_predictor, val_seen_episodes, val_unseen_episodes, args, device)

    writer.close()


def main():
    from gsamllavanav.parser import parse_args

    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
