import json

import torch


import time
from datetime import datetime

from gsamllavanav.parser import parse_args
from gsamllavanav.evaluate import eval_goal_predictor
from gsamllavanav.cityreferobject import get_city_refer_objects
from gsamllavanav.dataset.generate import generate_episodes_from_mturk_trajectories
from gsamllavanav.dataset.mturk_trajectory import load_mturk_trajectories
from gsamllavanav.models.goal_predictor import GoalPredictor
from gsamllavanav.models.seq2seq_with_map import Seq2SeqwithMap
from gsamllavanav.models.cma_with_map import CMAwithMap
from gsamllavanav.evaluate import save_visualizations
# from gsamllavanav.goal_selection import goal_selection_gdino, goal_selection_llava
import gc

torch.cuda.empty_cache()
gc.collect()
DEVICE = 'cuda:1'

class TrainingTimer:
    def __enter__(self):
        self.start_time = datetime.now()
        print(f"\n🚀 Training Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        total_duration = self.end_time - self.start_time
        print(f"\n🎉 Training End: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total Time: {total_duration} [HH:MM:SS]")

args = parse_args()

if args.model == 'mgp':
    from gsamllavanav.train import train
    from gsamllavanav.evaluate import run_episodes_batch
else:
    from gsamllavanav.train_baseline_with_map import train
    from gsamllavanav.evaluate_baseline_with_map import run_episodes_batch

Model = {
    'mgp': GoalPredictor,
    'seq2seq_with_map': Seq2SeqwithMap,
    'cma_with_map': CMAwithMap,
}[args.model]

if args.mode == 'train':
    with TrainingTimer():
        train(args, DEVICE)

if args.mode == 'eval':

    print("Device:",DEVICE)
    model_trajectory = args.checkpoint.split('/')[-2]
    epoch = args.checkpoint.split('/')[-1].split('.')[0]

    objects = get_city_refer_objects()

    # load predictor
    model : GoalPredictor | CMAwithMap | Seq2SeqwithMap = Model(args.map_size).to(DEVICE)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu')['predictor_state_dict']
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)

    for split in ('val_seen', 'val_unseen', 'test_unseen'):
    # for split in ['test_unseen']:
        
        test_episodes = generate_episodes_from_mturk_trajectories(objects, load_mturk_trajectories(split, 'all', args.altitude))

        trajectory_logs, pred_goal_logs, pred_progress_logs= run_episodes_batch(args, model, test_episodes, DEVICE)

        
        metrics = eval_goal_predictor(
            args, test_episodes, trajectory_logs, pred_goal_logs, pred_progress_logs,
        )

        print(
            f"{split} -- "
            f"mean_final_pos_to_goal_dist: {metrics.mean_final_pos_to_goal_dist:.1f}, "
            f"success_rate_final_pos_to_goal: {metrics.success_rate_final_pos_to_goal*100:.2f}, "
            f"success_rate_oracle_pos_to_goal: {metrics.success_rate_oracle_pos_to_goal*100:.2f}, "
            f"success_rate_weighted_by_path_length:{metrics.success_rate_weighted_by_path_length*100:.2f},"
            f"{split} -- "
        )
        # from gsamllavanav.observation import cropclient
        # save_visualizations(test_episodes, trajectory_logs, cropclient)

        noise = f"noise_{args.gps_noise_scale}" if args.gps_noise_scale > 0 else ""
        alt_env = f"_{args.alt_env}" if args.alt_env else ""
        with open(f'{args.model}_{model_trajectory}_{split}_{args.progress_stop_val}{noise}{alt_env}_{args.eval_goal_selector}.json', 'w') as f:
            json.dump({
                'metrics': metrics.to_dict(),
                'trajectory_logs': {str(eps_id): [tuple(pose) for pose in trajectory] for eps_id, trajectory in trajectory_logs.items()},
                'pred_goal_logs': {str(eps_id): [tuple(pos) for pos in pred_goals] for eps_id, pred_goals in pred_goal_logs.items()},
                'pred_progress_logs': {str(eps_id): pred_progresses for eps_id, pred_progresses in pred_progress_logs.items()},
            }, f)

