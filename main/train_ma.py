import os
import torch
import argparse
import multiprocessing
multiprocessing.set_start_method('forkserver', force=True)
from multiprocessing import Process

import retro

from common.const import *
from common.utils import SubprocVecEnv2P, VecTransposeImage2P
from common.algorithms import LeaguePPO
from common.retro_wrappers import SFWrapper, Monitor2P
from common.league import PayoffManager, League, FSPLeague, PSROLeague, Learner


STATE = "Champion.RyuVsRyu.2Player.align"


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action)
        env = Monitor2P(env)
        env.seed(seed)
        return env
    return _init


def worker(idx, learner, total_steps, rollout_opponent_num):
    print(f"worker {learner.player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        learner.player.construct_agent()
        learner.run(total_timesteps=total_steps, rollout_opponent_num=rollout_opponent_num)


def restore_worker(idx, learner, total_steps, rollout_opponent_num):
    print(f"restore_worker {learner.player.name} start")
    with torch.cuda.device(idx % torch.cuda.device_count()):
        learner.player.construct_agent()
        learner.player._initial_weights = learner.player._initial_weights_restore # restore the initial weights to the reset weights
        learner.run(total_timesteps=total_steps, rollout_opponent_num=rollout_opponent_num, reset_num_timesteps=False) # NOTE: do not reset num_timesteps so that the timesteps are restored


def constructor(args, side, log_name=None, single_env=False):
    num_env = 1 if single_env else args.num_env
    env = [make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=i) for i in range(num_env)]
    env = VecTransposeImage2P(SubprocVecEnv2P(env))
    return LeaguePPO(
        side,
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=1024, # 512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=1e-4, # lr_schedule,
        clip_range=0.1, # clip_range_schedule,
        tensorboard_log=None if log_name is None else os.path.join(args.log_dir, log_name),
        # seed=args.seed,
        other_learning_rate=1e-4, # other_lr_schedule,
    )


def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    # parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models/ma")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs/ma")
    # parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    # parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=24)
    # parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    # parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e10)) # 1e5
    # parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    # parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    # parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    # parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    # parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    # parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    # parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    # parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    # parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument('--rollout-opponent-num', type=int, help='Numbers of opponents to interact for each update', default=5) # 2
    parser.add_argument('--fsp-league', action='store_true', help='Fictitious self-play league')
    parser.add_argument('--psro-league', action='store_true', help='PSRO league')
    
    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.video_dir, exist_ok=True)
    # os.makedirs(args.finetune_dir, exist_ok=True)

    left_model = constructor(args, "left", log_name=None, single_env=True)
    right_model = constructor(args, "right", log_name=None, single_env=True)

    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        left_model.set_parameters_2p(args.left_model_file, args.right_model_file)
        right_model.set_parameters_2p(args.left_model_file, args.right_model_file)
    
    initial_agents = {
        'left': left_model,
        'right': right_model,
    }
    
    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        if args.fsp_league:
            league = FSPLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        elif args.psro_league:
            league = PSROLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        else:
            league = League(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, main_exploiters=1, league_exploiters=2)
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            learner = Learner(player)
            process = Process(target=worker, args=(idx, learner, args.total_steps, args.rollout_opponent_num))
            # process.daemon=True  # all processes closed when the main stops
            processes.append(process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


def restore():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    # parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models/ma")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs/ma")
    # parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    # parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=24)
    # parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    # parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e10)) # 1e5
    # parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    # parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    # parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    # parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    # parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    # parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    # parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    # parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    # parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    parser.add_argument('--rollout-opponent-num', type=int, help='Numbers of opponents to interact for each update', default=5) # 2
    parser.add_argument('--fsp-league', action='store_true', help='Fictitious self-play league')
    parser.add_argument('--psro-league', action='store_true', help='PSRO league')
    
    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # os.makedirs(args.video_dir, exist_ok=True)
    # os.makedirs(args.finetune_dir, exist_ok=True)

    model_files = {
        "LE0_left": "trained_models/ma_231218/LE0_left_230214912.pt",
        "LE0_right": "trained_models/ma_231218/LE0_right_235170576.pt",
        "LE1_left": "trained_models/ma_231218/LE1_left_216107520.pt",
        "LE1_right": "trained_models/ma_231218/LE1_right_228268080.pt",
        "MA0_left": "trained_models/ma_231218/MA0_left_237694104.pt",
        "MA0_right": "trained_models/ma_231218/MA0_right_224946696.pt",
        "ME0_left": "trained_models/ma_231218/ME0_left_230108016.pt",
        "ME0_right": "trained_models/ma_231218/ME0_right_230112360.pt",
    }
    payoff_file = "trained_models/ma_231218/payoff_20231218_20_38.pt"
    
    left_model = constructor(args, "left", log_name=None, single_env=True)
    right_model = constructor(args, "right", log_name=None, single_env=True)

    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        left_model.set_parameters_2p(args.left_model_file, args.right_model_file)
        right_model.set_parameters_2p(args.left_model_file, args.right_model_file)
    
    initial_agents = {
        'left': left_model,
        'right': right_model,
    }
    
    with PayoffManager() as manager:
        shared_payoff = manager.Payoff(args.save_dir)
        if args.fsp_league:
            league = FSPLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        elif args.psro_league:
            league = PSROLeague(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1)
        else:
            league = League(args=args, initial_agents=initial_agents, constructor=constructor, payoff=shared_payoff, main_agents=1, main_exploiters=1, league_exploiters=2)
        processes = []
        for idx in range(league.size()):
            player = league.get_player(idx)
            player.load(model_files[player.name])
        shared_payoff.load(payoff_file)
        for idx in range(league.size()):
            player = league.get_player(idx)
            learner = Learner(player)
            process = Process(target=restore_worker, args=(idx, learner, args.total_steps, args.rollout_opponent_num))
            # process.daemon=True  # all processes closed when the main stops
            processes.append(process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
    # restore() # NOTE: backup checkpoint before running this