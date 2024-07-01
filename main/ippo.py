import os
import av
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import copy

import retro
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from common.const import *
from common.utils import linear_schedule, SubprocVecEnv2P, VecTransposeImage2P
from common.game import get_next_level
from common.algorithms import IPPO
from common.retro_wrappers import SFWrapper, Monitor2P


STATE = "Champion.RyuVsRyu.2Player.align"


def constructor(args, side, log_name=None, single_env=False):
    pass


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


@torch.no_grad()
def evaluate(args, model, greedy=0.99, record=True):
    win_cnt = 0
    
    for i in range(1, args.num_episodes + 1):
        env = make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=None)().env

        done = False
        
        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            if np.random.uniform() > greedy:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=False)
            else:
                (action, _states), (action_other, _states_other) = model.predict(obs, deterministic=True)

            obs, reward, reward_other, done, info = env.step(np.hstack([action, action_other]))
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

            if done:
                if record:
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/episode_{i}.mp4", mode='w')
                    stream = container.add_stream('h264', rate=10)
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = 'yuv420p'
                    for img in video_log:
                        frame = av.VideoFrame.from_image(img)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    remain_packets = stream.encode(None)
                    container.mux(remain_packets)
                    container.close()

        if info['enemy_hp'] < info['agent_hp']:
            print("Victory!")
            win_cnt += 1

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward
    
        env.close()
    
    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    return win_rate


def main():
    parser = argparse.ArgumentParser(description='Reset game stats')
    parser.add_argument('--reset', choices=['round', 'match', 'game'], help='Reset stats for a round, a match, or the whole game', default='round')
    parser.add_argument('--model-file', help='The model to continue to learn from')
    parser.add_argument('--save-dir', help='The directory to save the trained models', default="trained_models")
    parser.add_argument('--log-dir', help='The directory to save logs', default="logs")
    parser.add_argument('--model-name-prefix', help='The prefix of the model names to save', default="ppo_ryu")
    parser.add_argument('--state', help='The state file to load. By default Champion.Level1.RyuVsGuile', default=SF_DEFAULT_STATE)
    parser.add_argument('--side', help='The side for AI to control. By default both', default='both', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e7))
    parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--seed', type=int, help='Seed', default=0)
    parser.add_argument('--update-left', type=int, help='Update left policy', default=1)
    parser.add_argument('--update-right', type=int, help='Update right policy', default=1)
    parser.add_argument('--left-model-file', help='The left model to continue to learn from')
    parser.add_argument('--right-model-file', help='The right model to continue to learn from')
    parser.add_argument('--other-timescale', type=float, help='Other agent learning rate scale', default=1.0)
    parser.add_argument('--fsp', action='store_true', help='Fictitious self-play')
    parser.add_argument('--fsp-threshold', type=float, help='Fictitious self-play threshold', default=0.5)
    parser.add_argument('--async-update', action='store_true', help='Update left and right asynchronously')
    
    args = parser.parse_args()
    print("command line args:" + str(args))

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.finetune_dir, exist_ok=True)
                                 
    # Set up the environment and model
    def env_generator():
        env = [make_env(sf_game, state=STATE, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, seed=i) for i in range(args.num_env)]
        return VecTransposeImage2P(SubprocVecEnv2P(env))
        # return SubprocVecEnv2P(env)

    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint

    def finetune_model_generator(model_file=None, lr_schedule=linear_schedule(5.0e-5, 2.5e-6), other_lr_schedule=linear_schedule(5.0e-5, 2.5e-6), clip_range_schedule=linear_schedule(0.075, 0.025)):
        finetune_env = env_generator()
        finetune_model = IPPO(
            "CnnPolicy", 
            finetune_env,
            device="cuda", 
            verbose=1,
            n_steps=512,
            batch_size=1024, # 512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=args.log_dir,
            seed=args.seed,
            update_left=bool(args.update_left),
            update_right=bool(args.update_right),
            other_learning_rate=other_lr_schedule,
        )
        if model_file:
            print("load model from " + model_file)
            if model_file.endswith(".pt"):
                model_file = torch.load(model_file, map_location=torch.device('cpu'))["kwargs"]["agent_dict"]
            finetune_model.set_parameters(model_file)
        return finetune_model

    finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_final_steps")
    lr_schedule = 1e-4 # if args.async_update else linear_schedule(2.5e-4, 2.5e-6)
    other_lr_schedule = 1e-4 # if args.async_update else linear_schedule(2.5e-4/args.other_timescale, 2.5e-6/args.other_timescale)
    clip_range_schedule = 0.1 # if args.async_update else linear_schedule(0.15, 0.025)
    model = finetune_model_generator(args.model_file, lr_schedule=lr_schedule, other_lr_schedule=other_lr_schedule, clip_range_schedule=clip_range_schedule)

    if args.left_model_file and args.right_model_file:
        print("load model from " + args.left_model_file + " and " + args.right_model_file)
        model.set_parameters_2p(args.left_model_file, args.right_model_file)
    model.save(os.path.join(args.save_dir, args.model_name_prefix + f"_0_steps"))
    
    results = evaluate(args, model, record=True)
    print(results)
    # assert False

    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=f"{args.model_name_prefix}")
    if args.async_update:
        model.async_learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback],
            fsp=args.fsp,
            fsp_threshold=args.fsp_threshold,
        )
    else:
        model.learn( 
            total_timesteps=args.total_steps*args.other_timescale,
            callback=[checkpoint_callback]
        )
    model.save(finetune_epoch_model_path)
    results = evaluate(args, model, record=True)
    print(results)
    with open(f"{args.finetune_dir}/{args.model_name_prefix}_start_results.txt", 'w') as f:
        f.write(str(results))


if __name__ == "__main__":
    main()
