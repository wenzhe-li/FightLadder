import os
import av
import sys
import torch
import argparse
from PIL import Image

import retro
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from common.const import *
from common.utils import linear_schedule, AnnealDenseCallback, AnnealAgressiveCallback
from common.retro_wrappers import SFWrapper


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, num_stack=12, num_step_frames=8, seed=0):
    def _init():
        players = 2
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            players=players
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


@torch.no_grad()
def evaluate(args, model=None, left_model=None, right_model=None, greedy=0.99, record=True, suffix=""):
    reset_type = 'round'
    win_cnt = 0
    
    for i in range(1, args.num_episodes + 1):

        if 'starall' in args.state:
            states = [args.state.replace('starall', f'star{i}') for i in range(1, 9)]
            env = make_env(sf_game, state=states[i % len(states)], side=args.side, reset_type=reset_type, rendering=False, verbose=True, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i)().env
        else:
            env = make_env(sf_game, state=args.state, side=args.side, reset_type=reset_type, rendering=False, verbose=True, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i)().env

        done = False
        
        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            if model is not None:
                if np.random.uniform() > greedy:
                    action, _states = model.predict(obs, deterministic=False)
                else:
                    action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            elif left_model is not None and right_model is not None:
                if np.random.uniform() > greedy:
                    left_action, left_states = left_model.predict(obs, deterministic=False)
                else:
                    left_action, left_states = left_model.predict(obs, deterministic=True)
                if np.random.uniform() > greedy:
                    right_action, right_states = right_model.predict(obs, deterministic=False)
                else:
                    right_action, right_states = right_model.predict(obs, deterministic=True)
                action = np.hstack([left_action, right_action])
                obs, reward, reward_other, done, info = env.step(action)
            else:
                raise ValueError("No model provided")

            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            # print(info)
            # if done:
            #     video_log[-1].save(f"{args.video_dir}/episode_{i}.png")

            if done:
                if record:
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/episode_{suffix}_{i}.mp4", mode='w')
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
    parser.add_argument('--side', help='The side for AI to control. By default left', default='left', choices=['left', 'right', 'both'])
    parser.add_argument('--render', action='store_true', help='Whether to render the game screen')
    parser.add_argument('--num-env', type=int, help='How many envirorments to create', default=64)
    parser.add_argument('--num-episodes', type=int, help='In evaluation, play how many episodes', default=20)
    parser.add_argument('--num-epoch', type=int, help='Finetune how many epochs', default=50)
    parser.add_argument('--total-steps', type=int, help='How many total steps to train', default=int(1e7))
    parser.add_argument('--video-dir', help='The path to save videos', default='videos')
    # parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--transfer-model-dir', type=str, help='Transfer model dir', default=None)
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--num-stack', type=int, help='Number of frames to stack', default=12)
    parser.add_argument('--num-step-frames', type=int, help='Number of frames per step', default=8)
    parser.add_argument('--anneal-dense-coeff', action='store_true', help='Anneal dense_coeff')
    parser.add_argument('--anneal-agressive-coeff', action='store_true', help='Anneal agressive_coeff')
    
    args = parser.parse_args()
    print("command line args:" + str(args))
    if args.transfer_model_dir is not None:
        assert args.enable_combo, "Transfer learning requires combo"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    # os.makedirs(args.finetune_dir, exist_ok=True)
                                 
    # Set up the environment and model
    if 'starall' in args.state:
        states = [args.state.replace('starall', f'star{i}') for i in range(1, 9)]
        env = SubprocVecEnv([make_env(sf_game, state=state, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i) for i, state in enumerate(states)] * (args.num_env // 8))
    else:
        env = SubprocVecEnv([make_env(sf_game, state=args.state, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i) for i in range(args.num_env)])

    # Set linear schedule for learning rate
    # Start
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    # fine-tune
    # lr_schedule = linear_schedule(5.0e-5, 2.5e-6)

    # Set linear scheduler for clip range
    # Start
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # fine-tune
    # clip_range_schedule = linear_schedule(0.075, 0.025)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=1024, # 512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=args.log_dir
    )
    
    if (args.model_file):
        print("load model from " + args.model_file)
        model.set_parameters(args.model_file)

    # Set up callbacks
    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=args.model_name_prefix)
    callbacks = [checkpoint_callback]
    if args.anneal_dense_coeff:
        anneal_dense_callback = AnnealDenseCallback(anneal_fraction=0.1, anneal_initial_coeff=1.0, anneal_final_coeff=0.0)
        callbacks.append(anneal_dense_callback)
    if args.anneal_agressive_coeff:
        anneal_agressive_callback = AnnealAgressiveCallback(anneal_fraction=0.5, anneal_initial_coeff=3.0, anneal_final_coeff=1.0)
        callbacks.append(anneal_agressive_callback)

    model.learn(
        total_timesteps=args.total_steps, 
        callback=callbacks
    )
    env.close()

    # Save the final model
    model.save(os.path.join(args.save_dir, args.model_name_prefix + "_final_steps.zip"))

    # Evaluate the model
    evaluate(args, model)


if __name__ == "__main__":
    main()
