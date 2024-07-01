import os
import av
import sys
import torch
import random
import argparse
import numpy as np
from PIL import Image

import retro
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn

from common.const import *
from common.utils import linear_schedule
from common.game import get_next_level
from common.retro_wrappers import SFWrapper


with open(f'curriculum/curriculum_latest.txt', 'r') as f:
    CURRICULUM = eval(f.readline())
print(f"Init CURRICULUM: {CURRICULUM}")


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
def evaluate(args, model, init_level, target_level, greedy=0.99, record=True):
    reset_type = 'game'
    win_cnt = 0
    
    for i in range(1, args.num_episodes + 1):
        idx = random.choice(range(len(CURRICULUM[f'Level{init_level}'])))
        state = "curriculum/" + CURRICULUM[f'Level{init_level}'][idx]
        print(f"Select state {state}")
        env = make_env(sf_game, state=state, side=args.side, reset_type=reset_type, init_level=init_level, rendering=False, state_dir=SF_STATE_DIR, verbose=True, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i)().env

        done = False
        
        obs = env.reset()
        if record:
            video_log = [Image.fromarray(env.render(mode="rgb_array"))]

        while not done:
            if np.random.uniform() > greedy:
                action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)
            if record:
                video_log.append(Image.fromarray(env.render(mode="rgb_array")))
            if env.level >= target_level:
                done = True

            if record and env.level == target_level and env.save_state:
                env.save_state = False
                env.save_state_to_file(f"Level{env.level}.{i}.state")
                # CURRICULUM[f'Level{env.level}'].append(f"Level{env.level}.{i}")
            
            if done:
                print(f"Episode done at level {env.level}", flush=True)
                if env.level >= target_level:
                    win_cnt += 1
                if record and env.level >= target_level:
                    height, width, layers = np.array(video_log[0]).shape
                    container = av.open(f"{args.video_dir}/target_{target_level}_episode_{i}_done_{env.level}.mp4", mode='w')
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

        # if info['enemy_hp'] < 0:
        #     print("Victory!")
        #     num_victory += 1

        # print("Total reward: {}\n".format(total_reward))
        # episode_reward_sum += total_reward
    
        env.close()
    win_rate = win_cnt / args.num_episodes
    print("Winning rate: {}".format(win_rate))
    # with open('curriculum/curriculum_latest.txt', 'w') as f:
    #     f.write(str(CURRICULUM))
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
    parser.add_argument('--finetune-dir', help='The path to save finetune results', default='finetune')
    parser.add_argument('--init-level', type=int, help='Initial level to load from. By default 0, starting from pretrain', default=0)
    parser.add_argument('--resume-epoch', type=int, help='Resume epoch. By default 0, starting from pretrain', default=0)
    parser.add_argument('--enable-combo', action='store_true', help='Enable special move action space for environment')
    parser.add_argument('--transfer-model-dir', type=str, help='Transfer model dir', default=None)
    parser.add_argument('--null-combo', action='store_true', help='Null action space for special move')
    parser.add_argument('--transform-action', action='store_true', help='Transform action space to MultiDiscrete')
    parser.add_argument('--num-stack', type=int, help='Number of frames to stack', default=12)
    parser.add_argument('--num-step-frames', type=int, help='Number of frames per step', default=8)
    
    args = parser.parse_args()
    print("command line args:" + str(args))
    if args.transfer_model_dir is not None:
        assert args.enable_combo, "Transfer learning requires combo"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.finetune_dir, exist_ok=True)
                                 
    # Set up the environment and model
    def env_generator(max_level, curriculum_schedule):
        env = []
        select = []
        probs = list(curriculum_schedule.values())
        # Sample env levels according to curriculum schedule
        samples = np.random.choice(len(probs), size=args.num_env, p=probs)
        levels = [list(curriculum_schedule.keys())[i] for i in samples]
        print('Curricilum levels: ', levels)

        for i, level in enumerate(levels):
            idx = random.choice(range(len(CURRICULUM[f'Level{level}'])))
            state = "curriculum/" + CURRICULUM[f'Level{level}'][idx]
            select.append(state)
            env.append(make_env(sf_game, state=state, side=args.side, reset_type=args.reset, rendering=args.render, enable_combo=args.enable_combo, null_combo=args.null_combo, transform_action=args.transform_action, num_stack=args.num_stack, num_step_frames=args.num_step_frames, seed=i))
        print("worker env: ", select)

        return SubprocVecEnv(env)

    checkpoint_interval = 31250 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    plot_in_eval = True

    def finetune_model_generator(curriculum_schedule, max_level=15, model_file=None, lr_schedule=linear_schedule(5.0e-5, 2.5e-6), clip_range_schedule=linear_schedule(0.075, 0.025)):
    # def finetune_model_generator(curriculum_schedule, max_level=15, model_file=None, lr_schedule=1e-4, clip_range_schedule=0.1):
        finetune_env = env_generator(max_level, curriculum_schedule)
        finetune_model = PPO(
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
        )
        if model_file:
            print("load model from " + model_file)
            finetune_model.set_parameters(model_file, exact_match=False)
        return finetune_model

    def uniform_schedule():
        levels = CURRICULUM.keys()
        valid_levels = []
        for l in levels:
            level = int(l.split('Level')[1])
            if level not in SF_BONUS_LEVEL: # skip bonus level
                valid_levels.append(level)
        probs = np.ones(len(valid_levels)) / len(valid_levels)
        schedule = {level: prob for level, prob in zip(valid_levels, probs)}

        return schedule

    def rate_weight_schedule(rates):
        levels = CURRICULUM.keys()
        valid_levels = []
        for l in levels:
            level = int(l.split('Level')[1])
            if level not in SF_BONUS_LEVEL:
                valid_levels.append(level)
        schedule = {}
        rate_values = np.array(list(rates.values()))
        norm_reverse_rates = (1 - rate_values) / np.sum(1 - rate_values)  # norm to be prob mass
        for level, r in zip(valid_levels, norm_reverse_rates):
            schedule[level] = r
        # print("rate_weight_schedule: ", rates, schedule)
        return schedule

    curriculum_schedule = uniform_schedule()
    if args.resume_epoch > 0:
        finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_epoch{args.resume_epoch}_final_steps.zip")
        with open(f"{args.finetune_dir}/{args.model_name_prefix}_epoch{args.resume_epoch}_results.txt", 'r') as f:
            results = eval(f.readline())
        curriculum_schedule = rate_weight_schedule(results)
        start_epoch = args.resume_epoch + 1
    else:
        finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_final_steps")
        model = finetune_model_generator(curriculum_schedule, model_file=args.transfer_model_dir, lr_schedule=linear_schedule(2.5e-4, 2.5e-6), clip_range_schedule=linear_schedule(0.15, 0.025))
        checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=f"{args.model_name_prefix}")
        if args.transfer_model_dir is None:
            model.learn(
                total_timesteps=args.total_steps,
                callback=[checkpoint_callback]
            )
        model.save(finetune_epoch_model_path)
        # model = finetune_model_generator(curriculum_schedule, model_file=finetune_epoch_model_path)
        results = {}
        for level in range(1, 16):
            # gather win rates over all levels
            if level in SF_BONUS_LEVEL: # skip bonus level
                continue
            next_level = get_next_level(level)
            print(f"Evaluate on level {level}, next level {next_level}")
            results[level] = evaluate(args, model, level, next_level, record=False)
        print(results)
        curriculum_schedule = rate_weight_schedule(results)
        with open(f"{args.finetune_dir}/{args.model_name_prefix}_start_results.txt", 'w') as f:
            f.write(str(results))
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epoch):
        checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=args.save_dir, name_prefix=f"{args.model_name_prefix}_epoch{epoch}")
        model = finetune_model_generator(curriculum_schedule, model_file=finetune_epoch_model_path)

        # fine-tune the model
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback]
        )

        results = {}
        for level in range(1, 16):
            # gather win rates over all levels
            if level in SF_BONUS_LEVEL: # skip bonus level
                continue
            next_level = get_next_level(level)
            print(f"Evaluate on level {level}, next level {next_level}")
            results[level] = evaluate(args, model, level, next_level, record=False)
        print(results)
        curriculum_schedule = rate_weight_schedule(results)

        with open(f"{args.finetune_dir}/{args.model_name_prefix}_epoch{epoch}_results.txt", 'w') as f:
            f.write(str(results))

        # plot the results
        if plot_in_eval:
            with open(f"{args.finetune_dir}/{args.model_name_prefix}_epoch{epoch}_results.txt", 'r') as f:
                results = eval(f.readline())
            x_list = [k for k, v in results.items()]
            y_list = [v for k, v in results.items()]
            print('product: ', np.prod(y_list))
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.bar(x_list, y_list, color='maroon')
            plt.xlabel('level')
            plt.ylabel('sucess rate')
            plt.savefig(f"{args.finetune_dir}/{args.model_name_prefix}_epoch{epoch}_results.pdf")

        finetune_epoch_model_path = os.path.join(args.save_dir, args.model_name_prefix + f"_epoch{epoch}_final_steps.zip")
        model.save(finetune_epoch_model_path)
    

if __name__ == "__main__":
    main()