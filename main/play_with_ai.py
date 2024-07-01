import os
import time
import math
import torch
from common.interactive import RetroInteractive 

import retro
from stable_baselines3 import PPO
from common.algorithms import LeaguePPO
from common.retro_wrappers import SFWrapper
from common.const import *


STATE = "Champion.RyuVsRyu.2Player.align"
MODEL_PATH = "/your/model/path.pt"

class SFPlayWrapper(SFWrapper):
    def __init__(self, env, side, reset_type="round", init_level=1, rendering=False, num_stack=12, num_step_frames=8, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False):
        assert side == 'both', 'SFPlayWrapper only supports side=both'
        super().__init__(env, side, reset_type, init_level, rendering, num_stack, num_step_frames, state_dir, verbose, enable_combo, null_combo, transform_action)
        self.ai_action_seq = None
    
    def reset(self):
        self.ai_action_seq = None
        return super().reset()

    def step(self, action):
        ai_action = action[:-12]
        human_action = action[-12:] # no combo bits for human
        if self.action_transformer is not None:
            assert ai_action.shape[-1] == 1, f"ai_action.shape[-1]={ai_action.shape[-1]}, self.action_dim={self.action_dim}"
            ai_action = self.action_transformer(ai_action)
        assert ai_action.shape[-1] == self.action_dim, f"ai_action.shape[-1]={ai_action.shape[-1]}, self.action_dim={self.action_dim}"
        
        if self.level in SF_BONUS_LEVEL: # skip bonus level
            skip_level = self.level
            no_op = np.zeros_like(action[:24])
            while self.level == skip_level:
                obs, _reward, _done, info = self.env.step(no_op)
                self.update_status(info, bonus=True)
            if self.verbose:
                print(f"Skip bonus level {skip_level}")
        
        custom_done = False

        if self.total_timesteps % self.num_step_frames == 0:
            ai_action[3] = 0 # Filter out the "START/PAUSE" button  
            if self.enable_combo:
                combo_id = int(4 * ai_action[self.action_dim - 3] + 2 * ai_action[self.action_dim - 2] + ai_action[self.action_dim - 1])
            else:
                combo_id = len(SF_COMBOS)

            if combo_id >= len(SF_COMBOS):
                self.ai_action_seq = [ai_action[:12] for _ in range(self.num_step_frames)]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                self.ai_action_seq = combo
        
        joint_action = np.hstack([self.ai_action_seq[self.total_timesteps % self.num_step_frames], human_action])
        obs, _reward, _done, info = self.env.step(joint_action)
        self.update_status(info)
        if self.rendering:
            self.env.render()
            time.sleep(0.01)
        
        agent_hp = info['agent_hp']
        enemy_hp = info['enemy_hp']
        agent_victories = info['agent_victories']
        enemy_victories = info['enemy_victories']
        round_countdown = info['round_countdown']
        timesup = (round_countdown <= 0)

        self.total_timesteps += 1

        if self.during_transation and (self.match_status == END_STATUS or self.round_status == END_STATUS):
            # During transation between episodes, do nothing
            custom_done = False
            custom_reward = 0
            custom_reward_inverse = 0
            if (enemy_victories == 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                # Player loses the game
                custom_done = not self.reset_type == "never"
            if (agent_victories == 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                # Player wins the match
                custom_done = self.reset_type == "match"
                if self.level == 15:
                    print(f"Player wins the game")
                    custom_done = True
                    self.save_state = True
                    self.level += 1
        else:
            self.during_transation = False
            # if self.save_state and self.state_dir is not None:
            #     self.save_state = False
            #     self.save_state_to_file(f"Level{self.level}.{self.total_timesteps}.state")

            if (agent_hp < 0 and enemy_hp < 0) or (timesup and agent_hp == enemy_hp):
                custom_reward = 1
                custom_reward_inverse = 1
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transation = True
            elif agent_hp < 0 or (timesup and agent_hp < enemy_hp):
                custom_reward = -math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1))     
                custom_reward_inverse = math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (enemy_victories >= 2) or ((self.match_status == END_STATUS) and (enemy_victories >= agent_victories)): # also need to handle 2nd condition during transation
                        # Player loses the game
                        # if self.verbose:
                        #     print("Player loses the game")
                        custom_done = not self.reset_type == "never"
            elif enemy_hp < 0 or (timesup and agent_hp > enemy_hp):
                custom_reward = math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1)) * self.aggresive_coeff
                custom_reward_inverse = -math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))

                if (self.reset_type == "round"):
                    custom_done = True
                else:
                    self.during_transation = True
                    if (agent_victories >= 2) or ((self.match_status == END_STATUS) and (agent_victories > enemy_victories)): # also need to handle 2nd condition during transation
                        # Player wins the match
                        # if self.verbose:
                        #     print("Player wins the match")
                        custom_done = self.reset_type == "match"
                        if self.level == 15:
                            print(f"Player wins the game")
                            custom_done = True
                            self.save_state = True
                            self.level += 1
            # While the fighting is still going on
            else:
                custom_reward = self.aggresive_coeff * (self.prev_enemy_hp - enemy_hp) - (self.prev_agent_hp - agent_hp)
                custom_reward_inverse = self.aggresive_coeff * (self.prev_agent_hp - agent_hp) - (self.prev_enemy_hp - enemy_hp)
                self.prev_agent_hp = agent_hp
                self.prev_enemy_hp = enemy_hp
                custom_done = False

        # if custom_reward != 0:
        #     print("reward:{}".format(custom_reward))

        info['level'] = self.level
        info['match'] = 'start' if self.match_status == START_STATUS else 'end'
        info['round'] = 'start' if self.round_status == START_STATUS else 'end'
        if custom_done:
            info['outcome'] = 'win' if (agent_hp > enemy_hp) else ('lose' if (agent_hp < enemy_hp) else 'draw')

        if self.side == 'left':
            return self._get_obs(obs), 0.001 * custom_reward, custom_done, info 
        elif self.side == 'right':
            return self._get_obs(obs), 0.001 * custom_reward_inverse, custom_done, info 
        else:
            return self._get_obs(obs), 0.001 * custom_reward, 0.001 * custom_reward_inverse, custom_done, info 


class LeaguePPOPlay(LeaguePPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        return self.policy.predict(observation, state, episode_start, deterministic) if self.side == 'left' else self.policy_other.predict(observation, state, episode_start, deterministic)


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
        env = SFPlayWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action)
        return env
    return _init


env = make_env(sf_game, state=STATE, side="both", reset_type="round", rendering=False, enable_combo=True, null_combo=True, transform_action=True, seed=None)()
# model = LeaguePPOPlay(
#         "left",
#         "CnnPolicy", 
#         env,
#         device="cuda", 
#         verbose=1,
#         n_steps=512,
#         batch_size=1024, # 512,
#         n_epochs=4,
#         gamma=0.94,
#         learning_rate=1e-4, # lr_schedule,
#         clip_range=0.1, # clip_range_schedule,
#         tensorboard_log=None,
#         # seed=args.seed,
#         other_learning_rate=1e-4, # other_lr_schedule,
#     )
# load_kwargs = torch.load(MODEL_PATH, map_location=torch.device('cpu'))["kwargs"]
# load_weights = load_kwargs["agent_dict"]
# model.set_parameters(load_weights)
model = PPO.load("/your/model/path.zip", env=env)
ia = RetroInteractive(env, model)

ia.run()
