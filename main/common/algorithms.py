import sys
import time
import random
import torch as th
import numpy as np
import torch.nn as nn
from gym import spaces
from copy import deepcopy
from collections import deque
from torch.nn import functional as F
from typing import Any, Dict, Mapping, Optional, Tuple, Union, Type, List, TypeVar

from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance, get_schedule_fn, update_learning_rate, is_vectorized_observation
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.vec_env import VecEnv

from .const import *
from .nash import compute_nash


SelfIPPO = TypeVar("SelfIPPO", bound="IPPO")
SelfLeaguePPO = TypeVar("SelfLeaguePPO", bound="LeaguePPO")


class IPPO(PPO):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        update_left = True,
        update_right = True,
        other_learning_rate = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.update_left = update_left
        self.update_right = update_right
        self.other_learning_rate = other_learning_rate

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        
        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer_other = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.other_lr_schedule = self.lr_schedule if self.other_learning_rate is None else get_schedule_fn(self.other_learning_rate)
        self.policy_other = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.other_lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy_other = self.policy_other.to(self.device)
    
    def _update_other_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        self.logger.record("train/other_learning_rate", self.other_lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.other_lr_schedule(self._current_progress_remaining))
    
    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "policy_other",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "rollout_buffer_other",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer", "policy_other", "policy_other.optimizer"]

        return state_dicts, []

    def set_parameters_2p(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        load_path_or_dict_other: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)
        params_other = None
        if isinstance(load_path_or_dict_other, dict):
            params_other = load_path_or_dict_other
        else:
            _, params_other, _ = load_from_zip_file(load_path_or_dict_other, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)
        
        for name in params_other:
            attr = None
            name_other = name.replace("policy", "policy_other")
            try:
                attr = recursive_getattr(self, name_other)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name_other} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params_other[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params_other[name], strict=exact_match)
            updated_objects.add(name_other)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )
    
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, deterministic), self.policy_other.predict(observation, state, episode_start, deterministic)
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_other: RolloutBuffer,
        n_rollout_steps: int,
        policy = None,
        policy_other = None,
        coordinate_fn = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        rollout_policy = self.policy if policy is None else policy
        rollout_policy_other = self.policy_other if policy_other is None else policy_other
        rollout_policy.set_training_mode(False)
        rollout_policy_other.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            rollout_policy.reset_noise(env.num_envs)
            rollout_policy_other.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                rollout_policy.reset_noise(env.num_envs)
                rollout_policy_other.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = rollout_policy(obs_tensor)
                actions_other, values_other, log_probs_other = rollout_policy_other(obs_tensor)
            actions = actions.cpu().numpy()
            actions_other = actions_other.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low, self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and coordinate_fn is not None
                ):
                    coordinate_fn(infos[idx]["outcome"])
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
                    terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = rollout_policy.predict_values(terminal_obs)[0]
                        terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
                    rewards[idx] += self.gamma * terminal_value
                    rewards_other[idx] += self.gamma * terminal_value_other                        

            # from IPython import embed; embed()
            if self.update_left:
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values, log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other, log_probs_other)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = rollout_policy.predict_values(obs_as_tensor(new_obs, self.device))
            values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        if self.update_left:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.update_right:
            rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        self.policy_other.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self._update_other_learning_rate(self.policy_other.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        policies = [self.policy, self.policy_other]
        rollout_buffers = [self.rollout_buffer, self.rollout_buffer_other]
        suffixes = ["", "_other"]
        update_flags = [self.update_left, self.update_right]
        # policies = [self.policy_other, self.policy]
        # rollout_buffers = [self.rollout_buffer_other, self.rollout_buffer]
        # suffixes = ["_other", ""]
        # update_flags = [self.update_right, self.update_left]

        for policy, rollout_buffer, suffix, update_flag in zip(policies, rollout_buffers, suffixes, update_flags):
            if not update_flag:
                continue
            
            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True

            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Optimization step
                    policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                    policy.optimizer.step()

                if not continue_training:
                    break

            self._n_updates += self.n_epochs
            explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

            # Logs
            self.logger.record(f"train/entropy_loss{suffix}", np.mean(entropy_losses))
            self.logger.record(f"train/policy_gradient_loss{suffix}", np.mean(pg_losses))
            self.logger.record(f"train/value_loss{suffix}", np.mean(value_losses))
            self.logger.record(f"train/approx_kl{suffix}", np.mean(approx_kl_divs))
            self.logger.record(f"train/clip_fraction{suffix}", np.mean(clip_fractions))
            self.logger.record(f"train/loss{suffix}", loss.item())
            self.logger.record(f"train/explained_variance{suffix}", explained_var)
            if hasattr(policy, "log_std"):
                self.logger.record(f"train/std{suffix}", th.exp(policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfIPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfIPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
    
    def async_learn(
        self: SelfIPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        fsp: bool = False, # NOTE: this method implements an approximate version of FSP, the full version is implemented in league.py
        max_fsp_num: int = 50,
        fsp_threshold: float = 0.3,
    ) -> SelfIPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps * 10, # Async learning is much slower
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        if fsp:
            left_state_dicts = [deepcopy(self.policy.state_dict())]
            right_state_dicts = [deepcopy(self.policy_other.state_dict())]
            tmp_left_policy = deepcopy(self.policy)
            tmp_right_policy = deepcopy(self.policy_other)

        while self.num_timesteps < total_timesteps:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            self.update_left = True
            self.update_right = False
            rew_diff = 0
            while (rew_diff < fsp_threshold) and (self.num_timesteps < total_timesteps):
                rew_diff = 0
                for _ in range(10):
                    if fsp:
                        tmp_right_policy.load_state_dict(random.choice(right_state_dicts))
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy_other=tmp_right_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]) - safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer])
                    self.train()
                rew_diff = rew_diff / 10
                if continue_training is False:
                    break
            print("[Left] rew_diff: ", rew_diff, flush=True)
            if continue_training is False:
                break
            if fsp:
                left_state_dicts.append(deepcopy(self.policy.state_dict()))
                if len(left_state_dicts) > max_fsp_num:
                    left_state_dicts.pop(random.randrange(len(left_state_dicts)))

            self.update_left = False
            self.update_right = True
            rew_diff = 0
            while (rew_diff < fsp_threshold) and (self.num_timesteps < total_timesteps):
                rew_diff = 0
                for _ in range(10):
                    if fsp:
                        tmp_left_policy.load_state_dict(random.choice(left_state_dicts))
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy=tmp_left_policy)
                    else:
                        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps)
                    if continue_training is False:
                        break
                    iteration += 1
                    # Display training infos
                    if log_interval is not None and iteration % log_interval == 0:
                        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                        self.logger.record("time/iterations", iteration, exclude="tensorboard")
                        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                        self.logger.record("time/fps", fps)
                        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                        self.logger.dump(step=self.num_timesteps)
                    rew_diff = rew_diff + safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]) - safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    self.train()
                rew_diff = rew_diff / 10
                if continue_training is False:
                    break
            print("[Right] rew_diff: ", rew_diff, flush=True)
            if continue_training is False:
                break
            if fsp:
                right_state_dicts.append(deepcopy(self.policy_other.state_dict()))
                if len(right_state_dicts) > max_fsp_num:
                    right_state_dicts.pop(random.randrange(len(right_state_dicts)))

        callback.on_training_end()

        return self


class BRIPPO(IPPO):

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_other: RolloutBuffer,
        n_rollout_steps: int,
        policy = None,
        policy_other = None,
        # coordinate_fn = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        rollout_policy = self.policy if policy is None else policy
        rollout_policy_other = self.policy_other if policy_other is None else policy_other
        rollout_policy.set_training_mode(False)
        rollout_policy_other.set_training_mode(False)

        round_results = {'win': 0, 'lose': 0, 'draw': 0}
        round_start_steps = self.num_timesteps

        n_steps = 0
        rollout_buffer.reset()
        rollout_buffer_other.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            rollout_policy.reset_noise(env.num_envs)
            rollout_policy_other.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                rollout_policy.reset_noise(env.num_envs)
                rollout_policy_other.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = rollout_policy(obs_tensor)
                actions_other, values_other, log_probs_other = rollout_policy_other(obs_tensor)
            actions = actions.cpu().numpy()
            actions_other = actions_other.cpu().numpy()

            # Rescale and perform action
            clipped_actions = np.hstack([actions, actions_other])
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(np.hstack([actions, actions_other]), self.action_space.low, self.action_space.high)

            new_obs, rewards, rewards_other, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
                actions_other = actions_other.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    # and coordinate_fn is not None
                ):
                    round_results[infos[idx]["outcome"]] += 1
                    # coordinate_fn(infos[idx]["outcome"])
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # print(f"[PPO] idx: {idx}, done: {done}, outcome: {infos[idx]['outcome']}", flush=True)
                    terminal_obs = rollout_policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs_other = rollout_policy_other.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = rollout_policy.predict_values(terminal_obs)[0]
                        terminal_value_other = rollout_policy_other.predict_values(terminal_obs_other)[0]
                    rewards[idx] += self.gamma * terminal_value
                    rewards_other[idx] += self.gamma * terminal_value_other                        

            # from IPython import embed; embed()
            if self.update_left:
                rollout_buffer.add(self._last_obs.copy(), actions, rewards, self._last_episode_starts, values, log_probs)
            if self.update_right:
                rollout_buffer_other.add(self._last_obs.copy(), actions_other, rewards_other, self._last_episode_starts, values_other, log_probs_other)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = rollout_policy.predict_values(obs_as_tensor(new_obs, self.device))
            values_other = rollout_policy_other.predict_values(obs_as_tensor(new_obs, self.device))

        if self.update_left:
            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        if self.update_right:
            rollout_buffer_other.compute_returns_and_advantage(last_values=values_other, dones=dones)

        callback.on_rollout_end()

        round_end_steps = self.num_timesteps
        round_results['start_steps'] = round_start_steps
        round_results['end_steps'] = round_end_steps
        with open(os.path.join(self.tensorboard_log, "round_results.txt"), "a") as f:
            f.write(str(round_results) + "\n")

        return True


class LeaguePPO(IPPO):

    def __init__(
        self,
        side,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        other_learning_rate = None,
    ):
        if side == "left":
            update_left = True
            update_right = False
        elif side == "right":
            update_left = False
            update_right = True
        else:
            raise ValueError("side should be 'left' or 'right'")
        self.side = side

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            update_left=update_left,
            update_right=update_right,
            other_learning_rate=other_learning_rate,
        )
    
    def train(self, rollout_buffer: RolloutBuffer) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        policy = self.policy if self.side == "left" else self.policy_other
        suffix = "" if self.side == "left" else "_other"
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    policy.reset_noise(self.batch_size)

                values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        # Logs
        self.logger.record(f"train/entropy_loss{suffix}", np.mean(entropy_losses))
        self.logger.record(f"train/policy_gradient_loss{suffix}", np.mean(pg_losses))
        self.logger.record(f"train/value_loss{suffix}", np.mean(value_losses))
        self.logger.record(f"train/approx_kl{suffix}", np.mean(approx_kl_divs))
        self.logger.record(f"train/clip_fraction{suffix}", np.mean(clip_fractions))
        self.logger.record(f"train/loss{suffix}", loss.item())
        self.logger.record(f"train/explained_variance{suffix}", explained_var)
        if hasattr(policy, "log_std"):
            self.logger.record(f"train/std{suffix}", th.exp(policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
    
    def learn(
        self: SelfLeaguePPO,
        total_timesteps: int,
        rollout_opponent_num: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "IPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        get_kwargs_fn = None,
    ) -> SelfLeaguePPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer
        all_rollouts = buffer_cls(
            self.n_steps * rollout_opponent_num,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        while self.num_timesteps < total_timesteps:
            
            all_rollouts.reset()

            for i in range(rollout_opponent_num):
                kwargs = get_kwargs_fn()

                # NOTE: reset env before each rollout to avoid cross-episodic interference among different opponents
                self._last_obs = self.env.reset()
                self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
                if self._vec_normalize_env is not None:
                    self._last_original_obs = self._vec_normalize_env.get_original_obs()

                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, self.rollout_buffer_other, n_rollout_steps=self.n_steps, policy=kwargs.get("policy"), policy_other=kwargs.get("policy_other"), coordinate_fn=kwargs.get("coordinate_fn"))
                if continue_training is False:
                    break

                collected_rollouts = self.rollout_buffer if self.side == "left" else self.rollout_buffer_other
                assert collected_rollouts.full, "rollout buffer should be full"
                curr_pos = all_rollouts.pos
                next_pos = all_rollouts.pos + collected_rollouts.size()
                all_rollouts.observations[curr_pos:next_pos] = collected_rollouts.observations[:]
                all_rollouts.actions[curr_pos:next_pos] = collected_rollouts.actions[:]
                all_rollouts.rewards[curr_pos:next_pos] = collected_rollouts.rewards[:]
                all_rollouts.returns[curr_pos:next_pos] = collected_rollouts.returns[:]
                all_rollouts.episode_starts[curr_pos:next_pos] = collected_rollouts.episode_starts[:]
                all_rollouts.values[curr_pos:next_pos] = collected_rollouts.values[:]
                all_rollouts.log_probs[curr_pos:next_pos] = collected_rollouts.log_probs[:]
                all_rollouts.advantages[curr_pos:next_pos] = collected_rollouts.advantages[:]
                all_rollouts.pos = next_pos
                if all_rollouts.pos == all_rollouts.buffer_size:
                    all_rollouts.full = True

            if continue_training is False:
                break    

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_other_mean", safe_mean([ep_info["ro"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train(all_rollouts)
            kwargs["sync_fn"]()

        callback.on_training_end()

        return self

    def get_steps(self) -> int:
        return self.num_timesteps
    
    def set_steps(self, steps: int) -> None:
        self.num_timesteps = steps
    
    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        self.policy.to("cpu")
        self.policy_other.to("cpu")
        params = super().get_parameters()
        self.policy.to(self.device)
        self.policy_other.to(self.device)
        return params
