# FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning

## Setup

Platform: Linux

Python: 3.8

Create environment:

```bash
conda env create -f environment.yml
```

Find out the gym-retro game folder:

```python
import os
import retro

retro_directory = os.path.dirname(retro.__file__)
game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
print(os.path.join(retro_directory, game_dir))
```

Add state files in `data/sf` and ROM file into the game folder.

Disclaimer: We are unable to provide you with any game ROMs. It is the users own legal responsibility to acquire a game ROM for emulation. This library should only be used for non-commercial research purposes.

## Key concepts

**Environment** is specified in `main/common/retro_wrappers.py`. It tracks the inner states of the game, and is compatible with Gym interface and popular RL packages such as stable-baselines.

**Algorithms** is implemented in `main/common/algorithms.py` and `main/common/league.py`. Specifically, `IPPO` in `algorithms.py` implements IPPO and 2Timescale methods, and League, PSRO, and FSP is implemented in `league.py`. We use PPO in stable-baselines as the backbone algorithm for all these implementations. The League implementation adapts the pseudocode in `main/common/pseudocode`, which is from previous work AlphaStar.

## Run the experiment

RL against built-in CPU player:

```bash
python train.py --reset=round \
--state=stars/Champion.Level1.RyuVsRyu.${side}_star${state} \ # difficulty level
--side=${side} \ # left/right
--model-name-prefix=ppo_ryu_${side}_star${state} \
--save-dir=trained_models/ppo_ryu_${side}_star${state} \
--log-dir=logs/ppo_ryu_${side}_star${state} \
--video-dir=videos/ppo_ryu_${side}_star${state} \
--num-epoch=50 \
--enable-combo --null-combo --transform-action
```

RL with curriculum learning:

```bash
python finetune.py --reset=round \
--model-name-prefix=ppo_ryu_finetune \
--save-dir=trained_models/ppo_ryu_finetune \
--log-dir=logs/ppo_ryu_finetune \
--video-dir=videos/ppo_ryu_finetune \
--finetune-dir=finetune/ppo_ryu_finetune \
--num-epoch=25
```

IPPO / 2Timescale:

```bash
python ippo.py --reset=${task} \
--model-name-prefix=ippo_ryu_2p_scale_${scale}_${seed} \
--save-dir=trained_models/ippo_ryu_2p_scale_${scale}_${seed} \
--log-dir=logs/ippo_ryu_2p_scale_${scale}_${seed} \
--video-dir=videos/ippo_ryu_2p_scale_${scale}_${seed} \
--finetune-dir=finetune/ippo_ryu_2p_scale_${scale}_${seed} \
--num-epoch=50 \
--enable-combo --null-combo --transform-action \
--other-timescale=${scale} \ # scale=1 equivalent to IPPO
--seed=${seed} \
```

League / PSRO / FSP:

```python
python train_ma.py --reset=round \
--save-dir=trained_models/ma \
--log-dir=logs/ma \
--left-model-file=trained_models/ppo_ryu_left_star8/ppo_ryu_left_star8_final_steps \
--right-model-file=trained_models/ppo_ryu_right_star8/ppo_ryu_right_star8_final_steps \
--enable-combo --null-combo --transform-action \
--seed=${seed}
# --psro-league for PSRO, --fsp-league for FSP
```

Single-Agent RL Exploiters:

```bash
python best_response.py --reset=round \
--model-name-prefix=br_${model}/seed_${seed} \
--save-dir=trained_models/ma_br/${model}/seed_${seed} \
--log-dir=logs/ma_br/${model}/seed_${seed} \
--video-dir=videos/ma_br/${model}/seed_${seed} \
--finetune-dir=finetune/ma_br/${model}/seed_${seed} \
--model-file=/path/to/model \ # --model-file is for 2P policies, also support load left and right 1P policies seperately, by --left-model-file and --right-model-file
--num-epoch=50 \
--enable-combo --null-combo --transform-action \
--update-right=0 \ # exploit the right policy, then do not update it 
--seed=${seed}
```

Play with trained policies:

```bash
python play_with_ai.py # change the model path in play_with_ai.py, the key mapping is in common/interactive.py
```

Stay tuned for supports on more fighting games! You could also integrate your own games via implementing a wrapper environment similar in `main/common/retro_wrappers.py`.

## Citation

If you find our repo useful, please consider cite our work:

```
@inproceedings{lifightladder,
  title={FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning},
  author={Li, Wenzhe and Ding, Zihan and Karten, Seth and Jin, Chi},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
