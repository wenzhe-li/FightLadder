import torch
import collections
import numpy as np
from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing.managers import BaseManager

from stable_baselines3.common.type_aliases import MaybeCallback

from .const import *
from .algorithms import LeaguePPO
from .nash import NashEquilibriumECOSSolver


PER_HISTORICAL_STEPS = 1e7 # 5e3


def remove_monotonic_suffix(win_rates, players):
    # if not win_rates:
    #     return win_rates, players
    assert win_rates.shape == (len(players),), "win_rates must be a 1D array"

    for i in range(len(win_rates) - 1, 0, -1):
        if win_rates[i - 1] < win_rates[i]:
            return win_rates[:i + 1], players[:i + 1]

    return np.array([]), []


def pfsp(win_rates, weighting="linear"):
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x)**2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


class Payoff:

    def __init__(self, save_dir="trained_models/ma"):
        self._players = {}
        self._players_other = {}
        self._wins = collections.defaultdict(lambda: 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._decay = 0.99
        self.save_dir = save_dir

    def _win_rate(self, _home, _away, side):
        if self._games[_home, _away] == 0:
            return 0.5

        if side == "left":
            return (self._wins[_home, _away] +
                    0.5 * self._draws[_home, _away]) / self._games[_home, _away]
        elif side == "right":
            return (self._losses[_home, _away] +
                    0.5 * self._draws[_home, _away]) / self._games[_home, _away]
        else:
            raise ValueError("side must be either 'left' or 'right'")

    def get_item(self, home, away, side):
        if not isinstance(home, list):
            home = [home]
        if not isinstance(away, list):
            away = [away]

        if side == "left":
            win_rates = np.array([[self._win_rate(h, a, side) for a in away] for h in home])
        elif side == "right":
            win_rates = np.array([[self._win_rate(h, a, side) for h in home] for a in away])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        # print(f"home: {home} away: {away}")
        # print(f"win_rates: {win_rates}")

        return win_rates

    def update(self, home, away, result):
        for stats in (self._games, self._wins, self._draws, self._losses):
            stats[home, away] *= self._decay

        self._games[home, away] += 1
        if result == "win":
            self._wins[home, away] += 1
        elif result == "draw":
            self._draws[home, away] += 1
        else:
            self._losses[home, away] += 1

    def add_player(self, cls_name, kwargs):
        save_kwargs = {
            "cls_name": cls_name,
            "kwargs": kwargs,
        }
        print(f"[Payoff] Saving model {kwargs['name']} at step {kwargs['checkpoint_step']}", flush=True)
        torch.save(save_kwargs, f"{self.save_dir}/{kwargs['name']}_{kwargs['checkpoint_step']}.pt")
        save_payoff = {
            "wins": dict(self._wins),
            "draws": dict(self._draws),
            "losses": dict(self._losses),
            "games": dict(self._games),
        }
        print("[Payoff] Saving payoff", flush=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M")
        torch.save(save_payoff, f"{self.save_dir}/payoff_{timestamp}.pt")

        player = construct_player(cls_name, kwargs)
        # NOTE: player.agent is not constructed here
        # print(">" * 10 + "Payoff" + ">" * 10)
        # print("wins", dict(self._wins))
        # print("draws", dict(self._draws))
        # print("losses", dict(self._losses))
        # print("games", dict(self._games))
        # print("<" * 10 + "Payoff" + "<" * 10)
        if player.side == "left":
            self._players.update({player.name: player})
        elif player.side == "right":
            self._players_other.update({player.name: player})
        else:
            raise ValueError("side must be either 'left' or 'right'")

    def players(self, side):
        if side == "left":
            return self._players
        elif side == "right":
            return self._players_other
        else:
            raise ValueError("side must be either 'left' or 'right'")
    
    def get_names(self, side, filter_class=None, filter_parent=None):
        if side == "left":
            if filter_parent is None:
                names = [name for name, player in self._players.items() if isinstance(player, filter_class)] if filter_class is not None else list(self._players.keys())
            else:
                assert filter_class is not None, "filter_class must be specified if filter_parent is specified"
                if isinstance(filter_parent, str):
                    filter_parent = [filter_parent]
                names = [name for name, player in self._players.items() if isinstance(player, filter_class) and player.parent in filter_parent]
        elif side == "right":
            if filter_parent is None:
                names = [name for name, player in self._players_other.items() if isinstance(player, filter_class)] if filter_class is not None else list(self._players_other.keys())
            else:
                assert filter_class is not None, "filter_class must be specified if filter_parent is specified"
                if isinstance(filter_parent, str):
                    filter_parent = [filter_parent]
                names = [name for name, player in self._players_other.items() if isinstance(player, filter_class) and player.parent in filter_parent]
        else:
            raise ValueError("side must be either 'left' or 'right'")
        
        # print(f"names: {names}")

        return names
    
    def get_player_by_name(self, name):
        # print(f"get_player_by_name: {name}", flush=True)
        if name in self._players:
            player = self._players[name]
        elif name in self._players_other:
            player = self._players_other[name]
        else:
            raise ValueError("Player not found")
        # print(f"player: {player.name}, target: {name}", flush=True)
        cls_name, kwargs = get_player_config(player)
        return cls_name, kwargs

    def get_historical_nash(self, side):
        historical_players = self.get_names("left", Historical)
        historical_players_other = self.get_names("right", Historical)
        win_rates_normalized = self.get_item(historical_players, historical_players_other, "left") - 0.5
        win_rates_normalized = win_rates_normalized.reshape(len(historical_players), len(historical_players_other))
        # print(f"[Payoff] players: {historical_players}")
        # print(f"[Payoff] players_other: {historical_players_other}")
        # print(f"[Payoff] win_rates_normalized: {win_rates_normalized}", flush=True)
        ne, ne_v = NashEquilibriumECOSSolver(win_rates_normalized)
        # print(f"[Payoff] ne: {ne}", flush=True)
        if side == "left":
            return historical_players, ne[0]
        elif side == "right":
            return historical_players_other, ne[1]
        else:
            raise ValueError("side must be either 'left' or 'right'")

    def load(self, path):
        models_dir = "/".join(path.split("/")[:-1])
        assert "LE0_left" in self._players and "LE1_left" in self._players and "LE0_right" in self._players_other and "LE1_right" in self._players_other, "League must be initialized with 2 LeagueExploiters on each side"
        assert "ME0_left" in self._players and "ME0_right" in self._players_other, "League must be initialized with 1 MainExploiter on each side"
        assert "MA0_left" in self._players and "MA0_right" in self._players_other, "League must be initialized with 1 MainPlayer on each side"
        load_payoff = torch.load(path, map_location=torch.device('cpu'))
        self._wins.update(load_payoff["wins"])
        self._draws.update(load_payoff["draws"])
        self._losses.update(load_payoff["losses"])
        self._games.update(load_payoff["games"])
        print(f"[Payoff] Loading payoff from {path}", flush=True)
        for name in set([name for name, _ in self._games.keys()]):
            if "historical" in name:
                load_kwargs = torch.load(f"{models_dir}/{name}_0.pt", map_location=torch.device('cpu'))
                cls_name, kwargs = load_kwargs["cls_name"], load_kwargs["kwargs"]
                player = construct_player(cls_name, kwargs)
                self._players.update({player.name: player})
        print(f"[Payoff] Loading payoff._players: {list(self._players.keys())}", flush=True)
        for name in set([name for _, name in self._games.keys()]):
            if "historical" in name:
                load_kwargs = torch.load(f"{models_dir}/{name}_0.pt", map_location=torch.device('cpu'))
                cls_name, kwargs = load_kwargs["cls_name"], load_kwargs["kwargs"]
                player = construct_player(cls_name, kwargs)
                self._players_other.update({player.name: player})
        print(f"[Payoff] Loading payoff._players_other: {list(self._players_other.keys())}", flush=True)


class PayoffManager(BaseManager):
    pass


PayoffManager.register('Payoff', Payoff)


def get_player_config(player):
    if isinstance(player, MainPlayer):
        cls_name = "MainPlayer"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    elif isinstance(player, FSPPlayer):
        cls_name = "FSPPlayer"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    elif isinstance(player, PSROPlayer):
        cls_name = "PSROPlayer"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    elif isinstance(player, MainExploiter):
        cls_name = "MainExploiter"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    elif isinstance(player, LeagueExploiter):
        cls_name = "LeagueExploiter"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    elif isinstance(player, Historical):
        cls_name = "Historical"
        kwargs = {
            "name": player.name,
            "side": player.side,
            "constructor": player.constructor,
            "args": player.args,
            "agent": None,
            "payoff": None,
            "parent_name": player.parent,
            "agent_dict": deepcopy(player.get_parameters()),
            "checkpoint_step": player._checkpoint_step,
        }
    else:
        raise ValueError("player must be either MainPlayer, MainExploiter, LeagueExploiter, or Historical")
    return cls_name, kwargs


def construct_player(cls_name, kwargs, construct_agent=False):
    if cls_name == "MainPlayer":
        player = MainPlayer(**kwargs)
    elif cls_name == "FSPPlayer":
        player = FSPPlayer(**kwargs)
    elif cls_name == "PSROPlayer":
        player = PSROPlayer(**kwargs)
    elif cls_name == "MainExploiter":
        player = MainExploiter(**kwargs)
    elif cls_name == "LeagueExploiter":
        player = LeagueExploiter(**kwargs)
    elif cls_name == "Historical":
        player = Historical(**kwargs)
    else:
        raise ValueError("cls_name must be either MainPlayer, MainExploiter, LeagueExploiter, or Historical")
    if construct_agent:
        player.construct_agent()
    return player


class Player(object):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        self.name = name
        self.side = side
        self.constructor = constructor
        self.args = args
        self.agent = None
        if agent_dict is not None:
            self._initial_weights = agent_dict
        else:
            self._initial_weights = agent.get_parameters()
        self._payoff = payoff
        self._checkpoint_step = checkpoint_step
    
    def construct_agent(self):
        self.agent = self.constructor(self.args, self.side, log_name=self.name)
        self.agent.set_parameters(self._initial_weights)
        self.agent.set_steps(self._checkpoint_step)
    
    def get_parameters(self):
        return self._initial_weights if self.agent is None else self.agent.get_parameters()

    def get_match(self) -> 'Player':
        pass

    def ready_to_checkpoint(self):
        return False

    def _create_checkpoint(self):
        return Historical(f"{self.name}_historical_step_{self._checkpoint_step}", self.side, self.constructor, self.args, None, None, self.name, agent_dict=self.get_parameters(), checkpoint_step=0)

    @property
    def payoff(self):
        return self._payoff

    def checkpoint(self):
        raise NotImplementedError
    
    def add_player(self, player):
        cls_name, kwargs = get_player_config(player)
        print(f"[Player] Adding player {kwargs['name']}")
        self._payoff.add_player(cls_name, kwargs)

    def get_player_by_name(self, name):
        cls_name, kwargs = self._payoff.get_player_by_name(name)
        return construct_player(cls_name, kwargs, construct_agent=True)

    def send_outcome(self, opponent_name, outcome):
        # print(f"[Player] send_outcome: {self.name} vs {opponent_name} = {outcome}")
        if self.side == "left":
            self._payoff.update(self.name, opponent_name, outcome)
        elif self.side == "right":
            self._payoff.update(opponent_name, self.name, outcome)
        else:
            raise ValueError("side must be either 'left' or 'right'")
        if self.ready_to_checkpoint():
            self.add_player(self.checkpoint())
    
    def sync(self):
        self.add_player(self)
    
    def load(self, path):
        load_kwargs = torch.load(path, map_location=torch.device('cpu'))["kwargs"]
        self._initial_weights_restore = deepcopy(self._initial_weights)
        self._initial_weights = load_kwargs["agent_dict"]
        self._checkpoint_step = load_kwargs["checkpoint_step"]
        print(f"[Player] Loading player {load_kwargs['name']} at step {load_kwargs['checkpoint_step']}", flush=True)
        self.sync()


class MainPlayer(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)

    def _pfsp_branch(self):
        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return self.get_player_by_name(np.random.choice(
            historical_opponents, p=pfsp(win_rates, weighting="squared"))), True

    def _selfplay_branch(self, opponent):
        # Play self-play match
        win_rate = self._payoff.get_item(self.name, opponent, "left") if self.side == "left" else self._payoff.get_item(opponent, self.name, "right")
        if win_rate > 0.3:
            return self.get_player_by_name(opponent), False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical_opponents = self._payoff.get_names("right", Historical, opponent) if self.side == "left" else self._payoff.get_names("left", Historical, opponent)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return self.get_player_by_name(np.random.choice(
            historical_opponents, p=pfsp(win_rates, weighting="variance"))), True

    def _verification_branch(self, opponent):
        # Check exploitation
        # opponents = self._payoff.players("right") if self.side == "left" else self._payoff.players("left")
        # exploiters = set([
        #     player for player in opponents
        #     if isinstance(player, MainExploiter)
        # ])
        exploiters = self._payoff.get_names("right", MainExploiter) if self.side == "left" else self._payoff.get_names("left", MainExploiter)
        # exp_historical = [
        #     player for player in opponents
        #     if isinstance(player, Historical) and player.parent in exploiters
        # ]
        exp_historical = self._payoff.get_names("right", Historical, exploiters) if self.side == "left" else self._payoff.get_names("left", Historical, exploiters)
        win_rates = self._payoff.get_item(self.name, exp_historical, "left") if self.side == "left" else self._payoff.get_item(exp_historical, self.name, "right")
        if len(win_rates) and win_rates.min() < 0.3:
            return self.get_player_by_name(np.random.choice(
                exp_historical, p=pfsp(win_rates, weighting="squared"))), True

        # Check forgetting
        historical_opponents = self._payoff.get_names("right", Historical, opponent) if self.side == "left" else self._payoff.get_names("left", Historical, opponent)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        win_rates, historical_opponents = remove_monotonic_suffix(win_rates, historical_opponents)
        if len(win_rates) and win_rates.min() < 0.7:
            return self.get_player_by_name(np.random.choice(
                historical_opponents, p=pfsp(win_rates, weighting="squared"))), True

        return None

    def get_match(self):
        coin_toss = np.random.random()

        # Make sure you can beat the League
        if coin_toss < 0.5:
            return self._pfsp_branch()

        main_opponents = self._payoff.get_names("right", MainPlayer) if self.side == "left" else self._payoff.get_names("left", MainPlayer)
        opponent = np.random.choice(main_opponents)

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < PER_HISTORICAL_STEPS:
            return False

        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return win_rates.min() > 0.7 or steps_passed > PER_HISTORICAL_STEPS * 2

    def checkpoint(self):
        if self.agent is not None:
            self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class MainExploiter(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)

    def get_match(self):
        coin_toss = np.random.random()

        main_opponents = self._payoff.get_names("right", MainPlayer) if self.side == "left" else self._payoff.get_names("left", MainPlayer)
        opponent = np.random.choice(main_opponents)

        win_rate = self._payoff.get_item(self.name, opponent, "left") if self.side == "left" else self._payoff.get_item(opponent, self.name, "right")
        if coin_toss < 0.5 or win_rate > 0.1:
            return self.get_player_by_name(opponent), True

        historical_opponents = self._payoff.get_names("right", Historical, opponent) if self.side == "left" else self._payoff.get_names("left", Historical, opponent)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")

        return self.get_player_by_name(np.random.choice(
            historical_opponents, p=pfsp(win_rates, weighting="variance"))), True

    def checkpoint(self):
        self._checkpoint_step = self.agent.get_steps()
        ckpt_agent = self._create_checkpoint()
        self.agent.set_parameters(self._initial_weights) # NOTE: best to sync, but not necessary
        return ckpt_agent

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < PER_HISTORICAL_STEPS:
            return False

        main_opponents = self._payoff.get_names("right", MainPlayer) if self.side == "left" else self._payoff.get_names("left", MainPlayer)
        win_rates = self._payoff.get_item(self.name, main_opponents, "left") if self.side == "left" else self._payoff.get_item(main_opponents, self.name, "right")
        return win_rates.min() > 0.7 or steps_passed > PER_HISTORICAL_STEPS * 2


class LeagueExploiter(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)

    def get_match(self):
        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return self.get_player_by_name(np.random.choice(
            historical_opponents, p=pfsp(win_rates, weighting="linear_capped"))), True

    def checkpoint(self):
        self._checkpoint_step = self.agent.get_steps()
        ckpt_agent = self._create_checkpoint()
        if np.random.random() < 0.25:
            self.agent.set_parameters(self._initial_weights) # NOTE: best to sync, but not necessary
        return ckpt_agent

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < PER_HISTORICAL_STEPS:
            return False
        
        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return win_rates.min() > 0.7 or steps_passed > PER_HISTORICAL_STEPS * 2


class Historical(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, parent_name: str, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)
        self._parent = parent_name

    @property
    def parent(self):
        return self._parent

    def get_match(self):
        raise ValueError("Historical players should not request matches")

    def ready_to_checkpoint(self):
        return False


class League(object):

    def __init__(self,
                args,
                initial_agents,
                constructor,
                payoff=None,
                main_agents=1,
                main_exploiters=1,
                league_exploiters=2):
        if payoff is None:
            self._payoff = Payoff()
        else:
            self._payoff = payoff
        self._learning_agents = []
        for side in initial_agents:
            for idx in range(main_agents):
                main_agent = MainPlayer(f"MA{idx}_{side}", side, constructor, args, initial_agents[side], self._payoff)
                self._learning_agents.append(main_agent)
                self.add_player(main_agent.checkpoint())
            for idx in range(main_exploiters):
                self._learning_agents.append(
                    MainExploiter(f"ME{idx}_{side}", side, constructor, args, initial_agents[side], self._payoff))
            for idx in range(league_exploiters):
                self._learning_agents.append(
                    LeagueExploiter(f"LE{idx}_{side}", side, constructor, args, initial_agents[side], self._payoff))
        for player in self._learning_agents:
            self.add_player(player)

    # def update(self, home, away, result):
    #     return self._payoff.update(home, away, result)

    def get_player(self, idx):
        return self._learning_agents[idx]

    def add_player(self, player):
        cls_name, kwargs = get_player_config(player)
        print(f"[League] Adding player {kwargs['name']}", flush=True)
        self._payoff.add_player(cls_name, kwargs)
    
    def size(self):
        return len(self._learning_agents)


class FSPPlayer(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)

    def get_match(self):
        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        return self.get_player_by_name(np.random.choice(
            historical_opponents)), True

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < PER_HISTORICAL_STEPS:
            return False

        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return win_rates.min() > 0.7 or steps_passed > PER_HISTORICAL_STEPS * 2

    def checkpoint(self):
        if self.agent is not None:
            self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class FSPLeague(League):

    def __init__(self,
                args,
                initial_agents,
                constructor,
                payoff=None,
                main_agents=1):
        if payoff is None:
            self._payoff = Payoff()
        else:
            self._payoff = payoff
        self._learning_agents = []
        for side in initial_agents:
            for idx in range(main_agents):
                main_agent = FSPPlayer(f"FSP{idx}_{side}", side, constructor, args, initial_agents[side], self._payoff)
                self._learning_agents.append(main_agent)
                self.add_player(main_agent.checkpoint())
        for player in self._learning_agents:
            self.add_player(player)


class PSROPlayer(Player):

    def __init__(self, name, side, constructor, args, agent: LeaguePPO, payoff: Payoff, agent_dict=None, checkpoint_step=0):
        super().__init__(name, side, constructor, args, agent, payoff, agent_dict, checkpoint_step)

    def get_match(self):
        historical_opponents, mixed_weights = self._payoff.get_historical_nash("right") if self.side == "left" else self._payoff.get_historical_nash("left")
        return self.get_player_by_name(np.random.choice(
            historical_opponents, p=mixed_weights)), True

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < PER_HISTORICAL_STEPS:
            return False

        historical_opponents = self._payoff.get_names("right", Historical) if self.side == "left" else self._payoff.get_names("left", Historical)
        win_rates = self._payoff.get_item(self.name, historical_opponents, "left") if self.side == "left" else self._payoff.get_item(historical_opponents, self.name, "right")
        return win_rates.min() > 0.7 or steps_passed > PER_HISTORICAL_STEPS * 2

    def checkpoint(self):
        if self.agent is not None:
            self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()

    def send_outcome(self, opponent_name, outcome):
        # print(f"[Player] send_outcome: {self.name} vs {opponent_name} = {outcome}")
        if self.side == "left":
            self._payoff.update(self.name, opponent_name, outcome)
        elif self.side == "right":
            self._payoff.update(opponent_name, self.name, outcome)
        else:
            raise ValueError("side must be either 'left' or 'right'")
        if self.ready_to_checkpoint():
            historical_agent = self.checkpoint()
            self.add_player(historical_agent)
            historical_agent.construct_agent()
            historical_agent._payoff = self._payoff
            # update payoff
            historical_opponents = historical_agent._payoff.get_names("right", Historical) if historical_agent.side == "left" else historical_agent._payoff.get_names("left", Historical)
            for opponent_name in historical_opponents:
                opponent = historical_agent.get_player_by_name(opponent_name)
                coordinate_fn = partial(historical_agent.send_outcome, opponent.name)
                if historical_agent.side == "left":
                    opponent_policy = None
                    opponent_policy_other = opponent.agent.policy_other
                elif historical_agent.side == "right":
                    opponent_policy = opponent.agent.policy
                    opponent_policy_other = None
                else:
                    raise ValueError("side must be either 'left' or 'right'")
                # reset env
                _, callback = historical_agent.agent._setup_learn(
                    total_timesteps=0,
                    callback=None,
                    reset_num_timesteps=True,
                    tb_log_name="run",
                    progress_bar=False,
                )
                callback.on_training_start(locals(), globals())
                historical_agent.agent.collect_rollouts(historical_agent.agent.env, callback, historical_agent.agent.rollout_buffer, historical_agent.agent.rollout_buffer_other, n_rollout_steps=historical_agent.agent.n_steps, policy=opponent_policy, policy_other=opponent_policy_other, coordinate_fn=coordinate_fn)
                callback.on_training_end()


class PSROLeague(League):

    def __init__(self,
                args,
                initial_agents,
                constructor,
                payoff=None,
                main_agents=1):
        if payoff is None:
            self._payoff = Payoff()
        else:
            self._payoff = payoff
        self._learning_agents = []
        for side in initial_agents:
            for idx in range(main_agents):
                main_agent = PSROPlayer(f"PSRO{idx}_{side}", side, constructor, args, initial_agents[side], self._payoff)
                self._learning_agents.append(main_agent)
                self.add_player_with_evaluation(main_agent.checkpoint())
        for player in self._learning_agents:
            self.add_player(player)
    
    def add_player_with_evaluation(self, historical_agent):
        self.add_player(historical_agent)
        historical_agent.construct_agent()
        historical_agent._payoff = self._payoff
        # update payoff
        historical_opponents = historical_agent._payoff.get_names("right", Historical) if historical_agent.side == "left" else historical_agent._payoff.get_names("left", Historical)
        for opponent_name in historical_opponents:
            opponent = historical_agent.get_player_by_name(opponent_name)
            coordinate_fn = partial(historical_agent.send_outcome, opponent.name)
            if historical_agent.side == "left":
                opponent_policy = None
                opponent_policy_other = opponent.agent.policy_other
            elif historical_agent.side == "right":
                opponent_policy = opponent.agent.policy
                opponent_policy_other = None
            else:
                raise ValueError("side must be either 'left' or 'right'")

            _, callback = historical_agent.agent._setup_learn(
                total_timesteps=0,
                callback=None,
                reset_num_timesteps=True,
                tb_log_name="run",
                progress_bar=False,
            )
            callback.on_training_start(locals(), globals())
            historical_agent.agent.collect_rollouts(historical_agent.agent.env, callback, historical_agent.agent.rollout_buffer, historical_agent.agent.rollout_buffer_other, n_rollout_steps=historical_agent.agent.n_steps, policy=opponent_policy, policy_other=opponent_policy_other, coordinate_fn=coordinate_fn)
            callback.on_training_end()


class Learner:

    def __init__(self, player: Player):
        self.player = player

    def run(
        self,
        total_timesteps: int,
        rollout_opponent_num: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "Learner",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        def get_kwargs():
            opponent, _ = self.player.get_match()
            print(f"[Learner] Self({self.player.name}) vs Opponent({opponent.name})")
            kwargs = {
                "policy_other": opponent.agent.policy_other,
                "coordinate_fn": partial(self.player.send_outcome, opponent.name),
                "sync_fn": self.player.sync,
            } if self.player.side == "left" else {
                "policy": opponent.agent.policy,
                "coordinate_fn": partial(self.player.send_outcome, opponent.name),
                "sync_fn": self.player.sync,
            }
            return kwargs
        
        self.player.agent.learn(total_timesteps, rollout_opponent_num, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar, get_kwargs)

