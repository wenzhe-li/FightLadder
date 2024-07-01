import retro
import os
import numpy as np


SF_BONUS_LEVEL = [4, 8, 12]

SF_DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

retro_directory = os.path.dirname(retro.__file__)
sf_game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"
SF_STATE_DIR = os.path.join(retro_directory, sf_game_dir)

sf_game = "StreetFighterIISpecialChampionEdition-Genesis"

START_STATUS = 0

END_STATUS = 1

BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

SF_COMBOS_BUTTONS = [
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['X']], # 'Hadouken-R'
    [['RIGHT'], ['DOWN'], ['DOWN', 'RIGHT'], ['X']], # 'Shoryuken-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['A']], # 'Tatsumaki-R'
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['X']], # 'Hadouken-L'
    [['LEFT'], ['DOWN'], ['DOWN', 'LEFT'], ['X']], #'Shoryuken-L'
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['A']], # 'Tatsumaki-L'
]
SF_COMBOS = []
for combo_buttons in SF_COMBOS_BUTTONS:
    action_seq = []
    for combo_button in combo_buttons:
        button = [int(b in combo_button) for b in BUTTONS]
        for _ in range(2):
            action_seq.append(np.array(button))
    SF_COMBOS.append(action_seq)

DIRECTIONS_BUTTONS = [
    [], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], 
    ['UP', 'LEFT'], ['UP', 'RIGHT'], ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'], 
]

ATTACKS_BUTTONS = [
    [], ['B'], ['A'], ['C'], ['Y'], ['X'], ['Z'],
]

SELECT_CHARACTER_MOVEMENTS = {
    'NO_OP': [],
    'START': ['START'],
    'LEFT': ['LEFT'],
    'RIGHT': ['RIGHT'],
    'UP': ['UP'],
    'DOWN': ['DOWN'],
}
SELECT_CHARACTER_BUTTONS = {}
for k, v in SELECT_CHARACTER_MOVEMENTS.items():
    SELECT_CHARACTER_BUTTONS[k] = np.array([int(b in v) for b in BUTTONS])
SELECT_CHARACTER_SEQUENCES = {
    'Ryu': [],
    'Honda': ['RIGHT'],
    'Blanka': ['RIGHT'] * 2,
    'Guile': ['RIGHT'] * 3,
    'Balrog': ['RIGHT'] * 4,
    'Vega': ['RIGHT'] * 5,
    'Ken': ['DOWN'],
    'Chunli': ['DOWN'] + ['RIGHT'],
    'Zangief': ['DOWN'] + ['RIGHT'] * 2,
    'Dhalsim': ['DOWN'] + ['RIGHT'] * 3,
    'Sagat': ['DOWN'] + ['RIGHT'] * 4,
    'Bison': ['DOWN'] + ['RIGHT'] * 5,
}