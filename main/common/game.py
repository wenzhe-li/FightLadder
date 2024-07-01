from .const import *


def get_next_level(level):
    next_level = level + 1
    if next_level in SF_BONUS_LEVEL:
        next_level += 1
    return next_level
