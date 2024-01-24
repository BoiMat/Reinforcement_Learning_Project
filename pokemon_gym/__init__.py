import gym
# from envs import PokemonBattleEnv
from gym.envs.registration import register

register(
    id='PokemonBattleEnv-v0',
    entry_point='pokemon_gym.envs:PokemonBattleEnv_v0',
)

register(
    id='PokemonBattleEnv-v1',
    entry_point='pokemon_gym.envs:PokemonBattleEnv_v1',
)

register(
    id='PokemonBattleEnv-v2',
    entry_point='pokemon_gym.envs:PokemonBattleEnv_v2',
)