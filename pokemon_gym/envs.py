import numpy as np
import gym

class PokemonBattleEnv_v0(gym.Env):
    def __init__(self, pokemon1, pokemon2, effectiveness_chart, opponent="random", additional_reward=True):
        super().__init__()
        
        # Agent is pokemon1, Opponent is pokemon2
        self.pokemon1 = pokemon1
        self.pokemon2 = pokemon2
        
        # Probability that the opponent will choose the most effective move:
        # 0 = random, 1 = always perfect
        self.perfect_opponent_prob = 0
        if opponent == "random_perfect":
            self.perfect_opponent_prob = 0.5
        
        # Whether to give additional reward for super effective moves
        # if False, reward is only given at the end of the episode
        self.additional_reward = additional_reward
        
        # Type effectiveness chart
        self.effectiveness_chart = effectiveness_chart
    
        # Action space: 4 moves in the moveset
        self.action_space = gym.spaces.Discrete(len(self.pokemon1.moveset))
        
        # Observation space: type of pokemon1, type of pokemon2
        self.observation_space = gym.spaces.Discrete(len(self.pokemon1.types_map))
        
        # Keeps track of reward and whether the battle is over
        self.reward = 0
        self.done = False
        
    def reset(self):
        self.pokemon1.reset()
        self.pokemon2.reset()
        self.reward = 0
        self.done = False
        return self.get_obs()
    
    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action = self.pokemon1.moveset[action]
        
        # pokemon1 attacks pokemon2
        self.attack(action, self.pokemon2)
        
        # pokemon2 attacks pokemon1 if it is not fainted
        if not self.pokemon2.is_fainted():
            if np.random.rand() < self.perfect_opponent_prob:
                opp_move = np.argmax(self.effectiveness_chart[:,self.pokemon1.type])
            else:
                opp_move = np.random.randint(0,len(self.pokemon2.moveset))
            
            self.attack(opp_move, self.pokemon1)
        else:
            opp_move = "fainted"
        
        if self.is_battle_over():
            self.done = True
            self.reward += self.pokemon1.health
            info = {"agent action": action, "opponent action": opp_move}
            return self.get_obs(), self.reward, self.done, info
        
        if self.additional_reward:
            self.reward += self.partial_reward(action)
            
        obs = self.get_obs()
        info = {"agent action": action, "opponent action": opp_move}
        
        return obs, self.reward, self.done, info

    def calculate_damage(self, move_type, defender_type):
        damage = 10 * self.effectiveness_chart[move_type][defender_type]
        return int(damage)
    
    def attack(self, action, defender):
        damage = self.calculate_damage(action, defender.type)
        defender.health -= damage
        if defender.health <= 0:
            defender.health = 0

    def partial_reward(self, action):
        if self.effectiveness_chart[action][self.pokemon2.type] == 2:
            return 1
        elif self.effectiveness_chart[action][self.pokemon2.type] == 1:
            return 0
        else:
            return -1

    def get_obs(self):
        return self.pokemon2.type
        
    def is_battle_over(self):
        return self.pokemon1.is_fainted() or self.pokemon2.is_fainted()
        
    def render(self):
        print(self.pokemon1, "vs", self.pokemon2)
        
        
        


class PokemonBattleEnv_v1(gym.Env):
    def __init__(self, pokemon1, pokemon2, effectiveness_chart, opponent="random", additional_reward=True, selftype_dmg=True):
        super().__init__()
        
        # Agent is pokemon1, Opponent is pokemon2
        self.pokemon1 = pokemon1
        self.pokemon2 = pokemon2
        
        # Probability that the opponent will choose the most effective move:
        # 0 = random, 1 = always perfect
        self.perfect_opponent_prob = 0
        if opponent == "random_perfect":
            self.perfect_opponent_prob = 0.5
        
        # Whether to give additional reward for super effective moves
        # if False, reward is only given at the end of the episode
        self.additional_reward = additional_reward
        
        # Whether moves of the same type as the pokemon deal extra damage
        self.selftype_dmg = selftype_dmg
        
        # Type effectiveness chart
        self.effectiveness_chart = effectiveness_chart
    
        # Action space: 4 moves in the moveset
        self.action_space = gym.spaces.Discrete(len(self.pokemon1.moveset))
        
        # Observation space: type of pokemon1, type of pokemon2
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(len(self.pokemon1.types_map)),
            gym.spaces.Discrete(len(self.pokemon2.types_map))
        ))
        
        # Keeps track of reward and whether the battle is over
        self.reward = 0
        self.done = False
        
    def reset(self):
        self.pokemon1.reset()
        self.pokemon2.reset()
        self.reward = 0
        self.done = False
        return self.get_obs()
    
    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action = self.pokemon1.moveset[action]
        
        # pokemon1 attacks pokemon2
        self.attack(action, self.pokemon1, self.pokemon2)
        
        # pokemon2 attacks pokemon1 if it is not fainted
        if not self.pokemon2.is_fainted():
            if np.random.rand() < self.perfect_opponent_prob:
                opp_move = np.argmax(self.effectiveness_chart[:,self.pokemon1.type])
            else:
                opp_move = np.random.randint(0,4)
            
            self.attack(opp_move, self.pokemon2, self.pokemon1)
        else:
            opp_move = "fainted"
        
        if self.is_battle_over():
            self.done = True
            self.reward += self.pokemon1.health
            info = {"agent action": action, "opponent action": opp_move}
            return self.get_obs(), self.reward, self.done, info
        
        if self.additional_reward:
            self.reward += self.partial_reward(action)
            
        obs = self.get_obs()
        info = {"agent action": action, "opponent action": opp_move}
        
        return obs, self.reward, self.done, info

    def calculate_damage(self, move_type, attacker_type, defender_type):
        damage = 10 * self.effectiveness_chart[move_type][defender_type]
        if self.selftype_dmg and move_type == attacker_type:
            damage += 5
        return int(damage)
    
    def attack(self, action, attacker, defender):
        damage = self.calculate_damage(action, attacker.type, defender.type)
        defender.health -= damage
        if defender.health <= 0:
            defender.health = 0

    def partial_reward(self, action):
        reward = 0
        if self.effectiveness_chart[action][self.pokemon2.type] == 2:
            reward += 1
        elif self.effectiveness_chart[action][self.pokemon2.type] == 1:
            reward += 0
        else:
            reward += -1
            
        if self.selftype_dmg and action == self.pokemon1.type:
            reward += 0.5
            
        return reward
        

    def get_obs(self):
        return [
            self.pokemon1.type,
            self.pokemon2.type
        ]
        
    def is_battle_over(self):
        return self.pokemon1.is_fainted() or self.pokemon2.is_fainted()
        
    def render(self):
        print(self.pokemon1, "vs", self.pokemon2)
        
        
        
class PokemonBattleEnv_v2(gym.Env):
    def __init__(self, pokemon1, pokemon2, effectiveness_chart, opponent="random", additional_reward=True, selftype_dmg=True):
        super().__init__()
        
        # Agent is pokemon1, Opponent is pokemon2
        self.pokemon1 = pokemon1
        self.pokemon2 = pokemon2
        
        # Probability that the opponent will choose the most effective move:
        # 0 = random, 1 = always perfect
        self.perfect_opponent_prob = 0
        if opponent == "random_perfect":
            self.perfect_opponent_prob = 0.5
        
        # Whether to give additional reward for super effective moves
        # if False, reward is only given at the end of the episode
        self.additional_reward = additional_reward
        
        # Whether moves of the same type as the pokemon deal extra damage
        self.selftype_dmg = selftype_dmg
        
        # Type effectiveness chart
        self.effectiveness_chart = effectiveness_chart
    
        # Action space: position of the move in the moveset
        self.action_space = gym.spaces.Discrete(len(self.pokemon1.moveset))
        
        # Observation space: type of pokemon1 = type of the first move, type of pokemon2, and the remaining 3 move types
        self.observation_space = gym.spaces.Box(
            low=0,  # Minimum value for each type (0)
            high=len(self.pokemon1.types_map),  # Maximum value for each type (6 types)
            shape=(5,)  # 6 dimensions: own type, opponent type, and 4 move types
        )
        
        # Keeps track of reward and whether the battle is over
        self.reward = 0
        self.done = False
        
    def reset(self):
        self.pokemon1.reset()
        self.pokemon2.reset()
        self.reward = 0
        self.done = False
        return self.get_obs()
    
    def step(self, action):
        
        assert self.action_space.contains(action)
        
        action = self.pokemon1.moveset[action]
        
        # pokemon1 attacks pokemon2
        self.attack(action, self.pokemon1, self.pokemon2)
        
        # pokemon2 attacks pokemon1 if it is not fainted
        if not self.pokemon2.is_fainted():
            if np.random.rand() < self.perfect_opponent_prob:
                opp_move = np.argmax(self.effectiveness_chart[:,self.pokemon1.type])
            else:
                opp_move = np.random.randint(0,4)
            
            self.attack(opp_move, self.pokemon2, self.pokemon1)
        else:
            opp_move = "fainted"
        
        if self.is_battle_over():
            self.done = True
            self.reward += self.pokemon1.health
            info = {"agent action": action, "opponent action": opp_move}
            return self.get_obs(), self.reward, self.done, info
        
        if self.additional_reward:
            self.reward += self.partial_reward(action)
            
        obs = self.get_obs()
        info = {"agent action": action, "opponent action": opp_move}
        
        return obs, self.reward, self.done, info

    def calculate_damage(self, move_type, attacker_type, defender_type):
        damage = 10 * self.effectiveness_chart[move_type][defender_type]
        if self.selftype_dmg and move_type == attacker_type:
            damage += 5
        return int(damage)
    
    def attack(self, action, attacker, defender):
        damage = self.calculate_damage(action, attacker.type, defender.type)
        defender.health -= damage
        if defender.health <= 0:
            defender.health = 0

    def partial_reward(self, action):
        reward = 0
        if self.effectiveness_chart[action][self.pokemon2.type] == 2:
            reward += 1
        elif self.effectiveness_chart[action][self.pokemon2.type] == 1:
            reward += 0
        else:
            reward += -1
            
        if self.selftype_dmg and action == self.pokemon1.type:
            reward += 0.5
            
        return reward
        

    def get_obs(self):
        return [self.pokemon1.type,
                self.pokemon2.type, 
                *self.pokemon1.moveset[1:]]
        
    def is_battle_over(self):
        return self.pokemon1.is_fainted() or self.pokemon2.is_fainted()
        
    def render(self):
        print(self.pokemon1, "vs", self.pokemon2)