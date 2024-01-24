import numpy as np

class Pokemon:
    def __init__(self, type=0, health=100, allowed_types=None):
        self.type = type
        self.health = health
        self.max_health = health
        self.moveset = []

        if allowed_types is not None:
            self.types_map = allowed_types
        else:
            self.types_map = ["fire", "water", "grass"]
        
        
    def random_pkm(self, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            
        self.type = np.random.randint(0,len(self.types_map))
        self.health = self.max_health
        self.moveset = [self.type]
        
        # if there are only 3 or 4 types, then the pokemon will have one move of each type
        # Otherwise, it will have 4 random moves between all the allowed types, with the first one being the pokemon's type
        if len(self.types_map) == 3 or len(self.types_map) == 4:
            self.moveset = np.arange(0,len(self.types_map))
            return self
        else:
            while len(self.moveset) < 4:
                move = np.random.randint(0,len(self.types_map))
                if move not in self.moveset:
                    self.moveset.append(move)
        
        return self
        
    def type_as_str(self):
        return self.types_map[self.type]
        
    def is_fainted(self):
        return self.health <= 0
    
    def restore(self):
        self.health = self.max_health
        
    def reset(self):
        return self.random_pkm()
        
    def __str__(self) -> str:
        return f"{self.type_as_str()} ({self.health}/{self.max_health}) {[self.types_map[t] for t in self.moveset]}"