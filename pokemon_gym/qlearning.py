import numpy as np

class Qlearning_TDControl():
    def __init__(self, 
                 space_size, 
                 action_size, 
                 gamma=1, 
                 lr_v=0.01):
        """
        Calculates optimal policy using off-policy Temporal Difference control
        Evaluates Q-value for (S,A) pairs, using one-step updates.
        """            
        self.gamma = gamma
   
        self.space_size = space_size
        self.action_size = action_size

        self.lr_v = lr_v
        self.Qvalues = np.zeros( (*self.space_size, self.action_size) )
    
    # -------------------   
    def single_step_update(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Uses the BEST (evaluated) action in the new state <- Q(S_new, A*) = max_A Q(S_new, A).
        """
        if done:     
            deltaQ = (r + 0 - self.Qvalues[ (*s, a) ])
        else:
            maxQ_over_actions = np.max(self.Qvalues[ (*new_s,) ])
            deltaQ = (r + 
                      self.gamma * maxQ_over_actions 
                                 - self.Qvalues[ (*s,a) ])
        
        self.Qvalues[ (*s, a) ] += self.lr_v * deltaQ
            
    # ---------------------
    def get_action_epsilon_greedy(self, s, eps):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        ran = np.random.rand()
        
        if (ran < eps):
            prob_actions = np.ones(self.action_size) / self.action_size 
        
        else:
            best_value = np.max(self.Qvalues[ (*s,) ])
            best_actions = ( self.Qvalues[ (*s,) ] == best_value )
            prob_actions = best_actions / np.sum(best_actions)
            
        a = np.random.choice(self.action_size, p=prob_actions)
        return a
        
    def greedy_policy(self):
        a = np.argmax(self.Qvalues, axis = -1)
        return a