import random as rand                   
                   
import numpy as np                   
                   
                   
class QLearner(object):                   
    """                   
    This is a Q learner object.                   
                   
    :param num_states: The number of states to consider.                   
    :type num_states: int                   
    :param num_actions: The number of actions available..                   
    :type num_actions: int                   
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.                   
    :type alpha: float                   
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.                   
    :type gamma: float                   
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.                   
    :type rar: float                   
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.                   
    :type radr: float                   
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.                   
    :type dyna: int                   
    :param verbose: If “verbose” is True, your code can print out information for debugging.                   
    :type verbose: bool                   
    """                   
    def __init__(                   
        self,                   
        num_states=100,                   
        num_actions=4,                   
        alpha=0.2,                   
        gamma=0.9,                   
        rar=0.5,                   
        radr=0.99,                   
        dyna=0,                   
        verbose=False,                   
    ):                   
        """                   
        Constructor method                   
        """                   
        self.verbose = verbose                   
        self.num_actions = num_actions                                                                                      
        self.num_states = num_states                   
        self.state_old = 0                   
        self.action_old = 0                   
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros(shape=(num_states, num_actions))
        if dyna != 0:
            self.T_c = np.ones(shape=(num_states, num_actions, num_states)) * 0.00001
            self.T = self.T_c/np.sum(self.T_c, axis=2, keepdims=True)
            # Initialize R to -1 as every move by default gets a -1 reward
            self.R = np.ones(shape=(num_states, num_actions)) * -1.0

    def update_Q(self, state, action, state_next, reward):
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + \
            self.alpha * (reward + self.gamma * self.Q[state_next][np.argmax(self.Q[state_next])])

    # https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
    def vectorized(self, prob_matrix, items):
        s = prob_matrix.cumsum(axis=0)
        r = np.random.rand(prob_matrix.shape[1])
        # print(r.shape)
        k = (s < r).sum(axis=0)
        return items[k]

    # https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    def random_choice_prob_index(self, a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        print(a.cumsum(axis=axis))
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)

    def execute_dyna(self, state, action, state_next, reward):
        self.T_c[state, action, state_next] += 1
        self.T = self.T_c / np.sum(self.T_c, axis=2, keepdims=True)
        self.R[state, action] = (1 - self.alpha) * self.R[state, action] + self.alpha * reward

        """ Vectorized implementation
        hallucinated_state = np.random.randint(0, self.num_states, self.dyna)
        hallucinated_action = np.random.randint(0, self.num_actions, self.dyna)
        # print(self.T[hallucinated_state, hallucinated_action].T.shape)
        # hallucinated_s_prime = self.T[hallucinated_state, hallucinated_action].argmax(axis=1)
        # hallucinated_s_prime = self.vectorized(self.T[hallucinated_state, hallucinated_action].T, np.arange(self.num_states, dtype=int))
        hallucinated_r = self.R[hallucinated_state, hallucinated_action]
        hallucinated_s_prime = self.random_choice_prob_index(self.T[hallucinated_state, hallucinated_action])

        check_hallucinated_s_prime = np.empty(self.dyna)
        check_hallucinated_r = np.empty(self.dyna)
        for i in range(self.dyna):
            check_hallucinated_s_prime[i] = np.random.multinomial(1, self.T[hallucinated_state[i]][hallucinated_action[i]]).argmax()
            check_hallucinated_r[i] = self.R[hallucinated_state[i]][hallucinated_action[i]]

        # print(hallucinated_s_prime[:5])
        # print(check_hallucinated_s_prime[:5].astype(int))
        print(hallucinated_r == check_hallucinated_r)
        # print(hallucinated_state.shape, hallucinated_action.shape, hallucinated_s_prime.shape, hallucinated_r.shape)
        # print(self.Q[hallucinated_s_prime].argmax(axis=1).shape)
        # print(self.Q[hallucinated_s_prime][np.argmax(self.Q[hallucinated_s_prime])].shape)
        # print(self.Q[hallucinated_s_prime][np.argmax(self.Q[hallucinated_s_prime], axis=1)].shape)
        # print(((1 - self.alpha) * self.Q[hallucinated_state, hallucinated_action]).shape)
        # print((self.alpha * (reward + self.gamma * self.Q[hallucinated_s_prime, self.Q[hallucinated_s_prime].argmax(axis=1)])).shape)
        self.Q[hallucinated_state, hallucinated_action] = (1 - self.alpha) * self.Q[hallucinated_state, hallucinated_action] + \
            self.alpha * (reward + self.gamma * self.Q[hallucinated_s_prime, self.Q[hallucinated_s_prime].argmax(axis=1)])
        """
        """Non-vectorized"""
        hallucinated_state = np.random.randint(0, self.num_states, self.dyna)
        hallucinated_action = np.random.randint(0, self.num_actions, self.dyna)
        hallucinated_r = self.R[hallucinated_state, hallucinated_action]
        for i in range(self.dyna):
            # prob = rand.uniform(0.0, 1.0)
            # if prob < self.rar:
            hallucinated_s_prime = np.random.multinomial(1, self.T[hallucinated_state[i]][hallucinated_action[i]]).argmax()
            # hallucinated_s_prime = np.random.choice(self.num_states, p=self.T[hallucinated_state, hallucinated_action][i])
            # else:
            # hallucinated_s_prime = self.T[hallucinated_state[i]][hallucinated_action[i]].argmax()
            self.update_Q(hallucinated_state[i], hallucinated_action[i], hallucinated_s_prime, hallucinated_r[i])
                   
    def querysetstate(self, s):                   
        """                   
        Update the state without updating the Q-table                   
                   
        :param s: The new state                   
        :type s: int                   
        :return: The selected action                   
        :rtype: int                   
        """                   
        self.state_old = s                   
        # action = np.random.randint(0, self.num_actions)
        action = np.argmax(self.Q[s])
        self.action_old = action                   
        if self.verbose:                   
            print(f"state = {s}")
            print(f"action = {action}")                   
        return action                   
                   
    def query(self, s_prime, r):                   
        """                   
        Update the Q table and return an action                   
                   
        :param s_prime: The new state                   
        :type s_prime: int                   
        :param r: The immediate reward                   
        :type r: float                   
        :return: The selected action                   
        :rtype: int                   
        """                   
        self.update_Q(self.state_old, self.action_old, s_prime, r)

        # self.memory.append((self.state_old, self.action_old, s_prime, r))

        if self.dyna > 0:
            self.execute_dyna(self.state_old, self.action_old, s_prime, r)

        action = np.random.randint(0, self.num_actions) if np.random.uniform(0.0, 1.0) < self.rar else np.argmax(self.Q[s_prime])

        self.rar *= self.radr

        if self.verbose:                   
            print(f"s = {s_prime}, a = {action}, r={r}")                   

        self.state_old = s_prime
        self.action_old = action
        return action
                   
                   
if __name__ == "__main__":                   
    print("Usage")                   
