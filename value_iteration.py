class ValueIteration:

    def __init__(self,env,theta=0.001,gamma=0.9):
        self.env = env

        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        self.v = [0.0] * self.state_num
        self.theta = theta
        self.gamma = gamma 
        self.pi = [None]*self.state_num


    def value_iteration(self):
        while True:

            max_diff = 0
            new_v = [0.0]*self.state_num

            for s in range(self.state_num):
                qsa_list = []

                for a in range(self.action_num):
                    qsa = 0
                    for res in self.env.unwrapped.P[s][a]:
                        p,next_state,r,done = res

                        qsa += p*(r+self.gamma*self.v[next_state]*(1-done))

                    qsa_list.append(qsa)

                new_v[s] = max(qsa_list)

                max_diff = max(max_diff,abs(new_v[s]-self.v[s]))

            self.v = new_v  #update whole new_v list

            if max_diff < self.theta:
                break
        self.get_policy()

    def get_policy(self):

        for s in range(self.state_num):
            qsa_list = []
            for a in range(self.action_num):
                qsa = 0
                for res in self.env.unwrapped.P[s][a]:

                    p, next_state, r, done = res

                    qsa += p * (r+ self.gamma* self.v[next_state]* (1 - done))

                qsa_list.append(qsa)

            max_q = max(qsa_list)
            count_max = qsa_list.count(max_q)

            self.pi[s] = [
                1/count_max if q == max_q else 0
                for q in qsa_list
            ]
