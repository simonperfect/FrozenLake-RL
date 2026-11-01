'''
S = Start
F = Frozen
H = Hole
G = Goal
'''


import gymnasium as gym 
from value_iteration import ValueIteration
from visualizer import visualize_agent
import numpy as np


def print_agent(agent, holes, goals):

    action_symbols = ["<", "v", ">", "^"]

    # =========================================
    # 顯示 State Value
    # =========================================

    print("\n===== State Values =====")

    for row in range(4):
        for col in range(4):

            state_id = row * 4 + col

            print(
                f"{agent.v[state_id]:.3f}",
                end="\t"
            )

        print()


    print("\n===== Policy =====")

    for row in range(4):
        for col in range(4):

            state_id = row * 4 + col

            # Hole
            if state_id in holes:
                print("****", end="\t")

            # Goal
            elif state_id in goals:
                print("EEEE", end="\t")

            else:
                policy = agent.pi[state_id]

                policy_string = ""

                for action in range(4):

                    if policy[action] > 0:
                        policy_string += action_symbols[action]
                    else:
                        policy_string += "o"

                print(policy_string, end="\t")

        print()


def test_agent(env, agent, episodes=1000):

    success_count = 0

    for episode in range(episodes):

        
        state, info = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):

            policy = agent.pi[state]
            action = np.random.choice(len(policy),p=policy)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

           
            if reward == 1:
                success_count += 1

    success_rate = success_count / episodes

    print("\n===== Agent Test =====")
    print("Episodes:", episodes)
    print("Success:", success_count)
    print(f"Success Rate: {success_rate:.2%}")



def main():

    env = gym.make("FrozenLake-v1",map_name="4x4",is_slippery=True,render_mode="ansi")

    state, info = env.reset(seed=0)

    print('============FrozenLake=============')
    print(env.render())


    print("number of states:",env.observation_space.n)
    print("number of actions:",env.action_space.n)
    print("start state:",state)


    print("Action define:")
    print("0 = left")
    print("1 = down")
    print("2 = right")
    print("3 = up")




    P = env.unwrapped.P
    # ============================================================
    # Custom Reward
    # Normal step = -0.1
    # Hole        = -1
    # Goal        = +1
    # ============================================================

    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):

            new_results = []

            for res in P[s][a]:
                p, next_state, r, done = res

                # Goal
                if done and r == 1.0:
                    new_reward = 1.0

                # Hole
                elif done:
                    new_reward = -1.0

                # Normal step
                else:
                    new_reward = -0.1

                new_results.append(
                    (p, next_state, new_reward, done)
                )

            P[s][a] = new_results
        
    

    print("\n===== State 0 Transition =====")

    action_names = ["left", "down", "right", "up"]

    for action in range(4):
        print(
            f"Action {action} ({action_names[action]}):",
            P[0][action]
        )

    holes = set()
    goals = set()
    terminal_states = set()

    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for res in P[s][a]:
                p,next_state,r,done = res

                if r == 1.0:    #mean it reach the goal
                    goals.add(next_state)

                if done:
                    terminal_states.add(next_state)
    holes = terminal_states - goals

    print("\n===== Special States =====")

    print("Hole States:", holes)

    print("Goal State:", goals)



    agent = ValueIteration(env,theta=0.001,gamma=0.9)
    agent.value_iteration()
    


    print_agent(agent,holes,goals)

    test_agent(env,agent,episodes=1000)



    reward, steps = visualize_agent(
        agent,
        seed=None,
        delay=0.5
    )

    print("\n===== Visual Test =====")
    print("Steps:", steps)

    if reward == 1:
        print("Result: SUCCESS")
    else:
        print("Result: FAILED")

    env.close()





if __name__ == "__main__":
    main()

