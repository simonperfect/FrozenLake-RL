import time
import numpy as np
import gymnasium as gym


# ============================================================
# 如果沒有打滑，計算理論上應該到達的 State
#
# FrozenLake Action：
# 0 = LEFT
# 1 = DOWN
# 2 = RIGHT
# 3 = UP
# ============================================================

def get_expected_state(state, action):

    row = state // 4
    col = state % 4

    # LEFT
    if action == 0:
        col = max(0, col - 1)

    # DOWN
    elif action == 1:
        row = min(3, row + 1)

    # RIGHT
    elif action == 2:
        col = min(3, col + 1)

    # UP
    elif action == 3:
        row = max(0, row - 1)

    return row * 4 + col


# ============================================================
# FrozenLake Visualizer
# ============================================================

def visualize_agent(
    agent,
    seed=None,
    delay=0.5,
    max_steps=100
):

    # ========================================================
    # 建立 FrozenLake
    # 使用 Gymnasium 原生 human 畫面
    # ========================================================

    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
        render_mode="human"
    )

    # 回到起點
    state, info = env.reset(seed=seed)

    terminated = False
    truncated = False

    reward = 0
    steps = 0

    action_names = [
        "LEFT",
        "DOWN",
        "RIGHT",
        "UP"
    ]

    print("\n===== Visualizer =====")

    # 讓初始畫面停一下
    time.sleep(delay)

    # ========================================================
    # Agent Loop
    # ========================================================

    while not (terminated or truncated):

        # 防止一直走不完
        if steps >= max_steps:
            print("\nReached maximum steps.")
            break

        # ====================================================
        # 記住目前 State
        # ====================================================

        old_state = state

        # ====================================================
        # 根據 Value Iteration Policy 選 Action
        # ====================================================

        policy = agent.pi[state]

        action = np.random.choice(
            len(policy),
            p=policy
        )

        # ====================================================
        # 如果完全沒有打滑
        # 理論上應該到哪個 State
        # ====================================================

        expected_state = get_expected_state(
            old_state,
            action
        )

        # ====================================================
        # FrozenLake 真正執行 Action
        # ====================================================

        state, reward, terminated, truncated, info = env.step(
            action
        )

        steps += 1

        # ====================================================
        # 判斷是否打滑
        #
        # 預期 State != 實際 State
        # 就代表環境沒有按照 Agent 想要的方向移動
        # ====================================================

        slipped = state != expected_state

        # ====================================================
        # 顯示結果
        # ====================================================

        if slipped:

            print(
                f"Step {steps}: "
                f"State {old_state} "
                f"-> Choose {action_names[action]} "
                f"-> Expected {expected_state} "
                f"-> Actual {state} "
                f"🧊 SLIPPED!"
            )

            # 打滑時停久一點
            # 方便肉眼看到角色突然滑向其他方向
            time.sleep(delay * 1.8)

        else:

            print(
                f"Step {steps}: "
                f"State {old_state} "
                f"-> {action_names[action]} "
                f"-> State {state}"
            )

            time.sleep(delay)

    # ========================================================
    # Episode 結束
    # ========================================================

    time.sleep(1)

    env.close()

    return reward, steps