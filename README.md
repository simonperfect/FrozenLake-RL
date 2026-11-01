# FrozenLake Reinforcement Learning

Implementing Value Iteration from scratch on Gymnasium FrozenLake.

## Features

- FrozenLake 4×4 environment
- MDP transition analysis
- Custom reward design
- Value Iteration from scratch
- Optimal state value function
- Optimal policy extraction
- 1000-episode evaluation
- Gymnasium visual simulation

## Algorithm

V(s) = max_a Q(s,a)

Q(s,a) = Σ P(s'|s,a)[r + γV(s')]

## Results

Episodes: 1000
Success Rate: ~72%

## Project Structure

- `main.py` - experiment entry point
- `value_iteration.py` - Value Iteration algorithm
- `visualizer.py` - FrozenLake visualization