import numpy as np
import os
import time
from collections import deque
from generate_cube_transitions import generate_all_solved_orientations, format_time

#Hyperparameters - I need help tuning them
alpha = 0.1
gamma = 0.99
epsilon = 0.1

max_episodes = 1000000
max_distance = 9
convergence_acceptance_rate = 0.95
steps_to_take = 20

#Load transition data - will need to supply this. file is ~ 3.17GB
print("\nLoading in Matrix Data")
data = np.load('cube_output/npz/transitions_depth_None.npz')
transitions = data['transitions']
moves_list = data['moves']
id_to_state = data['id_to_state']
print("Finished Loading in Matrix Data")

#Generate all solved orientations from this given solved state
base_solved = [1, 2, 1, 2, 1, 2, 1, 2,
               3, 4, 3, 4, 3, 4, 3, 4,
               5, 6, 5, 6, 5, 6, 5, 6]
solved_orientations = generate_all_solved_orientations(base_solved)

solved_state_ids = set()
for orientation in solved_orientations:
    for state_id, state in enumerate(id_to_state):
        if np.array_equal(state, orientation):
            solved_state_ids.add(state_id)
            break

print(f"Found {len(solved_state_ids)} solved orientations")

num_states = len(transitions)
num_actions = len(moves_list)

#BFS to find distance from each state to nearest solved state similar to how the transition matrix was formed
print("Computing distances from solved states...")
distances = np.full(num_states, -1, dtype=np.int32)  #-1 = not visited
queue = deque()

# Initialize with solved states
for state_id in solved_state_ids:
    distances[state_id] = 0
    queue.append(state_id)

while queue:
    current_state = queue.popleft()
    current_dist = distances[current_state]
    
    if current_dist >= max_distance:
        continue
    
    #Check all possible moves from this state
    for action_id in range(num_actions):
        next_state = transitions[current_state, action_id]
        
        if distances[next_state] == -1:  # Not visited yet
            distances[next_state] = current_dist + 1
            queue.append(next_state)
    
    if len(queue) % 100000 == 0 and len(queue) > 0:
        print(f"  Queue size: {len(queue):,}, Distance: {current_dist}")

for dist in range(max_distance + 1):
    count = np.sum(distances == dist)
    print(f"  Distance {dist}: {count:,} states")

unreachable = np.sum(distances == -1)
print(f"  Unreachable (>{max_distance}): {unreachable:,} states")

#Define reward function based on distance
#Give substantially better rewards closer to a solved state
def get_reward(state_id):
    dist = distances[state_id]
    if dist in range(max_distance):
        return round((1 - (dist / 10)**.5), 3)
    else:
        return -0.01

#Load existing Q-table or create new one
q_table_file = 'q_table.npz'
if os.path.exists(q_table_file):
    data = np.load(q_table_file)
    Q = data['Q']
    print(f"Loaded existing Q-table from {q_table_file}")
else:
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    print("Created new Q-table")

solve_count = 0
recent_solves = []
start_time = time.time()
last_checkpoint_time = start_time

for episode in range(max_episodes):
    state_id = np.random.randint(0, num_states)
    solved_this_episode = False
    
    for step in range(steps_to_take):
        if np.random.random() < epsilon:
            action_id = np.random.randint(0, num_actions)
        else:
            action_id = np.argmax(Q[state_id])
        
        next_state_id = transitions[state_id, action_id]
        reward = get_reward(next_state_id)
        
        Q[state_id, action_id] += alpha * (
            reward + gamma * np.max(Q[next_state_id]) - Q[state_id, action_id]
        )
        
        state_id = next_state_id
        
        if distances[state_id] == 0:  # Solved
            solved_this_episode = True
            solve_count += 1
            break
    
    recent_solves.append(1 if solved_this_episode else 0)
    if len(recent_solves) > 100:
        recent_solves.pop(0)
    
    if (episode + 1) % 10000 == 0:
        current_time = time.time()
        elapsed = current_time - start_time
        success_rate = sum(recent_solves) / len(recent_solves) if recent_solves else 0
        
        remaining_episodes = max_episodes - (episode + 1)
        
        np.savez_compressed(q_table_file, Q=Q)
        print(f"Episode {episode + 1:,}/{max_episodes:,} ({(episode+1)/max_episodes*100:.2f}%) completed")
        print(f"  Solves: {solve_count}, Success: {success_rate:.2%} Elapsed Time:{format_time(elapsed)}")
                
        if success_rate > convergence_acceptance_rate and len(recent_solves) == 100:
            print(f"Training converged!")
            break

np.savez_compressed(q_table_file, Q=Q)
print(f"Training complete! Total solves: {solve_count}/{episode + 1} in {format_time(time.time() - start_time)}")