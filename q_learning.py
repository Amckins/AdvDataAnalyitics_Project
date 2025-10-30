import numpy as np
import os
import time
from collections import deque
from generate_cube_transitions import generate_all_solved_orientations, format_time

#Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon_max = 1.0
epsilon_min = 0.01
decay_episodes = 450000      #decay from max to min over this many episodes
reset_episodes = 500000      #reset decay every this many episodes
max_episodes = 10000000
max_depth = 9
convergence_acceptance_rate = 0.97
convergence_window = 100     #number of recent episodes to track for convergence
steps_to_take = 100
log_interval = 10000
save_at_end = True

### Load transition data - will need to supply this. file is ~ 3.17GB ### 
print("\nLoading in Matrix Data")
current_time = time.time()
data = np.load('cube_output/npz/transitions_depth_None.npz')
transitions = data['transitions']
moves_list = data['moves']
id_to_state = data['id_to_state']
num_states = len(transitions)
num_actions = len(moves_list)
print(f"Finished Loading in Matrix Data in {format_time(time.time() - current_time)}")

#invalid moves to perform for invalid action masking 
#i.e. U is undone by U' and therefore counterproductive
invalid_move_pairs = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4,
                      6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}

#This is the base solved state - Need to find all others to apply rewards to
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
print(f"\nFound {len(solved_state_ids)} solved orientations")

print("\nComputing Number of States at each depth")

#create empty array where all states are set to -1
depth = np.full(num_states, -1, dtype=np.int32)
queue = deque()

#set the solved states to a value of 0
for state_id in solved_state_ids:
    depth[state_id] = 0
    queue.append(state_id)

#for every state, take every turn possible
#check if it has already been found
#if not, add it to the queue
#record the depth of each state as it has been found
while queue:
    current_state = queue.popleft()
    current_depth = depth[current_state]
    
    #stop if we are at max depth
    if current_depth >= max_depth:
        continue
    
    for action_id in range(num_actions):
        next_state = transitions[current_state, action_id]
        
        if depth[next_state] == -1:                  #have we visited this state?
            depth[next_state] = current_depth + 1    #if not, update its depth
            queue.append(next_state)                 #add it to the queue
    
    if len(queue) == 0 or depth[queue[0]] != current_depth:     #are we out of states at this depth?
        count = np.sum(depth == current_depth)                   #how many states are at this depth?
        print(f"\tDepth {current_depth}: {count:,} states")

unreachable = np.sum(depth == -1)                               #find # of all unvisited states
print(f"\tUnreachable (>{max_depth}): {unreachable:,} states")


def get_reward(old_state_id, new_state_id):
    #the reward is shaped to encourage moving from deeper states to shallower states
    new_depth = depth[new_state_id]
    old_depth = depth[old_state_id]

    if new_depth == 0:                  #largest reward as depth 0 is solved
        return 100
    if new_depth < old_depth:           #give incremented reward based on how close to solved state is
        return max_depth - new_depth
    elif new_depth > old_depth:         #punish actions that result in moving deeper
        return -5
    else:
        return -1.0                     #every other reward ~costing 1 turn


def get_valid_actions(last_move):
    #get valid actions excluding inverse of last move
    valid_actions = list(range(num_actions))
    if last_move is not None and last_move in invalid_move_pairs:
        invalid_action = invalid_move_pairs[last_move]
        valid_actions.remove(invalid_action)
    return valid_actions


def calculate_epsilon(episode):
    #exponential decay from max to min over decay_episodes, resetting every reset_episodes
    episode_in_cycle = episode % reset_episodes
    
    if episode_in_cycle < decay_episodes:
        #calculate decay rate to reach epsilon_min at decay_episodes
        decay_rate = (epsilon_min / epsilon_max) ** (1 / decay_episodes)
        return epsilon_max * (decay_rate ** episode_in_cycle)
    else:
        #after decay period, stay at minimum
        return epsilon_min


#check and load a Q-Table if it exists and create one if it doesn't
q_table_file = 'q_table.npz'            
if os.path.exists(q_table_file):
    current_time = time.time()
    data = np.load(q_table_file)
    Q = data['Q']
    print(f"\nLoaded Existing Q-table from {q_table_file} in {format_time(time.time() - current_time)}")
else:
    Q = np.zeros((num_states, num_actions), dtype=np.float32)  #set it as empty
    print("\nCreated New Q-table")

#start a log file to track training rates and hyperparameter performance
training_log_file = 'training_log.txt'
cumulative_episode = 0
cumulative_solves = 0

if os.path.exists(training_log_file):
    with open(training_log_file, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('Episode,')]
        if data_lines:
            last_line = data_lines[-1].split(',')
            cumulative_episode = int(last_line[0])
            cumulative_solves = int(last_line[2])
            print(f"Resuming from episode {cumulative_episode:,} with {cumulative_solves:,} cumulative solves")
        else:
            print("Log file exists but has no data rows. Starting from episode 0.")
    log_file = open(training_log_file, 'a')
else:
    log_file = open(training_log_file, 'w')
    log_file.write("Episode,Interval_Solves,Cumulative_Solves,Success_rate,Epsilon,Elapsed_Time\n")
    log_file.flush()
    print(f"Created new Training log: {training_log_file}")

#initialize counters
solve_count = 0
interval_solves = 0
checkpoint_counter = 1
solve_history = []

print(f"\nStarting training for {max_episodes:,} episodes of {steps_to_take:,} random turns")

start_time = time.time()                #ready
last_checkpoint_time = start_time       #set
for episode in range(max_episodes):     #go
    
    #calculate epsilon with exponential decay and automatic reset every 500k
    epsilon = calculate_epsilon(episode)
    
    #display reset notification every 500k episodes
    if episode > 0 and episode % reset_episodes == 0:
        print(f"{f'\nEpsilon Reset at Episode {episode:,}\n':^60}")

    state_id = np.random.randint(0, num_states)  #choose random state to start at
    solved_this_episode = False                  #reset solved condition
    last_move = None                             #reset last move   

    for step in range(steps_to_take):

        #epsilon-greedy: increasingly likely over time to choose optimal policy
        if np.random.random() < epsilon:
            #choose randomly from valid actions
            action_id = np.random.choice(get_valid_actions(last_move))     
        else:
            #mask the values of Q for policy decision only
            #if you choose U put a block on U', but don't train on that
            q_values_masked = Q[state_id].copy()
            if last_move is not None:
                q_values_masked[invalid_move_pairs[last_move]] = -np.inf
            action_id = np.argmax(q_values_masked)
        
        #what is the next state chosen and what is its reward?
        next_state_id = transitions[state_id, action_id]       
        reward = get_reward(state_id, next_state_id)
        
        #update Q-table based on action chosen
        Q[state_id, action_id] += alpha * (
            reward + gamma * np.max(Q[next_state_id]) - Q[state_id, action_id]
        )
        
        #update values for next iteration
        state_id = next_state_id
        last_move = action_id

        #is the cube in a solved state?
        if depth[state_id] == 0: 
            solved_this_episode = True
            solve_count += 1
            interval_solves += 1
            break
    
    #keep track of recent solves for convergence checking
    solve_history.append(1 if solved_this_episode else 0)
    if len(solve_history) > convergence_window:
        solve_history.pop(0)
    
    #periodically update terminal
    if (episode + 1) % log_interval == 0:
        current_time = time.time()
        total_elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - last_checkpoint_time

        #what was the success rate of the entire interval?
        interval_success_rate = interval_solves / log_interval

        #what was the success rate of the most recent episodes?
        recent_success_rate = sum(solve_history) / len(solve_history) if solve_history else 0

        #increase counter for logging purposes
        cumulative_episode += log_interval
        cumulative_solves += interval_solves

        def write_to_log():
            #format the metrics to save to the log file
            metrics_to_write = [cumulative_episode, interval_solves, cumulative_solves,
                                interval_success_rate, epsilon, interval_elapsed_time]
            format_specs = ["{}", "{}", "{}",
                            "{:.3f}", "{:.3f}", "{:.3f}"]
            log_str = ",".join(fmt.format(val) for fmt, val in zip(format_specs, metrics_to_write))
            log_file.write(log_str + "\n")
            log_file.flush()
        
        print(f"Episode {episode + 1:,}/{max_episodes:,} ({(episode + 1) / max_episodes:.2%}) completed. {solve_count:,} solves this session.")
        print(f"\tSolves: {interval_solves}, Success: {interval_success_rate:.2%}, Epsilon: {epsilon:.4f}, Elapsed: {format_time(interval_elapsed_time)}", end='')

        #has the learning converged?
        if recent_success_rate >= convergence_acceptance_rate and len(solve_history) == convergence_window:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            print("\n\n" + "=" * 60)
            print(f"{f'Training Converged after {cumulative_solves:,} solves!':^60}")
            print("=" * 60)
            print(f"\t\tTime Elapsed: {format_time(total_elapsed_time)}")
            print(f"\t\tTotal Solves: {solve_count:,} / {episode + 1:,}")
            print(f"\t\tSuccess Rate: {solve_count / (episode + 1):.2%}\n\n")
            break
        #has it finished running?
        elif (episode + 1) >= max_episodes:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            print("\n\n" + "=" * 60)
            print(f"{'Training Complete!':^60}")
            print("=" * 60)
            print(f"\t\tTime Elapsed: {format_time(total_elapsed_time)}")
            print(f"\t\tTotal Solves: {solve_count:,} / {episode + 1:,}")
            print(f"\t\tSuccess Rate: {solve_count / (episode + 1):.2%}\n\n")
            break
        #periodic updates every 10 * interval length
        elif (checkpoint_counter % 10) == 0 and not save_at_end:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            print(" ✓")
            checkpoint_counter += 1
        else:
            write_to_log()
            print()
            checkpoint_counter += 1
        
        interval_solves = 0
        last_checkpoint_time = time.time()

log_file.close()