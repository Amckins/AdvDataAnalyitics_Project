import numpy as np
import os
import time
from util import format_time


#Hyperparameters - Adjusted for unique 3.4M state space
alpha = 0.1                         
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.999995
max_episodes = 10000000
max_depth = 10
convergence_acceptance_rate = 0.95
steps_to_take_start = 200
log_interval = 10000
save_at_end = True
masking = True

#Load transition data
print("\nLoading unique Transition Matrix")
current_time = time.time()
data = np.load('cube_output/npz/transitions_depth_None.npz')
transitions = data['transitions']
moves_list = data['moves']
id_to_state = data['id_to_state']
solved_state_id = int(data['solved_state_id'][0])
num_states = len(transitions)
num_actions = len(moves_list)
print(f"Loaded {num_states:,} unique states in {format_time(time.time() - current_time)}")
print(f"Solved state ID: {solved_state_id}")

#Invalid move pairs for action masking
#U is undone by U' and therefore counterproductive
invalid_move_pairs = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4,
                      6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}

print("\nLoading depths from transition matrix")
depth_raw = data['depths']
depth = np.full(num_states, -1, dtype = np.int32)
for state_id in range(num_states):
    if (val := depth_raw[state_id]) <= max_depth:
        depth[state_id] = depth_raw[state_id]


# # Print depth statistics
# for d in range(max_depth + 1):
#     count = np.sum(depth == d)
#     if count > 0:
#         print(f"\tDepth {d}: {count:,} states")

# if states_beyond_max > 0:
#     print(f"\tMasked (>{max_depth}): {states_beyond_max:,} states set to unreachable")

def get_reward(old_state_id, new_state_id):
    new_depth, old_depth = int(depth[new_state_id]), int(depth[old_state_id])

    if new_depth == 0:                  #Solved!
        return 100
    elif new_depth < old_depth:         #Moving closer to solved
        return 14 - new_depth
    elif new_depth >= old_depth:        #Moving away or staying same depth
        return -(new_depth**2)
    else:                               #Fallback
        return -1.0

def get_valid_actions(last_move):
    valid_actions = list(range(num_actions))
    if last_move is not None and last_move in invalid_move_pairs:
        invalid_action = invalid_move_pairs[last_move]
        valid_actions.remove(invalid_action)
    return valid_actions

#Load or create Q-table
q_table_file = 'q_table.npz'
if os.path.exists(q_table_file):
    current_time = time.time()
    data = np.load(q_table_file)
    Q = data['Q']
    print(f"\nLoaded Q-table from {q_table_file} in {format_time(time.time()-current_time)}")
else:
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    print("\nCreated new Q-table")

#Training log
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
    log_file.write("Episode,Interval_Solves,Cumulative_Solves,Success_Rate,Avg_Solve_Length,Alpha,Gamma,Epsilon,Steps_Per_Episode,Interval_Time\n")
    log_file.flush()
    print(f"Created new training log: {training_log_file}")

# Initialize counters
solve_count = 0
interval_solves = 0
interval_solve_lengths = []  # Track steps taken for each solve in interval
full_save_counter = 1
recent_solves = []
min_epsilon_count = 0
faux_episode = 0
times_converged = 0
steps_to_take = steps_to_take_start

print(f"\nStarting training for {max_episodes:,} episodes")
print(f"Max steps per episode: {steps_to_take:,}")
print(f"unique state space: {num_states:,} states\n")

start_time = time.time()
last_checkpoint_time = start_time

for episode in range(max_episodes):

    # Epsilon-greedy decay
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** (faux_episode + 1)))
    
    # Choose random starting state (prefer deeper states for harder scrambles)
    state_id = max(np.random.randint(0, num_states),
                   np.random.randint(0, num_states))
    solved_this_episode = False
    last_move = None

    for step in range(steps_to_take):

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore: choose randomly from valid actions
            action_id = np.random.choice(get_valid_actions(last_move))
        elif masking:
            # Exploit with masking: mask invalid actions
            q_values_masked = Q[state_id].copy()
            if last_move is not None:
                q_values_masked[invalid_move_pairs[last_move]] = -np.inf
            action_id = np.argmax(q_values_masked)
        else:
            # Exploit without masking
            action_id = np.argmax(Q[state_id, get_valid_actions(last_move)])

        # Take action
        next_state_id = transitions[state_id, action_id]
        reward = get_reward(state_id, next_state_id)
        
        # Q-learning update
        Q[state_id, action_id] += alpha * (
            reward + gamma * np.max(Q[next_state_id]) - Q[state_id, action_id]
        )
        
        # Update state and last move
        state_id = next_state_id
        last_move = action_id

        # Check if solved (depth 0 = solved state)
        if depth[state_id] == 0:
            solved_this_episode = True
            solve_count += 1
            interval_solves += 1
            interval_solve_lengths.append(step + 1)  # Record number of steps to solve
            break
    
    # Track recent solves for convergence checking
    recent_solves.append(1 if solved_this_episode else 0)
    if len(recent_solves) > log_interval:
        recent_solves.pop(0)
    
    # Periodic logging
    if (episode + 1) % log_interval == 0:
        current_time = time.time()
        total_elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - last_checkpoint_time

        interval_success_rate = interval_solves / log_interval
        recent_success_rate = sum(recent_solves) / len(recent_solves) if recent_solves else 0

        cumulative_episode += log_interval
        cumulative_solves += interval_solves

        def write_to_log():
            # Calculate average solve length for this interval
            avg_solve_length = np.mean(interval_solve_lengths) if interval_solve_lengths else 0
            
            # Format all metrics for CSV
            metrics_to_write = [
                cumulative_episode,
                interval_solves,
                cumulative_solves,
                interval_success_rate,
                avg_solve_length,
                alpha,
                gamma,
                epsilon,
                steps_to_take,
                interval_elapsed_time
            ]
            format_specs = ["{}", "{}", "{}", "{:.4f}", "{:.2f}", "{:.3f}", "{:.3f}", "{:.4f}", "{}", "{:.2f}"]
            log_str = ",".join(fmt.format(val) for fmt, val in zip(format_specs, metrics_to_write))
            log_file.write(log_str + "\n")
            log_file.flush()
        
        avg_solve_length = np.mean(interval_solve_lengths) if interval_solve_lengths else 0
        min_solve_length =np.min(interval_solve_lengths) if interval_solve_lengths else 0
        max_solve_length = np.max(interval_solve_lengths) if interval_solve_lengths else 0
        percent_complete = (episode + 1) / max_episodes
        
        output_str = ''
        output_str += f"Ep {episode + 1:>10,} "
        output_str += f"({percent_complete:>6.2%}) "
        output_str += f"| Solves: {interval_solves:>5,} "
        output_str += f"({interval_success_rate:>6.2%}) "
        output_str += f"| Avg Len: {avg_solve_length:>6.2f} "
        output_str += f"({min_solve_length:>3.0f} - {max_solve_length:>3.0f}) "
        output_str += f"| ε: {epsilon:.4f} "
        output_str += f"| {format_time(interval_elapsed_time)}"
        
        print(output_str, end='')

        # Check for convergence
        if recent_success_rate >= convergence_acceptance_rate and len(recent_solves) == log_interval:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            times_converged += 1
            
            if times_converged >= 20:
                print("\n" + "="*60)
                print(f"Training Converged! ({cumulative_episode:,} episodes, {format_time(total_elapsed_time)})")
                print(f"Success rate: {interval_success_rate:.2%} | Total solves: {cumulative_solves:,}")
                print("="*60 + "\n")
                break
            elif (episode + 1) < max_episodes:
                print(f"\n\n[Convergence {times_converged}/20] Reducing steps to {max(14, int(steps_to_take - 0.05 * steps_to_take_start * times_converged))}\n")
                steps_to_take = max(14, int(steps_to_take - 0.05 * steps_to_take_start * times_converged))
                recent_solves = []

        elif (episode + 1) >= max_episodes:
            write_to_log()
            print("\n" + "="*60)
            print(f"Max episodes reached ({format_time(total_elapsed_time)})")
            print(f"Solves: {solve_count:,}/{episode + 1:,} ({solve_count / (episode + 1):.2%})")
            print("Saving Q-table...")
            current_time = time.time()
            np.savez_compressed(q_table_file, Q=Q)
            print(f"Saved in {format_time(time.time()-current_time)}")
            print("="*60)
            break
            
        elif (full_save_counter % 10) == 0 and not save_at_end:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            print(" ✓")
        else:
            write_to_log()
            print()
            
            # Reset epsilon decay if stuck at minimum
            if epsilon == epsilon_end:
                min_epsilon_count += 1
                if min_epsilon_count >= 2:
                    faux_episode = 0
                    min_epsilon_count = 0
                    print(f"\n[Epsilon reset]\n")
            
        interval_solves = 0
        interval_solve_lengths = []  # Reset solve length tracker
        last_checkpoint_time = time.time()
        full_save_counter += 1
        
    faux_episode += 1

log_file.close()