import numpy as np
import os
import time
from util import format_time


#Hyperparameters
alpha = 0.1                         
gamma = 0.99
epsilon_start = .5
epsilon_end = 0.001
epsilon_decay = 0.999999
max_episodes = 10000000
max_depth = 14
convergence_acceptance_rate = .9995
steps_to_take_start = 30
step_increments = int(.025*steps_to_take_start)
log_interval = 100000

masking = True
learning = True
full_agent_control = True
reset_solve_tracker = False
use_solve_tracker = True
save_at_end = True
create_backup = False

#Load transition data
print("\nLoading Transition Matrix")
current_time = time.time()
file = 'cube_output\\npz\\depth_None.npz'
data = np.load(file)
transitions = data['transitions']
moves_list = data['moves']
id_to_state = data['id_to_state']
solved_state_id = int(data['solved_state_id'][0])
num_states = len(transitions)
num_actions = len(moves_list)
print(f"Loaded {num_states:,} states in {format_time(time.time() - current_time)}")
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



def get_reward(old_state_id, new_state_id):
    new_depth, old_depth = int(depth[new_state_id]), int(depth[old_state_id])

    if new_depth == 0:
        return 10000                      #Massive reward for solving
    depth_change = old_depth - new_depth
    if depth_change > 0:
        return 100 * depth_change        #Big reward for progress
    else:
        return -50    #Reasonable penalty

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
    if 'starting_states_solved' in data:
        starting_states_solved = data['starting_states_solved']
    else:
        #Track which starting states have been solved within 14 steps
        starting_states_solved = np.zeros(num_states, dtype=bool)
    print(f"\nLoaded Q-table from {q_table_file} in {format_time(time.time()-current_time)}")
else:
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    starting_states_solved = np.zeros(num_states, dtype=bool)
    print("\nCreated new Q-table")

if create_backup:
    print("Creating Backup Q-Table")
    from datetime import datetime
    #Ensure the backup directory exists
    backup_dir = "cube_output/q_table_backups"
    os.makedirs(backup_dir, exist_ok=True)

    #Generate timestamped filename
    name, ext = q_table_file.rsplit('.', 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file_name = f"q_table_{timestamp}.{ext}"

    #Full path to save the backup
    backup_path = os.path.join("cube_output","q_table_backups", backup_file_name)

    np.savez_compressed(backup_path, Q=Q, starting_states_solved = starting_states_solved)
    print(f"Created Backup Q-Table: {backup_path}")
    
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

#Initialize counters
solve_count = 0
interval_solves = 0
interval_solve_lengths = []  #Track steps taken for each solve in interval
full_save_counter = 1
recent_solves = []
min_epsilon_count = 0
faux_episode = 0
times_converged = 0
steps_to_take = steps_to_take_start

print(f"\nStarting training for {max_episodes:,} episodes with a max of {steps_to_take:,} steps per episode")
print(f"Total State Space: {num_states:,} states")

if reset_solve_tracker:
    starting_states_solved = np.zeros(num_states, dtype=bool)
    
num_solves = np.sum(~starting_states_solved)
starting_solves = num_solves
if num_solves > 0:
    print(f"Solve States Found Previously: {num_states-num_solves:,} ({(num_states-num_solves)/num_states:.2%})")
    print(f"Remaining Left to Find: {num_solves:,}\n")

start_time = time.time()
last_checkpoint_time = start_time
unsolved_indices = np.where(~starting_states_solved)[0]
for session_episode in range(max_episodes):

    #Epsilon-greedy decay
    if not full_agent_control:
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** (faux_episode + 1)))
    else:
        epsilon = 0

    #Choose random starting state (95% unsolved, 5% random)
    if np.random.random() < 0.95 and use_solve_tracker:
        #Choose from states that haven't been solved within 14 steps yet
        if len(unsolved_indices) > 0:
            state_id = np.random.choice(unsolved_indices)
        else:
            #Fallback to random if all states somehow solved
            state_id = np.random.randint(0, num_states)
    else:
        #Choose completely randomly (5% of the time)
        state_id = np.random.randint(0, num_states)

    #Store the starting state for this episode
    starting_state_id = state_id
    step_hist = [starting_state_id]
    solved_this_episode = False
    last_move = None

    for step in range(steps_to_take):

        # #Epsilon-greedy action selection
        if np.random.random() < epsilon:
            #Explore: choose randomly from valid actions
            action_id = np.random.choice(get_valid_actions(last_move))
        elif masking:
            #Exploit with masking: mask invalid actions
            q_values_masked = Q[state_id].copy()
            if last_move is not None:
                q_values_masked[invalid_move_pairs[last_move]] = -np.inf
            action_id = np.argmax(q_values_masked)
        else:
            #Exploit without masking
            action_id = np.argmax(Q[state_id, get_valid_actions(last_move)])
            
        #Take action
        next_state_id = transitions[state_id, action_id]
        reward = get_reward(state_id, next_state_id)
        
        #Q-learning update
        if learning:
            Q[state_id, action_id] += alpha * (
                reward + gamma * np.max(Q[next_state_id]) - Q[state_id, action_id]
            )
        
        #Update state and last move
        state_id = next_state_id
        last_move = action_id
        step_hist.append(state_id)
        #Check if solved (depth 0 = solved state)
        if depth[state_id] == 0:
            solved_this_episode = True
            solve_count += 1
            interval_solves += 1
            interval_solve_lengths.append(step + 1)  #Record number of steps to solve
            #Mark starting state as solved if within 14 steps
            if len(step_hist)-1 <= 14 and use_solve_tracker:
                for s in step_hist:
                    starting_states_solved[s] = True
            break

    #Track recent solves for convergence checking
    recent_solves.append(1 if solved_this_episode else 0)
    if len(recent_solves) > log_interval:
        recent_solves.pop(0)
    
    #Periodic logging
    if (session_episode + 1) % log_interval == 0:
        current_time = time.time()
        total_elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - last_checkpoint_time

        interval_success_rate = interval_solves / log_interval
        recent_success_rate = sum(recent_solves) / len(recent_solves) if recent_solves else 0

        cumulative_episode += log_interval
        cumulative_solves += interval_solves

        def write_to_log():
            #Calculate average solve length for this interval
            avg_solve_length = np.mean(interval_solve_lengths) if interval_solve_lengths else 0
            
            #Format all metrics for CSV
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
        percent_complete = (session_episode + 1) / max_episodes
        
        #Count unsolved states for display
        num_unsolved = np.sum(~starting_states_solved)
        unsolved_indices = np.where(~starting_states_solved)[0]
        
        output_str = ''
        output_str += f"Ep {session_episode + 1:>10,} "
        output_str += f"({percent_complete:>6.2%}) "
        output_str += f"| Solves: {interval_solves:>5,} "
        output_str += f"({interval_success_rate:>6.2%}) "
        output_str += f"| Avg Len: {avg_solve_length:>6.2f} "
        output_str += f"({min_solve_length:>3.0f} - {max_solve_length:>3.0f}) "
        output_str += f"| Unsolved: {num_unsolved:>8,} "
        output_str += f"| ε: {epsilon:.4f} "
        output_str += f"| {format_time(interval_elapsed_time)}"
        
        print(output_str, end='')

        #Check for convergence
        if recent_success_rate >= convergence_acceptance_rate and len(recent_solves) == log_interval:
            np.savez_compressed(q_table_file, Q=Q, starting_states_solved = starting_states_solved)
            write_to_log()
            times_converged += 1
            
            if times_converged >= 30:
                print("\n" + "="*60)
                print(f"Training Converged! ({cumulative_episode:,} episodes, {format_time(total_elapsed_time)})")
                print(f"Success rate: {interval_success_rate:.2%} | Total solves: {cumulative_solves:,}")
                print("="*60 + "\n")
                break
            elif (session_episode + 1) < max_episodes:
                steps_to_take -= step_increments
                print(f"\n\n[Convergence {times_converged}/30] Reducing steps to {max(14, steps_to_take)}")
                faux_episode = 0
                min_epsilon_count = 0
                print(f"[Epsilon reset]\n")
                steps_to_take = max(14, steps_to_take)
                recent_solves = []

        elif (session_episode + 1) >= max_episodes:
            write_to_log()
            print("\n" + "="*60)
            print(f"Max episodes reached ({format_time(total_elapsed_time)})")
            print(f"Solves: {solve_count:,}/{session_episode + 1:,} ({solve_count / (session_episode + 1):.2%})")
            print(f"Found Solves (len <14): {starting_solves - num_unsolved:,}")
            print("Saving Q-table...")
            current_time = time.time()
            np.savez_compressed(q_table_file, Q=Q, starting_states_solved = starting_states_solved)
            print(f"Saved in {format_time(time.time()-current_time)}")
            print("="*60)
            break
            
        elif (full_save_counter % 10) == 0 and not save_at_end:
            np.savez_compressed(q_table_file, Q=Q, starting_states_solved = starting_states_solved)
            write_to_log()
            print(" ✓")
        else:
            write_to_log()
            print()
            
            #Reset epsilon decay if stuck at minimum
            if epsilon == epsilon_end:
                min_epsilon_count += 1
                if min_epsilon_count >= 2:
                    faux_episode = 0
                    min_epsilon_count = 0
                    print(f"\n[Epsilon reset]\n")
            
        interval_solves = 0
        interval_solve_lengths = []  #Reset solve length tracker
        last_checkpoint_time = time.time()
        full_save_counter += 1
        
    faux_episode += 1

log_file.close()