import numpy as np
import os
import time
from collections import deque
from generate_cube_transitions import generate_all_solved_orientations, format_time

#Hyperparameters - I need help tuning them
alpha = 0.1
gamma = 0.99
epsilon_start = 1
epsilon_end = .03
epsilon_decay = 0.9999
max_episodes = 10000000
max_depth = 9
convergence_acceptance_rate = 0.99
steps_to_take_start = 500
log_interval = 10000
save_at_end = True
masking = True

### Load transition data - will need to supply this. file is ~ 3.17GB ### 
print("\nLoading in Matrix Data")
current_time = time.time()
data = np.load('cube_output/npz/transitions_depth_None.npz')
transitions = data['transitions']
moves_list = data['moves']
id_to_state = data['id_to_state']
num_states = len(transitions)
num_actions = len(moves_list)
print(f"Finished Loading in Matrix Data in {format_time(time.time()- current_time)}")

#invalid moves to perform for invalid action masking 
#i.e. U is undone by U' and therfore counterproductive
invalid_move_pairs = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4,
                      6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}

#This is the base solved state - Need to find all others to apply rewards to
base_solved = [1, 2, 1, 2, 1, 2, 1, 2,
               3, 4, 3, 4, 3, 4, 3, 4,
               5, 6, 5, 6, 5, 6, 5, 6]

solved_orientations = generate_all_solved_orientations(base_solved)

solved_state_ids = set()
for orientation in solved_orientations: #spin
    for state_id, state in enumerate(id_to_state): #spinny
        if np.array_equal(state, orientation): #spin
            solved_state_ids.add(state_id) #spinny
            break #no spin
print(f"\nFound {len(solved_state_ids)} solved orientations")

print("\nComputing Number of States at each depth ")

#create empty array where all states are set to -1
depth = np.full(num_states, -1, dtype=np.int32)
queue = deque()

#set the solved states to a value of 0
for state_id in solved_state_ids:
    depth[state_id] = 0
    queue.append(state_id)

#for every state, take every turn possible
#check if it has already been found. 
#if not. add it to the queue
#record the depth of each state as it has been found
while queue:
    current_state = queue.popleft()
    current_depth = depth[current_state]
    
    #Go ahead and stop if we are at max depth
    if current_depth >= max_depth:
        continue
    
    for action_id in range(num_actions):
        next_state = transitions[current_state, action_id]
        
        if depth[next_state] == -1:                  #Have we visited this state?
            depth[next_state] = current_depth + 1    #if not, update its depth
            queue.append(next_state)                 #add it to the queue
    
    if len(queue) == 0 or depth[queue[0]] != current_depth:     #Are the states left the explore or are we out of our depth?
        count = np.sum(depth == current_depth)                   #How many states are at this depth?
        print(f"\tDepth {current_depth}: {count:,} states")


unreachable = np.sum(depth == -1)                               #find # of all unvisited states
print(f"\tUnreachable (>{max_depth}): {unreachable:,} states")  

def get_reward(old_state_id, new_state_id):
    #The reward is shaped to encourage moving from deeper states to shallower states
    #reward_schema.py outputs a visualization of the rewards
    new_depth, old_depth = depth[new_state_id], depth[old_state_id]

    if new_depth == 0:                  #largest reward as depth 0 is solved
        return 100
    elif new_depth < old_depth:         #give incremented reward based on how close to solved state is
        return 14 - new_depth
    elif new_depth >= old_depth:        #punish actions that result in moving deeper or staying at the same depth
        return -(new_depth**2)
    else:                               #every other reward ~costing 1 turn
        return -1.0                     

def get_valid_actions(last_move):
    valid_actions = list(range(num_actions))
    if last_move is not None and last_move in invalid_move_pairs:
        invalid_action = invalid_move_pairs[last_move]
        valid_actions.remove(invalid_action)
    return valid_actions
#check and load a Q-Table if it exists and create one if it doesnt
q_table_file = 'q_table.npz'            
if os.path.exists(q_table_file):
    current_time = time.time()
    data = np.load(q_table_file)
    Q = data['Q']
    print(f"\nLoaded Existing Q-table from {q_table_file} in {format_time(time.time()-current_time)}")
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
full_save_counter = 1
recent_solves = []
min_epsilon_count = 0
faux_episode = 0
times_converged = 0
steps_to_take = steps_to_take_start
print(f"\nStarting training for {max_episodes:,} episodes of {steps_to_take:,} random turns\n")

start_time = time.time()                #ready
last_checkpoint_time = start_time       #set
for episode in range(max_episodes):     #go

    #used in gradient decay, to initially prefer taking actions based on optimal policy
    #decays into taking random actions for the sake of learning
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** (faux_episode + 1)))
    
    state_id = max(np.random.randint(0, num_states),
                   np.random.randint(0, num_states))  #choose random state to start at
    solved_this_episode = False                       #reset solved condition
    last_move = None                                  #reset last move   

    for step in range(steps_to_take):

        #increasingly likely as time goes on to choose
        if np.random.random() < epsilon:

            #choose randomly from valid actions
            action_id = np.random.choice(get_valid_actions(last_move))     

        elif masking:
            #mask the values of Q for policy decision only. 
            #if you choose U put a block on U', but don't train on that
            q_values_masked = Q[state_id].copy()
            if last_move is not None: q_values_masked[last_move] = -np.inf
            action_id = np.argmax(q_values_masked)
        else:
            action_id = np.argmax(Q[get_valid_actions(last_move)])
            pass

        #what is the next state chosen and what is it's reward?
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
    
    #keep track of the most recent 100 solves for convergence checking
    recent_solves.append(1 if solved_this_episode else 0)
    if len(recent_solves) > log_interval:
        recent_solves.pop(0)
    
    #periodically update terminal
    if (episode + 1) % log_interval == 0:
        current_time = time.time()
        total_elapsed_time = current_time - start_time
        interval_elapsed_time = current_time - last_checkpoint_time

        #what was the success rate of the entire interval?
        interval_success_rate = interval_solves / log_interval

        #what was the sucess rate of the most 100 recent solves?
        recent_success_rate = sum(recent_solves) / len(recent_solves) if recent_solves else 0

        #increase counter for logging purposes
        cumulative_episode += log_interval
        cumulative_solves += interval_solves if interval_solves else 0

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
        print(f"\tSolves: {interval_solves}, Success: {interval_success_rate:.2%}, Epsilon: {epsilon:.4f}, Elapsed: {format_time(interval_elapsed_time)}", end = '')

        #has the learning converged?
        if recent_success_rate >= convergence_acceptance_rate and len(recent_solves) == log_interval:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()          
            times_converged += 1
            if times_converged >= 20:
                print("\n\n"+"="*80)
                print(f"{f"Training Converged after {cumulative_solves:,} Episodes!":^80}")
                print("="*80)
                print(f"\t\t\tTime Elapsed: {format_time(total_elapsed_time)}")
                print(f"\t\t\tIn the last {len(recent_solves):,} episodes")
                print(f"\t\t\t\tit had a success rate greater than {convergence_acceptance_rate:.2%}")
                print(f"\t\t\tTotal Solves: {solve_count:,} / {episode + 1:,}")
                print(f"\t\t\t\tSuccess Rate: {solve_count / (episode + 1):.2%}\n\n")
                break
            else:
                if (episode + 1) < max_episodes:
                    print("\n\n"+"="*80)
                    print(f"{f"Training Converged after {cumulative_episode:,} Episodes!":^80}")
                    print("="*80)
                    print(f"\t\t\tTime Elapsed: {format_time(total_elapsed_time)}")
                    print(f"\t\t\tTotal Solves: {interval_solves:,} / {log_interval:,}")
                    print(f"\t\t\tSuccess Rate: {interval_success_rate:.2%}\n")
                    print(f"\t\t\tIt has recently converged {times_converged} times.")
                    print(f"\t\t\tRepeating until it has converged {'20'} times \n\t\t\t\tor max episodes reached")
                    steps_to_take = max(14, int(steps_to_take - .05 * steps_to_take_start * times_converged))
                    print(f"\t\t\tReducing allowed turns per episode to {steps_to_take}.\n\n")
                    recent_solves = []

                

        #has it finished running?
        elif (episode + 1) >= max_episodes:
            write_to_log()
            print("\n\n"+"="*80)
            print(f"{"Max Number of Allowed Episodes Complete!":^80}")
            print("="*80)
            print(f"\t\t\tTime Elapsed: {format_time(total_elapsed_time)}")
            print(f"\t\t\tTotal Solves: {solve_count:,} / {episode + 1:,}")
            print(f"\t\t\tSuccess Rate: {solve_count / (episode + 1):.2%}\n\n")
            print(f"{"Saving Q-Table":^80}")
            current_time = time.time()
            np.savez_compressed(q_table_file, Q=Q)
            print(f"{f"Q-Table Saved in {format_time(time.time()-current_time())}!":^80}")

            break
        #periodic updates every 10 * interval length
        elif (full_save_counter % 10) == 0 and not save_at_end:
            np.savez_compressed(q_table_file, Q=Q)
            write_to_log()
            print(" ✓")
        else:
            write_to_log()
            print()
            #if epsilon has remained at min value for 3 intervals,
            #reset decay from beginning without altering episode tracking
            if epsilon == epsilon_end:
                min_epsilon_count += 1
                if min_epsilon_count >= 3:
                    faux_episode = 0
                    min_epsilon_count = 0
                    print(f"\n{"Resetting Epsilon Decay":^80}\n")
            
        interval_solves = 0
        last_checkpoint_time = time.time()
    faux_episode += 1