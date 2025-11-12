import numpy as np
import os
import time
from collections import deque
from cube import MOVES, BASE_STATE
from cube import turn, get_symmetric_states
from util import format_time

def build_transition_matrix(max_depth=None, output_dir='cube_outputs'):
    #Create output directories
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    #Initialize data structures
    state_to_id = {}        #tuple(state) -> state_id
    id_to_state = []        #list of states
    depth_array = []        #depth of each state
    transition_matrix = []  #will be shape (num_states, 12)
    
    #BFS queue: (state, depth)
    queue = deque()
    
    #Initialize with symmetric states at depth 0
    print(f"Starting BFS from 24 initial states at depth 0...")
    if max_depth is not None:
        print(f"Max depth set to: {max_depth}")
    print()
    
    start_time = time.time()
    last_checkpoint_time = start_time
    
    for state in get_symmetric_states(BASE_STATE):
        state_tuple = tuple(state)
        if state_tuple not in state_to_id:
            state_id = len(state_to_id)
            state_to_id[state_tuple] = state_id
            id_to_state.append(list(state))
            depth_array.append(0)
            queue.append((list(state), 0))
            #Placeholder for transitions (will fill during BFS)
            transition_matrix.append([-1] * 12)
    
    current_depth = 0
    states_at_current_depth = len(state_to_id)
    total_states = len(state_to_id)
    progress_counter = 0
    
    #Save depth 0
    checkpoint_time = time.time()
    time_elapsed = format_time(checkpoint_time - last_checkpoint_time)
    print(f"Depth 0 complete: {states_at_current_depth} states | Total: {total_states} states | Time: {time_elapsed}")
    save_checkpoint(checkpoint_dir, 0, id_to_state, depth_array, transition_matrix)
    print(f"Saved: {os.path.join(checkpoint_dir, 'depth_0.npz')}")
    print()
    last_checkpoint_time = checkpoint_time
    
    #BFS
    while queue:
        state, depth = queue.popleft()
        state_tuple = tuple(state)
        state_id = state_to_id[state_tuple]
        
        #Check if we've moved to a new depth
        if depth > current_depth:
            #Save checkpoint for completed depth
            checkpoint_time = time.time()
            time_elapsed = format_time(checkpoint_time - last_checkpoint_time)
            print(f"Depth {current_depth} complete: {states_at_current_depth} states | Total: {total_states} states | Time: {time_elapsed}")
            save_checkpoint(checkpoint_dir, current_depth, id_to_state, depth_array, transition_matrix)
            print(f"Saved: {os.path.join(checkpoint_dir, f'depth_{current_depth}.npz')}")
            print()
            
            current_depth = depth
            states_at_current_depth = 0
            last_checkpoint_time = checkpoint_time
            
            #Check if we've reached max depth
            if max_depth is not None and depth > max_depth:
                break
        
        #Explore all 12 moves
        for move_idx in range(12):
            move = MOVES[move_idx]
            next_state = turn(state, move)
            next_state_tuple = tuple(next_state)
            
            #Check if this is a new state
            if next_state_tuple not in state_to_id:
                next_state_id = len(state_to_id)
                state_to_id[next_state_tuple] = next_state_id
                id_to_state.append(list(next_state))
                depth_array.append(depth + 1)
                queue.append((list(next_state), depth + 1))
                transition_matrix.append([-1] * 12)
                
                states_at_current_depth += 1
                total_states += 1
                progress_counter += 1
                
                #Print progress every 100k states
                if progress_counter >= 100000:
                    interval_time = time.time()
                    time_elapsed = format_time(interval_time - last_checkpoint_time)
                    print(f"Depth {depth + 1}: {total_states:,} states | Time: {time_elapsed}")
                    progress_counter = 0
                    last_checkpoint_time = interval_time
            else:
                next_state_id = state_to_id[next_state_tuple]
            
            #Record transition
            transition_matrix[state_id][move_idx] = next_state_id
    
    #Save final checkpoint for last depth if there were states added
    if states_at_current_depth > 0 or current_depth == 0:
        checkpoint_time = time.time()
        time_elapsed = format_time(checkpoint_time - last_checkpoint_time)
        print(f"Depth {current_depth} complete: {states_at_current_depth} states | Total: {total_states} states | Time: {time_elapsed}")
        save_checkpoint(checkpoint_dir, current_depth, id_to_state, depth_array, transition_matrix)
        print(f"Saved: {os.path.join(checkpoint_dir, f'depth_{current_depth}.npz')}")
        print()
    
    #Convert to numpy arrays
    id_to_state_array = np.array(id_to_state, dtype=np.uint8)
    depths_array = np.array(depth_array, dtype=np.uint8)
    transitions_array = np.array(transition_matrix, dtype=np.int32)
    
    #Solved state ID is the first state (BASE_STATE after symmetry)
    solved_state_id = 0
    
    #Save final consolidated file
    end_time = time.time()
    total_time = format_time(end_time - start_time)
    
    print(f"BFS complete!" + (f" (stopped at max_depth={max_depth})" if max_depth is not None else ""))
    print(f"Total states discovered: {total_states:,}")
    print(f"Total time: {total_time}")
    
    final_filename = os.path.join(npz_dir, f"depth_{max_depth if max_depth is not None else 'None'}.npz")
    np.savez_compressed(
        final_filename,
        transitions=transitions_array,
        moves=MOVES,
        id_to_state=id_to_state_array,
        solved_state_id=np.array([solved_state_id], dtype=np.int32),
        depths=depths_array
    )
    print(f"Final save: {final_filename}")
    
    return {
        'transitions': transitions_array,
        'moves': MOVES,
        'id_to_state': id_to_state_array,
        'solved_state_id': solved_state_id,
        'depths': depths_array,
        'state_to_id': state_to_id
    }

def save_checkpoint(checkpoint_dir, depth, id_to_state, depth_array, transition_matrix):
    """Save checkpoint after completing a depth level."""
    id_to_state_array = np.array(id_to_state, dtype=np.uint8)
    depths_array = np.array(depth_array, dtype=np.uint8)
    transitions_array = np.array(transition_matrix, dtype=np.int32)
    
    #Solved state ID is always 0 (first state)
    solved_state_id = 0
    
    filename = os.path.join(checkpoint_dir, f'depth_{depth}.npz')
    np.savez_compressed(
        filename,
        transitions=transitions_array,
        moves=MOVES,
        id_to_state=id_to_state_array,
        solved_state_id=np.array([solved_state_id], dtype=np.int32),
        depths=depths_array
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting transition matrix generation...")
    print("="*60 + "\n")
    
    max_depth = None  
    output_dir = 'cube_output'
    
    result = build_transition_matrix(max_depth=max_depth, output_dir=output_dir)
    
    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}")