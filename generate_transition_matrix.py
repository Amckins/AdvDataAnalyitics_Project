import numpy as np
import os
import time
import csv
from cube import FACES, RINGS
from cube import turn, rotate_cube
from cube import get_symmetric_states, get_opposite_move
from util import format_time

def normalize_state(state):
    orientations = get_symmetric_states(state)              #find all symmetrical states
    orientation_tuples = [tuple(o) for o in orientations]   
    unique = min(orientation_tuples)                        #find the lowest found state id
    return np.array(unique)

def state_to_unique(state):
    return normalize_state(state)

def validate_matrix(transitions, moves, id_to_state, freq_of_updates = 100000):
    #Verify that every transition matches turn() output
    print("\nValidating transition matrix...")
    UNKNOWN = np.iinfo(np.uint32).max
    errors = 0
    
    for state_id in range(len(id_to_state)):                    #Look at every single state discovered
        if state_id % freq_of_updates == 0 and state_id > 0:    #periodic updating
            print(f"  Validated {state_id:,} states...")
        
        current_state = id_to_state[state_id]
        
        for move_id, move in enumerate(moves):                  #for every possible move
            next_state_id = transitions[state_id, move_id]      #what does the matrix say is the next state?
            
            if next_state_id == UNKNOWN:                        #pass over if max_depth != None
                continue
            
            expected_raw = turn(current_state, move)            #what does the turn look like regularly?
            expected = normalize_state(expected_raw)            #find the unique version of this new state
            actual = id_to_state[next_state_id]                 #convert state into id
            
            if not np.array_equal(expected, actual):            #does the expected match matrix?
                errors += 1
                if errors <= 5:
                    print(f"ERROR: State {state_id}, Move {move}")
    
    if errors == 0:
        print("✓ Validation passed - all transitions correct!")
    else:
        print(f"✗ Validation failed - {errors} errors found!")
        raise ValueError("Matrix validation failed")

def generate_transition_matrix(output_dir='output', max_moves=None, resume_from=None):
    moves = ["U", "U'", "D", "D'", "B", "B'",
             "F", "F'", "R", "R'", "L", "L'"]
    
    #Create output structure
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    #Create CSV to log progress
    csv_file = os.path.join(output_dir, 'progress_log.csv')

    if resume_from:
        print(f"Resuming from: {resume_from}")
        data = np.load(resume_from)
        
        id_to_state = data['id_to_state'].tolist()
        solved_state_id = int(data['solved_state_id'][0])
        depths = data['depths'].tolist()
        depth = int(data['depth'][0])
        start_time = time.time() - float(data['elapsed_time'][0])
        
        #Reconstruct state_to_id from id_to_state
        state_to_id = {}
        for state_id, state in enumerate(id_to_state):
            state_to_id[tuple(state)] = state_id
        
        #Reconstruct current_level from arrays
        current_level_ids = data['current_level_ids']
        current_level_moves = data['current_level_moves']
        current_level = []
        for state_id, move_idx in zip(current_level_ids, current_level_moves):
            move = moves[move_idx] if move_idx >= 0 else None
            current_level.append((int(state_id), move))
    else:
        #Start with ONLY ONE unique solved state
        base_solved = np.array([1, 2, 1, 2, 1, 2, 1, 2,
                                3, 4, 3, 4, 3, 4, 3, 4,
                                5, 6, 5, 6, 5, 6, 5, 6])
        
        #Normalize to get the unique form
        unique_solved = normalize_state(base_solved)
        
        state_to_id = {}
        id_to_state = []
        
        state_tuple = tuple(unique_solved)
        state_id = 0
        state_to_id[state_tuple] = state_id
        id_to_state.append(unique_solved)
        solved_state_id = state_id
        depths = [0]
        
        current_level = [(state_id, None)]
        depth = 0
        start_time = time.time()

        #Start the CSV to log data
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['depth', 'states_discovered', 'time_elapsed'])

    print(f"Starting from depth {depth} with {len(state_to_id):,} states")
    
    while current_level:
        print(f"\nDepth {depth}: {len(state_to_id):,} total states, {len(current_level):,} at this level")
        
        # Save checkpoint (only if doesn't exist)
        checkpoint_file = os.path.join(checkpoint_dir, f'depth_{depth}.npz')
        if not os.path.exists(checkpoint_file):
            # Convert current_level to arrays
            current_level_ids = np.array([state_id for state_id, _ in current_level], dtype=np.uint32)
            current_level_moves = np.array([moves.index(move) if move else -1 for _, move in current_level], dtype=np.int8)
            
            np.savez_compressed(checkpoint_file,
                                id_to_state=np.array(id_to_state, dtype=np.uint8),
                                solved_state_id=np.array([solved_state_id], dtype=np.uint32),
                                current_level_ids=current_level_ids,
                                current_level_moves=current_level_moves,
                                depth=np.array([depth], dtype=np.uint32),
                                depths=np.array(depths, dtype=np.uint8),
                                elapsed_time=np.array([time.time() - start_time], dtype=np.float64))
        
        if max_moves is not None and depth >= max_moves:
            break
        
        next_level = []
        for state_id, last_move in current_level:
            current_state = id_to_state[state_id]
            
            for move in moves:
                if last_move is not None:
                    opposite = get_opposite_move(last_move)
                    if move == opposite:  #Only skip opposite moves
                        continue
                
                #Apply move
                next_state_raw = turn(current_state, move)
                
                #Normalize to unique form
                next_state = normalize_state(next_state_raw)
                next_state_tuple = tuple(next_state)
                
                if next_state_tuple not in state_to_id:             #if we havent encountered this state yet
                    new_state_id = len(state_to_id)                 #append it to the end of what we have found thus far
                    state_to_id[next_state_tuple] = new_state_id
                    id_to_state.append(next_state)
                    depths.append(depth + 1)                        
                    next_level.append((new_state_id, move))
                    
                    #periodic updating
                    current_count = len(state_to_id)
                    if current_count % 100000 == 0: 
                        elapsed_time = time.time() - start_time
                        print(f"\tProgress: {current_count:,} states discovered, {format_time(elapsed_time)} elapsed")

                        with open(csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([depth, current_count, elapsed_time])
        
        current_level = next_level
        current_count = len(state_to_id)
        elapsed_time = time.time() - start_time
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([depth, current_count, elapsed_time])
        depth += 1
    
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Total unique states discovered: {len(state_to_id):,}")
    print(f"Time elapsed: {format_time(elapsed_time)}")
    print(f"Expected: ~3,674,160 states for 2x2x2 cube")
    print(f"{'='*60}")
    
    # Build transition matrix
    print("\nBuilding transition matrix...")
    UNKNOWN = np.iinfo(np.uint32).max
    transitions = np.full((len(state_to_id), 12), UNKNOWN, dtype=np.uint32)
    
    for state_id, state in enumerate(id_to_state):
        if state_id % 100000 == 0 and state_id > 0:
            print(f"  Built transitions for {state_id:,} states...")
            
        for move_id, move in enumerate(moves):
            next_state_raw = turn(state, move)
            next_state = normalize_state(next_state_raw)  # Normalize
            next_state_tuple = tuple(next_state)
            
            if next_state_tuple in state_to_id:
                transitions[state_id, move_id] = state_to_id[next_state_tuple]
    
    # VALIDATE
    validate_matrix(transitions, moves, id_to_state)
    
    # Save
    filename = os.path.join(npz_dir, f'transitions_depth_{max_moves}.npz')
    print(f"\nSaving to {filename}...")
    
    np.savez_compressed(filename,
                        transitions=transitions,
                        moves=moves,
                        id_to_state=np.array(id_to_state, dtype=np.uint8),
                        solved_state_id=np.array([solved_state_id], dtype=np.uint32),
                        depths=np.array(depths, dtype=np.uint8))
    
    print(f"Done!")
    return filename


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("Starting state space generation...")
    print("="*60)
    
    output_dir = 'cube_output'
    max_moves = 6
    resume_from = None  
    
    generate_transition_matrix(output_dir, max_moves, resume_from)