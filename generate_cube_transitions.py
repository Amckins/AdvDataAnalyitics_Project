import numpy as np
import os
import time
import csv
import pickle
import math

FACES = {
    "U": [4, 6, 2, 0],
    "D": [5, 1, 3, 7],
    "B": [8, 12, 14, 10],
    "F": [13, 9, 11, 15],
    "R": [16, 20, 22, 18],
    "L": [23, 21, 17, 19]    
} 

RINGS = {
    "U": [12, 8, 20, 16, 9, 13, 17, 21],
    "D": [23, 19, 15, 11, 18, 22, 10, 14],
    "B": [7, 3, 22, 20, 2, 6, 21, 23],
    "F": [19, 17, 4, 0, 16, 18, 1, 5],
    "R": [3, 1, 11, 9, 0, 2, 8, 10],
    "L": [14, 12, 6, 4, 13, 15, 5, 7]
}
def format_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    time_str = ""
    if hours > 0:
        time_str += f"{hours}h "
    if minutes > 0:
        time_str += f"{minutes}m "
    if seconds > 0:
        time_str += f"{seconds:.1f}s"
    return time_str
def turn(state, move):
    new_state = state.copy()
    face_name = move.rstrip("'2")
    clockwise = not move.endswith("'")
    cube_size = 2
    
    # Rotate the ring
    dot_ids = RINGS[face_name]
    current_colors = [new_state[dot_id] for dot_id in dot_ids]
    
    if clockwise:
        rotated_colors = current_colors[-cube_size:] + current_colors[:-cube_size]
    else:
        rotated_colors = current_colors[cube_size:] + current_colors[:cube_size]
    
    for dot_id, color in zip(dot_ids, rotated_colors):
        new_state[dot_id] = color

    # Rotate the face
    dot_ids = FACES[face_name]
    current_colors = [new_state[dot_id] for dot_id in dot_ids]
    
    if clockwise:
        rotated_colors = current_colors[-1:] + current_colors[:-1]
    else:
        rotated_colors = current_colors[1:] + current_colors[:1]
    
    for dot_id, color in zip(dot_ids, rotated_colors):
        new_state[dot_id] = color
    
    return new_state

def rotate_cube(state, rotation):
    rotation_map = {
        "x": ["R", "L'"], "x'": ["R'", "L"],
        "y": ["U", "D'"], "y'": ["U'", "D"],
        "z": ["F", "B'"], "z'": ["F'", "B"]
    }
    moves = rotation_map[rotation]
    new_state = turn(state, moves[0])
    new_state = turn(new_state, moves[1])
    return new_state

def generate_all_solved_orientations(base_solved):
    orientations = []
    current = base_solved.copy()
    
    for _ in range(4):
        state_top = current.copy()
        orientations.append(state_top.copy())
        
        state_top = rotate_cube(state_top, "x")
        orientations.append(state_top.copy())
        
        state_top = rotate_cube(state_top, "x")
        orientations.append(state_top.copy())
        
        state_top = rotate_cube(state_top, "x")
        orientations.append(state_top.copy())
        
        state_top = current.copy()
        state_top = rotate_cube(state_top, "z")
        orientations.append(state_top.copy())
        
        state_top = rotate_cube(state_top, "z")
        state_top = rotate_cube(state_top, "z")
        orientations.append(state_top.copy())
        
        current = rotate_cube(current, "y")
    
    return orientations

def get_opposite_move(move):
    if move.endswith("'"):
        return move[:-1]
    else:
        return move + "'"

def validate_matrix(transitions, moves, id_to_state, freq_of_updates = 100000):
    ###Verify that every transition matches turn() output###
    print("\nValidating transition matrix...")
    UNKNOWN = np.iinfo(np.uint32).max
    errors = 0
    
    for state_id in range(len(id_to_state)):
        if state_id % freq_of_updates == 0 and state_id > 0:
            print(f"  Validated {state_id:,} states...")
        
        current_state = id_to_state[state_id]
        
        for move_id, move in enumerate(moves):
            next_state_id = transitions[state_id, move_id]
            
            if next_state_id == UNKNOWN:
                continue
            
            expected = turn(current_state, move)
            actual = id_to_state[next_state_id]
            
            if not np.array_equal(expected, actual):
                errors += 1
                if errors <= 5:
                    print(f"ERROR: State {state_id}, Move {move}")
    
    if errors == 0:
        print(" Validation passed - all transitions correct!")
    else:
        print(f" Validation failed - {errors} errors found!")
        raise ValueError("Matrix validation failed")

def generate_transition_matrix(output_dir='output', max_moves=None, resume_from=None):
    moves = ["U", "U'", "D", "D'", "B", "B'",
             "F", "F'", "R", "R'", "L", "L'"]
    
    # Create output structure
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    npz_dir = os.path.join(output_dir, 'npz')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    # Create CSV to log progress
    csv_file = os.path.join(output_dir, 'progress_log.csv')
    csv_exists = os.path.exists(csv_file)

    if resume_from:
        print(f"Resuming from: {resume_from}")
        with open(resume_from, 'rb') as f:
            data = pickle.load(f)
        state_to_id = data['state_to_id']
        id_to_state = data['id_to_state']
        solved_state_ids = data['solved_state_ids']
        current_level = data['current_level']
        depth = data['depth']
        start_time = time.time() - data['elapsed_time']
        last_logged_count = (len(state_to_id) // 100000) * 100000
    else:
        base_solved = [1, 2, 1, 2, 1, 2, 1, 2,
                       3, 4, 3, 4, 3, 4, 3, 4,
                       5, 6, 5, 6, 5, 6, 5, 6]
        
        solved_orientations = generate_all_solved_orientations(base_solved)
        state_to_id = {}
        id_to_state = []
        solved_state_ids = []
        
        for orientation in solved_orientations:
            state_tuple = tuple(orientation)
            if state_tuple not in state_to_id:
                state_id = len(state_to_id)
                state_to_id[state_tuple] = state_id
                id_to_state.append(orientation)
                solved_state_ids.append(state_id)
        
        current_level = [(state_id, None) for state_id in solved_state_ids]
        depth = 0
        start_time = time.time()
        last_logged_count = 0

        #start the csv to log data
        with open(csv_file, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow(['depth', 'states_discovered', 'time_elapsed'])

    print(f"Starting from depth {depth} with {len(state_to_id):,} states")
    
    while current_level:
        print(f"\nDepth {depth}: {len(state_to_id):,} total states, {len(current_level):,} at this level")
        
        # Save checkpoint (only if doesn't exist)
        checkpoint_file = os.path.join(checkpoint_dir, f'depth_{depth}.pkl')
        if not os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'state_to_id': state_to_id,
                    'id_to_state': id_to_state,
                    'solved_state_ids': solved_state_ids,
                    'current_level': current_level,
                    'depth': depth,
                    'elapsed_time': time.time() - start_time
                }, f)
        
        if max_moves is not None and depth >= max_moves:
            break
        
        next_level = []
        for state_id, last_move in current_level:
            current_state = id_to_state[state_id]
            
            for move in moves:
                if last_move is not None:
                    opposite = get_opposite_move(last_move)
                    if move == opposite:  # Only skip opposite moves
                        continue
                
                next_state = turn(current_state, move)
                next_state_tuple = tuple(next_state)
                
                if next_state_tuple not in state_to_id:
                    new_state_id = len(state_to_id)
                    state_to_id[next_state_tuple] = new_state_id
                    id_to_state.append(next_state)
                    next_level.append((new_state_id, move))
                    
                    current_count = len(state_to_id)
                    if current_count % 100000 == 0:
                        elapsed_time = time.time() - start_time
                        print(f" Progress: {current_count:,} states discovered, {format_time(elapsed_time)} elapsed")

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
    print(f"\nTotal states discovered: {len(state_to_id):,}, {format_time(elapsed_time)} elapsed")
    
    # Build transition matrix
    print("\nBuilding transition matrix...")
    UNKNOWN = np.iinfo(np.uint32).max
    transitions = np.full((len(state_to_id), 12), UNKNOWN, dtype=np.uint32)
    
    for state_id, state in enumerate(id_to_state):
        for move_id, move in enumerate(moves):
            next_state = turn(state, move)
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
                        solved_state_ids=np.array(solved_state_ids, dtype=np.uint32))
    
    print(f"Done! Size: {os.path.getsize(filename) / (1024**2):.1f} MB")
    return filename

if __name__ == "__main__":
    
    output_dir = 'cube_output'
    max_moves = None  
    resume_from = None  
    
    generate_transition_matrix(output_dir, max_moves, resume_from)