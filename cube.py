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

def get_symmetric_states(base_state):
    orientations = []
    current = base_state.copy()
    
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