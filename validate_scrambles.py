import numpy as np
from scramble import generate_random_scramble
import math
import random

depth = None
npz_file = f"cube_output/npz/transitions_depth_{depth}.npz"
num_scrambles = 100000
freq_of_outputs = int(.05 * 10 **int(math.log10(num_scrambles)))

FACES = {"U": [4, 6, 2, 0], 
         "D": [5, 1, 3, 7],
         "B": [8, 12, 14, 10],
         "F": [13, 9, 11, 15],
         "R": [16, 20, 22, 18],
         "L": [23, 21, 17, 19]} 
RINGS = {"U": [12, 8, 20, 16, 9, 13, 17, 21],
         "D": [23, 19, 15, 11, 18, 22, 10, 14],
         "B": [7, 3, 22, 20, 2, 6, 21, 23],
         "F": [19, 17, 4, 0, 16, 18, 1, 5],
         "R": [3, 1, 11, 9, 0, 2, 8, 10],
         "L": [14, 12, 6, 4, 13, 15, 5, 7]}

def expand_moves(scramble_str):
    """Convert 'R2 U2 F' to 'R R U U F'"""
    moves = scramble_str.strip().split()
    expanded = []
    for move in moves:
        if move.endswith('2'):
            base = move[:-1]
            expanded.extend([base, base])
        else:
            expanded.append(move)
    return expanded

def turn(state, move):
    new_state = state.copy()
    face_name = move.rstrip("'2")
    clockwise = not move.endswith("'")
    
    dot_ids = RINGS[face_name]
    current = [new_state[i] for i in dot_ids]
    rotated = current[-2:] + current[:-2] if clockwise else current[2:] + current[:2]
    for i, color in zip(dot_ids, rotated):
        new_state[i] = color
    
    dot_ids = FACES[face_name]
    current = [new_state[i] for i in dot_ids]
    rotated = current[-1:] + current[:-1] if clockwise else current[1:] + current[:1]
    for i, color in zip(dot_ids, rotated):
        new_state[i] = color
    
    return new_state

# Load
print(f"Loading Transition Matrix from {npz_file}")
data = np.load(npz_file)
print(f"\nLoaded Transition Matrix")
transitions = data['transitions']
moves = list(data['moves'])
id_to_state = data['id_to_state']
solved_id = data['solved_state_ids'][0]

move_to_id = {m: i for i, m in enumerate(moves)}
UNKNOWN = np.iinfo(np.uint32).max

print(f"Testing {num_scrambles} scrambles at depth ≤ {depth}...")

matches = 0
mismatches = 0
unknown = 0
too_long = 0
unknown_scrambles = []

for i in range(num_scrambles):
    if depth is None:
        len = random.randint(14,30)
        scramble = expand_moves(generate_random_scramble(len))
    else:
        scramble = expand_moves(generate_random_scramble(depth))
    
    # Skip if expanded scramble exceeds matrix depth
    if depth is not None:
        if len(scramble) > depth:
            too_long += 1
            continue
    
    # Via matrix - track states
    matrix_states = [solved_id]
    state_id = solved_id
    unknown_hit = False
    for move in scramble:
        state_id = transitions[state_id, move_to_id[move]]
        if state_id == UNKNOWN:
            unknown_hit = True
            break
        matrix_states.append(state_id)
    
    if unknown_hit:
        unknown += 1
        unknown_scrambles.append(' '.join(scramble))
        continue
    
    matrix_result = id_to_state[state_id]
    
    # Via turn - track states
    direct_states = [id_to_state[solved_id].copy()]
    state = id_to_state[solved_id].copy()
    for move in scramble:
        state = turn(state, move)
        direct_states.append(state.copy())
    
    if np.array_equal(matrix_result, state):
        matches += 1
    else:
        mismatches += 1
        print(f"\n{'='*70}")
        print(f"MISMATCH #{mismatches} on scramble {i+1}: {' '.join(scramble)}")
        print(f"{'='*70}")
        print(f"Start (solved): State ID {solved_id}")
        
        for step, move in enumerate(scramble, 1):
            matrix_state = id_to_state[matrix_states[step]]
            direct_state = direct_states[step]
            
            match = "✓" if np.array_equal(matrix_state, direct_state) else "✗"
            print(f"\nStep {step} - Move '{move}': {match}")
            print(f"  Matrix: State ID {matrix_states[step]} -> {list(matrix_state)}")
            print(f"  Direct:                      -> {list(direct_state)}")
        print(f"{'='*70}\n")
    
    if (i + 1) % freq_of_outputs == 0:
        tested = matches + mismatches + unknown
        print(f"  Progress: {i+1}/{num_scrambles} (Tested: {tested}, Matches: {matches}, Mismatches: {mismatches}, Unknown: {unknown}, Too long: {too_long})")

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Total generated: {num_scrambles}")
print(f"  Too long after expansion: {too_long}")
print(f"  Tested: {matches + mismatches + unknown}")
print(f"    Matches: {matches}")
print(f"    Mismatches: {mismatches}")
print(f"    Unknown states: {unknown}")

if unknown_scrambles:
    print(f"\n{'='*70}")
    print(f"SEQUENCES HITTING UNKNOWN STATES ({len(unknown_scrambles)} total):")
    print(f"{'='*70}")
    for seq in unknown_scrambles[:20]:  # Show first 20
        print(f"  {seq}")
    if len(unknown_scrambles) > 20:
        print(f"  ... and {len(unknown_scrambles) - 20} more")

print("\n" + ("✓ PASSED" if mismatches == 0 and unknown == 0 else f"✗ FAILED - {mismatches} mismatches, {unknown} unknown states"))