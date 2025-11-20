import numpy as np
from cube import MOVES
import os
matrix_file = 'cube_output/npz/depth_None.npz'
q_table_file = 'q_table.npz'

print("Loading in Data...")
data = np.load(matrix_file)
transition = data['transitions']
id_to_state = data['id_to_state']
depth = data['depths']
num_actions = len(MOVES)
print("Loaded in Transition Matrix file")

print("Loading in Q-Table")
if os.path.exists(q_table_file):
    data = np.load(q_table_file)
    Q = data['Q']
    print("Loaded Q-Table")
else:
    print("Q Table File Not Found")
    break_glass = True

def state_to_id(target_state, id_to_state = id_to_state):
    target = np.array(target_state)
    
    # Compare all states at once
    matches = np.all(id_to_state == target, axis=1)
    indices = np.where(matches)[0]
    
    if len(indices) > 0:
        return indices[0]
    return None

def condense_move_str(move_str):
    move_list = move_str.split()
    condensed_str = ''
    i = 0

    while i < len(move_list):
        # Count how many times the same move repeats in a row
        count = 1
        while i + count < len(move_list) and move_list[i] == move_list[i + count]:
            count += 1

        # Condense based on count
        if count == 1:
            condensed_str += move_list[i] + " "
        elif count == 2:
            condensed_str += move_list[i] + "2 "
        else:
            condensed_str += move_list[i] + str(count) + " "

        i += count
    return condensed_str

def get_valid_actions(last_move):
    valid_actions = list(range(num_actions))
    invalid_move_pairs = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4,
                      6: 7, 7: 6, 8: 9, 9: 8, 10: 11, 11: 10}
    if last_move is not None and last_move in invalid_move_pairs:
        invalid_action = invalid_move_pairs[last_move]
        valid_actions.remove(invalid_action)
    return valid_actions

def get_solve_str(state_vector, break_glass = False):
    print("Generating Solve String")
    try:
        state_id = state_to_id(state_vector)
        current_depth = int(depth[state_id])
        solved = False
        move_str = ''
        last_move = None

        if break_glass:
            while not solved:
                for action_id, move in enumerate(MOVES):
                    next_state_id = transition[state_id, action_id]
                    next_depth = int(depth[next_state_id])
                    if next_depth == 0:
                        solved = True
                        move_str += move + " "
                        break
                    elif next_depth - current_depth < 0:
                        state_id = next_state_id
                        current_depth = next_depth
                        move_str += move + " "
                        break        
        else:
            print("This has not been implemented yet")
            pass
            fail_safe_counter = 0
            while not solved:
                action_id = np.argmax(Q[state_id, get_valid_actions(last_move)])
                next_state_id = transition[state_id, action_id]
                move_str += MOVES[action_id] + " "

                if depth[state_id] == 0 or fail_safe_counter >= 50:
                    solved = True
                fail_safe_counter += 1
                
        return condense_move_str(move_str)
    except:
        response_str = 'INVALID | '
        color_count_wrong = False
        color_dict = {1:"W", 
                2: "Y",  
                3: "O",  
                4: "R",  
                5: "B",  
                6: "G"}  
        color_count = []
        for i in range(1,7):
            color_count.append(state_vector.count(i))
        for i, count in enumerate(color_count):
            if count > 4:
                response_str += f"{color_dict[i + 1]}: +{count - 4} "
                color_count_wrong = True
            elif count < 4:
                response_str += f"{color_dict[i + 1]}: -{4 - count} "
                color_count_wrong = True
        if not color_count_wrong:
            response_str += "Check Orientations"
        return response_str

