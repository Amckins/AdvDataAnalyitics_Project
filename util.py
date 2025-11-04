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

def generate_random_scramble(scramble_length=10):
    import random
    
    move_pairs = [("U", "D"), ("R", "L"), ("F", "B")] #don't want to allow U2 to be undone with D2
    all_moves = ["U", "D", "R", "L", "F", "B"]
    modifiers = ["", "'", "2"] #cw, ccw and repeat twice
    modifier_probabilities = [0.4, 0.4, 0.2]
    
    scramble_sequence = ""
    last_move = ""
    move_count = 0
    
    while move_count < scramble_length:
        move = random.choice(all_moves)
        
        #find opposite of LAST move
        opposite = ""
        for pair in move_pairs:
            if last_move in pair:
                opposite = pair[1] if last_move == pair[0] else pair[0]
                break
        
        #skip if same move OR opposite of last move (prevents cube rotation)
        if move == last_move or move == opposite:
            continue
        
        #choose modifier to add if any
        modifier = random.choices(modifiers, weights=modifier_probabilities, k = 1)[0]
        
        #count moves used (2 for double moves)
        moves_to_add = 2 if modifier == "2" else 1
        
        #don't exceed scramble length: UR2 should have the same move lengths as URR
        if move_count + moves_to_add > scramble_length:
            modifier = random.choice(modifiers[:-1])
            moves_to_add = 1
        
        scramble_sequence += move + modifier + " "
        last_move = move
        move_count += moves_to_add
    
    return scramble_sequence[:-1]

def reward_schema(max_depth = 14):
    import pandas as pd

    def get_reward(old_depth, new_depth):
        if new_depth == 0:
            return 100
        elif new_depth < old_depth:
            return 14 - new_depth
        elif new_depth >= old_depth:
            return -(2*new_depth)
        else:
            return -1.0

    depths = list(range(15))
    
    rewards = []

    for i in depths:
        row = []
        for j in depths:
            if i > max_depth:
                row.append(-1)
            else:
                row.append(get_reward(i, j))
        rewards.append(row)

    df = pd.DataFrame(rewards, 
                    index=[f"{i}" for i in depths], 
                    columns=[f"{j}" for j in depths])

    print(df.to_string())
