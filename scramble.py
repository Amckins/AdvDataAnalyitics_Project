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

# for i in range(20):
#     print(generate_random_scramble()) 