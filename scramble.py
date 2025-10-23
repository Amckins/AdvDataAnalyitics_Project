def generate_random_scramble(scramble_length = 10):
    import random
    valid_moves = ["U", "D", "F", "B", "R", "L"]
    modifiers = ["", "'", "2"]
    modifier_probabilities = [0.4, 0.4, 0.2]
    i =0

    last_move, scramble_sequence = "", ""
    while i < scramble_length:
        # print(last_move)
        modifier = random.choices(modifiers, weights = modifier_probabilities, k = 1)[0]
        move = random.choice([m for m in valid_moves if (m + modifier) != last_move or m != (last_move + modifier)])
        last_move = move

        scramble_sequence += move + modifier + " "

        i += 1
        if modifier.isdigit():
            i += 1

    scramble_sequence = scramble_sequence[:-1]
    # print(scramble_sequence)
    return scramble_sequence

# generate_random_scramble(20)