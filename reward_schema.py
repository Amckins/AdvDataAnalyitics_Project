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
max_depth = 14
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
