import numpy as np
from itertools import combinations, product
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, Button, Div, TextInput
from bokeh.layouts import column, row
from bokeh.io import curdoc
from scramble import generate_random_scramble
#labels intended to diagnose issues w/ indexing
use_labels = False

#when scrambling, how many moves should be generated
scramble_length = 20

#shape and formatting parameters
step = 0.4
val = 0.5
cube_size = 2
a, b, c = 0.4, 0, 0
d = a * np.tan(np.deg2rad(60))

rings = [val + i * step for i in range(cube_size)]
lims = 2.5 * max(rings)

x_centers = [a, -a, c]
y_centers = [b, -b, d]
x_shift = np.mean(x_centers)
y_shift = np.mean(y_centers)

centers = [
    (a - x_shift, b - y_shift),
    (-a - x_shift, -b - y_shift),
    (c - x_shift, d - y_shift)]

colors = ["#ffffff", "#ffff00",
          "#ff6400", "#ff0000",
          "#0000bb", "#00bb00"]

#Between any two circles, find the coordinates where they intersect
#For general cases in this use case, there should always be 2 points
def circle_intersections(x0, y0, r0, x1, y1, r1):
    """Derived from https://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return []
    a = (r0**2 - r1**2 + d**2) / (2*d)
    h = np.sqrt(max(r0**2 - a**2, 0))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    xs1 = xm + h * dy / d
    ys1 = ym - h * dx / d
    xs2 = xm - h * dy / d
    ys2 = ym + h * dx / d
    return [(xs1, ys1), (xs2, ys2)]

#create the graph portion of the display
p = figure(width = 450,
           height = 450,
           match_aspect = True,
           x_range = (-lims, lims),
           y_range = (-lims, lims))
p.axis.visible = False
p.grid.visible = False

#for each center create rings around them. their intersections will form the faces
for (cx, cy) in centers:
    for r in rings:
        theta = np.linspace(0, 2 * np.pi, 100)
        p.line(cx + np.sqrt(r) * np.cos(theta),
               cy + np.sqrt(r) * np.sin(theta),
               line_color= "black",
               line_width = 1)
        
x_vals, y_vals, color_vals = [], [], []
dot_colors = []  # 1D vector of colors for each dot

#initialize colors
group_index = 0
for (c1, c2) in combinations(centers, 2):
    for (r_sq1, r_sq2) in product(rings, rings):
        r1, r2 = np.sqrt(r_sq1), np.sqrt(r_sq2)
        pts = circle_intersections(c1[0], c1[1], r1, c2[0], c2[1], r2)
        if len(pts) == 2:
            color1 = colors[group_index % len(colors)]
            color2 = colors[(group_index + 1) % len(colors)]
            
            for i, (x, y) in enumerate(pts):
                x_vals.append(x)
                y_vals.append(y)
                hex_color = color1 if i == 0 else color2
                color_vals.append(hex_color)
                dot_colors.append(hex_color)

    group_index += 2

#create lists of indices that correspond to the faces/rings 
#when a ring 'turns' the corresponding face will need to turn as well
FACES = {
    "U" : [4, 6, 0, 2],
    "D" : [1, 3, 5, 7],
    "B" : [8, 12, 10, 14],
    "F" : [9, 11, 13, 15],
    "R" : [16, 20, 18, 22],
    "L" : [17, 19, 21, 23]    
} 
RINGS = {
    "Ur" : [12, 8, 20, 16, 9, 13, 17, 21],
    "Dr" : [23, 19, 15, 11, 18, 22, 10, 14],
    "Br" : [7, 3, 22, 20, 2, 6, 21, 23],
    "Fr" : [19, 17, 4, 0, 16, 18, 1, 5],
    "Rr" : [3, 1, 11, 9, 0, 2, 8, 10],
    "Lr" : [14, 12, 6, 4, 13, 15, 5, 7]
}
def rotate_face(face_move):
    #Read which face is being turned and which direction should be turned
    face_name = face_move.rstrip("'")
    clockwise = not face_move.endswith("'")
    if (face_name + "r") in RINGS and face_name in FACES:
        # Rotate the ring
        dot_ids = RINGS[face_name + "r"]
        current_colors = [dot_colors[dot_id] for dot_id in dot_ids]
        
        if clockwise:
            rotated_colors = current_colors[-cube_size:] + current_colors[:-cube_size]
        else:
            rotated_colors = current_colors[cube_size:] + current_colors[:cube_size]
        
        for dot_id, color in zip(dot_ids, rotated_colors):
            dot_colors[dot_id] = color

        # Rotate the face (2x2 grid)
        dot_ids = FACES[face_name]
        current_colors = [dot_colors[dot_id] for dot_id in dot_ids]
        n = len(current_colors)
        
        rotated_colors = [0] * n
        for i in range(n):
            row = i // cube_size
            col = i % cube_size
            
            if clockwise:
                # Clockwise: (row, col) -> (col, size-1-row)
                new_row = col
                new_col = cube_size - 1 - row
            else:
                # Counter-clockwise: (row, col) -> (size-1-col, row)
                new_row = cube_size - 1 - col
                new_col = row
            
            new_idx = new_row * cube_size + new_col
            rotated_colors[new_idx] = current_colors[i]
        
        for dot_id, color in zip(dot_ids, rotated_colors):
            dot_colors[dot_id] = color
    else:
        raise ValueError(f"{face_name} is not a valid Face Name!")
    
    # Update the data source
    source.data = dict(x = x_vals,
                      y = y_vals,
                      color = dot_colors.copy(),
                      dot_id = [str(i) for i in range(len(x_vals))])

def parse_move_sequence(sequence):
    """when setting up the messed up state and when solving it, a sequence is needed"""
    sequence = sequence.upper().replace(" ", "")
    moves = []
    i = 0
    while i < len(sequence):
        if sequence[i] in FACES:
            move = sequence[i]
            i += 1
            
            # Check for a digit
            if i < len(sequence) and sequence[i].isdigit():
                repeat = int(sequence[i])
                i += 1
            else:
                repeat = 1
            
            # Check for prime
            if i < len(sequence) and sequence[i] == "'":
                move += "'"
                i += 1
            
            # Add the move 'repeat' times
            moves.extend([move] * repeat)
        else:
            i += 1
    
    return moves

def execute_sequence():
    """execute a sequence of moves from the text input"""
    sequence = text_input.value
    moves = parse_move_sequence(sequence)
    for move in moves:
        rotate_face(move)
    
    #clear the text field
    text_input.value = ""

def execute_scramble():
    """execute a sequence of moves from the text input"""
    sequence = generate_random_scramble(scramble_length = scramble_length)
    moves = parse_move_sequence(sequence)
    for move in moves:
        rotate_face(move)

#draw dots
source = ColumnDataSource(data = dict(x = x_vals,
                                      y = y_vals,
                                      color = color_vals,
                                      dot_id = [str(i) for i in range(len(x_vals))]))
p.circle(x = "x",
         y = "y",
         color = "color",
         size = 20,
         line_color = "black",
         source = source)

# Add labels with dot numbers
if use_labels:
    labels = LabelSet(x = "x",
                      y = "y",
                      text = "dot_id",
                      source=source,
                      x_offset=-5,
                      y_offset=-4,
                      text_font_size="10pt",
                      text_align="center",
                      text_baseline="middle")
    p.add_layout(labels)

# Create text input for move sequences
text_input = TextInput(
                       value="", 
                       width=400)

# Create execute button
execute_button = Button(label="Execute Sequence:", button_type="success", width=150)
execute_button.on_click(execute_sequence)

# Create buttons for each different move
button_U = Button(label = "U", button_type = "default", width = 50)
button_U_prime = Button(label = "U'", button_type = "default", width = 50)
button_D = Button(label = "D", button_type = "default", width = 50)
button_D_prime = Button(label = "D'", button_type = "default", width = 50)
button_B = Button(label = "B", button_type = "default", width = 50)
button_B_prime = Button(label = "B'", button_type = "default", width = 50)
button_F = Button(label = "F", button_type = "default", width = 50)
button_F_prime = Button(label = "F'", button_type = "default", width = 50)
button_R = Button(label = "R", button_type = "default", width = 50)
button_R_prime = Button(label = "R'", button_type = "default", width = 50)
button_L = Button(label = "L", button_type = "default", width = 50)
button_L_prime = Button(label = "L'", button_type = "default", width = 50)

# create functionality for those buttons
button_U.on_click(lambda: rotate_face("U"))
button_U_prime.on_click(lambda: rotate_face("U'"))
button_D.on_click(lambda: rotate_face("D"))
button_D_prime.on_click(lambda: rotate_face("D'"))
button_B.on_click(lambda: rotate_face("B"))
button_B_prime.on_click(lambda: rotate_face("B'"))
button_F.on_click(lambda: rotate_face("F"))
button_F_prime.on_click(lambda: rotate_face("F'"))
button_R.on_click(lambda: rotate_face("R"))
button_R_prime.on_click(lambda: rotate_face("R'"))
button_L.on_click(lambda: rotate_face("L"))
button_L_prime.on_click(lambda: rotate_face("L'"))

#create scramble button
button_scramble = Button(label = "Scramble", button_type = "warning", width = 110)
button_scramble.on_click(execute_scramble)

#formatting buttons to pack into the display
button_col = column(
    row(button_U, button_U_prime),
    row(button_D, button_D_prime),
    row(button_B, button_B_prime),
    row(button_F, button_F_prime),
    row(button_R, button_R_prime),
    row(button_L, button_L_prime),
    row(button_scramble)
)

# Layout with text input at the top
layout = column(
    row(p, button_col),
    row(execute_button, text_input)
)


curdoc().add_root(layout)