import numpy as np
from itertools import combinations, product
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet, Button, TextInput, CustomJS
from bokeh.layouts import column, row
from bokeh.io import curdoc
from cube import FACES, RINGS
from util import generate_random_scramble
from solver import get_solve_str

#configuration
use_labels = True
view_scramble_progress = True
diagnose_state = False
scramble_length = 30
animation_delay = 300
grid_visible = [False]
break_glass = False

#circle graph display parameters
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
    ( a - x_shift,  b - y_shift),
    (-a - x_shift, -b - y_shift),
    ( c - x_shift,  d - y_shift)]

color_dict = {"#ffffff": 1, 
              "#ffed00": 2,  
              "#ff5900": 3,  
              "#c41e3a": 4,  
              "#0051ba": 5,  
              "#009b48": 6}  
color_list = list(color_dict.keys())

def circle_intersections(x0, y0, r0, x1, y1, r1):
    """Derived from https://mathworld.wolfram.com/Circle-CircleIntersection.html"""
    dx, dy = x1 - x0, y1 - y0
    d = np.hypot(dx, dy)
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0:
        return []
    a_val = (r0**2 - r1**2 + d**2) / (2*d)
    h = np.sqrt(max(r0**2 - a_val**2, 0))
    xm = x0 + a_val * dx / d
    ym = y0 + a_val * dy / d
    xs1 = xm + h * dy / d
    ys1 = ym - h * dx / d
    xs2 = xm - h * dy / d
    ys2 = ym + h * dx / d
    return [(xs1, ys1), (xs2, ys2)]

p_circle = figure(width=450, height=450, match_aspect=True,
                x_range=(-lims, lims), y_range=(-lims, lims))
p_circle.axis.visible = False
p_circle.grid.visible = False
p_circle.outline_line_color = None
p_circle.toolbar_location = None

for (cx, cy) in centers:
    for r in rings:
        theta = np.linspace(0, 2 * np.pi, 100)
        p_circle.line(cx + np.sqrt(r) * np.cos(theta),
                    cy + np.sqrt(r) * np.sin(theta),
                    line_color="black", line_width=3)

x_vals, y_vals, color_vals = [], [], []
dot_colors = []

group_index = 0
for (c1, c2) in combinations(centers, 2):
    for (r_sq1, r_sq2) in product(rings, rings):
        r1, r2 = np.sqrt(r_sq1), np.sqrt(r_sq2)
        pts = circle_intersections(c1[0], c1[1], r1, c2[0], c2[1], r2)
        if len(pts) == 2:
            color1 = color_list[group_index % len(color_list)]
            color2 = color_list[(group_index + 1) % len(color_list)]
            
            for i, (x, y) in enumerate(pts):
                x_vals.append(x)
                y_vals.append(y)
                hex_color = color1 if i == 0 else color2
                color_vals.append(hex_color)
                dot_colors.append(hex_color)
    group_index += 2

source_circle = ColumnDataSource(data=dict(x = x_vals,
                                           y = y_vals,
                                           color = dot_colors.copy(),
                                           dot_id = [str(i) for i in range(len(x_vals))]))
p_circle.circle(x = "x",
                y = "y",
                color = "color",
                size = 20,
                line_color = "black",
                line_width = 2,
                source = source_circle)

if use_labels:
    labels = LabelSet(x = "x",
                      y = "y",
                      text = "dot_id",
                      source = source_circle,
                      x_offset = -5,
                      y_offset = -4,
                      text_font_size = "10pt",
                      text_align = "center",
                      text_baseline = "middle")
    p_circle.add_layout(labels)

#Mapping from circle IDs to square IDs
matching_ids = {0:2, 1:19, 2:3, 3:18, 
                4:0, 5:17, 6:1, 7:16, 
                8:22, 9:11, 10:23, 11:10,
                12:20, 13:9, 14:21, 15:8,
                16:13, 17:7, 18:12, 19:6,
                20:15, 21:5, 22:14, 23:4
}

#Reverse mapping of square IDs to circle IDs
square_to_circle = {v: k for k, v in matching_ids.items()}

#Convert square_to_circle dict to a format JS can use
square_to_circle_js = {str(k): v for k, v in square_to_circle.items()}

#Create square cube net for color input
faces_square = {
    "U": (2, 8),
    "L": (0, 6),
    "F": (2, 6),
    "R": (4, 6),
    "D": (2, 4),
    "B": (2, 2)
}

xs_square, ys_square, colors_square, labels_square = [], [], [], []
square_num = 0
for fx, fy in faces_square.values():
    for j in range(2):
        for i in range(2):
            xs_square.append(fx + j + 0.5)
            ys_square.append(fy + i + 0.5)
            
            if square_num in square_to_circle:
                circle_id = square_to_circle[square_num]
                colors_square.append(dot_colors[circle_id])
            else:
                colors_square.append(color_list[0])
            
            labels_square.append(str(square_num))
            square_num += 1

source_square = ColumnDataSource(data=dict(x = xs_square,
                                           y = ys_square, 
                                           color = colors_square, 
                                           label = labels_square))

p_square = figure(width = 300,
                  height = 400,
                  tools = "tap",
                  title = "Click to cycle colors")
p_square.x_range.start, p_square.x_range.end = -0.5, 6.5
p_square.y_range.start, p_square.y_range.end = 0.5, 10.5
p_square.grid.visible = False
p_square.axis.visible = False
p_square.outline_line_color = None
p_square.toolbar_location = None
p_square.title.align = "center"

rects_square = p_square.rect(x = "x",
                             y = "y",
                             width = 0.95, 
                             height = 0.95,
                             fill_color = "color", 
                             line_color = "black", 
                             source = source_square)
if use_labels:
    labels_grid = LabelSet(x = 'x',
                           y = 'y', 
                           text = 'label', 
                           source = source_square,
                           text_align = 'center',
                           text_baseline = 'middle',
                           text_font_size = '12pt',
                           text_color = 'black')
    p_square.add_layout(labels_grid)


face_borders = [
    {"x": 3, "y": 9, "width": 2, "height": 2},
    {"x": 1, "y": 7, "width": 2, "height": 2},
    {"x": 3, "y": 7, "width": 2, "height": 2},
    {"x": 5, "y": 7, "width": 2, "height": 2},
    {"x": 3, "y": 5, "width": 2, "height": 2},
    {"x": 3, "y": 3, "width": 2, "height": 2},
]

for border in face_borders:
    p_square.rect(x = border["x"],
                  y = border["y"], 
                  width = border["width"],
                  height = border["height"],
                  fill_color = None,
                  line_color = "black",
                  line_width = 4)

inner_borders = []
for face_name, (fx, fy) in faces_square.items():
    inner_borders.append({"x0": fx + 1,
                          "y0": fy,
                          "x1": fx + 1,
                          "y1": fy + 2})
    inner_borders.append({"x0": fx,
                          "y0": fy + 1,
                          "x1": fx + 2, 
                          "y1": fy + 1})

for border in inner_borders:
    p_square.line(x = [border["x0"], border["x1"]],
                  y = [border["y0"], border["y1"]],
                  line_color = "black",
                  line_width = 2)
    

#This part was created using claude as i could not figure out how to do it in python by itself
#Create a CustomJS callback for handling both left and right clicks
tap_callback = CustomJS(args=dict(source_square=source_square, source_circle=source_circle,
                                   colors=color_list, square_to_circle=square_to_circle_js), code="""
    // This will be triggered by the TapTool, but we'll handle the actual cycling in Python
""")

#Add a JS callback for right-click detection
right_click_callback = CustomJS(args=dict(source_square=source_square, source_circle=source_circle,
                                          colors=color_list, square_to_circle=square_to_circle_js,
                                          p_square=p_square), code="""
    // Get the canvas element
    const canvas = p_square.canvas_view.primary_canvas_view.ctx.canvas;
    
    // Add context menu handler
    canvas.addEventListener('contextmenu', function(event) {
        event.preventDefault();
        
        // Get click coordinates relative to canvas
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Convert to data coordinates
        const xv = p_square.x_scale.invert(x);
        const yv = p_square.y_scale.invert(y);
        
        // Find which square was clicked
        const data = source_square.data;
        for (let i = 0; i < data['x'].length; i++) {
            const dx = Math.abs(data['x'][i] - xv);
            const dy = Math.abs(data['y'][i] - yv);
            
            if (dx < 0.5 && dy < 0.5) {
                // Cycle color backwards
                const current = data['color'][i];
                const idx = colors.indexOf(current);
                const prev = (idx - 1 + colors.length) % colors.length;
                data['color'][i] = colors[prev];
                
                // Sync to circle display
                const circle_id = square_to_circle[i.toString()];
                if (circle_id !== undefined) {
                    source_circle.data['color'][circle_id] = colors[prev];
                }
                
                source_square.change.emit();
                source_circle.change.emit();
                break;
            }
        }
    });
""")

#We need to execute the right-click callback once to set up the event listener
p_square.js_on_event('tap', right_click_callback)

#Python callback to sync dot_colors when source_circle changes (for right-clicks)
def sync_dot_colors_from_circle(attr, old, new):
    """Update dot_colors list when circle display changes"""
    if 'color' in new:
        colors = new['color']
        for i in range(min(len(dot_colors), len(colors))):
            dot_colors[i] = colors[i]

source_circle.on_change('data', sync_dot_colors_from_circle)

#Python callback for cycling colors on square net and syncing to circle (left-click)
def cycle_square_color():
    """Called when a square is clicked"""
    inds = source_square.selected.indices
    if len(inds) > 0:
        colors_square = list(source_square.data['color'])
        colors_circle = list(source_circle.data['color'])
        
        for i in inds:
            #Cycle the color forward
            current = colors_square[i]
            idx = color_list.index(current)
            next_color = color_list[(idx + 1) % len(color_list)]
            colors_square[i] = next_color
            
            #Sync to circle display and dot_colors list
            if i in square_to_circle:
                circle_id = square_to_circle[i]
                dot_colors[circle_id] = next_color
                colors_circle[circle_id] = next_color
        
        #Update only the color data
        source_square.data['color'] = colors_square
        source_circle.data['color'] = colors_circle
        
        #Clear selection
        source_square.selected.indices = []

#Connect the tap tool to the Python callback
source_square.selected.on_change('indices', lambda attr, old, new: cycle_square_color())

def sync_colors():
    colors_from_square = list(source_square.data['color'])
    
    #Map each square net square to its corresponding circle
    for net_id in range(len(colors_from_square)):
        if net_id in square_to_circle:
            circle_id = square_to_circle[net_id]
            dot_colors[circle_id] = colors_from_square[net_id]
    
    #Update the circle display - create completely new dict
    new_data = {
        'x': list(x_vals),
        'y': list(y_vals),
        'color': list(dot_colors),
        'dot_id': [str(i) for i in range(len(x_vals))]
    }
    source_circle.data = new_data
    
    if diagnose_state:
        print("Synced state:", read_current_state())

def update_square_from_circle():
    """Update square net to match current circle display state"""
    current_colors_square = list(source_square.data['color'])
    
    for net_id in range(len(current_colors_square)):
        if net_id in square_to_circle:
            circle_id = square_to_circle[net_id]
            if circle_id < len(dot_colors):
                current_colors_square[net_id] = dot_colors[circle_id]
    
    source_square.data = dict(x=xs_square, y=ys_square,
                             color=current_colors_square,
                             label=labels_square)
###this is where the ai use stops

def rotate_face(face_move):
    face_name = face_move.rstrip("'")
    clockwise = not face_move.endswith("'")
    if face_name in RINGS and face_name in FACES:
        dot_ids = RINGS[face_name]
        current_colors = [dot_colors[dot_id] for dot_id in dot_ids]
        
        if clockwise:
            rotated_colors = current_colors[-cube_size:] + current_colors[:-cube_size]
        else:
            rotated_colors = current_colors[cube_size:] + current_colors[:cube_size]
        
        for dot_id, color in zip(dot_ids, rotated_colors):
            dot_colors[dot_id] = color

        dot_ids = FACES[face_name]
        current_colors = [dot_colors[dot_id] for dot_id in dot_ids]
        if clockwise:
            rotated_colors = current_colors[-1:] + current_colors[:-1]
        else:
            rotated_colors = current_colors[1:] + current_colors[:1]
        
        for dot_id, color in zip(dot_ids, rotated_colors):
            dot_colors[dot_id] = color
    else:
        raise ValueError(f"{face_name} is not a valid Face Name!")
    
    source_circle.data = dict(x = x_vals,
                              y = y_vals,
                              color = dot_colors.copy(),
                              dot_id = [str(i) for i in range(len(x_vals))])
    
    #Update square net to reflect the move
    update_square_from_circle()
    
    if diagnose_state:
        print(read_current_state())

def rotate_cube(move):
    rotations = {
        "x" : ("R",  "L'"),
        "x'": ("R'", "L"),
        "y" : ("U",  "D'"),
        "y'": ("U'", "D"),
        "z" : ("F",  "B'"),
        "z'": ("F'", "B")
    }
    rotate_face(rotations[move][0])
    rotate_face(rotations[move][1])

def parse_move_sequence(sequence):
    sequence = sequence.replace(" ", "")
    moves = []
    i = 0
    while i < len(sequence):
        char = sequence[i]
        if char.upper() in FACES or char in ['x', 'y', 'z']:
            move = char
            i += 1
            if i < len(sequence) and sequence[i].isdigit():
                repeat = int(sequence[i])
                i += 1
            else:
                repeat = 1
            if i < len(sequence) and sequence[i] == "'":
                move += "'"
                i += 1
            moves.extend([move] * repeat)
        else:
            i += 1
    return moves

def execute_sequence():
    if sequence_executing[0]:
        return
    
    sequence = text_input.value
    moves = parse_move_sequence(sequence)
    if not moves:
        return
    
    sequence_executing[0] = True
    disable_all_buttons()
    
    def execute_next_move(move_index):
        if move_index < len(moves):
            move = moves[move_index]
            if move.rstrip("'") in ['x', 'y', 'z']:
                rotate_cube(move)
            else:
                rotate_face(move)
            
            curdoc().add_timeout_callback(lambda: execute_next_move(move_index + 1), animation_delay)
        else:
            text_input.value = ""
            sequence_executing[0] = False
            enable_all_buttons()
    
    #Start executing moves
    execute_next_move(0)

def execute_scramble():
    if sequence_executing[0]:
        return
    
    sequence = generate_random_scramble(scramble_length=scramble_length)
    moves = parse_move_sequence(sequence)
    
    sequence_executing[0] = True
    disable_all_buttons()
    
    def execute_next_move(move_index):
        if move_index < len(moves):
            move = moves[move_index]
            rotate_face(move)
            
            curdoc().add_timeout_callback(lambda: execute_next_move(move_index + 1), animation_delay)
        else:
            if view_scramble_progress:
                print(read_current_state())
            sequence_executing[0] = False
            enable_all_buttons()
    
    execute_next_move(0)

def read_current_state():
    state = []
    for hex_code in source_circle.data['color']:
        state.append(color_dict[hex_code])
    return state

sequence_executing = [False]

execute_button = Button(label = "Execute Sequence",
                        button_type = "success",
                        width = 150)
execute_button.on_click(execute_sequence)

text_input = TextInput(value = "",
                       width = 300)


toggle_grid_button = Button(label = "Hide Grid",
                            button_type = "warning",
                            width = 110)



def toggle_grid():
    if grid_visible[0]:
        grid_column.visible = False
        toggle_grid_button.label = "Show Grid"
        grid_visible[0] = False
    else:
        grid_column.visible = True
        toggle_grid_button.label = "Hide Grid"
        grid_visible[0] = True

toggle_grid_button.on_click(toggle_grid)

def set_solve_str():
    disable_all_buttons()
    text_input.value = ''
    state_vector = read_current_state()
    if 'Q' not in locals() or 'Q' not in globals():
        break_glass = True
    solve_str = get_solve_str(state_vector, break_glass = break_glass)
    if solve_str:
        text_input.value = solve_str
    enable_all_buttons()

button_U = Button(label="U", button_type="default", width=50)
button_U_prime = Button(label="U'", button_type="default", width=50)
button_D = Button(label="D", button_type="default", width=50)
button_D_prime = Button(label="D'", button_type="default", width=50)
button_B = Button(label="B", button_type="default", width=50)
button_B_prime = Button(label="B'", button_type="default", width=50)
button_F = Button(label="F", button_type="default", width=50)
button_F_prime = Button(label="F'", button_type="default", width=50)
button_R = Button(label="R", button_type="default", width=50)
button_R_prime = Button(label="R'", button_type="default", width=50)
button_L = Button(label="L", button_type="default", width=50)
button_L_prime = Button(label="L'", button_type="default", width=50)

button_X = Button(label="x", button_type="default", width=50)
button_X_prime = Button(label="x'", button_type="default", width=50)
button_Y = Button(label="y", button_type="default", width=50)
button_Y_prime = Button(label="y'", button_type="default", width=50)
button_Z = Button(label="z", button_type="default", width=50)
button_Z_prime = Button(label="z'", button_type="default", width=50)

#Connect Python callbacks to buttons
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

button_X.on_click(lambda: rotate_cube("x"))
button_X_prime.on_click(lambda: rotate_cube("x'"))
button_Y.on_click(lambda: rotate_cube("y"))
button_Y_prime.on_click(lambda: rotate_cube("y'"))
button_Z.on_click(lambda: rotate_cube("z"))
button_Z_prime.on_click(lambda: rotate_cube("z'"))

button_scramble = Button(label="Scramble",
                         button_type="warning",
                         width=110)
button_scramble.on_click(execute_scramble)

button_solver = Button(label = "Generate Solve String",
                      button_type = "success",
                      width = 150)
button_solver.on_click(set_solve_str)

#Functions to enable/disable all buttons during sequence execution
all_buttons = [
    button_U, button_U_prime, 
    button_D, button_D_prime,
    button_B, button_B_prime, 
    button_F, button_F_prime,
    button_R, button_R_prime, 
    button_L, button_L_prime,
    button_X, button_X_prime, 
    button_Y, button_Y_prime,
    button_Z, button_Z_prime, 
    button_scramble, 
    execute_button,
    button_solver
]

def disable_all_buttons():
    for btn in all_buttons:
        btn.disabled = True

def enable_all_buttons():
    for btn in all_buttons:
        btn.disabled = False

#Layout buttons
button_col = column(
    row(button_U, button_U_prime, button_X, button_X_prime),
    row(button_D, button_D_prime, button_Y, button_Y_prime),
    row(button_B, button_B_prime, button_Z, button_Z_prime),
    row(button_F, button_F_prime),
    row(button_R, button_R_prime),
    row(button_L, button_L_prime),
    row(button_scramble),
    row(toggle_grid_button)
)

#Grid column with toggle button
grid_column = column(p_square)


#Final layout
layout = column(
    row(),
    row(p_circle, button_col, grid_column),
    row(execute_button, text_input),
    row(button_solver)
)

#Add to Bokeh document
curdoc().add_root(layout)
curdoc().title = "Rubik's Cube Interface"
toggle_grid()