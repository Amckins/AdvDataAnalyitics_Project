This is for 2-D visualizaion of a 2x2 rubiks cube
I cannot find the original version of this to credit them, but this was inspired by 
https://www.reddit.com/r/interestingasfuck/comments/1cbw71j/rubiks_cube_explained_in_2d_model_is_easier_to/

You can visualize it by running in your terminal
bokeh serve visual.py
and ctl-click the local host option
visual.py is intended solely to visualize and verify the logic
that the agent will be using

to that end, the following environment will be created
(may need to run the program to visualize it better)
set use_labels = True before running to see the indices


State Map [0,...,23] with values 1-6
So a solved state = [1, 2, 1, 2, 1, 2, 1, 2, 
                     3, 4, 3, 4, 3, 4, 3, 4,
                     5, 6, 5, 6, 5, 6, 5, 6]

Faces (2x2 clusters) values stored in this indices
U = [4, 6, 0, 2]     Initially White  tf F = [1, 1, 1, 1]
D = [1, 3, 5, 7]     Initially Yellow tf D = [2, 2, 2, 2]
B = [8, 12, 10, 14]  Initially Orange tf B = [3, 3, 3, 3]
F = [9, 11, 13, 15]  Initially Red    tf F = [4, 4, 4, 4]
R = [16, 20, 18, 22] Initally Blue    tf R = [5, 5, 5, 5]
L = [17, 19, 21, 23] Initally Green   tf L = [6, 6, 6, 6]

Rings (the dots along the smallest circle around the inner 3 clusters 
       or the LARGEST circle closest to the outside cluster)
Ur = [12, 8, 20, 16, 9, 13, 17, 21] initially Ur = [3, 3, 5, 5, 4, 4, 6, 6]
Dr = [23, 19, 15, 11, 18, 22, 10, 14] Initially Dr = [6, 6, 4, 4, 5, 5, 3, 3]
Br = [7, 3, 22, 20, 2, 6, 21, 23] Initially Br = [2, 2, 5, 5, 1, 1, 6, 6]
Fr = [19, 17, 4, 0, 16, 18, 1, 5] Initially Fr = [6, 6, 1, 1, 5, 5, 2, 2]
Rr = [3, 1, 11, 9, 0, 2, 8, 10] Initially Rr = [2, 2, 4, 4, 1, 1, 3, 3]
Lr = [7, 5, 15, 13, 4, 6, 12, 14] Initially Lr = [3, 3, 1, 1, 4, 4, 2, 2]

Notice that the Ur and Dr are inverse of eachother
This comes from them being on opposite sides of the cube

Define What a Turn means:

The Letter by itself will dictate a 90 degree clockwise rotation
Letter + ' will mean 90 degree counter clockwise rotation
U will change the colors in [4, 6, 0, 2] into the values that were in [0, 4, 2, 6]
    as well as rotate the ring Ur 
    from [12, 8, 20, 16, 9, 13, 17, 21] to [17, 21, 12, 8, 20, 16, 9, 13]
    or   [3, 3, 5, 5, 4, 4, 6, 6]       to [6, 6, 3, 3, 5, 5, 4, 4] in colors
U' will reverse that action
