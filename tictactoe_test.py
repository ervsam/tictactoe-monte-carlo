import numpy as np
import pickle

f = open("data",'rb')
sv1, sv2 = pickle.load(f)
f.close()

def availablePositions(state_space):    
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if state_space[i, j] == 0]

BOARD_SIZE = 3

while True:
    try:
        inpt = input()
    except EOFError as e:
        break
    inpt = inpt.split()
    
    tot = sum(list(map(int, inpt))[1:])
    
    sv = sv1 if tot == 0 else sv2
    
    if tot == -1:
        inpt = [str(i*-1) for i in list(map(int, inpt))]
    elif tot == 0 and int(inpt[0]) == -1:
        inpt = [str(i*-1) for i in list(map(int, inpt))]

    in_iter = iter(inpt)
    player = int(next(in_iter))
            
    tictactoe = []
    for i in range(BOARD_SIZE):
        t = []
        for j in range(BOARD_SIZE):
            t.append(float(next(in_iter)))
        tictactoe.append(t)

    avail = availablePositions(np.array(tictactoe))
    max_val = -999
    for a in avail:
        tmp_state_space = np.copy(tictactoe)
        tmp_state_space[a[0]][a[1]] = player
        value = sv.get(str(tmp_state_space))
        if value >= max_val:
            max_val = value
            action = a
    print(action[1], action[0])


# SAMPLE INPUT
# 1 -1 0 0 1 1 -1 -1 0 1
# 1 0 0 0 -1 1 -1 1 -1 1




