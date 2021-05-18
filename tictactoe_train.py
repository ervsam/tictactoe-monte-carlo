import numpy as np
import random
import pickle

BOARD_SIZE = 3
ITERATIONS = 50000
gamma = 0.8
exp_rate = 0.1

def winner(state_space, avail_space_n):
    # rows, cols
    for i in range(BOARD_SIZE):
        if BOARD_SIZE in [sum(state_space[i, :]), sum(state_space[:, i])]:
            return 1, True
        if -BOARD_SIZE in [sum(state_space[i, :]), sum(state_space[:, i])]:
            return -1, True
    # diagonal
    diag_sum1 = sum([state_space[i, i] for i in range(BOARD_SIZE)])
    diag_sum2 = sum([state_space[i, BOARD_SIZE - i - 1] for i in range(BOARD_SIZE)])
    diag_sum = max(abs(diag_sum1), abs(diag_sum2))
    if diag_sum == BOARD_SIZE:
        if diag_sum1 == BOARD_SIZE or diag_sum2 == BOARD_SIZE:
            return 1, True
        else:
            return -1, True
    # tie
    if avail_space_n == 0:
        return 0, True
    return None, False

def availablePositions(state_space):    
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if state_space[i, j] == 0]

def generate_ep(it):
    state_space = np.zeros((BOARD_SIZE, BOARD_SIZE))
    avail_space_n = 9
    
    ep1 = []
    ep2 = []
    
    p1 = True
    p2 = not p1
    end = False
    
    while not end:
        avail = availablePositions(state_space)
        state_val = state_val_1 if p1 else state_val_2
        
        if it < 7000:
            action = random.choice(avail)
            state_space_tmp = np.copy(state_space)
            for a in avail:
                state_space_tmp[a[0]][a[1]] = 1 if p1 else -1
                if state_val.get(str(state_space_tmp)) is None:
                    action = a
                state_space_tmp[a[0]][a[1]] = 0
        elif np.random.uniform(0, 1) <= exp_rate:
            action = random.choice(avail)
        else:
            max_val = -999
            state_space_tmp = np.copy(state_space)
            for a in avail:
                state_space_tmp[a[0]][a[1]] = 1 if p1 else -1
                if state_val.get(str(state_space_tmp)) is None:
                    state_val[str(state_space_tmp)] = 0
                    
                value = state_val.get(str(state_space_tmp))
                if value >= max_val:
                    max_val = value
                    action = a
                state_space_tmp[a[0]][a[1]] = 0
        
        state_space[action[0]][action[1]] = 1 if p1 else -1
        avail_space_n -= 1
        
        win, end = winner(state_space, avail_space_n)
        if win != None:
            if win == 1:
                ep1.append([np.copy(state_space), 1])
                ep2.append([np.copy(state_space), -1])
            elif win == -1:
                ep1.append([np.copy(state_space), -1])
                ep2.append([np.copy(state_space), 1])
            else:
                ep1.append([np.copy(state_space), -0.5])
                ep2.append([np.copy(state_space), -0.5])
            return ep1, ep2
        else:
            if p1:
                ep1.append([np.copy(state_space), 0])
            else:
                ep2.append([np.copy(state_space), 0])
            
        p1, p2 = p2, p1

state_val_1 = {}
state_val_2 = {}
returns1 = {}
returns2 = {}
deltas = {}

for it in range(ITERATIONS):
    ep1, ep2 = generate_ep(it)
    
    G1 = 0
    G2 = 0
    
    for step in ep1[::-1]:
        s1, reward1 = (str(step[0]), step[1])
        
        G1 = gamma*G1 + reward1
        if returns1.get(s1) is None:
            returns1[s1] = []
            deltas[s1] = []
            state_val_1[s1] = 0

        returns1[s1].append(G1)
        deltas[s1].append(np.abs(state_val_1[s1] - np.average(returns1[s1])))
        state_val_1[s1] = np.average(returns1[s1])
        
    for step in ep2[::-1]:
        s2, reward2 = (str(step[0]), step[1])
        
        G2 = gamma*G2 + reward2
        if returns2.get(s2) is None:
            returns2[s2] = []
            state_val_2[s2] = 0
        returns2[s2].append(G2)
        state_val_2[s2] = np.average(returns2[s2])


f = open("data", "wb")
pickle.dump((state_val_1, state_val_2), f)
f.close()