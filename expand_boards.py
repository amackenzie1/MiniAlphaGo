import numpy as np
from symmetries import symmetries 

def unzip(games):
    moves = []
    for i in games:
        moves += i 
    boards = []
    policies = []
    results = []
    for i in moves:
        boards.append(i[0])
        policies.append(i[1])
        results.append(i[2])
    return boards, policies, results 

def expand(games):
    expboards = []
    expolicies = []
    expresults = []

    boards, policies, results = unzip(games)
    for b, p, r in zip(boards, policies, results):
        b8, p8 = symmetries(b, p)
        expboards += b8
        expolicies += p8 
        expresults += [r]*8 

    return np.array(expboards), np.array(expolicies), np.array(expresults)