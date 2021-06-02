import numpy as np
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Softmax, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.losses import CategoricalCrossentropy 
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from model_definition import get_model 
import time 
import pickle 
import random 
from uuid import uuid1
import sys 
from board import Board 

model = get_model()


def evaluate(board):
    if board.is_done():
        return board.score(), Softmax()(np.array([0]*26, dtype='float32'))

    val, probs = model(np.array([board.to_array()]))
    return val.numpy()[0][0], np.squeeze(probs)


class Node:
    def __init__(self, board, move, prob, parent=None, root=False, turn=1, done=False):
        self.board = board
        self.move = move 
        self.Q = 0
        self.P = prob
        self.N = 0
        self.W = 0 
        self.children = {}
        self.parent = parent 
        self.root = root 
        self.turn = turn 
        self.done = False 

c_puct = 1.5

class MonteCarloSearchTree:

    def fill(self, node):
        if node.move is not None:
            node.board.move(node.move, -1)
        val, probs = evaluate(node.board)
        probs = 0.75 * probs + 0.25 * np.random.dirichlet([0.4]*26)
        node.W = val
        node.Q = val
        node.N = 1
        node.done = node.board.is_done()
        for i in node.board.get_moves(1):
            newboard = node.board.clone()
            newboard.flip()
            node.children[i] = Node(newboard, move=i, prob=probs[i], parent=node, turn=-1*node.turn, done=node.done)

        if len(node.children.keys()) == 0:
            node.done = True
            node.W = node.board.score()
            node.Q = node.W
        
    def __init__(self, tau=1):
        self.tau = tau
        self.root = Node(Board(), move=None, prob=1, root=True)
        self.fill(self.root)
        self.policy = None
        self.num_moves = 0
    
        
    def get_move(self):
        distribution = [0 if i not in self.root.children.keys() else self.root.children[i].N for i in range(26)]
        for i in range(26):
            distribution[i] = distribution[i]**(1/self.tau)
        normalization = sum(distribution)
        for i in range(26):
            distribution[i] = distribution[i]/normalization
        self.policy = [round(i, 5) for i in distribution] 
        move = np.random.choice(a=26, p=distribution)
        return move
    
    def search_once(self, node):
        U = {}
        total_visits = 0

        for i in node.children.values():
            total_visits += i.N
        sqrt_total_visits = np.sqrt(total_visits + 1)

        for i in node.children.keys():
            child = node.children[i]
            child_u = c_puct * child.P * sqrt_total_visits / (1 + child.N)
            child_puct = child_u - child.Q 
            U[i] = child_puct 
        
        move = max(U, key=lambda key: U[key])

        if node.children[move].done:
            node.N += 1
            node.W = node.Q * node.N
            self.backup(node, node.Q)

        elif node.children[move].N == 0:
            self.fill(node.children[move])
            self.backup(node.children[move], node.children[move].W)

        else:
            self.search_once(node.children[move])

    def backup(self, node, value):
        if not node.root:
            node.parent.W -= value 
            node.parent.N += 1
            node.parent.Q = node.parent.W/node.parent.N 
            self.backup(node.parent, value*-1)
    
    def search(self, playouts=100):
        for i in range(playouts):
            self.search_once(self.root)

    def advance_root(self, move):
        for i in list(self.root.children.keys()):
            if i != move:
                del self.root.children[i]
        self.root = self.root.children[move]
        self.num_moves += 1
        if self.num_moves > 20:
            self.tau = 0.01
        if self.root.N == 0:
            self.fill(self.root)
        if self.root.board.is_done():
            return self.root.board.score()
        elif len(self.root.children.keys()) == 0:
            return self.root.board.score() 
        return 2
    
    def info(self):
        print(f"Probabilities: {[0 if i not in self.root.children.keys() else self.root.children[i].P for i in range(26)]}")
        print(f"Visits: {[0 if i not in self.root.children.keys() else self.root.children[i].N for i in range(26)]}")
        print(f"Policy: {self.policy}")

def play_game(tau, depth):
    def get_sequence(length, result):
        x = []
        for i in range(length):
            x.append(result)
            result *= -1
        return x[::-1]

    boards = []
    policies = []

    mcts = MonteCarloSearchTree(tau)
    mcts.search(depth)
    boards.append(mcts.root.board.to_array())
    model_move = mcts.get_move()
    policies.append(mcts.policy)
    x = mcts.advance_root(model_move)
    while x == 2:
        mcts.search(depth)
        boards.append(mcts.root.board.to_array())
        policies.append(mcts.policy)
        model_move = mcts.get_move() 
        x = mcts.advance_root(model_move)

    boards.append(np.array(mcts.root.board.to_array(), dtype='int8'))
    policies.append(mcts.policy)

    return list(zip(boards, policies, get_sequence(len(boards), x)))


def play_vs_random():
    mcts = MonteCarloSearchTree(0.01)
    mcts.search(400)
    model_move = mcts.get_move()
    x = mcts.advance_root(model_move)
    random_move = random.choice(list(mcts.root.children.keys()))
    mcts.advance_root(random_move)

    while x == 2:
        mcts.search(400)
        model_move = mcts.get_move() 
        x = mcts.advance_root(model_move)
        if x != 2:
            return 1

        random_move = random.choice(list(mcts.root.children.keys()))
        x = mcts.advance_root(random_move)

    return -1 

episode_length = 6

def iterate(episode_length):
    games = []

    for i in range(episode_length):
        print(f"Process {os.getpid()}, game {i+1}")
        games.append(play_game(1, 400))

    gamefile = str(uuid1())

    pickle.dump(games, open(f"games/{os.uname()[1]}:{gamefile}.p", "wb"))
    
def get_data():
    games = []
    for i in os.listdir("games"):
        games += pickle.load( open( f"games/{i}", "rb" ) )
    for i in os.listdir("games"):
        os.remove(f"games/{i}")
    return games 


def benchmark(length):
    score = 0
    for i in range(length):
        print(f"Benchmark game {i+1}")
        result = play_vs_random()
        print(result)
        score += result
    wins = (score + length)/2 
    return wins/length 

def getversion():
    return int(open("info.txt").readlines()[0].split()[1])


def work(episode_length):
    version = getversion()

    model.load_weights("baby_alphazero/v1")
    iterate(episode_length)
    
    while getversion() == version:
        print(f"Sleeping, I, {os.getpid()}, am not boss.")
        time.sleep(3)    

def process(episode_length):
    with tf.device("CPU:0"):
        while True:
            work(episode_length)


move_map = {"a1":0, "b1": 1, "c1":2, "d1":3, "e1": 4, "a2":5, "b2":6 , "c2":7, "d2":8,
            "e2":9, "a3":10, "b3":11, "c3":12, "d3":13, "e3":14, "a4":15, "b4":16,
            "c4":17, "d4":18, "e4":19, "a5":20, "b5":21, "c5":22, "d5":23, "e5":24, "pass":25}

def play_vs_human(depth):
    mcts = MonteCarloSearchTree(0.01)
    mcts.search(depth)
    model_move = mcts.get_move()
    mcts.info()
    x = mcts.advance_root(model_move)
    mcts.root.board.display()
    human_move = move_map[input("Your move: ").lower()]
    mcts.advance_root(human_move)
    mcts.root.board.flip()
    mcts.root.board.display()
    mcts.root.board.flip()
    
    while x == 2:
        mcts.search(depth)
        model_move = mcts.get_move()
        mcts.info()
        x = mcts.advance_root(model_move)
        mcts.root.board.display()
        if x != 2:
            print("I win!")
            return 1
        human_move = move_map[input("Your move: ").lower()]
        x = mcts.advance_root(human_move)
        mcts.root.board.flip()
        mcts.root.board.display()
        mcts.root.board.flip()

if "baby_alphazero" not in os.listdir():
    model.build(input_shape=(11, 11, 6))
    model.save_weights("baby_alphazero/v1")

episode_length = int(sys.argv[1])

#model.load_weights("baby_alphazero/v1")

try:
    process(episode_length)
except Exception as e:
    fp = open(f"{os.getpid()}_error.txt")
    fp.write(str(Exception))
