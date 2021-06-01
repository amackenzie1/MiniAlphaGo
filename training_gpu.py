import numpy as np
import os 
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
from expand_boards import expand
import random 
from uuid import uuid1
import sys 
from sklearn.utils import shuffle 

model = get_model()
model.load_weights("baby_alphazero/v1")

def get_data():
    games = []
    for i in os.listdir("games"):
        games += pickle.load( open( f"games/{i}", "rb" ) )
    for i in os.listdir("games"):
        os.remove(f"games/{i}")
    return games 


def getversion():
    return int(open("info.txt").readlines()[0].split()[1])

global num_processes
num_processes = 2
episode_length = 6

def train():
    version = getversion()

    model.load_weights("baby_alphazero/v1")
    
    while len(os.listdir("games")) < num_processes:
        print(f"Sleeping, I, {os.getpid()}, am boss but there aren't enough processes done yet.")
        time.sleep(20)
         
    print(f"Process {os.getpid()} starting training.")
    open("info.txt", "w").write(f"Version: {version+1}")

    window_size = min(20, max(min(version, 4), version//2))*episode_length 
    print(f"Window size: {window_size}")
    if "training_data.p" in os.listdir():
        training_data = pickle.load( open( f"training_data.p", "rb" ) )
    else:
        training_data = [] 
    
    training_data += get_data()
    training_data = training_data[-1*window_size:]
    pickle.dump(training_data, open(f"training_data.p", "wb"))
    print(f"Length of training data: {len(training_data)}")

    boards, policies, results = expand(training_data)
    del training_data
    boards, policies, results = shuffle(boards, policies, results)

    def generator():
        i = 0
        while i < len(boards):
            i += 300000 
            yield boards[(i-300000):i], policies[(i-300000):i], results[(i-300000):i]

    for boards0, policies0, results0 in generator():
    
        print(f"Shape of boards: {boards0.shape}")
        print(f"Shape of policies: {policies0.shape}") 
        print(f"Shape of results: {results0.shape}") 

        boards0 = np.array(boards0, dtype='float32')
        results0 = np.array(results0, dtype='float32')
        policies0 = np.array(policies0, dtype='float32')

        model.fit(boards0, {'policy': policies0, 'value': results0}, epochs=1, batch_size=32)

    model.save_weights("baby_alphazero/v1")
    
    
num_processes = int(sys.argv[1])
episode_length = int(sys.argv[2])
while True:
    train()