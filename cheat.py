import pickle
for i in range(72):
    pickle.dump([], open(f"games/{i}.p", "wb"))
