from board import Board 
import numpy as np

myboard = Board()
myboard.move(1, 1)
myboard.move(2, 1)
myboard.move(3, 1)

counter = 0
permboard = []
for i in range(11):
    permrow = []
    for j in range(11): 
        permrow.append(counter)
        counter += 1
    permboard.append(permrow)
permboard = np.array(permboard)

def permute(mylist, perm):
    newlist = np.zeros(121)
    for i in range(121):
        newlist[i] = mylist[perm[i]]
    return newlist

def symmetries(board, probs):
    boards = []
    probabilities = []
    
    boards.append(board)
    probabilities.append(permute(probs, permboard.flatten()))
    boards.append(np.flip(board, axis=0))
    probabilities.append(permute(probs, np.flip(permboard, axis=0).flatten()))
    boards.append(np.flip(board, axis=1))
    probabilities.append(permute(probs, np.flip(permboard, axis=1).flatten()))
    boards.append(np.flip(np.flip(board, axis=1), axis=0))
    probabilities.append(permute(probs, np.flip(permboard).flatten()))

    boards.append(np.transpose(board, axes=[1, 0, 2]))
    probabilities.append(permute(probs, permboard.T.flatten()))
    boards.append(np.flip(np.transpose(board, axes=[1, 0, 2]), axis=0))
    probabilities.append(permute(probs, np.flip(permboard.T, axis=0).flatten()))
    boards.append(np.flip(np.transpose(board, axes=[1, 0, 2]), axis=1))
    probabilities.append(permute(probs, np.flip(permboard.T, axis=1).flatten()))
    boards.append(np.flip(np.flip(np.transpose(board, axes=[1, 0, 2]), axis=1), axis=0))
    probabilities.append(permute(probs, np.flip(permboard.T).flatten()))

    return boards, [np.append(i, probs[-1]) for i in probabilities]

if __name__ == "__main__":
    preprobs = np.zeros(121)
    preprobs[1] = 1
    preprobs[2] = 1
    boards, probs = symmetries(myboard.to_array(), preprobs)
    print(myboard.to_array().shape)
    def render(probs):
        for i in range(11):
            for j in range(11):
                print(int(probs[i * 11 + j]), end="")
            print("")

    for i, j in zip(boards, probs):
        print(np.sum(i, axis=-1))
        render(j)