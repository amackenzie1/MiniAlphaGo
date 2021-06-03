import numpy as np 
import time 
import random 


def clean(board, priority):

    def live(board, number):
        for i in kids(number):
            if board[i//5][i%5] == 0:
                return True
        return False
    
    def kids(position):
        potentials = [position + 1, position - 1, position + 5, position - 5]
        potentials = [i for i in potentials if i >= 0 and i < 25 and abs(position % 5 - i % 5) < 2]
        return potentials
    
    def search(root, all_positions):
        result = [root]
        for i in kids(root):
            if i in all_positions:
                all_positions.remove(i)
                result += search(i, all_positions)
        return result 

    def get_connected_components(board, who):
        ccs = []
        all_positions = list(range(25))
        for i in list(all_positions):
            if board[i//5][i%5] != who:
                all_positions.remove(i)
        
                
        while len(all_positions) > 0:
            root = all_positions[0]
            all_positions.remove(root)
            ccs.append(search(root, all_positions))
        
        return ccs 
    
    deaths = []

    def half_clean(who):
        components = get_connected_components(board, who)
        for component in components:
            if True not in [live(board, i) for i in component]:
                for i in component:
                    board[i//5][i%5] = 0
                deaths.extend(component)

    if priority == 1:
        half_clean(-1)
        half_clean(1)
    else:
        half_clean(1)
        half_clean(-1)

    return board, deaths 
    
class Board:
    def __init__(self):
        oneboard = [[0]*5]*5
        self.board = np.array([oneboard]*3)
        self.board[0][1][1] = 1
        self.board[1][2][2] = -1
        self.moves = 0
        self.recent_onedeaths = []

    def display(self):
        print("")
        print("   ---------------------")
        counter = 1
        for row in self.board[-1]:
            print(f"{counter}  |", end="") if len(str(counter)) == 1 else print(f"{counter} |", end="")
            counter += 1
            for i in row:
                if i == 1:
                    print(f" X |", end="")
                elif i == -1:
                    print(f" O |", end="")
                else:
                    print(f"   |", end="")
            print("")
            print("   ---------------------")
        print("    ", end="")
        for i in "ABCDE":
            print(f" {i}  ", end="")
        print()
        print(f"Moves: {self.moves}")
        print(f"Done: {self.is_done()}")

    def from_letters(self, move):
        row = int("".join([i for i in move if i.isnumeric()]))
        column = "ABCDE".index("".join([i.upper() for i in move if i.isalpha()]))
        return row-1, column

    def from_numbers(self, move):
        return move//5, move%5

    def move(self, move, who):

        if move == 25 or str(move).lower() == "pass":
            self.moves += 1
            self.board = np.concatenate([self.board[1:], np.array([self.board[-1]])])
            return 2

        if type(move) == str:
            to_modify = self.board[-1].copy()
            row, column = self.from_letters(move)
    
        elif type(move) == int:
            to_modify = self.board[-1].copy()
            row, column = self.from_numbers(move)
        else:
            print(move)
            print("WTF")
            exit()
        if to_modify[row][column] == 0:
            to_modify[row][column] = who 
            cleaned, deaths = clean(to_modify, who)
            if len(deaths) == 1:
                self.recent_onedeaths += deaths 
                self.recent_onedeaths = self.recent_onedeaths[-2:]
            newboard = np.array([cleaned])
            if np.all(newboard[0] == self.board[1]) and not np.all(newboard[0] == self.board[2]):
                return -2
            self.board = np.concatenate([self.board[1:], newboard])
            self.moves += 1
            return 2
        else:
            return -2

    def clone(self):
        newboard = Board()
        newboard.board = list(self.board)
        newboard.moves = self.moves
        newboard.recent_onedeaths = list(self.recent_onedeaths)
        return newboard
    
    def flip(self):
        for i in range(len(self.board)):
            self.board[i] = -1 * self.board[i]

    def is_done(self):
        if self.moves > 45:
            return True
        if np.all(self.board[-1] == self.board[0]):
            return True
        else:
            print(self.board[0])
            print(self.board[-1])
        return False 
    
    def get_moves(self, who):
        moves = [25]
        for i in range(25):
            if self.board[-1][i//5][i%5] == 0:
                if i not in self.recent_onedeaths:
                    moves.append(i)
                else:
                    if self.clone().move(i, who) == 2:
                        moves.append(i)

        return moves

    def to_array(self):
        pos0 = (self.board[0] + np.abs(self.board[0]))/2 
        neg0 = (np.abs(self.board[0]) - self.board[0])/2 
        pos0 = np.expand_dims(pos0, axis=2)
        neg0 = np.expand_dims(neg0, axis=2)
        pos1 = (self.board[1] + np.abs(self.board[1]))/2 
        neg1 = (np.abs(self.board[1]) - self.board[1])/2 
        pos1 = np.expand_dims(pos1, axis=2)
        neg1 = np.expand_dims(neg1, axis=2)
        pos2 = (self.board[2] + np.abs(self.board[2]))/2 
        neg2 = (np.abs(self.board[2]) - self.board[2])/2 
        pos2 = np.expand_dims(pos2, axis=2)
        neg2 = np.expand_dims(neg2, axis=2)
        return np.concatenate([pos0, neg0, pos1, neg1, pos2, neg2], axis=2)
    
    def score(self):
        return 1 if np.sum(self.board[-1]) > 0 else -1

if __name__ == "__main__":
    board = Board()
    turn = 1
    board2 = board.clone() 

    t1 = time.time()
    for i in range(30):
        board.display()
        x = int(input("Move: "))
        if board.move(x, turn) == 2:
            turn *= -1
    board.display()
    board.flip()
    board.display()
    print(board.score())
    print(board.to_array().shape)