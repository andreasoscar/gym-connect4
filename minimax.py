import random
import numpy as np
import copy 

class minimax:

    def __init__(self,depth):
        self.COL = 7
        self.COL = 6
        self.depth = depth
        
        self.SAMPLE_SIZE = 4

    def is_column_available(self, state, column):
        return any(state[i][column] == ' ' for i in range(self.COL))

    def availableColumns(self, state):
        l = []
        for i in range(self.COL):
                if self.is_column_available(state,i):
                        l.append(i)
        return l

    def score_sample(self, sample):
        score = 0
        scoring = {'X':0, ' ':0, 'O':0}
        playerScoring = [0,0,5,15,100]
        botScoring = [0,0,0,-10,-100]
        for i in sample:
                scoring[i] = scoring[i] + 1
        if scoring['O'] + scoring[' '] == 4:
                score += playerScoring[scoring['O']]
        elif scoring['X'] + scoring[' '] == 4:
                score += botScoring[scoring['X']]
        return score


    def score(self,state):
        score = 0

        center_column = []
        for r in range(self.COL):
                center_column.append(state[r][3])
        center = {' ':0, 'X':0, 'O':0}
        for i in center_column:
                center[i] = center[i] + 1
        #center should be weighted more, but if there are -1's in the row, the chances of winning is lower, so reduce its weight in that case
        score += (4-center['X'])*center['O']

        #horizontal samples [x,x,x,x]
        for row in range(self.COL):
                horizontal_column = []
                for col in range(self.COL):
                        horizontal_column.append(state[row][col])
                for col in range(self.COL-3):
                        sample = horizontal_column[col:col+self.SAMPLE_SIZE]
                        score += self.score_sample(sample)

        #vertical sample
        #[x]
        #[x]
        #[x]
        #[x]
        for col in range(self.COL):
                vertical_column = []
                for row in range(self.COL):
                        vertical_column.append(state[row][col])
                for row in range(self.COL-3):
                        sample = vertical_column[row:row+self.SAMPLE_SIZE]
                        score += self.score_sample(sample)
        # 0 0 0 x
        # 0 0 x 0
        # 0 x 0 0
        # x 0 0 0
        # positive sloped sample
        # for row in range(self.COL-3):
                for col in range(self.COL-3):
                        positive_sloped_column = [state[row+slope][col+slope] for slope in range(self.SAMPLE_SIZE)]
                        score += self.score_sample(positive_sloped_column)

        # x 0 0 0
        # 0 x 0 0
        # 0 0 x 0
        # 0 0 0 x
        # negative sloped sample
        for row in range(self.COL-3):
                for col in range(self.COL-3):
                        negative_sloped_column = [state[row+3-slope][col+slope] for slope in range(self.SAMPLE_SIZE)]
                        score += self.score_sample(negative_sloped_column)
        

                return score
    def alphabeta(self, depth, alpha, beta, maximize, state, col):
        available = self.availableColumns(state)
        #Initially set to the child denoted 'col', this variable will change if a better decision (0-6) is found among the children of this node
        beneficial_move = col
        if depth == 0 or len(available) == 0:
                return self.score(state), col

        if maximize:
                value = float('-inf')
                for child in available:
                        #1 = player
                        modified_board = self.make_theoretical_move(child,state,'O')
                        _value = self.alphabeta(depth-1, alpha, beta, False, modified_board, child)[0]
                        if _value > value:
                                beneficial_move = child
                                value = _value
                        alpha = max(alpha,value)
                        if alpha >= beta:
                                break
                return value, beneficial_move

        elif not maximize:
                value = float('inf')
                for child in available:
                        #-1 = bot
                        modified_board = self.make_theoretical_move(child, state, 'X')
                        _value = self.alphabeta(depth-1, alpha, beta, True, modified_board, child)[0]
                        if _value < value:
                                value = _value
                                beneficial_move = child
                        beta = min(beta, value)
                        if beta <= alpha:
                                break
                return value, beneficial_move

    #insert piece in selected column
    def make_theoretical_move(self, col, state, player):
        state = copy.deepcopy(state)
        #move from bottom-up, insert the piece in the first available slot in each column
        for row in reversed(range(0,self.COL,1)):
                if state[row][col] == ' ':
                        state[row][col] = player
                        return state

    def student_move(self,state):
        possible_moves = self.availableColumns(state)
        random.shuffle(possible_moves)
        #since this is the first time we place a piece, all moves 0-6 should be available, pick a random column to place a piece in
        
        #print("initial col",possible_moves)
        initial_col = random.choice(possible_moves)
        # value is not used here
        value, col = self.alphabeta(self.depth, float('-inf'), float('inf'), True, state, initial_col)
        return col
