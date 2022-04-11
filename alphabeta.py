
class alphabeta:
    COL = 7
    ROW = 6
    def is_column_available(Self, state,column):
        return any(state[i][column] == 0 for i in range(ROW))

    def availableColumns(self, state):
        l = []
        for i in range(COL):
                if is_column_available(state,i):
                        l.append(i)
        return l
        
    def score_sample(self, sample):
        score = 0
        scoring = {-1:0, 0:0, 1:0}
        playerScoring = [0,0,5,15,100]
        botScoring = [0,0,0,-10,-100]
        for i in sample:
                scoring[i] = scoring[i] + 1
        if scoring[1] + scoring[0] == 4:
                score += playerScoring[scoring[1]]
        elif scoring[-1] + scoring[0] == 4:
                score += botScoring[scoring[-1]]
        return score
                

    def score(self, state):
        score = 0
        
        center_column = []
        for r in range(ROW):
                center_column.append(state[r][3])
        center = {-1:0, 0:0, 1:0}
        for i in center_column:
                center[i] = center[i] + 1
        #center should be weighted more, but if there are -1's in the row, the chances of winning is lower, so reduce its weight in that case
        score += (4-center[-1])*center[1]
        
        #horizontal samples [x,x,x,x]
        for row in range(ROW):
                horizontal_column = []
                for col in range(COL):
                        horizontal_column.append(state[row][col])
                for col in range(COL-3):
                        sample = horizontal_column[col:col+SAMPLE_SIZE]
                        score += score_sample(sample)
                
        #vertical sample
        #[x]
        #[x]
        #[x]
        #[x]
        for col in range(COL):
                vertical_column = []
                for row in range(ROW):
                        vertical_column.append(state[row][col])
                for row in range(ROW-3):
                        sample = vertical_column[row:row+SAMPLE_SIZE]
                        score += score_sample(sample)
        #0 0 0 x
        #0 0 x 0
        #0 x 0 0
        #x 0 0 0
        #positive sloped sample
        for row in range(ROW-3):
                for col in range(COL-3):
                        positive_sloped_column = [state[row+slope][col+slope] for slope in range(SAMPLE_SIZE)]
                        score += score_sample(positive_sloped_column)
                        
        #x 0 0 0
        #0 x 0 0
        #0 0 x 0
        #0 0 0 x
        #negative sloped sample
        for row in range(ROW-3):
                for col in range(COL-3):
                        negative_sloped_column = [state[row+3-slope][col+slope] for slope in range(SAMPLE_SIZE)]
                        score += score_sample(negative_sloped_column)
        
        return score
    def alphabeta(self, depth, alpha, beta, maximize, state, col):
        
        available = availableColumns(state)
        #Initially set to the child denoted 'col', this variable will change if a better decision (0-6) is found among the children of this node
        beneficial_move = col
        if depth == 0 or len(available) == 0:
                return score(state), col
            
        if maximize:
                value = float('-inf')   
                for child in available:
                        #1 = player
                        modified_board = make_theoretical_move(child,state,1)
                        _value = alphabeta(depth-1, alpha, beta, False, modified_board, child)[0]
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
                        modified_board = make_theoretical_move(child, state, -1)
                        _value = alphabeta(depth-1, alpha, beta, True, modified_board, child)[0]
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
        for row in reversed(range(0,ROW,1)):
                if state[row][col] == 0:
                        state[row][col] = player
                        return state
            
    def student_move(self, state):
        
        possible_moves = availableColumns(state)
        random.shuffle(possible_moves)
        #since this is the first time we place a piece, all moves 0-6 should be available, pick a random column to place a piece in 
        initial_col = random.choice(possible_moves)
        #value is not used here
        value, col = alphabeta(5, float('-inf'), float('inf'), True, state, initial_col)
        return col
        
    