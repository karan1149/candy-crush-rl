#!/usr/bin/env python

'''
Features:
1.     + min row
2.     + max row
3.     + same col switch?
4.     + num valid moves
5.     + num in a row from making switch
6.     + median utility from NSARS episodes with limited DEPTH
7-15.  + max count of any 1 color in rows 1-9
16-24. + max count of any 1 color in cols 1-9
'''

import random
import numpy as np
import collections
from sklearn.neural_network import MLPRegressor

# NSCORES = 100
NFEATURES = 24
NSARS = 25       # number of SARS' sequences to generate at each turn
DEPTH = 5 	#max depth of SARS' sequence
NUMSAMPLES = 10000	# number of total trials to run
NTURNS = 50		# number of turns in the game
NROWS = 9
NCOLS = 9
NCOLORS = 5
DISCOUNT = 0.5		# discount in computing Q_opt
# STEP_SIZE = 0.00000000001		# eta in computing Q_opt
# EPSILON = 0.5		# epsilon in epsilon-greedy (used in generating SARS')
# SCORE(X) = 10(X^2 - X)

net = MLPRegressor(hidden_layer_sizes=(10,))
net.coefs_ = [list([[  1.41064131e+00,   1.46948691e+00,  -2.24180998e-02,
         -3.78213189e-02,  -7.88690564e-47,   1.06024932e+00,
          3.02749706e-77,   1.01993214e-65,   1.57168527e-45,
          1.33264699e+00],
       [  1.36787233e+00,   1.48466498e+00,  -2.24184017e-02,
         -2.70854246e-01,  -3.20318084e-90,   1.43620710e+00,
         -6.97627193e-56,   8.66637950e-63,   1.30390091e-58,
          1.36911041e+00],
       [  1.13074871e+00,   1.14861820e+00,  -2.24064805e-02,
         -3.57387868e-02,  -3.63739675e-84,   1.25371029e+00,
          2.74808685e-88,   7.95821987e-72,  -6.16637575e-95,
          9.14102736e-01],
       [  1.56892593e+00,   1.66921605e+00,  -2.24187276e-02,
          3.97464248e-02,  -5.00913299e-96,   1.32831486e+00,
          5.23613549e-47,   7.96319992e-47,   1.11270815e-97,
          1.58048805e+00],
       [  1.46609153e+00,   1.64144575e+00,  -2.24185198e-02,
         -3.70583893e-02,  -1.48994534e-83,   1.48847808e+00,
         -2.86366630e-94,   5.36510314e-98,   1.61654579e-54,
          1.78469820e+00],
       [  4.27915262e-01,   6.13957196e-01,  -2.33231506e-02,
         -7.66902167e-03,  -3.89765017e-48,   5.10179927e-01,
          8.11190958e-68,   1.77924760e-50,  -1.81057442e-58,
          4.42326647e-01],
       [  1.72333659e+00,   1.64688949e+00,  -2.24184017e-02,
         -3.79464251e-02,  -5.55046234e-99,   1.55477628e+00,
          1.21867458e-96,  -9.09268946e-63,  -7.93446998e-74,
          1.66607761e+00],
       [  1.65764539e+00,   1.81899065e+00,  -2.24311935e-02,
         -4.40808094e-02,   5.77931344e-94,   1.49321653e+00,
         -2.22456818e-71,   2.47084981e-99,  -2.91387914e-99,
          1.74245198e+00],
       [  1.77452049e+00,   1.59332372e+00,  -2.24185198e-02,
         -8.41958797e-03,  -1.33421184e-72,   1.47806937e+00,
          3.21925485e-63,   2.84528628e-89,   7.20529734e-98,
          1.67939888e+00],
       [  1.72283323e+00,   1.81996851e+00,  -2.24186142e-02,
         -4.08878027e-02,  -5.87322523e-74,   1.70523929e+00,
         -2.39415123e-60,   1.37579163e-90,  -3.34827922e-98,
          1.62981667e+00],
       [  1.49334508e+00,   1.64927013e+00,  -2.36990836e-02,
         -1.14629487e-01,  -8.11789175e-56,   1.52021563e+00,
          5.35984102e-61,  -1.43319115e-59,  -2.33143502e-93,
          1.73770059e+00],
       [  1.75491696e+00,   1.73139747e+00,  -2.24184017e-02,
         -4.84019108e-02,  -2.16493726e-98,   1.54103295e+00,
         -1.35152193e-76,   9.97747019e-58,  -1.70468225e-96,
          1.59777217e+00],
       [  1.72605524e+00,   1.88481937e+00,   1.92892970e-03,
         -3.69851379e-02,   3.37490590e-93,   1.49437858e+00,
          2.94218424e-98,   6.42548932e-87,  -1.16766145e-89,
          1.77561187e+00],
       [  1.90768844e+00,   1.93244124e+00,  -2.74564574e-02,
         -2.71667538e-01,  -2.18666923e-49,   1.55576580e+00,
         -1.58598149e-75,   3.78152642e-98,   4.23037119e-99,
          1.73901334e+00],
       [  1.55641842e+00,   1.79417984e+00,  -5.33388254e-02,
         -3.65821918e-02,  -4.33876657e-76,   1.38924817e+00,
         -4.44999122e-91,   2.60235079e-78,  -1.34557746e-92,
          1.65395741e+00],
       [  1.80106140e+00,   1.71822717e+00,  -2.24185198e-02,
         -6.08623498e-02,  -2.63200165e-95,   1.63469537e+00,
         -1.59828045e-75,   8.96513524e-57,  -1.54236694e-91,
          1.63753035e+00],
       [  1.63285069e+00,   1.69965960e+00,  -2.24185788e-02,
         -9.00696543e-04,   3.80723743e-70,   1.64507097e+00,
          1.53521926e-58,  -6.48996765e-73,  -2.38599167e-61,
          1.43841903e+00],
       [  1.66784122e+00,   1.64395019e+00,  -1.94467567e-02,
          3.32720907e-02,  -4.44534570e-59,   1.50816903e+00,
         -7.64643820e-98,   2.75376561e-97,   1.07638351e-88,
          1.70558378e+00],
       [  1.61329682e+00,   1.63271565e+00,  -4.29314297e-02,
          1.94655265e-01,  -1.12144809e-50,   1.27333655e+00,
          1.82673396e-54,   3.71140173e-98,  -1.62296556e-59,
          1.61163731e+00],
       [  1.59510539e+00,   1.81980312e+00,  -3.30870058e-02,
         -3.71166967e-02,  -7.87829729e-85,   1.42915028e+00,
         -4.36115173e-69,   2.03217920e-97,   3.22513628e-94,
          1.62940354e+00],
       [  1.79143343e+00,   1.57172023e+00,  -2.24185198e-02,
         -1.51838680e-01,  -4.99287499e-66,   1.73919962e+00,
         -1.17830405e-98,  -2.28552643e-78,  -4.83889897e-56,
          1.83747478e+00],
       [  1.74400131e+00,   1.57415224e+00,  -3.35400574e-02,
         -5.86277565e-02,  -4.11793646e-94,   1.37246761e+00,
          1.05188608e-90,  -1.92722372e-92,   1.73788699e-89,
          1.56492990e+00],
       [  1.60255981e+00,   1.70390259e+00,  -2.50406772e-02,
         -2.13121414e-01,   1.63413144e-69,   1.43694087e+00,
         -9.56922597e-98,   8.74895335e-85,  -1.80515318e-50,
          1.60100365e+00],
       [  1.53168374e+00,   1.84291014e+00,  -2.24414250e-02,
          3.46318652e-02,   3.69355801e-75,   1.53155037e+00,
          9.85110048e-51,  -4.91456836e-54,  -1.06266143e-47,
          1.46674520e+00]]), list([[  7.61211153e-01],
       [  5.29023058e-01],
       [ -6.76783513e-01],
       [ -1.23527535e-01],
       [  1.04599422e-01],
       [  1.06178562e+00],
       [ -1.09977597e-43],
       [ -1.22990539e-90],
       [ -3.14851814e-21],
       [  7.33380751e-01]])];

net.intercepts_ = [list([ 1.44847648,  1.47542637,  0.51003163,  0.45278632, -0.0056204 ,
        1.53020242, -0.23453891, -0.00187764, -0.21982535,  1.69397764]), list([ 1.9355952])];

net.n_outputs_ = 1;
net.n_layers_ = 3;
net.out_activation_ = "identity";


def dotProd(a,b):
    return sum(a[i]*b[i] for i in xrange(len(a)))

def arrToTuple(arr):
    tupArr = [tuple(elem) for elem in arr]
    return tuple(tupArr)

def isEndState(state):
    return state[1] == 0

def actions(state):
    actions = []
    grid = state[0]
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            if isValidMove(grid,(i,j),(i,j+1)):
                coord1 = (i,j)
                coord2 = (i,j+1)
                actions.append((coord1, coord2))
            elif isValidMove(grid,(i,j),(i+1,j)):
                coord1 = (i,j)
                coord2 = (i+1,j)
                actions.append((coord1, coord2))
    return actions

def dropCols(grid, colsToRowsMap):
    for col in colsToRowsMap:
        currRows = colsToRowsMap[col]
        minRow = currRows[0]
        maxRow = currRows[1]
        diff = maxRow - minRow + 1
        for i in xrange(minRow):
            grid[minRow-1-i+diff, col] = grid[minRow-1-i, col]
        for i in xrange(diff):
            grid[i, col] = random.randint(1, NCOLORS)
    return

def exploreCoord(grid, i, j):
    color = grid[i, j]
    colorRowsSet = set([(i, j)])
    colorColsSet = set([(i, j)])
    #explore above
    for row in xrange(i):
        if color != grid[i-row-1, j]:
            break
        colorRowsSet.add((i-row-1, j))
    #explore below
    for row in xrange(i+1, NROWS):
        if color != grid[row, j]:
            break
        colorRowsSet.add((row, j))
    #explore left
    for col in xrange(j):
        if color != grid[i, j-col-1]:
            break
        colorColsSet.add((i, j-col-1))
    #explore right
    for col in xrange(j+1, NCOLS):
        if color != grid[i, col]:
            break
        colorColsSet.add((i, col))
    return colorRowsSet, colorColsSet

def cascadeCoords(grid):
    coords = set()
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            colorRowsSet, colorColsSet = exploreCoord(grid, i, j)
            colorAllSet = set()
            if len(colorRowsSet) >= 3:
                colorAllSet = colorRowsSet
            if len(colorColsSet) >= 3:
                colorAllSet = colorAllSet.union(colorColsSet)
            if len(colorAllSet) > len(coords):
                coords = colorAllSet
    return coords

def cascade(grid):
    turnScore = 0
    combo = 1
    while (True):
        coords = cascadeCoords(grid)
        cascadeSize = len(coords)
        if cascadeSize < 3:
            break
        addScore = 10*(cascadeSize**2-cascadeSize)*combo
        turnScore += addScore
        combo += 1
        colsToRowsMap = {}
        for coord in coords:
            if coord[1] not in colsToRowsMap:
                colsToRowsMap[coord[1]] = [coord[0], coord[0]]
            else:
                currRows = colsToRowsMap[coord[1]]
                currRowMin = currRows[0]
                currRowMax = currRows[1]
                if coord[0] < currRowMin:
                    currRows[0] = coord[0]
                elif coord[0] > currRowMax:
                    currRows[1] = coord[0]
        dropCols(grid, colsToRowsMap)
        # if True:
        if False:
            print coords
            print 'deleted', cascadeSize, 'positions'
            print grid
            print addScore
    return grid, turnScore

def testSwitch(state, action):
    grid = np.array(state[0])
    coord1, coord2 = action
    grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
    coords = cascadeCoords(grid)
    return len(coords)

def maxCount(state):
    grid = np.array(state[0])
    rowCount = collections.Counter()
    colCount = collections.Counter()
    max_row = []
    max_col = []
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            rowCount[(i, grid[i,j])] += 1
        max_row.append(max(rowCount[(i, color)] for color in xrange(1, NCOLORS)))
    for j in xrange(NCOLS):
        for i in xrange(NROWS):
            colCount[(j, grid[i,j])] += 1
        max_col.append(max(colCount[(j, color)] for color in xrange(1, NCOLORS)))
    return max_row + max_col

def initialize(grid):
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            grid[i, j] = random.randint(1, NCOLORS)
    print grid
    grid, turnScore = cascade(grid)
    return grid

def isValidCoord(coord):
    if (coord[0] < 0 or coord[0] >= NROWS) or (coord[1] < 0 or coord[1] >= NCOLS):
        return False
    else:
        return True

def isValidMove(grid,coord1,coord2):
    #coord is (x,y)
    if coord1 == coord2:
        return False
    if isValidCoord(coord1) and isValidCoord(coord2):
        if (abs(coord1[0] - coord2[0]) == 1 and coord1[1] == coord2[1]) or (abs(coord1[1] - coord2[1]) == 1 and coord1[0] == coord2[0]):
            gridCopy = np.copy(grid)
            gridCopy[coord1], gridCopy[coord2] = gridCopy[coord2], gridCopy[coord1]
            colorRowsSet1, colorColsSet1 = exploreCoord(gridCopy, coord1[0], coord1[1])
            if len(colorRowsSet1) >= 3 or len(colorColsSet1) >= 3:
                return True
            colorRowsSet2, colorColsSet2 = exploreCoord(gridCopy, coord2[0], coord2[1])
            if len(colorRowsSet2) >= 3 or len(colorColsSet2) >= 3:
                return True
    return False

def numValidMoves(grid):
    count = 0
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            if isValidMove(grid,(i,j),(i,j+1)) or isValidMove(grid,(i,j),(i+1,j)):
                count += 1
    return count

def validMoveExists(grid):
    for i in xrange(NROWS):
        for j in xrange(NCOLS):
            if isValidMove(grid,(i,j),(i,j+1)) or isValidMove(grid,(i,j),(i+1,j)):
                return True
    return False

def makeSwitch(state, action):
    grid = np.array(state[0])
    coord1, coord2 = action
    grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
    grid, turnScore = cascade(grid)
    return grid, turnScore

def generateSARSA(currState, action):
    prevState = currState
    depth = DEPTH
    utility = 0
    depth = 0
    while ((not isEndState(prevState)) and depth < DEPTH):
        if depth != 0:
            action = random.choice(actions(prevState))
        grid, reward = makeSwitch(prevState, action)
        newState = (arrToTuple(grid), prevState[1]-1)
        prevState = newState
        utility += reward * (DISCOUNT**depth)
        depth += 1
    return utility

def getFeatureVec(state, action):
    minRow = min(action[0][0], action[1][0])
    maxRow = max(action[0][0], action[1][0])
    sameCol = 1 if action[0][1] == action[1][1] else 0
    nValidMoves = numValidMoves(state[0])
    maxDelete = testSwitch(state, action)
    medUtil = np.median([generateSARSA(state, action) for i in xrange(NSARS)])
    phi = [minRow, maxRow, sameCol, nValidMoves, maxDelete, medUtil]
    phi += maxCount(state)
    return np.array(phi)

def getQopt(state, action):
    global net
    if isEndState(state): return 0.
    return net.predict([getFeatureVec(state, action)])[0]


# def updateQ(state, action, reward, newState):
# 	Vopt, pi_opt = max((Qopt[(newState, action)], action) for action in actions(newState))
# 	Qopt[(state, action)] = (1-STEP_SIZE)*Qopt[(state, action)] + STEP_SIZE*(reward + DISCOUNT*Vopt)
# 	updated.add(state)
# 	return

# def updateWeights(state, action, reward, newState):
# 	global weights
# 	Vopt, pi_opt = max((getQopt(newState, action), action) for action in actions(newState))
# 	weights = weights - STEP_SIZE * (getQopt(state, action) - (reward + DISCOUNT*Vopt)) * getFeatureVec(state, action)
# 	for i in xrange(NFEATURES):
# 		if weights[i] < 0:
# 			weights[i] = 0.
# 	return

def aiPlay(grid, sample):
    score = 0
    # prevWeights = np.zeros(24, dtype = float)
    for turn in xrange(NTURNS):
        while True:
            if validMoveExists(grid):
                break
            grid = np.resize((1,NROWS*NCOLS))
            np.random.shuffle(grid)
            grid = np.resize((NROWS, NCOLS))
        turnsLeft = NTURNS - turn
        print ''
        print 'Turns left:', NTURNS - turn
        print 'SCORE', score
        print grid
        # print 'generating SARSA...'
        currState = (arrToTuple(grid), turnsLeft)
        # sarsa = [generateSARSA(currState) for i in xrange(NSARSA)]
        # print 'updating weights'
        # for i in xrange(NSARSA):
        # 	print i
        # 	for (state, action, reward, newState) in sarsa[i]:
        # 		updateWeights(state, action, reward, newState)
        # print 'choosing action...'
        # action = None
        # if (random.random() < EPSILON):
        # 	print 'choosing random action...'
        # 	action = random.choice(actions(currState))
        # else:
        print 'choosing optimal action...'
        Vopt, pi_opt = max((getQopt(currState, action), action) for action in actions(currState))
        action = pi_opt
        print 'switched', action[0], 'and', action[1]
        grid, turnScore = makeSwitch(currState, action)
        score += turnScore
        # newState = (arrToTuple(grid), turnsLeft-1)
        # print 'updating weights'
        # updateWeights(currState, action, turnScore, newState)
        # print 'new weights (scaled):',
        # if weights.all() == 0:
        # 	scaledWeights = weights
        # else:
        # 	scaledWeights = weights/min(weights[i] for i in xrange(NFEATURES) if weights[i] > 0)
        # print np.round(scaledWeights, 3)
        # print 'weight diff:', scaledWeights - prevWeights
        # prevWeights = scaledWeights
    print ''
    print 'Turns left:', 0
    print grid
    print sample, 'FINAL SCORE', score
    return score

def main():
    random.seed()
    np.random.seed()
    scoreSum = 0
    for i in xrange(NUMSAMPLES):
        grid = np.zeros((NROWS, NCOLS), dtype = int)
        grid = initialize(grid)
        score = aiPlay(grid, i+1)
        scoreSum += score
        print i+1, 'AVERAGE SCORE:', float(scoreSum)/(i+1)
        print ''
    return

if __name__ == "__main__":
    main()
