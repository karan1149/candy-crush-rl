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

# 1. number of valid pairs in state
# 2. number in a row produced by switched: testSwitch
# 3. max count of a number in a row: maxCount[0]
# 4. Max count of a number in a col: maxCount[1]
# 5. average score from running cascadeGrid NSCORES times from input state and action 
'''

import random
import numpy as np
import collections

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
STEP_SIZE = 0.0001		# eta in computing Q_opt
EPSILON = 0.5		# epsilon in epsilon-greedy (used in generating SARS')
# SCORE(X) = 10(X^2 - X)
weights = np.zeros(NFEATURES, dtype = float)

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
	if isEndState(state): return 0.
	return dotProd(getFeatureVec(state, action), weights)


# def updateQ(state, action, reward, newState):
# 	Vopt, pi_opt = max((Qopt[(newState, action)], action) for action in actions(newState))
# 	Qopt[(state, action)] = (1-STEP_SIZE)*Qopt[(state, action)] + STEP_SIZE*(reward + DISCOUNT*Vopt)
# 	updated.add(state)
# 	return

def updateWeights(state, action, reward, newState):
	global weights
	Vopt, pi_opt = max((getQopt(newState, action), action) for action in actions(newState))
	weights = weights - STEP_SIZE * (getQopt(state, action) - (reward + DISCOUNT*Vopt)) * getFeatureVec(state, action)
	for i in xrange(NFEATURES):
		if weights[i] < 0:
			weights[i] = 0.
	return

def aiPlay(grid, sample):
	score = 0
	prevWeights = np.zeros(24, dtype = float)
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
		print 'choosing action...'
		action = None
		if (random.random() < EPSILON):
			action = random.choice(actions(currState))
		else:
			Vopt, pi_opt = max((getQopt(currState, action), action) for action in actions(currState))
			action = pi_opt
		print 'switched', action[0], 'and', action[1]
		grid, turnScore = makeSwitch(currState, action)
		score += turnScore
		newState = (arrToTuple(grid), turnsLeft-1)
		print 'updating weights'
		updateWeights(currState, action, turnScore, newState)
		print 'new weights:', weights
		print 'weight diff:', weights - prevWeights
		prevWeights = weights
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