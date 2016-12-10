#!/usr/bin/env python

import random
import numpy as np
import collections

NSARSA = 25         # number of SARS' sequences to generate at each turn
NUMSAMPLES = 1000	# number of total trials to run
NTURNS = 50			# number of turns in the game
NROWS = 9		
NCOLS = 9
NCOLORS = 5
DISCOUNT = 0.5		# discount in computing Q_opt
STEP_SIZE = 0.2		# eta in computing Q_opt
EPSILON = 0.2		# epsilon in epsilon-greedy (used in generating SARS')
# SCORE(X) = 10(X^2 - X)
Qopt = collections.defaultdict(int)
updated = set()

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

def validMoveExists(grid):
	for i in xrange(NROWS):
		for j in xrange(NCOLS):
			if isValidMove(grid,(i,j),(i,j+1)) or isValidMove(grid,(i,j),(i+1,j)):
				return True
	return False

def generateSARSA(currState):
	sarsa = []
	prevState = currState
	while (not isEndState(prevState)):
		action = None
		if prevState in updated:
			if (random.random() < EPSILON):
				action = random.choice(actions(prevState))
			else:
				Vopt, action = max((Qopt[(prevState, action)], action) for action in actions(prevState))
		else:
			action = random.choice(actions(prevState))
		grid, reward = makeSwitch(prevState, action)
		newState = (arrToTuple(grid), prevState[1]-1)
		sarsa.append((prevState, action, reward, newState))
		prevState = newState
	return sarsa

def makeSwitch(state, action):
	grid = np.array(state[0])
	coord1, coord2 = action
	grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
	grid, turnScore = cascade(grid)
	return grid, turnScore

def updateQ(state, action, reward, newState):
	Vopt, pi_opt = max((Qopt[(newState, action)], action) for action in actions(newState))
	Qopt[(state, action)] = (1-STEP_SIZE)*Qopt[(state, action)] + STEP_SIZE*(reward + DISCOUNT*Vopt)
	updated.add(state)
	return

def aiPlay(grid, sample):
	score = 0
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
		print 'generating SARSA...'
		currState = (arrToTuple(grid), turnsLeft)
		sarsa = [generateSARSA(currState) for i in xrange(NSARSA)]	
		print 'updating Qopt...'
		for i in xrange(NSARSA):
			for (state, action, reward, newState) in sarsa[i]:
				updateQ(state, action, reward, newState)
		print 'obtaining optimal action...'
		Vopt, pi_opt = max((Qopt[(currState, action)], action) for action in actions(currState))
		action = pi_opt
		print 'switched', action[0], 'and', action[1]
		grid, turnScore = makeSwitch(currState, action)
		score += turnScore
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