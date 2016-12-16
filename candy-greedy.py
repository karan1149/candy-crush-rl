#!/usr/bin/env python

import random
import numpy as np

HUMAN = False
NUMSAMPLES = 1000
NTURNS = 20
NROWS = 9
NCOLS = 9
NCOLORS = 5
# SCORE(X) = 10(X^2 - X)
SCORE = 0
COMBO = 1

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
	score = 0
	combo = 1
	while (True):
		coords = cascadeCoords(grid)
		cascadeSize = len(coords)
		if cascadeSize < 3:
			break		
		addScore = 10*(cascadeSize**2-cascadeSize)*combo
		score += addScore
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
		if HUMAN:
			print coords			
			print 'deleted', cascadeSize, 'positions'
			print grid
			print addScore
	return score, combo

def initialize(grid):
	for i in xrange(NROWS):
		for j in xrange(NCOLS):
			grid[i, j] = random.randint(1, NCOLORS)	
	# print grid
	cascade(grid)
	return

def isValidCoord(coord):
	if (coord[0] < 0 or coord[0] >= NROWS) or (coord[1] < 0 or coord[1] >= NCOLS):
		return False
	else:
		return True

	if coord1 == coord2:
		return False
def isValidMove(grid,coord1,coord2):
	#coord is (x,y)
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

def aiPlay(grid, sample):
	global SCORE
	global COMBO
	score = 0
	for turn in xrange(NTURNS):	
		# print ''	
		# print 'Turns left:', NTURNS - turn
		# print 'SCORE', SCORE
		# print grid	
		combo = 1
		coord1 = []
		coord2 = []
		maxScore = 0
		for i in xrange(NROWS):
			for j in xrange(NCOLS):
				tempGrid = np.array(grid)
				if isValidMove(grid,(i,j),(i,j+1)):
					candidateCoord1 = (i,j)
					candidateCoord2 = (i,j+1)
					tempGrid[candidateCoord1], tempGrid[candidateCoord2] = tempGrid[candidateCoord2], tempGrid[candidateCoord1]
					turnScore, combo = cascade(tempGrid)
					# print "Considering score:", turnScore, "which is obtained after switching", candidateCoord1, candidateCoord2
					if turnScore > maxScore:
						maxScore = turnScore
						coord1 = candidateCoord1
						coord2 = candidateCoord2
					break
				elif isValidMove(grid,(i,j),(i+1,j)):
					candidateCoord1 = (i,j)
					candidateCoord2 = (i+1,j)
					tempGrid[candidateCoord1], tempGrid[candidateCoord2] = tempGrid[candidateCoord2], tempGrid[candidateCoord1]
					turnScore, combo = cascade(tempGrid)
					# print "Considering score:", turnScore, "which is obtained after switching", candidateCoord1, candidateCoord2
					if turnScore > maxScore:
						maxScore = turnScore
						coord1 = candidateCoord1
						coord2 = candidateCoord2
					break
		# print 'switched', coord1, 'and', coord2
		grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
		score, combo = cascade(grid)
		SCORE += score
		COMBO = combo
	# 	print 'Combo of %s!' % (combo-1)	
	# print ''		
	# print 'Turns left:', 0
	# print grid	
	print sample, 'FINAL SCORE', SCORE
	return

def main():
	global SCORE
	random.seed()
	np.random.seed()
	grid = np.zeros((NROWS, NCOLS), dtype = int)
	initialize(grid)
	SCORE = 0	
	#print isValidMove(grid, (2,3), (3,3))
	if HUMAN:
		humanPlay(grid)
	else:
		scoreSum = 0
		for i in xrange(NUMSAMPLES):
			grid = np.zeros((NROWS, NCOLS), dtype = int)
			initialize(grid)
			SCORE = 0	
			aiPlay(grid, i+1)
			scoreSum += SCORE
		print 'AVERAGE SCORE:', float(scoreSum)/NUMSAMPLES
	return

if __name__ == "__main__":
	main()