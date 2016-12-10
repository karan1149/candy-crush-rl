#!/usr/bin/env python

import random
import numpy as np

HUMAN = True	
NUMSAMPLES = 10
NTURNS = 10
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
	global SCORE
	global COMBO
	while (True):
		coords = cascadeCoords(grid)
		cascadeSize = len(coords)
		if cascadeSize < 3:
			break
		addScore = 10*(cascadeSize**2-cascadeSize)*COMBO
		SCORE += addScore
		COMBO += 1
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
	return

def initialize(grid):
	for i in xrange(NROWS):
		for j in xrange(NCOLS):
			grid[i, j] = random.randint(1, NCOLORS)
	print grid
	cascade(grid)
	return

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

def humanPlay(grid):
	global SCORE
	global COMBO
	for turn in xrange(NTURNS):
		print ''
		print 'Turns left:', NTURNS - turn
		print 'SCORE', SCORE
		print grid
		COMBO = 1
		while True:
			if validMoveExists(grid):
				break
			grid = np.resize((1,NROWS*NCOLS))
			np.random.shuffle(grid)
			grid = np.resize((NROWS, NCOLS))
		coord1 = []
		coord2 = []
		while True:
			while True:
				rawCoord1 = raw_input('Enter first coordinate to switch in the form "row,col": ')
				coord1 = rawCoord1.split(',')
				if len(coord1) != 2 or not coord1[0].isdigit() or not coord1[1].isdigit():
					print 'Invalid Coord'
					continue
				coord1[0] = int(coord1[0])
				coord1[1] = int(coord1[1])
				coord1 = tuple(coord1)
				if not isValidCoord(coord1):
					print 'Invalid Coord'
				else:
					break
			while True:
				rawCoord2 = raw_input('Enter second coordinate to switch in the form "row,col": ')
				coord2 = rawCoord2.split(',')
				if len(coord2) != 2 or not coord2[0].isdigit() or not coord2[1].isdigit():
					print 'Invalid Coord'
					continue
				coord2[0] = int(coord2[0])
				coord2[1] = int(coord2[1])
				coord2 = tuple(coord2)
				if not isValidCoord(coord2):
					print 'Invalid Coord'
				else:
					break
			if not isValidMove(grid, coord1, coord2):
				print 'Invalid move'
			else:
				break
		print 'switched', coord1, 'and', coord2
		grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
		cascade(grid)
		print 'Combo of %s!' % (COMBO-1)
	print ''
	print 'Turns left:', 0
	print grid
	print 'FINAL SCORE', SCORE
	return

def aiPlay(grid, sample):
	global SCORE
	global COMBO
	for turn in xrange(NTURNS):
		print ''
		print 'Turns left:', NTURNS - turn
		print 'SCORE', SCORE
		print grid
		COMBO = 1
		while True:
			if validMoveExists(grid):
				break
			grid = np.resize((1,NROWS*NCOLS))
			np.random.shuffle(grid)
			grid = np.resize((NROWS, NCOLS))
		coord1 = []
		coord2 = []
		for i in xrange(NROWS):
			if len(coord1) > 0 or len(coord2) > 0:
				break
			for j in xrange(NCOLS):
				if isValidMove(grid,(i,j),(i,j+1)):
					coord1 = (i,j)
					coord2 = (i,j+1)
					break
				elif isValidMove(grid,(i,j),(i+1,j)):
					coord1 = (i,j)
					coord2 = (i+1,j)
					break
		print 'switched', coord1, 'and', coord2
		grid[coord1], grid[coord2] = grid[coord2], grid[coord1]
		cascade(grid)
		print 'Combo of %s!' % (COMBO-1)
	print ''
	print 'Turns left:', 0
	print grid
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
