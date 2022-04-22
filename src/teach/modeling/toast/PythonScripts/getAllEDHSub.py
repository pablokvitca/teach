import json
import sys
import os

from getEDHSub import getEDHSub

print(sys.argv[1])
path = str(sys.argv[1])

def getAllEDHSub(path):
	
	files = os.listdir(path)

	x_set = []
	y_set = []

	for file in files:
		x, y = getEDHSub(path + "/" + file)

		x_set.append(x)
		y_set.append(y)

	return x_set, y_set
