import json
import sys

def splitIntoWords(sentence):
	return sentence.split()

def getEDHSub(file):

	f = open(file)

	data = json.load(f)

	sub_goals = []

	

	for i in data['history_subgoals']:
		sub_goals.append(i)

	for i in data['future_subgoals']:
		sub_goals.append(i)


	dialogue_history = []
	for i in data['dialog_history']:
		dialogue_history.append(i[1])

	return dialogue_history, sub_goals
