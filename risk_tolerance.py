import csv
import pandas
import numpy as np
import random

RISK_SCORE_MIN = 10
RISK_SCORE_MAX = 47

KEEP_COLS = [
	'financialRisk_score (N)',
	'question20 (S)',
	'question21 (S)',
	'question22 (S)',
	'question23 (S)',
	'question24 (S)',
	'question25 (S)',
	'question26 (S)',
	'question27 (S)',
	'question28 (S)'
]



def preprocess(file_name):
	"""
	Gets the data from the survey responses CSV file and preprocesses it.
	"""
	with open(file_name) as csvfile:
		response_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		columns = response_reader.next()
		
		# Get a list of column indices to keep
		column_indices = []
		for x in range(columns):
			col_name = columns[x]

			if col_name in KEEP_COLS:
				column_indices.append(x)

		data = []
     	for row in response_reader:
			row = [row[x] for x in range(row) if x in column_indices]
			data.append(row)

	return pandas.DataFrame(data=data, columns=columns)
		
# Preprocess data
def preprocess(file_name):
	data = raw_data.replace(1 with a).replace(2 with b)
	data = pandas.DataFrame(data=user_responses, columns=decision_choices + survey_score)

	return pre

# Run simulation
def simulate(file_name='survey_responses.csv'):
	data = preprocess(file_name)
	neutral_state = median(data.survey_scores)

	# Initialize decision points for all agents
	decision_points = [(base_rate, risk_sensitivity), ...]
	for d in decision:
		decision_points[decision] = initialize_decision_point(data[decision])

	# Simulate agent paths
	# String keys with 0, 1 representing the decision
	predictions = {keys = all_possible_paths, vals = []}
	for _ in range(agent_max):
		agent_risk_tolerance = random.randint(RISK_SCORE_MIN, RISK_SCORE_MAX + 1)
		path = []

		for decision_point in decision_points:
			choice = calculate_risk_tolerance(
				base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance
			)
			path.append(choice)

		predictions[path].append(agent_risk_tolerance)

def predict(predictions, user_path):
	"""
	Predict the user's risk tolerance based on the agent data.

	:param predictions: The predictions dictionary
	:param user_path: The user's path (a binary string)
	:returns: The predicted risk tolerance score
	"""
	return np.mean(predictions[user_path])

base_rate = n1 / (n1 + n0) where 0 < base_rate < 1
risk_sensitivity = s1 / s0 where 1 < risk_sensitivity < 4.7
neutral _state = median risk tolerance of all respondents (independent of decision)
agent_risk_tolerance = generated for agent, 10 < agent_risk_tolerance < 47 (range given by survey)

# An agent makes choice for a decision point
def calculate_risk_tolerance(base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance):
	agent_risk_factor = agent_risk_tolerance - neutral_state
	decision_risk_factor = risk_sensitivity - 1
	risk_adjustment = agent_risk_factor * decision_risk_factor
	probability_1 = base_rate + risk_adjustment
	return probability_1 < rand

data_col = column of preprocessed data (0 and 1 choices for one decision point)
# Initializes the risk sensitivity and base rate
def initialize_decision_point(data_col):
	s_a = avg(data_col == a)
	s_b = avg(data_col == b)
	n_a = count(data_col == a)
	n_b = count(data_col == b)

	if s_a > s_b:
		risk_sensitivity = s_a / s_b
		base_rate = n_a / (n_a + n_b)
	else:
		risk_sensitivity = s_b / s_a
		base_rate = n_b / (n_a + n_b)

	return base_rate, risk_sensitivity
