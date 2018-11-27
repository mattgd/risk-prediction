import csv
import json
import numpy as np
import pandas
import random
import statistics

# TODO: Create class for ML and Agent methods

RISK_SCORE_MIN = 10
RISK_SCORE_MAX = 47
NUM_DECISION_POINTS = 9
DEFAULT_AGENT_MAX = 512

SURVEY_SCORE_COLUMN = 'financialRisk_score (N)'
DECISIONS = [
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
KEEP_COLS = [SURVEY_SCORE_COLUMN] + DECISIONS

# In the dictionary below, a is considered risk-taking, and b is risk-adverse.
REPLACE_RESPONSES = {
    'a': [
        'No',
        'Speed up',
        'Car',
        'Sprint away as fast as you can',
        'Quiet',
        'Drag',
        'Sneak',
        'Bottle',
        'Plea',
        'Throw'
    ],
    'b': [
        'Yes',
        'Confront him',
        'Woods',
        'Slowly walk away (maybe you’ll lose him)',
        'Run',
        'Cut',
        'Outside',
        'Knife',
        'Fight',
        'Swing'
    ]
}

def categorize_decisions(decisions):
    """
    Categorizes the activity decisions (used in preprocessing).
    """
    for idx, decision in enumerate(decisions):
        for replace_key, replace_values in REPLACE_RESPONSES.items():
            if decision in replace_values:
                decisions[idx] = replace_key

    return decisions

def preprocess(file_name):
    """
    Gets the data from the survey responses CSV file and preprocesses it.
    """
    with open(file_name) as csvfile:
        response_reader = csv.reader(csvfile)
        columns = next(response_reader)
        
        # Get a list of column indices to keep
        new_columns = {}
        for idx, col_name in enumerate(columns):
            if col_name in KEEP_COLS:
                new_columns[col_name] = idx

        survey_score_col_idx = new_columns[SURVEY_SCORE_COLUMN]
        data = []
        for row in response_reader:
            row = [(int(row[idx]) if idx == survey_score_col_idx else row[idx]) for idx in range(len(row)) if idx in new_columns.values()]
            row = categorize_decisions(row)
            data.append(row)

    headers = [col[0] for col in sorted(new_columns.items(), key = lambda x : x[1])]
    data = pandas.DataFrame(data=data, columns=headers)

	  # Drop columns where all values are the same
    nunique = data.apply(pandas.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    data = data.drop(cols_to_drop, axis=1)

    return data

def get_possible_paths(num_decision_points):
    """
    :returns: All of the possible paths (binary strings) given the number of 
              decision points
    """
    paths = []

    for i in range(1 << num_decision_points):
        s = bin(i)[2:]
        s = '0' * (num_decision_points - len(s)) + s
        paths.append(s)

    return paths

def simulate(file_name='survey_responses.csv', agent_max=DEFAULT_AGENT_MAX):
    """
    Runs the risk tolerance simulation, given a data file.

    :param file_name: The name of the data file
    :param agent_max: The number of agents to simulate
    """
    data = preprocess(file_name)
    survey_scores = map(int, data[SURVEY_SCORE_COLUMN].tolist())
    neutral_state = statistics.median(survey_scores)

    # Initialize decision points for all agents
    decision_points = {}
    decision_cols = [col for col in list(data) if col != SURVEY_SCORE_COLUMN]
    for decision in decision_cols:
        decision_points[decision] = initialize_decision_point(data, decision)

    # Simulate agent paths where each path key is a binary String representing the decision
    predictions = {path: [] for path in get_possible_paths(NUM_DECISION_POINTS)}
    for _ in range(agent_max):
        agent_risk_tolerance = random.randint(RISK_SCORE_MIN, RISK_SCORE_MAX + 1)
        path = ''

        for decision_point in decision_points.values():
            base_rate, risk_sensitivity, toggled = decision_point
            choice = calculate_risk_tolerance(
                base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance
            )

            path += str(choice)

        predictions[path].append(agent_risk_tolerance)

    print(json.dumps(predictions))

def predict(predictions, user_path):
    """
    Predict the user's risk tolerance based on the agent data.

    :param predictions: The predictions dictionary
    :param user_path: The user's path (a binary string)
    :returns: The predicted risk tolerance score
    """
    return np.mean(predictions[user_path])

def calculate_risk_tolerance(base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance):
    """
    Function for an agent to make a choice for a decision point.

    :param base_rate: The base decision rate, defined as n1 / (n1 + n0), where 0 < base_rate < 1
    :param risk_sensitivity: The agent's sensitivity to risk, defined as s1 / s0 where 1 < risk_sensitivity < 4.7
    :param neutral_state: The neutral state for the set, defined as median risk tolerance of all respondents (independent of decision)
    :param agent_risk_tolerance: The agent's generated risk tolerance level, where 10 < agent_risk_tolerance < 47 (range given by survey)
    :returns: The agent's risk score
    """
    agent_risk_factor = agent_risk_tolerance - neutral_state
    decision_risk_factor = risk_sensitivity - 1
    risk_adjustment = agent_risk_factor * decision_risk_factor
    probability_1 = base_rate + risk_adjustment

    return 1 if probability_1 < random.random() else 0

def initialize_decision_point(data, column):
    """
    Initializes the risk sensitivity and base rate.

    :param data_col: Column of preprocessed data (0 and 1 choices for one decision point)
    :returns: The base rate and risk sensitivity for this decision point
    """
    groups = data.groupby(column)[SURVEY_SCORE_COLUMN]
    counts = groups.size()
    means = groups.mean()
    
    s_a = means['a']
    s_b = means['b']
    n_a = counts['a']
    n_b = counts['b']

    if s_a > s_b:
        risk_sensitivity = s_a / s_b
        base_rate = n_a / (n_a + n_b)
        toggled = True
    else:
        risk_sensitivity = s_b / s_a
        base_rate = n_b / (n_a + n_b)
        toggled = False

    return base_rate, risk_sensitivity, toggled

if __name__ == '__main__':
    simulate()
