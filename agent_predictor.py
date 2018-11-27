import json
import numpy as np
import pandas as pd
import random
from preprocessor import SURVEY_SCORE_COLUMN
import statistics

RISK_SCORE_MIN = 10
RISK_SCORE_MAX = 47
DEFAULT_AGENT_MAX = 512

class AgentPredictor:
    """
    Setup a new agent predictor.

    :param data: preprocessed training data
    """
    def __init__(self, data):
        self.decision_cols = [col for col in list(data) if col != SURVEY_SCORE_COLUMN]
        self.num_decision_points = len(self.decision_cols)
        self.predictions = {path: [] for path in self.get_possible_paths(self.num_decision_points)}
        self.simulate(data)

    def simulate(self, data, agent_max=DEFAULT_AGENT_MAX):
        """
        Runs the risk tolerance simulation, given a data file.

        :param data: preprocessed data frame
        :param agent_max: The number of agents to simulate
        """
        survey_scores = map(int, data[SURVEY_SCORE_COLUMN].tolist())
        neutral_state = statistics.median(survey_scores)

        # Initialize decision points for all agents
        decision_points = {}
        for decision in self.decision_cols:
            decision_points[decision] = self.initialize_decision_point(data, decision)

        # Simulate agent paths where each path key is a binary String representing the decision
        for _ in range(agent_max):
            agent_risk_tolerance = random.randint(RISK_SCORE_MIN, RISK_SCORE_MAX + 1)
            path = ''

            for decision_point in decision_points.values():
                base_rate, risk_sensitivity, toggled = decision_point
                choice = self.calculate_risk_tolerance(
                    base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance
                )

                path += str(choice)

            self.predictions[path].append(agent_risk_tolerance)

    def predict(self, user_path):
        """
        Predict the user's risk tolerance based on the agent data.

        :param predictions: The predictions dictionary
        :param user_path: The user's path (a binary string)
        :returns: The predicted risk tolerance score
        """
        predictions = []

        for row in user_path.iterrows():
            user_path_str = row[1].str.cat(sep='')
            predictions.append(np.mean(self.predictions[user_path_str]))
        
        return predictions

    def initialize_decision_point(self, data, column):
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

    def get_possible_paths(self, num_decision_points):
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

    def calculate_risk_tolerance(self, base_rate, risk_sensitivity, neutral_state, agent_risk_tolerance):
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