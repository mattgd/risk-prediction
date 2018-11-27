import ml_predictor as ml
import agent_predictor as ag
import preprocessor as pp
import pandas as pd
import numpy as np
import sys

RAW_DATA = 'survey_responses.csv' 
TEST_PATH = pandas.DataFrame([['a']*9, ['b']*9])

if __name__ == '__main__':

    data = pp.preprocess(RAW_DATA)

    # Generate prediction for specified user
    if len(sys.argv) > 1:
        user_path = pd.DataFrame([list(sys.argv[1])])
        if len(user_path) != 9:
            sys.exit('Path requires 9 decision points')
        p_ml = ml.MlPredictor(data)
        p_ag = ag.AgentPredictor(data)
        print("Machine Learning Prediction: " + str(p_ml.predict(user_path)[0]))
        print("Simulated Agent Prediction: " + str(p_ag.predict(user_path)[0]))

    # Repeatedly run comparison of predictors
    else:
        pass
