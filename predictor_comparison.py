import ml_predictor as ml
import agent_predictor as ag
import preprocessor as pp
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import r2_score as accuracy

RAW_DATA = 'survey_responses.csv' 

if __name__ == '__main__':

    data = pp.preprocess(RAW_DATA)

    # Generate prediction for specified user
    if len(sys.argv) > 1:
        user_path = pd.DataFrame([list(sys.argv[1])])
        if len(user_path.columns) != 9:
            sys.exit('Path requires 9 decision points')
        p_ml = ml.MlPredictor(data)
        p_ag = ag.AgentPredictor(data)
        print("Machine Learning Prediction: " + str(p_ml.predict(user_path)[0]))
        print("Simulated Agent Prediction: " + str(p_ag.predict(user_path)[0]))

    # Repeatedly run comparison of predictors
    else:
        ml_accuracies = []
        ag_accuracies = []
        ml_offness = []
        ag_offness = []
        for i in range(10):

            # Split into test and train data
            msk = np.random.rand(len(data)) < .8
            train = data[msk]
            test = data[~msk]
            expected_ml = list(test.iloc[:,0])
            expected_ag = list(expected_ml)
            actual_input = test.iloc[:,1:]

            # Predict with both predictors
            p_ml = ml.MlPredictor(train)
            p_ag = ag.AgentPredictor(train)
            actual_ml = p_ml.predict(actual_input)
            actual_ag = p_ag.predict(actual_input)
            if any(isinstance(x, str) for x in actual_ag):
                idxs = [i for i, x in enumerate(actual_ag) if isinstance(x, str)]
                for idx in reversed(idxs):
                    actual_ag.pop(idx)
                    expected_ag.pop(idx)

            # Calculate Accurcies
            acc_ml = accuracy(expected_ml, actual_ml)
            acc_ag = accuracy(expected_ag, actual_ag)
            ml_accuracies.append(acc_ml)
            ag_accuracies.append(acc_ag)
            off_ml = np.mean([abs(x[0]-x[1]) for x in zip(expected_ml, actual_ml)])
            off_ag = np.mean([abs(x[0]-x[1]) for x in zip(expected_ag, actual_ag)])
            ml_offness.append(off_ml)
            ag_offness.append(off_ag)

            # Report
            print("Run " + str(i))
            print("\tMachine Learning Accuracy: " + str(acc_ml))
            print("\tSimulated Agent Accuracy: " + str(acc_ag))
            print("\tMachine Learning Offness: " + str(off_ml))
            print("\tSimulated Agent Offness: " + str(off_ag))
            print()

        print("Overall Report")
        print("\tMachine Learning Accuracy: " + str(np.mean(ml_accuracies)))
        print("\tSimulated Agent Accuracy: " + str(np.mean(ag_accuracies)))
        print("\tMachine Learning Offness: " + str(np.mean(ml_offness)))
        print("\tSimulated Agent Offness: " + str(np.mean(ag_offness)))
        print()
