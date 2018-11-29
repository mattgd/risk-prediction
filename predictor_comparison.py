import ml_predictor as ml
import agent_predictor as ag
import preprocessor as pp
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
import math
from matplotlib import pyplot as plt

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
        print("Linear Regression Prediction: " + str(p_ml.predict(user_path)[0]))
        print("Simulated Agent Prediction: " + str(p_ag.predict(user_path)[0]))

    # Repeatedly run comparison of predictors
    else:
        ml_errors = []
        ag_errors = []
        for i in range(10):

            # Split into test and train data
            msk = np.random.rand(len(data)) < .8
            train = data[msk]
            while(any([len(np.unique(train[a]))==1 for a in train.columns])):
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

            # Calculate errors
            error_ml = math.sqrt(mean_squared_error(expected_ml, actual_ml))
            error_ag = math.sqrt(mean_squared_error(expected_ag, actual_ag))
            ml_errors.append(error_ml)
            ag_errors.append(error_ag)
            off_ml = np.mean([abs(x[0]-x[1]) for x in zip(expected_ml, actual_ml)])
            off_ag = np.mean([abs(x[0]-x[1]) for x in zip(expected_ag, actual_ag)])

            # Report
            print("Run " + str(i+1))
            print("\tLinear Regression Error: " + str(error_ml))
            print("\tSimulated Agent Error: " + str(error_ag))
            print()

        # Calculate averages
        ml_errors.append(np.mean(ml_errors))
        ag_errors.append(np.mean(ag_errors))

        print("Overall Report")
        print("\tLinear Regression Error: " + str(ml_errors[-1]))
        print("\tSimulated Agent Error: " + str(ag_errors[-1]))
        print()

        x_pos = np.arange(len(ml_errors))
        w = .3
        ml_bars = plt.bar(x_pos-w/2, ml_errors, w, label='Linear Regression', color='cornflowerblue')
        ag_bars = plt.bar(x_pos+w/2, ag_errors, w, label='Agent Simulation', color='salmon')
        ml_bars[-1].set_color('darkblue')
        ag_bars[-1].set_color('darkred')
        plt.xticks(x_pos, ['Run ' + str(x+1) for x in range(10)] + ['Average'])
        plt.ylabel('Root Mean Squared Error')
        plt.legend()
        plt.title('Prediction Errors')
        plt.show()
