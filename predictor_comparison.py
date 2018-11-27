import ml_predictor as ml
import preprocessor
import pandas

RAW_DATA = 'survey_responses.csv' 
TEST_PATH = pandas.DataFrame([['a']*9, ['b']*9])

if __name__ == '__main__':
    data = preprocessor.preprocess(RAW_DATA)
    predictor = ml.MlPredictor(data)
    prediction = predictor.predict(TEST_PATH)
    print(prediction)

