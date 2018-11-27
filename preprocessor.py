import csv
import pandas

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
    #nunique = data.apply(pandas.Series.nunique)
    #cols_to_drop = nunique[nunique == 1].index
    #data = data.drop(cols_to_drop, axis=1)

    return data
