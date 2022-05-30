import operator
import pandas as pd
from collections import defaultdict



def int_to_answers():
    data_path = 'data/Training Data QA.pickle'
    df = pd.read_pickle(data_path)
    answers = df[['multiple_choice_answer']].values.tolist()
    freq = defaultdict(int)
    for answer in answers:
        freq[answer[0].lower()] += 1
    int_to_answer = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)[0:1000]
    int_to_answer = [answer[0] for answer in int_to_answer]
    

    return int_to_answer