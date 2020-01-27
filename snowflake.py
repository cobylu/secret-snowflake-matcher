from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer


def similarity(corpus):
    tfidf_vectorizer = TfidfVectorizer(analyzer="word",
                                   ngram_range=(1,2), 
                                   max_df=0.95, 
                                   min_df=2,
                                   stop_words="english")
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    return squareform(pdist(tfidf.todense(), "cosine"))

def optimize(similarity):
    num_people = similarity.shape[0]
    c = np.reshape(similarity, -1)
    for i in range(num_people):
        for j in range(num_people):
            c[i * num_people + j] = c[i * num_people + j] if i != j else 99999
    
    # give to only 1
    give = np.zeros((num_people, num_people**2))
    for i in range(num_people):
        give[i, i * num_people:(i + 1) * num_people] = 1
    
    # receive from only 1
    receive = np.zeros((num_people, num_people**2))
    for i in range(num_people):
        receive[i, i:num_people ** 2:num_people] = 1

    # don't give and receive from same person
    giv_rec = np.zeros((num_people * (num_people - 1), num_people**2))
    i = 0
    for comb in combinations(range(num_people), 2):
        giv_rec[i, comb[0] * num_people + comb[1]] = 1
        giv_rec[i, comb[1] * num_people + comb[0]] = 1
        i += 1

    A_eq = np.vstack((give, receive))
    b_eq = np.ones(2 * num_people)

    A_ub = giv_rec
    b_ub = np.ones((num_people - 1) * num_people)
    return linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method="revised simplex")

if __name__ == "__main__":
    df = pd.read_csv("Engineering Secret Snowflake 2019 (Responses) - Form Responses 1.csv")
    going = df.loc[df["Do you want to participate in Engineering's Secret Snowflake?"] == "Yes", ["Email Address", "List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]].reset_index(drop=True)
    corpus = going["List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]

    interest_similarity = similarity(corpus)
    result = optimize(interest_similarity)

    solution = np.round(result.x)
    members = going["Email Address"]
    num_people = len(members)
    for i, g in enumerate(members):
        for j, r in enumerate(members):
            if solution[i * num_people + j] == 1:
                print(f"{g} gives to {r}")
