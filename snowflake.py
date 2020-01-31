from itertools import combinations

import numpy as np
import pandas as pd
from pulp import *
from scipy.optimize import linprog
import spacy


def similarity(df):
    nlp = spacy.load("en_core_web_md")
    stop_lemmas = set(['\n', '\n\n', ' ', '!', '"', '&', "'", '(', ')', '+', ',', 
    '-', '--', '-PRON-', '.', '..', '...', '/', ':', ':)', ';', '>', '?','_', 'a', 
    'an', 'also', 'candy', 'color', 'enjoy', 'favorite', 'hobby', 'honestly', 'i',
    'interested', 'interesting', 'into', 'like', 'love', 'lover', 'need', 
    'preferably','obsessed','really'])

    corpus = df["List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]
    spacy_corpus = corpus.apply(nlp)
    for i, sentence in enumerate(spacy_corpus):
        new_sentence = []
        for word in sentence:
            if not word.is_stop and str(word.lemma_) not in stop_lemmas:
                new_sentence.append(str(word.lemma_))
        df.loc[i, "interests (cleaned)"] = " ".join(new_sentence)

    corpus = df["interests (cleaned)"]
    spacy_corpus = corpus.apply(nlp)
    spacy_similarity = np.zeros((len(spacy_corpus), len(spacy_corpus)))
    for (i, doc) in enumerate(spacy_corpus):
        for (j, other_doc) in enumerate(spacy_corpus):
            spacy_similarity[i, j] = doc.similarity(other_doc)
    
    return -spacy_similarity

def optimize(similarity):
    num_people = similarity.shape[0]
    print(f"num people1: {num_people}")

    lp_vars = []
    c = []
    for i in range(num_people):
        for j in range(num_people):
            lp_vars.append(LpVariable(f"g_{i}_r_{j}", lowBound = 0, upBound = 1, cat=LpInteger))
            c.append(similarity[i][j] if i != j else 99999)
    
    prob = LpProblem("snowflake", LpMinimize)
    
    # minimization function
    prob += lpDot(c, lp_vars)

    # give only 1
    for i in range(num_people):
        prob += lpSum([lp_vars[i] for i in range(i * num_people, (i + 1) * num_people)]) == 1

    # receive only 1
    for i in range(num_people):
        prob += lpSum([lp_vars[i] for i in range(i, num_people ** 2, num_people)]) == 1

    # don't give and receive to the same person
    for comb in combinations(range(num_people), 2):
        prob += lp_vars[comb[0] * num_people + comb[1]] + lp_vars[comb[1] * num_people + comb[0]] <= 1
    
    return lp_vars, prob.solve()

if __name__ == "__main__":
    df = pd.read_csv("Engineering Secret Snowflake 2019 (Responses) - Form Responses 1.csv")
    going = df.loc[df["Do you want to participate in Engineering's Secret Snowflake?"] == "Yes", ["Email Address", "List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]].reset_index(drop=True)

    interest_similarity = similarity(going)
    lp_vars, result = optimize(interest_similarity)

    print(LpStatus[result])
    members = going["Email Address"]
    num_people = len(members)
    check = 0
    matching = pd.DataFrame(columns = ["Giver", "Receiver", "Giver's interests", "Receiver's Interests"])
    for i, g in enumerate(members):
        for j, r in enumerate(members):
            if lp_vars[i * num_people + j].varValue  >= 0.95:
                matching = matching.append({"Giver": g, "Receiver": r, "Giver's interests": going.loc[i, "List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "], "Receiver's Interests": going.loc[j, "List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]}, ignore_index = True)
                check += 1
    matching.to_csv("matching.csv",index = False, quoting=true)
    print(f"{check} number of snowflakes matched")
