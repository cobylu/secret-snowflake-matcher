from itertools import combinations

import numpy as np
import pandas as pd
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
    giv_rec = np.zeros((num_people * (num_people - 1) // 2, num_people**2))
    for i, comb in enumerate(combinations(range(num_people), 2)):
        giv_rec[i, comb[0] * num_people + comb[1]] = 1
        giv_rec[i, comb[1] * num_people + comb[0]] = 1

    A_eq = np.vstack((give, receive))
    b_eq = np.ones(2 * num_people)

    A_ub = giv_rec
    b_ub = np.ones((num_people - 1) * num_people // 2)
    return linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="revised simplex")

if __name__ == "__main__":
    df = pd.read_csv("Engineering Secret Snowflake 2019 (Responses) - Form Responses 1.csv")
    going = df.loc[df["Do you want to participate in Engineering's Secret Snowflake?"] == "Yes", ["Email Address", "List 4 facts about yourself to help your secret snowflake find the perfect gift for you! "]].reset_index(drop=True)

    interest_similarity = similarity(going)
    result = optimize(interest_similarity)

    solution = np.round(result.x)
    members = going["Email Address"]
    num_people = len(members)
    print(f"num people2: {num_people}")
    for i, g in enumerate(members):
        for j, r in enumerate(members):
            if solution[i * num_people + j] == 1:
                print(f"{g} gives to {r}")
