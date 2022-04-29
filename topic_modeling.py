import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import spacy
from datetime import datetime
import joblib

from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pyLDAvis.sklearn import prepare

def spacy_analysis():
    if not(os.path.exists("data/training_corr.pkl")):
        print("Please execute the preprocess.py file before.")
        exit(1)

    if not(os.path.exists("data/training_corr.pkl")):
        training_set = pd.read_pickle("data/training_corr.pkl")

        sents = []
        tokens = []
        lemma = []
        pos = []
        ner = []

        stop_words = set(STOP_WORDS)
        stop_words.update(punctuation)  # remove it if you need punctuation

        nlp = spacy.load("en_core_web_sm")

        t0 = datetime.now()

        # suppress numpy warnings
        np.warnings.filterwarnings("ignore")

        for essay in nlp.pipe(training_set["corrected"], batch_size=100):
            if essay.is_parsed:
                tokens.append([e.text for e in essay])
                sents.append([sent.text.strip() for sent in essay.sents])
                pos.append([e.pos_ for e in essay])
                ner.append([e.text for e in essay.ents])
                lemma.append([n.lemma_ for n in essay])
            else:
                # We want to make sure that the lists of parsed results have the
                # same number of entries of the original Dataframe, so add some blanks in case the parse fails
                tokens.append(None)
                lemma.append(None)
                pos.append(None)
                sents.append(None)
                ner.append(None)

        training_set["tokens"] = tokens
        training_set["lemma"] = lemma
        training_set["pos"] = pos
        training_set["sents"] = sents
        training_set["ner"] = ner

        t1 = datetime.now()
        print("Processing time: {}".format(t1 - t0))

        training_set.to_pickle("data/training_spacy.pkl")
    else: 
        training_set = pd.read_pickle("data/training_spacy.pkl")

    return training_set

def create_plots(training_set):
    print(training_set[["tokens", "pos", "sents", "ner"]].head())

    # Replace topic numbers with meaningful one-word summary:
    topic_dict = {
        "topic": {
            1: "computer",
            2: "censorship",
            3: "cyclist",
            4: "hibiscus",
            5: "mood",
            6: "dirigibles",
            7: "patience",
            8: "laughter",
        }
    }

    training_set.replace(topic_dict, inplace=True)

    # Lemmatized essays re-joined (list to essay)
    training_set["l_essay"] = training_set["lemma"].apply(" ".join)

    # Baseline: number of unique lemma
    vectorizer = CountVectorizer(
        max_df=0.2, min_df=3, stop_words=STOP_WORDS, max_features=2000
    )  # default: binary=False
    doc_term_matrix = vectorizer.fit_transform(
        training_set.l_essay
    )  # using lemmatized essays

    # Most frequent tokens:
    words = vectorizer.get_feature_names()

    doc_term_matrix_df = pd.DataFrame.sparse.from_spmatrix(doc_term_matrix, columns=words)
    word_freq = doc_term_matrix_df.sum(axis=0).astype(int)
    word_freq.sort_values(ascending=False).head(10)

    lda_base = LatentDirichletAllocation(
        n_components=8,
        n_jobs=-1,
        learning_method="batch",
        max_iter=40,
        perp_tol=0.01,
        verbose=1,
        evaluate_every=5,
    )
    lda_base.fit(doc_term_matrix)

    # save base model
    joblib.dump(lda_base, "data/lda_baseline.pkl")

    topic_labels = ["Topic {}".format(i) for i in range(1, 9)]
    topics_count = lda_base.components_
    topics_prob = topics_count / topics_count.sum(axis=1).reshape(-1, 1)
    topics = pd.DataFrame(topics_prob.T, index=words, columns=topic_labels)
    topics.sample(10)

    one_word = list(topic_dict["topic"].values())
    sns.heatmap(topics.reindex(one_word), cmap="Blues")
    plt.title("Topic probabilities for one-word-summary")
    plt.savefig("images/topic_one_word_summary")

    top_words = {}
    for topic, words_ in topics.items():
        top_words[topic] = words_.nlargest(10).index.tolist()
    pd.DataFrame(top_words)

    train_preds = lda_base.transform(doc_term_matrix)
    train_eval = pd.DataFrame(train_preds, columns=topic_labels, index=training_set.topic)
    train_eval.sample(10)

    train_eval.groupby(level="topic").mean().plot.bar(
        title="Avg. Topic Probabilities", rot=0, colormap="tab10", figsize=(10, 5)
    )
    plt.savefig("images/avg_topic_proba")

    df = train_eval.groupby(level="topic").agg("median")
    fig, ax = plt.subplots(figsize=(8, 8))
    g = sns.heatmap(
        df, annot=True, fmt=".1%", annot_kws={"size": 10}, cmap="Blues", square=True
    )
    loc, labels = plt.yticks()
    g.set_yticklabels(labels, rotation=0)
    g.set_title("Topic Assignments")
    fig = g.get_figure()
    fig.savefig("images/topic_assignements.png")

    df = (
        train_eval.idxmax(axis=1)
        .reset_index()
        .groupby("topic", as_index=False)
        .agg(lambda x: x.value_counts().index[0])
        .rename(columns={0: "assignment"})
    )
    print(df)

if __name__ == "__main__":
    training_set = spacy_analysis()
    create_plots(training_set)
