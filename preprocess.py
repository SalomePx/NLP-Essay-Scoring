import numpy as np
import pandas as pd
import re
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

import language_tool_python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
from sklearn.model_selection import train_test_split

import pyLDAvis
from pyLDAvis.sklearn import prepare

import swifter
from textblob import TextBlob


plt.style.use("seaborn-colorblind")

# Setup Pandas
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.notebook_repr_html", True)
pd.set_option("display.max_colwidth", 100)

# pyLDAvis.enable_notebook()
plt.rcParams["figure.figsize"] = [8, 5]
plt.rcParams["figure.dpi"] = 100

import warnings

warnings.simplefilter("ignore", DeprecationWarning)


def create_plots(training_set):
    # print(training_set.sample())
    training_set.info()

    training_set.groupby("topic").agg("count").plot.bar(y="essay", rot=0, legend=False)
    plt.title("Essay count by topic #")
    plt.ylabel("Count")
    plt.savefig("images/count_by_topics.png")

    # Count characters and words for each essay
    training_set["word_count"] = training_set["essay"].str.strip().str.split().str.len()

    training_set.hist(
        column="word_count",
        by="topic",
        bins=25,
        sharey=True,
        sharex=True,
        layout=(2, 4),
        figsize=(7, 4),
        rot=0,
    )
    plt.suptitle("Word count by topic #")
    plt.xlabel("Number of words")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/words_by_topic.png")

    training_set.groupby(["topic"])["target_score"].agg(
        ["min", "max", "count", "nunique"]
    )

    topic_number = 0
    fig, ax = plt.subplots(4, 2, figsize=(8, 10))
    for i in range(4):
        for j in range(2):
            topic_number += 1
            sns.violinplot(
                x="target_score",
                y="word_count",
                data=training_set[training_set["topic"] == topic_number],
                ax=ax[i, j],
            )
            ax[i, j].set_title("Topic %i" % topic_number)
    ax[3, 0].locator_params(nbins=10)
    ax[3, 1].locator_params(nbins=10)
    plt.suptitle("Word count by score")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/words_by_score.png")

    topic_number = 0
    fig, ax = plt.subplots(4, 2, figsize=(9, 9), sharey=False)
    for i in range(4):
        for j in range(2):
            topic_number += 1
            training_set[training_set["topic"] == topic_number].groupby("target_score")[
                "essay_id"
            ].agg("count").plot.bar(ax=ax[i, j], rot=0)
            ax[i, j].set_title("Topic %i" % topic_number)
    ax[3, 0].locator_params(nbins=10)
    ax[3, 1].locator_params(nbins=10)
    plt.suptitle("Histograms of essay scores")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/scores_histogram.png")


def correct_dataset(training_set, save=True):
    """
    use language tool to correct for most spelling and grammatical errors. Also count the applied corrections.
    Using language_check python wrapper for languagetool:
    https://www.languagetool.org/dev
    """
    tool = language_tool_python.LanguageTool("en-US")

    t0 = datetime.now()

    training_set["matches"] = training_set["essay"].apply(lambda txt: tool.check(txt))
    training_set["corrections"] = training_set.apply(
        lambda l: len(l["matches"]), axis=1
    )
    training_set["corrected"] = training_set.swifter.apply(
        lambda l: tool.correct(l["essay"]), axis=1
    )

    t1 = datetime.now()
    print("Processing time: {}".format(t1 - t0))

    # save work
    training_set.to_pickle("data/training_corr.pkl")

    if save:
        training_set["corrected"].to_csv(
            "data/corrected_training_set_rel3", index=False
        )

    print(training_set["corrected"])


if __name__ == "__main__":
    try:
        training_set = pd.read_csv(
            "data/training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1"
        ).rename(
            columns={
                "essay_set": "topic",
                "domain1_score": "target_score",
                "domain2_score": "topic2_target",
            }
        )
    except pd.errors.ParserError as e:
        print(f"Problem file: training_set_rel3.tsv caused Exception: {e}")
        pass
    training_set.sample()
    create_plots(training_set)
    if not (os.path.exists("data/training_corr.pkl")):
        correct_dataset(training_set)
