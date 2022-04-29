# Import modules and setup notebook

import numpy as np
import pandas as pd
import re
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pickle5 as pickle
plt.rcParams["figure.dpi"] = 100

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings

warnings.filterwarnings("ignore")

# kappa metric for measuring agreement of automatic to human scores
from skll.metrics import kappa
from bhkappa import mean_quadratic_weighted_kappa

plt.style.use("seaborn-colorblind")

# Setup Pandas
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.notebook_repr_html", True)
pd.set_option("display.max_colwidth", 100)

def get_features():
    # Read essay data processed in previous notebook
    if not(os.path.exists("data/training_spacy.pkl")):
        print("Please execute the topic_modelin.py file first.")
        exit(1)
    with open("data/training_spacy.pkl", "rb") as fh:
        training_set = pickle.load(fh)   

    training_set[["lemma", "pos", "ner"]].sample(3)

    """Choose arbitrary essay from highest available target_score for each topic.
    all other essays will be compared to these. 
    The uncorrected essays will be used since the reference essays should have fewer errors.
    """
    reference_essays = {
        1: 161,
        2: 3022,
        3: 5263,
        4: 5341,
        5: 7209,
        6: 8896,
        7: 11796,
        8: 12340,
    }  # topic: essay_id

    references = {}

    if not (os.path.exists("data/training_features.pkl")):
        t0 = datetime.now()

        nlp = spacy.load("en_core_web_sm")
        stop_words = set(STOP_WORDS)

        # generate nlp object for reference essays:
        for topic, index in reference_essays.items():
            references[topic] = nlp(training_set.iloc[index]["essay"])

        # generate document similarity for each essay compared to topic reference
        training_set["similarity"] = training_set.apply(
            lambda row: nlp(row["essay"]).similarity(references[row["topic"]]), axis=1
        )

        t1 = datetime.now()
        print("Processing time: {}".format(t1 - t0))

        # Plot document similarity vs target score for each topic
        topic_number = 0
        fig, ax = plt.subplots(4, 2, figsize=(7, 10))
        for i in range(4):
            for j in range(2):
                topic_number += 1
                sns.regplot(
                    x="target_score",
                    y="similarity",
                    data=training_set[training_set["topic"] == topic_number],
                    ax=ax[i, j],
                )
                ax[i, j].set_title("Topic %i" % topic_number)
        ax[3, 0].locator_params(nbins=10)
        ax[3, 1].locator_params(nbins=10)
        plt.suptitle("Document similarity by topic")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("images/document_sim_by_topic")

        # count various features

        t0 = datetime.now()

        training_set["token_count"] = training_set.apply(lambda x: len(x["tokens"]), axis=1)
        training_set["unique_token_count"] = training_set.apply(
            lambda x: len(set(x["tokens"])), axis=1
        )
        training_set["nostop_count"] = training_set.apply(
            lambda x: len([token for token in x["tokens"] if token not in stop_words]),
            axis=1,
        )
        training_set["sent_count"] = training_set.apply(lambda x: len(x["sents"]), axis=1)
        training_set["ner_count"] = training_set.apply(lambda x: len(x["ner"]), axis=1)
        training_set["comma"] = training_set.apply(
            lambda x: x["corrected"].count(","), axis=1
        )
        training_set["question"] = training_set.apply(
            lambda x: x["corrected"].count("?"), axis=1
        )
        training_set["exclamation"] = training_set.apply(
            lambda x: x["corrected"].count("!"), axis=1
        )
        training_set["quotation"] = training_set.apply(
            lambda x: x["corrected"].count('"') + x["corrected"].count("'"), axis=1
        )
        training_set["organization"] = training_set.apply(
            lambda x: x["corrected"].count(r"@ORGANIZATION"), axis=1
        )
        training_set["caps"] = training_set.apply(
            lambda x: x["corrected"].count(r"@CAPS"), axis=1
        )
        training_set["person"] = training_set.apply(
            lambda x: x["corrected"].count(r"@PERSON"), axis=1
        )
        training_set["location"] = training_set.apply(
            lambda x: x["corrected"].count(r"@LOCATION"), axis=1
        )
        training_set["money"] = training_set.apply(
            lambda x: x["corrected"].count(r"@MONEY"), axis=1
        )
        training_set["time"] = training_set.apply(
            lambda x: x["corrected"].count(r"@TIME"), axis=1
        )
        training_set["date"] = training_set.apply(
            lambda x: x["corrected"].count(r"@DATE"), axis=1
        )
        training_set["percent"] = training_set.apply(
            lambda x: x["corrected"].count(r"@PERCENT"), axis=1
        )
        training_set["noun"] = training_set.apply(lambda x: x["pos"].count("NOUN"), axis=1)
        training_set["adj"] = training_set.apply(lambda x: x["pos"].count("ADJ"), axis=1)
        training_set["pron"] = training_set.apply(lambda x: x["pos"].count("PRON"), axis=1)
        training_set["verb"] = training_set.apply(lambda x: x["pos"].count("VERB"), axis=1)
        training_set["noun"] = training_set.apply(lambda x: x["pos"].count("NOUN"), axis=1)
        training_set["cconj"] = training_set.apply(
            lambda x: x["pos"].count("CCONJ"), axis=1
        )
        training_set["adv"] = training_set.apply(lambda x: x["pos"].count("ADV"), axis=1)
        training_set["det"] = training_set.apply(lambda x: x["pos"].count("DET"), axis=1)
        training_set["propn"] = training_set.apply(
            lambda x: x["pos"].count("PROPN"), axis=1
        )
        training_set["num"] = training_set.apply(lambda x: x["pos"].count("NUM"), axis=1)
        training_set["part"] = training_set.apply(lambda x: x["pos"].count("PART"), axis=1)
        training_set["intj"] = training_set.apply(lambda x: x["pos"].count("INTJ"), axis=1)

        t1 = datetime.now()
        print("Processing time: {}".format(t1 - t0))

        # save to file
        training_set.to_pickle("data/training_features.pkl")
    else:
        with open("data/training_features.pkl", "rb") as fh:
            training_set = pickle.load(fh) 

    training_set[["lemma", "pos", "ner"]].sample(3)

    return training_set



def get_df(training_set):
    """Choose arbitrary essay from highest available target_score for each topic.
    all other essays will be compared to these. 
    The uncorrected essays will be used since the reference essays should have fewer errors.
    """
    reference_essays = {
        1: 161,
        2: 3022,
        3: 5263,
        4: 5341,
        5: 7209,
        6: 8896,
        7: 11796,
        8: 12340,
    }  # topic: essay_id

    references = {}

    t0 = datetime.now()

    nlp = spacy.load("en_core_web_sm")
    stop_words = set(STOP_WORDS)

    # generate nlp object for reference essays:
    for topic, index in reference_essays.items():
        references[topic] = nlp(training_set.iloc[index]["essay"])

    # generate document similarity for each essay compared to topic reference
    training_set["similarity"] = training_set.apply(
        lambda row: nlp(row["essay"]).similarity(references[row["topic"]]), axis=1
    )

    t1 = datetime.now()
    print("Processing time: {}".format(t1 - t0))

    # Plot correlation of essay-length related features
    usecols = [
        "word_count",
        "token_count",
        "unique_token_count",
        "nostop_count",
        "sent_count",
    ]
    g = sns.pairplot(
        training_set[training_set.topic == 4],
        hue="target_score",
        vars=usecols,
        plot_kws={"s": 20},
        palette="bright",
    )
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle("Pairplots of select features", fontsize=16)
    plt.savefig("images/pairplot_select_features")

    print(training_set.info())

    # Selecting k best features: Some features omitted due to high correlation

    predictors = [
        #                 'word_count',
        "corrections",
        "similarity",
        #                 'token_count',
        "unique_token_count",
        #                 'nostop_count',
        "sent_count",
        "ner_count",
        "comma",
        "question",
        "exclamation",
        "quotation",
        "organization",
        "caps",
        "person",
        "location",
        "money",
        "time",
        "date",
        "percent",
        "noun",
        "adj",
        "pron",
        "verb",
        "cconj",
        "adv",
        "det",
        "propn",
        "num",
        "part",
        "intj",
    ]

    # Create and fit selector
    selector = SelectKBest(
        f_regression, k=10
    )  # f_classif, chi2, f_regression, mutual_info_classif, mutual_info_regression

    # Create empty dataframe
    df = pd.DataFrame()

    for topic in range(1, 9):
        kpredictors = []

        # test for division by zero errors due to insufficient data:
        for p in predictors:
            if np.std(training_set[training_set.topic == topic][p], axis=0) != 0:
                kpredictors.append(p)

        # select k best for each topic:
        X = training_set[training_set.topic == topic][kpredictors]
        y = training_set[training_set.topic == topic].target_score

        selector.fit(X, y)

        # Get idxs of columns to keep
        mask = selector.get_support(indices=True)

        selected_features = training_set[training_set.topic == topic][predictors].columns[
            mask
        ]
        df["Topic " + str(topic)] = selected_features
    print(df)
    return df


def evaluate(df, topic, features, model):
    """Regression pipeline with kappa evaluation"""

    X = df[df["topic"] == topic][features]
    y = df[df["topic"] == topic]["target_score"].astype(np.float64)
    # token_ct = X.token_count
    # X = X.div(token_ct, axis=0)
    # X['token_count'] = X['token_count'].mul(token_ct, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=26
    )

    pipeline = Pipeline(model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return kappa(y_pred, y_test, weights="quadratic")

def get_predictors():
    return [
        "word_count",
        "corrections",
        "similarity",
        "token_count",
        "unique_token_count",
        "nostop_count",
        "sent_count",
        "ner_count",
        "comma",
        "question",
        "exclamation",
        "quotation",
        "organization",
        "caps",
        "person",
        "location",
        "money",
        "time",
        "date",
        "percent",
        "noun",
        "adj",
        "pron",
        "verb",
        "cconj",
        "adv",
        "det",
        "propn",
        "num",
        "part",
        "intj",
    ]

def evaluate_by_topic(df, training_set, predictors):

    # feature selection
    # fvalue_selector = SelectKBest(score_func=f_regression, k=10)

    # for use in pipeline
    models = [
        [("scaler", StandardScaler()), ("linearSVC", LinearSVC(C=0.01))],
        [("scaler", StandardScaler()), ("lm", LinearRegression())],
        [("rf", RandomForestRegressor(random_state=26))],
        [("en", ElasticNet(l1_ratio=0.01, alpha=0.1, max_iter=100000, random_state=26))],
    ]

    for steps in models:
        kappas = []
        weights = []
        for topic in range(1, 9):
            kappas.append(evaluate(training_set, topic, predictors, steps))
            weights.append(len(training_set[training_set.topic == topic]))

        mqwk = mean_quadratic_weighted_kappa(kappas, weights=weights)
        print(steps)
        print("Weighted by topic Kappa score: {:.4f}".format(mqwk))
        print("")

# ElasticNet with GridSearchCV for each individual topic


def en_evaluate(df, topic, features):
    # Regression pipeline with kappa evaluation
    paramgrid = {
        "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.7, 0.99],
        "alpha": [0.001, 0.01, 0.1, 1],
    }
    X = df[df["topic"] == topic][features]
    y = df[df["topic"] == topic]["target_score"].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=26
    )

    gs = GridSearchCV(
        ElasticNet(max_iter=100000, random_state=26), param_grid=paramgrid, cv=5
    )
    gs.fit(X_train, y_train)
    print("Topic", topic, "best parameters:", gs.best_params_)
    y_pred = gs.predict(X_test)

    return kappa(y_pred, y_test, weights="quadratic")

def en_evaluate_by_topic(df, training_set, predictors):
    kappas = []
    weights = []
    for topic in range(1, 9):
        kappas.append(en_evaluate(training_set, topic, predictors))
        weights.append(len(training_set[training_set.topic == topic]))

    mqwk = mean_quadratic_weighted_kappa(kappas, weights=weights)
    print("Weighted by topic Kappa score: {:.4f}".format(mqwk))

    # Individual topic kappa scores
    print(kappas)

def get_X_y(training_set, predictors):
    X = training_set[predictors]
    y = training_set["target_score"].astype(np.float64)
    return X,y


def forest_analysis(X,y, training_set, predictors):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=26)

    forest.fit(X, y)

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # plot feature importances

    features = pd.DataFrame(
        {"feature_name": X.columns, "importance": forest.feature_importances_, "std": std}
    )
    features.sort_values("importance").plot.barh(
        x="feature_name", y="importance", xerr="std", legend=False
    )
    plt.title("Gini importances of forest features")
    plt.xlabel("Gini-importance")
    plt.tight_layout()
    plt.savefig("images/features_gini")

    # best k features
    k = 15
    top_features = features.sort_values("importance", ascending=False)[
        "feature_name"
    ].tolist()[:k]

    # Linear regression with top k features
    kappas = []
    weights = []
    steps = [("scaler", StandardScaler()), ("lm", LinearRegression())]
    for topic in range(1, 9):
        kappas.append(evaluate(training_set, topic, top_features, steps))
        weights.append(len(training_set[training_set.topic == topic]))

    mqwk = mean_quadratic_weighted_kappa(kappas, weights=weights)
    print("Weighted by topic Kappa score: {:.4f}".format(mqwk))

    # Overview of correlating features
    corr = training_set[predictors].corr()  # default: Pearson
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(
        corr,
        mask=mask,
        cmap="Spectral",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.savefig("images/correlated_features")

    # Lemmatized essays re-joined (list to essay)
    training_set["l_essay"] = training_set["lemma"].apply(" ".join)

    vectorizer = TfidfVectorizer(
        max_df=0.2, min_df=3, max_features=2000, stop_words=STOP_WORDS
    )  # default: binary=False
    tfidf_matrix = vectorizer.fit_transform(training_set.l_essay)  # using lemmatized essays

    # Combine previous predictors with TF-IDF matrix
    combined_dense = pd.concat(
        [
            pd.DataFrame(tfidf_matrix.todense()),
            training_set[predictors],
            training_set["topic"],
            training_set["target_score"],
        ],
        axis=1,
    )
    return combined_dense

# ElasticNet with GridSearchCV for each individual topic


def tf_evaluate(df, topic):
    # Regression pipeline with kappa evaluation
    paramgrid = {"l1_ratio": [0.01, 0.1, 0.5, 0.9], "alpha": [0.01, 0.1, 1]}
    X = df[df["topic"] == topic].drop(["topic", "target_score"], axis=1)
    y = df[df["topic"] == topic]["target_score"].astype(np.float64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=26
    )

    gs = GridSearchCV(
        ElasticNet(max_iter=100000, random_state=26), param_grid=paramgrid, cv=5
    )
    gs.fit(X_train, y_train)
    print("Topic", topic, "best parameters:", gs.best_params_)
    y_pred = gs.predict(X_test)

    return kappa(y_pred, y_test, weights="quadratic")


# ElasticNet with GridSearchCV for each individual topic
def elasticnet_analysis(training_set, combined_dense):
    kappas = []
    weights = []
    for topic in range(1, 9):
        kappas.append(tf_evaluate(combined_dense, topic))
        weights.append(len(training_set[training_set.topic == topic]))

    mqwk = mean_quadratic_weighted_kappa(kappas, weights=weights)
    print("Weighted by topic Kappa score: {:.4f}".format(mqwk))

if __name__ == "__main__":
    print("Getting features...")
    training_set = get_features()
    print("All features done.")
    print("Building df...")
    df = get_df(training_set)
    print("df built.")
    predictors = get_predictors()
    print("Evaluation by topic...")
    evaluate_by_topic(df, training_set, predictors)
    print("Evaluation done.")
    print("En-Evaluation by topic...")
    en_evaluate_by_topic(df, training_set, predictors)
    print("Evaluation done.")
    X, y = get_X_y(training_set, predictors)
    print("Forest analysis...")
    combined_dense = forest_analysis(X,y, training_set, predictors)
    print("ElasticNet analysis...")
    elasticnet_analysis(training_set, combined_dense)

