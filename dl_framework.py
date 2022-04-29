# Import necessary modules

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import os
from sklearn.model_selection import train_test_split
import pickle5 as pickle

from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
)
from sklearn.model_selection import cross_val_score, KFold

from gensim.models.word2vec import Word2Vec

# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
import language_tool_python
import spacy

from sklearn.model_selection import train_test_split, KFold

from skll.metrics import kappa
from bhkappa import mean_quadratic_weighted_kappa

from scipy.sparse import csr_matrix

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    LSTM,
    Embedding,
    Bidirectional,
    Flatten,
)
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

nlp = spacy.load("en_core_web_sm")

stopwords = stopwords.words("english")

print("Numpy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)


def correct_language(df):
    """
    use language tool to correct for most spelling and grammatical errors. Also count the applied corrections.
    Using language_check python wrapper for languagetool:
    https://www.languagetool.org/dev
    """
    tool = language_tool_python.LanguageTool("en-US")

    df["matches"] = df["essay"].apply(lambda txt: tool.check(txt))
    df["corrections"] = df.apply(lambda l: len(l["matches"]), axis=1)
    df["corrected"] = df.apply(
        lambda l: language_tool_python.correct(l["essay"]), axis=1
    )

    df.to_pickle("data/training_corr.pkl")

    return df


# read essays from training_set

# apply spelling and grammar corrections

def get_training_combo():
    print("Getting training set...")
    if not (os.path.exists("data/training_corr.pkl")):
        training_set = pd.read_csv(
            "data/training_set_rel3.tsv", sep="\t", encoding="ISO-8859-1"
        ).rename(
            columns={
                "essay_set": "topic",
                "domain1_score": "target_score",
                "domain2_score": "topic2_target",
            }
        )
        training_set = correct_language(training_set)
    else:
        with open("data/training_corr.pkl", "rb") as fh:
            training_set = pickle.load(fh) 

    print(training_set.head())

    if not (os.path.exists("data/combo_set.pkl")):
        # read essays from validation and test sets

        valid_set  = pd.read_csv('data/valid_set.tsv', sep='\t', encoding = "ISO-8859-1")\
                    .rename(columns={'essay_set': 'topic'})
        test_set  = pd.read_csv('data/test_set.tsv', sep='\t', encoding = "ISO-8859-1")\
                    .rename(columns={'essay_set': 'topic'})

        combo_set = pd.concat([valid_set, test_set], sort=False)

        # apply spelling and grammar corrections
        combo_set = correct_language(combo_set)
        combo_set = pd.concat([combo_set, training_set], sort=False)
        combo_set.to_pickle('data/combo_set.pkl')
    else:
        with open("data/combo_set.pkl", "rb") as fh:
            combo_set = pickle.load(fh) 

    return training_set, combo_set

# Clean training_set essays before feeding them to the Word2Vec model.
punctuations = string.punctuation

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_essays(essays, logging=False):
    print("Cleaning essays...")
    texts = []
    counter = 1
    for essay in essays.corrected:
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(essays)))
        counter += 1
        essay = nlp(essay, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in essay if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


# Define function to preprocess text for a word2vec model
def cleanup_essay_word2vec(essays, logging=False):
    print("Cleaning word2vec essay...")
    sentences = []
    counter = 1
    for essay in essays:
        if counter % 2000 == 0 and logging:
            print("Processed %d out of %d documents" % (counter, len(essays)))
        # Disable tagger so that lemma_ of personal pronouns (I, me, etc) don't getted marked as "-PRON-"
        essay = nlp(essay, disable=['tagger'])
        # Grab lemmatized form of words and make lowercase
        essay = " ".join([tok.lemma_.lower() for tok in essay])
        # Split into sentences based on punctuation
        essay = re.split("[\.?!;] ", essay)
        # Remove commas, periods, and other punctuation (mostly commas)
        essay = [re.sub("[\.,;:!?]", "", sent) for sent in essay]
        # Split into words
        essay = [sent.split() for sent in essay]
        sentences += essay
        counter += 1
    return sentences

def create_word2vec(cleaned_word2vec, text_dim = 300):
    print("Training Word2Vec model...")
    wordvec_model = Word2Vec(cleaned_word2vec, vector_size=text_dim, window=5, min_count=3, workers=-1, sg=1)
    print("Word2Vec model created.")
    print("%d unique words represented by %d dimensional vectors" % (len(wordvec_model.wv.index_to_key), text_dim))
    wordvec_model.save('models/wordvec_model')
    print("Word2Vec model saved.")
    return wordvec_model

# Define function to create averaged word vectors given a cleaned text.
def create_average_vec(essay, wordvec_model, text_dim=300):
    average = np.zeros((text_dim,), dtype='float32')
    num_words = 0.
    for word in essay.split():
        if word in wordvec_model.wv.index_to_key:
            average = np.add(average, wordvec_model.wv[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average

def create_word_vec(training_set, train_cleaned, wordvec_model, text_dim=300):
    # Create word vectors
    cleaned_vec = np.zeros((training_set.shape[0], text_dim), dtype="float32")  
    for i in range(len(train_cleaned)):
        cleaned_vec[i] = create_average_vec(train_cleaned[i], wordvec_model)
    
    print("Word vectors for all essays in the training data set are of shape:", cleaned_vec.shape)
    return cleaned_vec

def load_features():
    # Read generated features from file:
    with open("data/training_features.pkl", "rb") as fh:
        additional_features = pickle.load(fh) 

    # Use select features from Gini feature importances
    feature_list = [
                    'word_count',
                    'corrections',
                    'similarity',
                    'token_count',
                    'unique_token_count',
                    'nostop_count',
                    'sent_count',
                    'ner_count',
                    'comma',
                    'question',
                    'exclamation',
                    'quotation',
                    'organization',
                    'caps',
                    'person',
                    'location',
                    'money',
                    'time',
                    'date',
                    'percent',
                    'noun',
                    'adj',
                    'pron',
                    'verb',
                    'cconj',
                    'adv',
                    'det',
                    'propn',
                    'num',
                    'part',
                    'intj'
                    ]

    additional_features = additional_features[feature_list]

    stdscaler = StandardScaler()
    additional_features = stdscaler.fit_transform(additional_features)
    additional_features.shape
    return additional_features


def combine_data_features(training_set, cleaned_vec, additional_features):
    # Combine topic number, target score, additional features and cleaned word vectors
    all_data = pd.concat([training_set[['topic','target_score']], pd.DataFrame(additional_features), pd.DataFrame(cleaned_vec)], axis=1)
    return all_data

def model_by_topic(all_data):
    # Build model
    output_dim = 1
    input_dim = all_data.shape[1]-2
    dropout = 0.2

    model = None
    model = Sequential()

    # Densely Connected Neural Network (Multi-Layer Perceptron)
    model.add(Dense(14, activation='relu', kernel_initializer='he_normal', input_dim=input_dim)) 
    model.add(Dropout(dropout))
    # model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    # model.add(Dropout(dropout))
    model.add(Dense(output_dim))
    model.summary()

    # Compile the model
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mse', metrics=['mse','mae'])

    return model

def train_topic(model, all_data):
    # Run each topic individually through neural network
    kappa_list = []
    weights = []
    epochs = 100

    for topic in range(1,9):
        # split data
        X = all_data[all_data.topic == topic].drop(['topic', 'target_score'], axis=1)
        y = all_data[all_data.topic == topic].target_score.to_frame()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26)
        estimator = model.fit(X_train, y_train,
    #                       validation_split=0.3,
                        epochs=epochs, batch_size=15, verbose=0)
        # get predictions
        y_pred = pd.DataFrame(model.predict(X_test).reshape(-1))
        
        # get topic kappa score
        kappa_list.append(kappa(y_test.values, y_pred.round(0).astype(int).values, weights='quadratic'))

        # get weights (number of essays)
        weights.append(y_test.shape[0]/all_data.shape[0])    

    # get weighted average kappa
    qwk = mean_quadratic_weighted_kappa(kappa_list, weights=1) # weights)
    print('Combined Kappa score: {:.2f}%'.format(qwk * 100))

def train_topic_crossval(model, all_data):
    input_dim = all_data.shape[1]-2
    dropout = 0.2
    # Cross-validation
    kappa_dict = {}
    for topic in range(1,9):
        
        model = None
        # create model
        model = Sequential()
        model.add(Dense(14, input_dim=input_dim, kernel_initializer='he_normal', activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')

        X = all_data[all_data.topic == topic].drop(['topic', 'target_score'], axis=1)
        y = all_data[all_data.topic == topic].target_score.to_frame()
        # split data
        kf = KFold(n_splits=5)
        kappa_list = []
        for train, test in kf.split(X):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            model.fit(X_train, y_train, epochs=200, batch_size=15, verbose=0) 
            y_pred = pd.DataFrame(model.predict(X_test).reshape(-1))
            kappa_list.append(kappa(y_pred.round(0).astype(int).values, 
                            y.iloc[test].values, 
                            weights='quadratic'))
        print("Kappa for topic", topic, ": {:.3f}%".format(np.mean(kappa_list)))
        kappa_dict[topic] = np.mean(kappa_list)

    mqwk = mean_quadratic_weighted_kappa(list(kappa_dict.values()), weights=1) # weights)
    print('Combined Kappa score: {:.4f}%'.format(mqwk))
    print(f"Kappa List : {kappa_list}")

def split_X_y_scores(X, y, scores):
    # Data to be split
    X_train, X_test, y_train, y_test, scores_train, scores_test = \
            train_test_split(
                    X, 
                    y, 
                    scores,
                    test_size=0.2, 
                    random_state=26
                    )

    print('X_train size: {}'.format(X_train.shape))
    print('X_test size: {}'.format(X_test.shape))
    print('y_train size: {}'.format(y_train.shape))
    print('y_test size: {}'.format(y_test.shape))
    print('scores_train size: {}'.format(scores_train.shape))
    print('scores_test size: {}'.format(scores_test.shape))

    return X_train, X_test, y_train, y_test, scores_train, scores_test

def build_model(X, y, architecture='mlp'):
    output_dim = y.shape[1]
    input_dim = X.shape[1]
    dropout = 0.2
    model = Sequential()
    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(14, activation='relu', kernel_initializer='he_normal', input_dim=input_dim)) 
        model.add(Dropout(dropout))
        model.add(Dense(output_dim))
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        inputs = Input(shape=(input_dim,1))

        x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)

        #Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x) 
        outputs = Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')
    elif architecture == 'lstm':
        # LSTM network
        inputs = Input(shape=(input_dim,1))

        x = Bidirectional(LSTM(64, return_sequences=True),
                        merge_mode='concat')(inputs)
        x = Dropout(dropout)(x)
        x = Flatten()(x)
        outputs = Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    else:
        print('Error: Model type not found.')
    return model

def train_model(model, X_train, X_test, y_train, y_test) :
    # If the model is a CNN then expand the dimensions of the training data
    if model.name == "CNN" or model.name == "LSTM":
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print('Text train shape: ', X_train.shape)
        print('Text test shape: ', X_test.shape)
        
    model.summary()

    # Compile the model
    # Optimizer
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Regression:
    model.compile(optimizer=adam, loss='mse', metrics=['mse','mae'])

    # Define number of epochs
    epochs = 100

    # Fit the model to the training data
    estimator = model.fit(X_train, y_train,
    #                       validation_split=0.3,
                        epochs=epochs, batch_size=15, verbose=0)

    y_pred = pd.DataFrame(model.predict(X_test).reshape(-1))
    """Reverse scaling back to original target score scales"""

    # Merge results
    results = scores_test.reset_index(drop=True)\
                        .join(y_pred)\
                        .rename(columns={0:'y_pred'})\
                        .sort_values(by='topic')\
                        .reset_index(drop=True)
    results.head()   
    topic_number = 0
    fig, ax = plt.subplots(4,2, figsize=(9,9), sharey=False)
    for i in range(4):
        for j in range(2):
            topic_number += 1
            results[results['topic'] == topic_number]\
                [['scaled', 'y_pred']]\
                .plot.hist(histtype='step', bins=20, ax=ax[i, j], rot=0)
            ax[i,j].set_title('Topic %i' % topic_number)
    ax[3,0].locator_params(nbins=10)
    ax[3,1].locator_params(nbins=10)
    plt.suptitle('Scaled prediction errors')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/scaled_pred_err" + model.name)

    return results

def min_max_analysis(results):
    """ Create list of tuples with min/max target scores sorted by topic number.
    Performed here on results in case min/max values didn't pass through train_test_split"""

    score_df = results.groupby('topic')['target_score'].agg(['min', 'max'])  
    score_ranges = list(zip(score_df['min'], score_df['max'])) 

    """Shrink back to original range by topic number:"""
    y_p_df = pd.Series()
    y_t_df = pd.Series()

    for topic in range(1,9):
        scaler = MinMaxScaler(score_ranges[topic-1])
        scaled_pred = results[results.topic == topic]['y_pred'].to_frame()
        y_pred_shrunk = scaler.fit_transform(scaled_pred).round(0).astype('int')
        scaled_true = results[results.topic == topic]['scaled'].to_frame()
        y_true_shrunk = scaler.fit_transform(scaled_true).round(0).astype('int')
        y_p_df = y_p_df.append(pd.Series(np.squeeze(np.asarray(y_pred_shrunk))), ignore_index=True)
        y_t_df = y_t_df.append(pd.Series(np.squeeze(np.asarray(y_true_shrunk))), ignore_index=True)
        
    # Append to results df
    results['pred'] = y_p_df
    results['y_true'] = y_t_df
    results.head()
    # score histogram

    results[['pred', 'y_true']].plot.hist(histtype='step', bins=20, logy=True)
    plt.title('Histogram of target scores')
    plt.xlabel('target score')
    plt.ylabel('log count')
    plt.savefig('images/target_scores.png', dpi=300)

def kappa_analysis(results):
    k = kappa(results.pred, results.target_score, weights='quadratic')
    print('Combined essay kappa score: {:.4f}'.format(k))

    qwk = []
    # weights = []
    for topic in range(1,9):
        qwk.append(
                kappa(results[results.topic == topic]['target_score'], 
                    results[results.topic == topic]['pred'],
                        weights='quadratic'))
    #     weights.append(len(results[results.topic==topic])/X_test.shape[0])    
    mqwk = mean_quadratic_weighted_kappa(qwk, weights=1)
    print('Weighted by topic Kappa score: {:.2f}%'.format(mqwk * 100))

    print(qwk)

    # kappa for two human raters
    qwk = []
    # weights = []
    for topic in range(1,9):
        qwk.append(
                kappa(training_set[training_set.topic == topic]['rater1_domain1'], 
                    training_set[training_set.topic == topic]['rater2_domain1'],
                        weights='quadratic'))
    #     weights.append(len(results[results.topic==topic])/X_test.shape[0])    
    mqwk = mean_quadratic_weighted_kappa(qwk, weights=1)
    print('Weighted by topic Kappa score: {:.4f}'.format(mqwk))

    topic_number = 0
    fig, ax = plt.subplots(4,2, figsize=(8,8), sharey=False)
    for i in range(4):
        for j in range(2):
            topic_number += 1
            results[results['topic'] == topic_number]\
                .groupby('y_true')['y_true']\
                .agg('count')\
                .plot.bar(ax=ax[i, j], rot=0, fill=False, ec='b', label='actual')
            results[results['topic'] == topic_number]\
                .groupby('pred')['pred']\
                .agg('count')\
                .plot.bar(ax=ax[i, j], rot=0, fill=False, ec='r', label='prediction')
            ax[i,j].set_title('Topic %i' % topic_number)
    ax[3,0].locator_params(nbins=10)
    ax[3,1].locator_params(nbins=10)
    plt.suptitle('Histograms of predicted essay scores')
    plt.legend(bbox_to_anchor=(1.0, 1.05))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("images/predicted_essay_scores")

    # Check for scaling errors
    errors = len(results.query('y_true != target_score')[['topic', 'target_score', 'y_true']])

    print('{:.1f}% of target scores did not revert back to their original value.'.format(errors/results.shape[0] * 100))



if __name__ == '__main__':
    training_set, combo_set = get_training_combo()
    # Cleanup text and make sure it retains original shape
    train_cleaned = cleanup_essays(training_set, logging=True)
    cleaned_word2vec = cleanup_essay_word2vec(combo_set['corrected'], logging=True)
    print('Cleaned up training data size (i.e. number of sentences): ', len(cleaned_word2vec))
    wordvec_model = create_word2vec(cleaned_word2vec)
    cleaned_vec = create_word_vec(training_set, train_cleaned, wordvec_model)
    additional_features = load_features()
    all_data = combine_data_features(training_set, cleaned_vec, additional_features)
    model_topic = model_by_topic(all_data)
    train_topic(model_topic, all_data)
    train_topic_crossval(model_topic, all_data)
    # DataFrame used to pass original values through train_test_split
    scores = all_data[['topic', 'target_score']].reset_index() 
    # Rescale target_score (essay grades) in range 0 - 60:
    scaler = MinMaxScaler((0,10))

    # Rescale and assign target variable y
    scaled = []
    for topic in range(1,9):
        topic_scores = scores[scores['topic'] == topic]['target_score'].to_frame()
        s = (scaler.fit_transform(topic_scores).reshape(-1))
        scaled = np.append(scaled, s)
        
    scores['scaled'] = scaled

    """Use this for regression"""
    y = scores['scaled'].to_frame()

    # Features
    X = all_data.drop(['topic', 'target_score'], axis=1)
    # score histogram
    y.hist(bins=61)
    plt.title('Histogram of scaled target scores')
    plt.xlabel('target score')
    plt.ylabel('count')
    plt.savefig('images/scaled_target_score.png', dpi=300)
    X_train, X_test, y_train, y_test, scores_train, scores_test = split_X_y_scores(X, y, scores)
    # Define keras model
    model = build_model(X,y,'mlp')
    # model = build_model('cnn')
    # model = build_model('lstm')

    results = train_model(model, X_train, X_test, y_train, y_test)

    min_max_analysis(results)
    kappa_analysis(results)




