import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.optimize import fmin_powell


def worker_init_fn(worker_id):
    np.random.seed(worker_id)


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None,
                             max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings



def get_senti(df, data, df_id):
    doc_sent_mag = []
    doc_sent_score = []
    nf_count = 0
    for pet in df_id:
        try:
            with open(f"../data/{data}_sentiment/{pet}.json", 'r',
                      encoding="utf8") as f:
                sentiment = json.load(f)
            doc_sent_mag.append(
                    sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            nf_count += 1
            doc_sent_mag.append(-1)
            doc_sent_score.append(-1)
    df.loc[:, 'doc_sent_mag'] = doc_sent_mag
    df.loc[:, 'doc_sent_score'] = doc_sent_score


def prep_data(n, pca=False, pca_scale=False, split=False,
              remove_low_variance=False, variacne_thresh=0.0001):
    train = pd.read_csv("../data/train/train.csv")
    test = pd.read_csv("../data/test/test.csv")
    # combine train and test
    target = train['AdoptionSpeed']
    train_id = train['PetID']
    test_id = test['PetID']
    train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
    test.drop(['PetID'], axis=1, inplace=True)

    # get sentiment data
    get_senti(train, 'train', train_id)
    get_senti(test, 'test', test_id)

    # split train validation set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_index, val_index = next(iter(splitter.split(train, target)))
    validation = train.iloc[val_index, :].copy().reset_index(drop=True)
    valid_target = target[val_index].copy().reset_index(drop=True)
    train = train.iloc[train_index, :].copy().reset_index(drop=True)
    target = target[train_index].copy().reset_index(drop=True)

    # data  description
    train_desc = train.Description.fillna("none").values
    test_desc = test.Description.fillna("none").values
    valid_desc = validation.Description.fillna("none").values
    tfv = TfidfVectorizer(min_df=3, max_features=1000,
                          strip_accents='unicode', analyzer='word',
                          token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 3),
                          use_idf=1, smooth_idf=1, sublinear_tf=True)
    # Fit TFIDF
    tfv.fit(list(train_desc))
    X = tfv.transform(train_desc)
    X_test = tfv.transform(test_desc)
    X_valid = tfv.transform(valid_desc)
    print(X)
    print("X (tfidf):", X.shape)
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    print(svd.explained_variance_ratio_.sum())
    print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    print("X (svd):", X.shape)
    X = pd.DataFrame(X, columns=[f'svd_{i}' for i in range(n)])
    train = pd.concat((train, X), axis=1)
    X_test = svd.transform(X_test)
    X_test = pd.DataFrame(X_test,
                          columns=[f'svd_{i}' for i in range(n)])
    test = pd.concat((test, X_test), axis=1)
    X_valid = svd.transform(X_valid)
    X_valid = pd.DataFrame(X_valid,
                           columns=[f'svd_{i}' for i in range(n)])
    validation = pd.concat((validation, X_valid), axis=1)
    print("train:", train.shape)
    print("test:", test.shape)
    train_len = len(train)
    valid_len = len(validation)
    all_data = train.append([validation, test])
    all_data.reset_index(inplace=True, drop=True)

    # dealing with breed mis labels
    breed_suspicious_index = list(
            all_data.loc[all_data.loc[:, 'Breed1'] == 0, 'Breed1'].index)
    all_data.loc[breed_suspicious_index, 'Breed2']
    all_data.loc[breed_suspicious_index, 'Breed1'] = all_data.loc[
            breed_suspicious_index, 'Breed2'].copy()
    all_data.loc[breed_suspicious_index, 'Breed2'] = 0

    all_data.loc[all_data.loc[:, 'Breed1'] == 307, 'Breed2']
    len(all_data.loc[(all_data.loc[:, 'Breed1'] == 307) |
                     (all_data.loc[:, 'Breed2'] != 0), 'Breed1'])

    # create mix_breed variable
    mix_breed_index = list(all_data.where(
                (all_data.loc[:, 'Breed1'] == 307) |
                (all_data.loc[:, 'Breed2'] != 0)).dropna().index)

    all_data.loc[mix_breed_index, 'mix_breed'] = 1
    all_data.fillna({'mix_breed': 0}, inplace=True)
    len(all_data.loc[all_data.loc[:, 'Breed1'] == 307, 'Breed1'])
    len(all_data.loc[all_data.loc[:, 'Breed2'] != 0, 'Breed1'])

    # add decription length
    all_data.fillna({'Description': ' '}, inplace=True)
    all_data.loc[:, 'des_len'] = all_data.Description.map(
            lambda x: len(x.split()))

    all_data.drop(['Name', 'RescuerID', 'Description'],
                  axis=1, inplace=True)
    # set the right data_type
    description = all_data.describe(include='all').append(
                    [all_data.isnull().sum().rename('null_vals'),
                     all_data.dtypes.rename('data_types')])
    categorical_des = description.loc[:, (all_data.dtypes == 'int64') |
                                         (all_data.dtypes == 'object')]

    categorical_variables = list(categorical_des.columns)
    all_data.loc[:, categorical_variables] =\
        all_data.loc[:, categorical_variables].astype(object)

    # deal with cat and numeric variables
    numeric_des = description.loc[:, (all_data.dtypes == 'float64') |
                                     (all_data.dtypes == 'int64')]
    numeric_variables = list(numeric_des.columns)

    # from numeric description
    categorical_des = description.loc[:, (all_data.dtypes == 'object')]
    cat_suspicious_list = ['Age', 'des_len', 'Quantity', 'Fee', 'MaturitySize',
                           'FurLength', 'Health', 'VideoAmt']

    # variable type adjustment
    categorical_variables = list(set(categorical_variables) -
                                 set(cat_suspicious_list))
    numeric_variables.extend(cat_suspicious_list)

    # seperate numeric and categorical data and handle them seperatly
    encoder = LabelEncoder()
    all_data = all_data.apply(encoder.fit_transform)
    all_cat_data = all_data.loc[:, categorical_variables]
    all_numeric_data = all_data.loc[:, numeric_variables]
    all_numeric_data[::] = all_numeric_data[::].astype('float64')

# =============================================================================
# 
# 
# 
#     # one hot encode cat variables
#     all_cat_data = pd.get_dummies(all_cat_data, columns=categorical_variables)
# #    all_cat_data.shape
# #    all_cat_data.dtypes
#     # pca cat variables
#     if pca:
#         pca = PCA().fit(all_cat_data)
#         explain = np.cumsum(pca.explained_variance_ratio_)
#         pca_dimenssion_k = np.where(explain > 0.999)[0][0] + 1
#         pca_k = PCA(n_components=pca_dimenssion_k)
#         cat_pca = pca_k.fit_transform(all_cat_data)
#         cat_pca = pd.DataFrame(cat_pca)
# #        cat_pca.describe()
# 
#     # Normalize
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     all_numeric_data[:] = scaler.fit_transform(all_numeric_data)
# #    all_numeric_data.isnull().sum()
#     if pca_scale:
#         cat_pca[:] = scaler.fit_transform(cat_pca)
#         
#     # concat all data
#     if pca:
#         cat_pca.index = all_numeric_data.index
#         all_data_transformed = pd.concat([cat_pca, all_numeric_data,
#                                           all_data.Response], axis=1)
#     else:
#         all_data_transformed = pd.concat([all_cat_data, all_numeric_data,
#                                           all_data.Response], axis=1)
#     if not pca and remove_low_variance:
#         variabel_variance = all_data_transformed.var(axis=0)
#         low_var_col = list(variabel_variance[
#                         variabel_variance < variacne_thresh].index)
#         all_data_transformed = all_data_transformed.drop(low_var_col, axis=1)
# =============================================================================
    all_data_transformed = pd.concat([all_numeric_data, all_cat_data], axis=1)
    # split train and test
    train = all_data_transformed.iloc[:train_len, :].copy()
    validation = all_data_transformed.iloc[train_len:train_len+valid_len,
                                           :].copy()
    test = (all_data_transformed.iloc[train_len+valid_len:, :].copy()
            ).reset_index(drop=True)
    return train, test, target, validation, valid_target, categorical_variables


def train_offset(x0, y, train_preds):
    '''
    Finding offsets
    '''
    res = digit(x0, train_preds)
    return -quadratic_weighted_kappa(y, res)


def digit(x0, train_preds):
    '''
    Digitize train list
    '''
    res = []
    for y in list(train_preds):
        limit = True
        for index, value in enumerate(x0):
            if y < value:
                res.append(index)
                limit = False
                break
        if limit:
            res.append(index + 1)
    return res

def feature_importance_plot(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 25))
    sns.barplot(feature_importance.values, feature_importance.index,
                orient='h', ax=ax).tick_params(labelsize=8)
    plt.show()


def simple_score(x0, target, train_y_pred, valid_y_pred, valid_target):
    offsets = fmin_powell(train_offset, x0, (target, train_y_pred),
                          maxiter=20000, disp=True)
    train_y_pred = digit(offsets, train_y_pred)
    print(f"train_score : {quadratic_weighted_kappa(target, train_y_pred)}")

    valid_y_pred = digit(offsets, valid_y_pred)
    quadratic_weighted_kappa(valid_target, valid_y_pred)
    print(
     f"valid_score : {quadratic_weighted_kappa(valid_target, valid_y_pred)}")
