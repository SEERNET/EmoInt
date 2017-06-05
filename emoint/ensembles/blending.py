import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression


def get_out(x, clf, regr):
    if regr:
        return clf.predict(x)
    else:
        return clf.predict_proba(x)[:, 1]


def blend(X, y, X_submission, clfs, blend_clf=LogisticRegression(), n_folds=10, shuffle=True, seed=0, regr=True):
    """
    This blending technique is used either for regression or binary classification
    :param X: cross validation X
    :param y: cross validation y
    :param X_submission: Submission X
    :param clfs: list of classifiers to blend
    :param blend_clf: classifier to blend all predictions
    :param n_folds: number of folds for cross validation
    :param shuffle: shuffle training data
    :param seed: seed to replicate results
    :param regr: regression or binary classification
    :return: y_submission. probability for class-1 in case of binary classification.
     regression value prediction in case of regression.
    """
    np.random.seed(seed)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(KFold(np.size(y), n_folds))

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            clf.fit(X_train, y_train)
            y_test_pred = get_out(X_test, clf, regr)

            dataset_blend_train[test, j] = y_test_pred
            dataset_blend_test_j[:, i] = get_out(X_submission, clf, regr)

        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    clf = blend_clf
    clf.fit(dataset_blend_train, y)
    y_submission = get_out(dataset_blend_test, clf, regr)

    if not regr:
        y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    return y_submission
