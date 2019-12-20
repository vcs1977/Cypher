import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from cypher import Cypher

data = load_breast_cancer()
X, y = data.data, data.target


def model0(para, X, y):
    model = ExtraTreesClassifier(
        n_estimators=para["n_estimators"],
        criterion=para["criterion"],
        max_features=para["max_features"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        bootstrap=para["bootstrap"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


def model1(para, X, y):
    model = RandomForestClassifier(
        n_estimators=para["n_estimators"],
        criterion=para["criterion"],
        max_features=para["max_features"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        bootstrap=para["bootstrap"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


def model2(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        learning_rate=para["learning_rate"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
        subsample=para["subsample"],
        max_features=para["max_features"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model0: {
        "n_estimators": range(10, 200, 10),
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
    },
    model1: {
        "n_estimators": range(10, 200, 10),
        "criterion": ["gini", "entropy"],
        "max_features": np.arange(0.05, 1.01, 0.05),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "bootstrap": [True, False],
    },
    model2: {
        "n_estimators": range(10, 200, 10),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": np.arange(0.05, 1.01, 0.05),
        "max_features": np.arange(0.05, 1.01, 0.05),
    },
}


opt = Cypher(search_config, n_iter=30, n_jobs=4)
opt.search(X, y)
