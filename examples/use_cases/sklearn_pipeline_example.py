from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from cypher import Cypher

data = load_breast_cancer()
X, y = data.data, data.target


def pipeline1(filter_, gbc):
    return Pipeline([("filter_", filter_), ("gbc", gbc)])


def pipeline2(filter_, gbc):
    return gbc


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    filter_ = SelectKBest(f_classif, k=para["k"])
    model_ = para["pipeline"](filter_, gbc)

    scores = cross_val_score(model_, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "k": range(2, 30),
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
        "min_samples_leaf": range(1, 11),
        "pipeline": [pipeline1, pipeline2],
    }
}


opt = Cypher(search_config, n_iter=100)
opt.search(X, y)
