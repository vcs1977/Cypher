from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from cypher import Cypher

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    model = CatBoostClassifier(
        iterations=10, depth=para["depth"], learning_rate=para["learning_rate"]
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {"depth": range(2, 22), "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0]}
}


opt = Cypher(search_config, n_iter=5)
opt.search(X, y)
