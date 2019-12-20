from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from cypher import Cypher

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

# Without scatter initialization
opt = Cypher(
    search_config,
    optimizer="HillClimbing",
    n_iter=10,
    random_state=0,
    scatter_init=False,
)
opt.search(X, y)


# With scatter initialization
opt = Cypher(
    search_config, optimizer="HillClimbing", n_iter=10, random_state=0, scatter_init=10
)
opt.search(X, y)
