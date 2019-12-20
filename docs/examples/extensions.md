## Scatter-initialization

```python
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
```

## Warm-start

```python
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

start_point = {
    model: {"n_estimators": [190], "max_depth": [2], "min_samples_split": [5]}
}


opt = Cypher(search_config, warm_start=start_point)
opt.search(X, y)
```

## Memory

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from cypher import Cypher

iris_data = load_iris()
X = iris_data.data
y = iris_data.target


def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"]
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {model: {"n_estimators": range(10, 200, 10), "max_depth": range(2, 15)}}

"""
The memory will remember previous evaluations done during the optimization process.
Instead of retraining the model, it accesses the memory and uses the saved score/loss.
This shows as a speed up during the optimization process, since the whole search space has been explored.
"""
opt = Cypher(search_config, n_iter=1000, memory=True)

# search best hyperparameter for given data
opt.search(X, y)
```
