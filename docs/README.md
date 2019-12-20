## Installation

?> PyPi always has the newest version of cypher:

[![pyversions](https://img.shields.io/pypi/pyversions/cypher.svg?style=for-the-badge&logo=python&color=blue&logoColor=white)](https://pypi.org/project/cypherbytanish)
[![PyPI version](https://img.shields.io/pypi/v/cypher?style=for-the-badge&logo=pypi&color=green&logoColor=white)](https://pypi.org/project/cypherbytanish/)
[![PyPI version](https://img.shields.io/pypi/dm/cypher?style=for-the-badge&color=red)](https://pypi.org/project/cypherbytanish/)

```bash
pip install cypherbytanish
```

## Minimal example

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from cypher import Cypher

data = load_breast_cancer()
X, y = data.data, data.target

def model(para, X, y):
    model = GradientBoostingClassifier(n_estimators=para['n_estimators'])
    scores = cross_val_score(model, X, y)

    return scores.mean()

search_config = {
    model: {'n_estimators': range(10, 200, 10)}
}

opt = Cypher(search_config)
opt.search(X, y)
```
