## Minimal Example

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
## Roadmap

<details>
<summary><b>v2.0.0</b>:heavy_check_mark:</summary>

  - [x] Change API
  - [x] Ray integration
</details>

<details open>
<summary><b>v2.1.0</b></summary>

  - [x] Save memory of evaluations for later runs (long term memory)
  - [x] Warm start sequence based optimizers with long term memory
  - [x] Gaussian process regressors from various packages (gpy, sklearn, GPflow, ...) via wrapper
</details>

<details>
<summary><b>v2.2.0</b></summary>

  - [ ] Tree-structured Parzen Estimator
  - [ ] Spiral optimization
  - [ ] Downhill-Simplex-Method
</details>

<details>
<summary><b>v2.3.0</b></summary>

  - [ ] Helper-classes for model pruning
  - [ ] Helper-classes for dataset approximation
</details>

<br>

## Experimental algorithms

The following algorithms are of my own design and, to my knowledge, do not yet exist in the technical literature.
If any of these algorithms already exist I would like you to share it with me in an issue.

#### Random Annealing

A combination between simulated annealing and random search.

#### Scatter Initialization

Inspired by hyperband optimization.

<br>

## References

#### [1] [Proxy Datasets for Training Convolutional Neural Networks](https://arxiv.org/pdf/1906.04887v1.pdf)

<br>
