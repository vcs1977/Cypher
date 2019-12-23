
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

## Random Annealing

An algorithm that chooses a new position within a large hypersphere around the current position. This hypersphere gets smaller over time.

---

**Use case/properties:**
- Disclaimer: I have not seen this algorithm before, but invented it myself. It seems to be a good alternative to the other random algorithms
- Good as a first method of optimization
- For a short optimization run to get an acceptable solution

<p align="center">
<img src="./plots/search_paths/RandomAnnealing [('epsilon', 0.1)].svg" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon', 0.3)].svg" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon', 0.5)].svg" width= 49%/>
<img src="./plots/search_paths/RandomAnnealing [('epsilon', 1)].svg" width= 49%/>
</p>

<br>

#### Scatter Initialization

Inspired by hyperband optimization.

## Scatter-initialization

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

# Without scatter initialization
opt = Hyperactive(X, y)
opt.search(
    search_config,
    optimizer="HillClimbing",
    n_iter=10,
    random_state=0,
    init_config=False,
)

init_config = {"scatter_init": 10}

# With scatter initialization
opt = Hyperactive(X, y)
opt.search(
    search_config,
    optimizer="HillClimbing",
    n_iter=10,
    random_state=0,
    init_config=init_config,
)
```


<br>

## References

#### [1] [Proxy Datasets for Training Convolutional Neural Networks](https://arxiv.org/pdf/1906.04887v1.pdf)

<br>
