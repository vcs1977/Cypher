# Author: Tanish Shinde
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from cypher import Cypher

data = load_iris()
X = data.data
y = data.target

n_iter = 30


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 11),
        "min_samples_leaf": range(1, 11),
    }
}


def test_HillClimbingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="HillClimbing")


def test_StochasticHillClimbingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticHillClimbing")


def test_TabuOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="TabuSearch")


def test_RandomSearchOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomSearch")


def test_RandomRestartHillClimbingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomRestartHillClimbing")


def test_RandomAnnealingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="RandomAnnealing")


def test_SimulatedAnnealingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="SimulatedAnnealing")


def test_StochasticTunnelingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="StochasticTunneling")


def test_ParallelTemperingOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="ParallelTempering")


def test_ParticleSwarmOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="ParticleSwarm")


def test_EvolutionStrategyOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="EvolutionStrategy")


def test_BayesianOptimizer():
    opt = Cypher(X, y)
    opt.search(search_config, n_iter=n_iter, optimizer="Bayesian")
