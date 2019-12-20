import numpy as np
from cypher import Cypher


def himmelblau(para, X, y):
    """Himmelblau's function"""

    return -(
        (para["x"] ** 2 + para["y"] - 11) ** 2 + (para["x"] + para["y"] ** 2 - 7) ** 2
    )


x_range = np.arange(0, 10, 0.1)

search_config = {himmelblau: {"x": x_range, "y": x_range}}

opt = Cypher(search_config, n_iter=1000000)
opt.search(0, 0)
