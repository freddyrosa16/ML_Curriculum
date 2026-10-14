"""Tests for 07-decision_tree."""
import numpy as np
import pytest

from helpers import load_task

BUILD = [f"{i}-build_decision_tree.py" for i in range(9)]


def build(i):
    return load_task(__file__, BUILD[i])


def from_task(first):
    return list(range(first, 9))


def example_0(m):
    leaf0 = m.Leaf(0, depth=1)
    leaf1 = m.Leaf(0, depth=2)
    leaf2 = m.Leaf(1, depth=2)
    inner = m.Node(feature=1, threshold=30000, left_child=leaf1, right_child=leaf2, depth=1)
    root = m.Node(feature=0, threshold=0.5, left_child=leaf0, right_child=inner,
                  depth=0, is_root=True)
    return m.Decision_Tree(root=root)


def example_1(m, depth):
    """Perfect tree on feature 0: leaf values 2**depth-1 ... 0 from left to right."""
    level = [m.Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def value(node):
        return node.value if node.is_leaf else node.threshold

    for d in range(depth):
        level = [m.Node(feature=0,
                        threshold=(value(level[2 * i]) + value(level[2 * i + 1])) / 2,
                        left_child=level[2 * i], right_child=level[2 * i + 1],
                        depth=depth - d - 1)
                 for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return m.Decision_Tree(root=root)


def clouds(n_clouds=6, per_cloud=30, sigma=0.12, seed=0):
    """Gaussian clouds with centers evenly spaced on the unit circle."""
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(n_clouds) / n_clouds
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    target = np.repeat(np.arange(n_clouds), per_cloud)
    explanatory = centers[target] + rng.normal(scale=sigma, size=(len(target), 2))
    return explanatory, target


def split(explanatory, target, proportion=0.2, seed=0):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(target), dtype=bool)
    test[rng.choice(len(target), int(len(target) * proportion), replace=False)] = True
    return explanatory[~test], target[~test], explanatory[test], target[test]


def clean(text):
    return "\n".join(line.rstrip() for line in str(text).rstrip("\n").splitlines())


# ------------------------------------------------------------------ basics

@pytest.mark.parametrize("i", from_task(0))
def test_class_attributes(i):
    m = build(i)
    n = m.Node(feature=2, threshold=1.5, depth=3)
    assert (n.feature, n.threshold, n.depth) == (2, 1.5, 3)
    assert n.left_child is None and n.right_child is None
    assert n.is_leaf is False and n.is_root is False and n.sub_population is None
    leaf = m.Leaf(7, depth=4)
    assert isinstance(leaf, m.Node)
    assert leaf.value == 7 and leaf.is_leaf is True and leaf.depth == 4
    t = m.Decision_Tree(max_depth=5, min_pop=2, seed=3)
    assert isinstance(t.rng, np.random.Generator)
    assert isinstance(t.root, m.Node) and t.root.is_root
    assert (t.max_depth, t.min_pop, t.split_criterion) == (5, 2, "random")
    assert t.explanatory is None and t.target is None and t.predict is None


@pytest.mark.parametrize("i", from_task(0))
def test_depth(i):
    m = build(i)
    assert example_0(m).depth() == 2
    assert example_1(m, 5).depth() == 5
    assert m.Leaf(1, depth=3).max_depth_below() == 3


@pytest.mark.parametrize("i", from_task(1))
def test_count_nodes(i):
    m = build(i)
    t = example_0(m)
    assert t.count_nodes() == 5
    assert t.count_nodes(only_leaves=True) == 3
    t = example_1(m, 4)
    assert t.count_nodes() == 31
    assert t.count_nodes(only_leaves=True) == 16


EXAMPLE_0_STR = """root [feature=0, threshold=0.5]
    +---> leaf [value=0]
    +---> node [feature=1, threshold=30000]
           +---> leaf [value=0]
           +---> leaf [value=1]"""

EXAMPLE_1_STR = """root [feature=0, threshold=1.5]
    +---> node [feature=0, threshold=2.5]
    |      +---> leaf [value=3]
    |      +---> leaf [value=2]
    +---> node [feature=0, threshold=0.5]
           +---> leaf [value=1]
           +---> leaf [value=0]"""


@pytest.mark.parametrize("i", from_task(2))
def test_str(i):
    m = build(i)
    assert clean(example_0(m)) == EXAMPLE_0_STR
    assert clean(example_1(m, 2)) == EXAMPLE_1_STR
    assert clean(m.Leaf(4)) == "-> leaf [value=4]"


@pytest.mark.parametrize("i", from_task(3))
def test_get_leaves(i):
    m = build(i)
    leaves = example_0(m).get_leaves()
    assert all(leaf.is_leaf for leaf in leaves)
    assert [leaf.value for leaf in leaves] == [0, 0, 1]
    assert [leaf.value for leaf in example_1(m, 3).get_leaves()] == list(range(7, -1, -1))


@pytest.mark.parametrize("i", from_task(4))
def test_update_bounds(i):
    m = build(i)
    t = example_0(m)
    t.update_bounds()
    got = [(leaf.lower, leaf.upper) for leaf in t.get_leaves()]
    assert got == [({0: 0.5}, {0: np.inf}),
                   ({0: -np.inf, 1: 30000}, {0: 0.5}),
                   ({0: -np.inf}, {0: 0.5, 1: 30000})]
    assert t.root.lower == {0: -np.inf} and t.root.upper == {0: np.inf}


@pytest.mark.parametrize("i", from_task(5))
def test_indicator(i):
    m = build(i)
    t = example_0(m)
    t.update_bounds()
    leaves = t.get_leaves()
    for leaf in leaves:
        leaf.update_indicator()
    A = np.array([[0, 22000], [0, 44000], [1, 22000], [0.5, 30000], [0.5, 30001]])
    got = np.array([leaf.indicator(A) for leaf in leaves])
    assert got.dtype == bool and got.shape == (3, 5)
    expected = np.array([[False, False, True, False, False],
                         [False, True, False, False, True],
                         [True, False, False, True, False]])
    assert np.array_equal(got, expected), \
        "lower bounds are strict (>), upper bounds are not (<=)"
    rng = np.random.default_rng(0)
    B = np.stack([rng.uniform(-1, 2, 200), rng.uniform(0, 60000, 200)], axis=1)
    assert np.all(np.sum([leaf.indicator(B) for leaf in leaves], axis=0) == 1), \
        "every individual must be in exactly one leaf"


@pytest.mark.parametrize("i", from_task(6))
def test_predict(i):
    m = build(i)
    t = example_0(m)
    t.update_predict()
    A = np.array([[1, 22000], [1, 44000], [0, 22000], [0, 44000]])
    assert list(t.predict(A)) == [0, 0, 1, 0]
    assert [t.pred(x) for x in A] == [0, 0, 1, 0]
    t = example_1(m, 4)
    t.update_predict()
    B = np.random.default_rng(1).uniform(-1, 17, size=(300, 1))
    B[:10, 0] = np.arange(10) + 0.5  # exactly on thresholds
    assert np.array_equal(t.predict(B), [t.pred(x) for x in B])
    assert np.array_equal(t.predict(B), np.clip(np.ceil(B[:, 0] - 0.5), 0, 15))


# ----------------------------------------------------------------- training

@pytest.fixture
def cloud_data():
    return split(*clouds())


def stats(t):
    return t.depth(), t.count_nodes(), t.count_nodes(only_leaves=True)


@pytest.mark.parametrize("i", from_task(7))
def test_fit_random(i, cloud_data):
    m = build(i)
    Xtr, ytr, Xte, yte = cloud_data
    t = m.Decision_Tree(split_criterion="random", max_depth=20, seed=0)
    t.fit(Xtr, ytr)
    assert t.accuracy(Xtr, ytr) == 1.0
    assert t.accuracy(Xte, yte) > 0.85
    assert t.depth() <= 20
    assert np.array_equal(t.predict(Xte), [t.pred(x) for x in Xte])
    # follows the documented random procedure exactly
    assert stats(t) == (12, 77, 39)


@pytest.mark.parametrize("i", from_task(7))
def test_fit_limits(i, cloud_data):
    m = build(i)
    Xtr, ytr, _, _ = cloud_data
    t = m.Decision_Tree(max_depth=3, seed=1)
    t.fit(Xtr, ytr)
    assert t.depth() <= 3
    big = m.Decision_Tree(max_depth=20, min_pop=15, seed=1)
    big.fit(Xtr, ytr)
    small = m.Decision_Tree(max_depth=20, min_pop=1, seed=1)
    small.fit(Xtr, ytr)
    assert big.count_nodes() < small.count_nodes()
    assert m.Decision_Tree(seed=0).np_extrema(np.array([3, -1, 7])) == (-1, 7)


@pytest.mark.parametrize("i", from_task(7))
def test_fit_verbose(i, cloud_data, capsys):
    m = build(i)
    Xtr, ytr, _, _ = cloud_data
    m.Decision_Tree(max_depth=20, seed=0).fit(Xtr, ytr, verbose=1)
    out = capsys.readouterr().out
    assert "Training finished." in out
    for label in ("Depth", "Number of nodes", "Number of leaves", "Accuracy on training data"):
        assert label in out


def test_possible_thresholds():
    m = build(8)
    t = m.Decision_Tree(split_criterion="Gini")
    t.explanatory = np.array([[3.0, 0], [1.0, 0], [2.0, 0], [3.0, 0], [10.0, 0]])
    node = m.Node(is_root=True)
    node.sub_population = np.array([True, True, True, True, False])
    assert np.allclose(t.possible_thresholds(node, 0), [1.5, 2.5])


def brute_force_gini(x, y):
    best = (None, np.inf)
    values = np.unique(x)
    for thr in (values[1:] + values[:-1]) / 2:
        score = 0
        for side in (x > thr, x <= thr):
            _, counts = np.unique(y[side], return_counts=True)
            p = counts / side.sum()
            score += side.sum() / len(y) * (1 - np.sum(p ** 2))
        if score < best[1]:
            best = (thr, score)
    return best


def test_gini_one_feature_and_split():
    m = build(8)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    y = (X[:, 1] + 0.3 * rng.normal(size=60) > 0).astype(int)
    t = m.Decision_Tree(split_criterion="Gini")
    t.explanatory, t.target = X, y
    node = m.Node(is_root=True)
    node.sub_population = np.ones(60, dtype=bool)
    for f in range(3):
        thr, score = t.Gini_split_criterion_one_feature(node, f)
        exp_thr, exp_score = brute_force_gini(X[:, f], y)
        assert np.isclose(thr, exp_thr) and np.isclose(score, exp_score)
    feature, threshold = t.Gini_split_criterion(node)
    assert feature == 1
    assert np.isclose(threshold, brute_force_gini(X[:, 1], y)[0])


def test_fit_gini(cloud_data):
    m = build(8)
    Xtr, ytr, Xte, yte = cloud_data
    g = m.Decision_Tree(split_criterion="Gini", max_depth=20, seed=0)
    g.fit(Xtr, ytr)
    r = m.Decision_Tree(split_criterion="random", max_depth=20, seed=0)
    r.fit(Xtr, ytr)
    assert g.accuracy(Xtr, ytr) == 1.0
    assert g.accuracy(Xte, yte) > 0.85
    assert g.count_nodes() < r.count_nodes()
    assert stats(g) == (3, 11, 6)


# ------------------------------------------------------------------- forests

def test_random_forest(cloud_data):
    RF = load_task(__file__, "9-random_forest.py", "Random_Forest")
    DT = build(8).Decision_Tree
    Xtr, ytr, Xte, yte = cloud_data
    f = RF(max_depth=10, seed=4)
    f.fit(Xtr, ytr, n_trees=7)
    pred = f.predict(Xte)
    assert pred.shape == (len(yte),)
    votes = []
    for k in range(7):
        t = DT(max_depth=10, seed=4 + k)
        t.fit(Xtr, ytr)
        votes.append(t.predict(Xte))
    votes = np.array(votes)
    for j in range(len(yte)):
        values, counts = np.unique(votes[:, j], return_counts=True)
        assert counts[values == pred[j]][0] == counts.max(), "predict the most common vote"
    assert f.accuracy(Xte, yte) > 0.85


def outlier_data():
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(size=(200, 2)) * 0.3, [[3, 3], [-3, 2.5]]])


def test_isolation_tree():
    IT = load_task(__file__, "10-isolation_tree.py", "Isolation_Random_Tree")
    P = outlier_data()
    t = IT(max_depth=15, seed=0)
    t.fit(P)
    depths = t.predict(P)
    assert depths.shape == (202,)
    assert depths.max() <= 15 and depths.min() >= 1
    for leaf in t.get_leaves():
        assert leaf.value == leaf.depth
    assert depths[-2:].mean() < depths[:200].mean()
    assert t.depth() <= 15


def test_isolation_forest():
    IRF = load_task(__file__, "11-isolation_forest.py", "Isolation_Random_Forest")
    IT = load_task(__file__, "10-isolation_tree.py", "Isolation_Random_Tree")
    P = outlier_data()
    f = IRF(max_depth=15, seed=0)
    f.fit(P, n_trees=30)
    mean_depth = f.predict(P)
    trees = []
    for k in range(3):
        t = IT(max_depth=15, seed=k)
        t.fit(P)
        trees.append(t.predict(P))
    small = IRF(max_depth=15, seed=0)
    small.fit(P, n_trees=3)
    assert np.allclose(small.predict(P), np.mean(trees, axis=0))
    rows, depths = f.suspects(P, 2)
    assert rows.shape == (2, 2) and depths.shape == (2,)
    assert {tuple(r) for r in rows} == {(3.0, 3.0), (-3.0, 2.5)}
    assert depths[0] <= depths[1]
    assert np.all(depths <= np.sort(mean_depth)[1] + 1e-12)
