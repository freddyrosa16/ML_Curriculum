"""Tests for 03-sequences-nlp/01-rnns.

Each scenario builds a small cell, overwrites its weights with known random
values, runs it, and compares against expected numbers. Expected values are
hard-coded so this file contains no implementation.
"""
import numpy as np

from helpers import load_task

I, H, O = 3, 4, 2   # input, hidden, output sizes
M, T = 2, 3         # batch size, time steps


def fill(cell, names, seed):
    """Overwrite each named weight with known random values of the same shape."""
    rng = np.random.default_rng(seed)
    for name in names:
        setattr(cell, name, rng.normal(size=getattr(cell, name).shape))
    return rng


# ---------- scenarios (inputs are fixed; outputs are compared to EXPECTED) ----------

def scenario_rnn_cell():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    cell = RNNCell(I, H, O)
    rng = fill(cell, ["Wh", "Wy", "bh", "by"], 0)
    h_prev, x_t = rng.normal(size=(M, H)), rng.normal(size=(M, I))
    return cell.forward(h_prev, x_t)


def scenario_rnn_init():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    np.random.seed(0)
    cell = RNNCell(10, 15, 5)
    return cell.Wh.ravel()[:3], cell.Wy.ravel()[:3]


def scenario_rnn():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    rnn = load_task(__file__, "1-rnn.py", "rnn")
    cell = RNNCell(I, H, O)
    rng = fill(cell, ["Wh", "Wy", "bh", "by"], 1)
    X, h_0 = rng.normal(size=(T, M, I)), rng.normal(size=(M, H))
    return rnn(cell, X, h_0)


def scenario_gru_cell():
    GRUCell = load_task(__file__, "2-gru_cell.py", "GRUCell")
    cell = GRUCell(I, H, O)
    rng = fill(cell, ["Wz", "Wr", "Wh", "Wy", "bz", "br", "bh", "by"], 2)
    h_prev, x_t = rng.normal(size=(M, H)), rng.normal(size=(M, I))
    return cell.forward(h_prev, x_t)


def scenario_lstm_cell():
    LSTMCell = load_task(__file__, "3-lstm_cell.py", "LSTMCell")
    cell = LSTMCell(I, H, O)
    rng = fill(cell, ["Wf", "Wu", "Wc", "Wo", "Wy", "bf", "bu", "bc", "bo", "by"], 3)
    h_prev, c_prev = rng.normal(size=(M, H)), rng.normal(size=(M, H))
    x_t = rng.normal(size=(M, I))
    return cell.forward(h_prev, c_prev, x_t)


def scenario_deep_rnn():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    deep_rnn = load_task(__file__, "4-deep_rnn.py", "deep_rnn")
    cells = [RNNCell(I, H, O), RNNCell(H, H, O), RNNCell(H, H, O)]
    rng = None
    for k, cell in enumerate(cells):
        rng = fill(cell, ["Wh", "Wy", "bh", "by"], 40 + k)
    X, h_0 = rng.normal(size=(T, M, I)), rng.normal(size=(3, M, H))
    return deep_rnn(cells, X, h_0)


def _bi_cell(filename, seed):
    Bi = load_task(__file__, filename, "BidirectionalCell")
    cell = Bi(I, H, O)
    rng = fill(cell, ["Whf", "Whb", "Wy", "bhf", "bhb", "by"], seed)
    return cell, rng


def scenario_bi_forward():
    cell, rng = _bi_cell("5-bi_forward.py", 5)
    return cell.forward(rng.normal(size=(M, H)), rng.normal(size=(M, I)))


def scenario_bi_backward():
    cell, rng = _bi_cell("6-bi_backward.py", 6)
    return cell.backward(rng.normal(size=(M, H)), rng.normal(size=(M, I)))


def scenario_bi_output():
    cell, rng = _bi_cell("7-bi_output.py", 7)
    return cell.output(rng.normal(size=(T, M, 2 * H)))


def scenario_bi_rnn():
    cell, rng = _bi_cell("7-bi_output.py", 8)
    bi_rnn = load_task(__file__, "8-bi_rnn.py", "bi_rnn")
    X = rng.normal(size=(T, M, I))
    h_0, h_t = rng.normal(size=(M, H)), rng.normal(size=(M, H))
    return bi_rnn(cell, X, h_0, h_t)


# BEGIN EXPECTED (generated from the answer key)
EXPECTED = {'bi_backward': [[0.4372273, 0.87116396, 0.66121611, 0.99990799],
                 [0.23442639, 0.99962605, 0.81836638, 0.99593828]],
 'bi_forward': [[-0.99948904, -0.97848938, 0.96036466, -0.39544933],
                [0.98521038, 0.23496924, 0.98723191, 0.75344926]],
 'bi_output': [[[0.26628167, 0.73371833], [0.98052493, 0.01947507]],
               [[0.00166296, 0.99833704], [0.57919351, 0.42080649]],
               [[0.00467309, 0.99532691], [0.26689657, 0.73310343]]],
 'bi_rnn': [[[[0.80511953, 0.826315, -0.51198969, -0.91591375, -0.99613577, -0.41410335, 0.99999158,
               -0.99495669],
              [0.76070994, 0.99960534, -0.99430849, 0.88262228, 0.99967204, 0.92947973, -0.98230906,
               0.98565787]],
             [[-0.99973436, -0.7296482, -0.88478913, 0.12041742, 0.99905809, -0.87151427,
               -0.14873319, -0.90328007],
              [-0.99868638, -0.94929623, -0.89275579, 0.97030106, -0.97505535, 0.99967647,
               0.81339437, -0.97768175]],
             [[0.99479038, 0.39724844, 0.95481863, 0.71706293, -0.63219212, -0.83742658, 0.97537846,
               -0.99985961],
              [0.99940893, 0.99679117, 0.96920763, 0.38692443, 0.87107473, 0.71944692, -0.77221639,
               -0.99989514]]],
            [[[0.01573547, 0.98426453], [0.99600205, 0.00399795]],
             [[0.94831228, 0.05168772], [0.53884963, 0.46115037]],
             [[0.33981184, 0.66018816], [0.87119238, 0.12880762]]]],
 'deep_rnn': [[[[[0.71122658, 0.79334724, -0.34872507, -0.46235179],
                 [0.85797588, -0.19130432, -1.27568632, -1.13328721]],
                [[-0.91945229, 0.49716074, 0.14242574, 0.69048535],
                 [-0.42725265, 0.15853969, 0.62559039, -0.30934654]],
                [[0.45677524, -0.66192594, -0.36305385, -0.38173789],
                 [-1.19583965, 0.48697248, -0.46940234, 0.01249412]]],
               [[[-0.95911362, -0.63622205, 0.03523616, 0.99931264],
                 [-0.56381001, 0.87492929, 0.41550704, 0.99773129]],
                [[0.95861529, 0.86701264, -0.18031014, 0.98760699],
                 [0.76978109, 0.98006239, -0.62432236, 0.972071]],
                [[0.99309827, -0.2080642, 0.9952111, 0.26143771],
                 [0.20579198, 0.66422565, 0.93741082, -0.97738855]]],
               [[[0.62457874, 0.96481316, -0.14654898, -0.99556368],
                 [-0.98041308, -0.81139594, -0.8447808, 0.9847236]],
                [[-0.98755584, 0.99341105, -0.61300203, 0.9804406],
                 [-0.07389003, 0.98131953, 0.69278384, 0.98876454]],
                [[0.91730689, 0.13496103, 0.99648459, 0.78364952],
                 [-0.58336703, -0.99336409, 0.99839156, 0.90295791]]],
               [[[-0.62740489, 0.77706708, 0.60999765, 0.85121667],
                 [0.97767138, 0.98591522, 0.00135007, -0.94044718]],
                [[0.99659933, 0.99948857, -0.50547002, 0.94402784],
                 [-0.96967187, 0.88665364, -0.90020159, 0.98768346]],
                [[0.91912024, -0.92238192, 0.9999304, 0.36387604],
                 [0.99289374, 0.99872363, 0.95938156, -0.67547147]]]],
              [[[0.65504114, 0.34495886], [0.85091343, 0.14908657]],
               [[0.41898871, 0.58101129], [0.40728378, 0.59271622]],
               [[0.69798676, 0.30201324], [0.79869023, 0.20130977]]]],
 'gru_cell': [[[-0.76068971, -0.42107031, 0.99428222, -0.8975597],
               [-0.42175922, -0.71729798, -0.21645335, -1.95329093]],
              [[0.30984832, 0.69015168], [0.00506385, 0.99493615]]],
 'lstm_cell': [[[-0.00366787, 0.00171952, 0.54435645, -0.39387556],
                [0.60372066, -0.00299468, 0.1192606, -0.02050533]],
               [[-0.0173238, 0.01673562, 0.72329189, -0.67009693],
                [1.221271, -0.41568049, 0.40703792, -0.15325286]],
               [[0.0652857, 0.9347143], [0.20824433, 0.79175567]]],
 'rnn': [[[[0.05410228, 0.27279134, -0.98218812, -1.10737305],
           [0.19958453, -0.46674962, 0.23550561, 0.75951952]],
          [[0.43413943, -0.80870291, -0.54005028, -0.95573587],
           [0.0356064, -0.679531, 0.18236667, 0.36816291]],
          [[-0.8912974, -0.99065801, 0.94918429, -0.82911179],
           [0.01137232, -0.73687409, 0.9671384, 0.81663078]],
          [[-0.99678252, -0.99944809, -0.88591557, -0.75043088],
           [-0.52698428, -0.70301503, -0.87280791, -0.4100686]]],
         [[[0.13764868, 0.86235132], [0.03967824, 0.96032176]],
          [[0.00241328, 0.99758672], [0.00507487, 0.99492513]],
          [[0.16630994, 0.83369006], [0.33323411, 0.66676589]]]],
 'rnn_cell': [[[-0.38215722, -0.96774198, 0.99908605, 0.98829076],
               [-0.99997673, -0.99946382, 0.98182813, 0.98611782]],
              [[0.07058146, 0.92941854], [0.10705523, 0.89294477]]],
 'rnn_init': [[1.76405235, 0.40015721, 0.97873798], [0.49383678, 0.64331447, -1.57062341]]}
# END EXPECTED


def close(actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"shape {actual.shape} != expected {expected.shape}"
    assert np.allclose(actual, expected, atol=1e-6), "values differ from the expected output"


# ---------- Task 0 ----------

def test_rnn_cell_weight_shapes_and_zero_biases():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    cell = RNNCell(10, 15, 5)
    assert cell.Wh.shape == (25, 15)
    assert cell.Wy.shape == (15, 5)
    assert cell.bh.shape == (1, 15) and not cell.bh.any()
    assert cell.by.shape == (1, 5) and not cell.by.any()


def test_rnn_cell_init_order():
    wh, wy = scenario_rnn_init()
    close(wh, EXPECTED["rnn_init"][0])
    close(wy, EXPECTED["rnn_init"][1])


def test_rnn_cell_forward():
    h_next, y = scenario_rnn_cell()
    close(h_next, EXPECTED["rnn_cell"][0])
    close(y, EXPECTED["rnn_cell"][1])
    assert np.allclose(y.sum(axis=1), 1), "softmax rows should sum to 1"


# ---------- Task 1 ----------

def test_rnn_forward_shapes_and_values():
    H_all, Y = scenario_rnn()
    assert H_all.shape == (T + 1, M, H)
    assert Y.shape == (T, M, O)
    close(H_all, EXPECTED["rnn"][0])
    close(Y, EXPECTED["rnn"][1])


def test_rnn_keeps_initial_state():
    RNNCell = load_task(__file__, "0-rnn_cell.py", "RNNCell")
    rnn = load_task(__file__, "1-rnn.py", "rnn")
    cell = RNNCell(I, H, O)
    h_0 = np.full((M, H), 0.5)
    H_all, _ = rnn(cell, np.zeros((T, M, I)), h_0)
    assert np.allclose(H_all[0], h_0)


# ---------- Task 2 ----------

def test_gru_cell_shapes():
    GRUCell = load_task(__file__, "2-gru_cell.py", "GRUCell")
    cell = GRUCell(10, 15, 5)
    for name in ["Wz", "Wr", "Wh"]:
        assert getattr(cell, name).shape == (25, 15), name
    assert cell.Wy.shape == (15, 5)
    for name in ["bz", "br", "bh"]:
        assert getattr(cell, name).shape == (1, 15) and not getattr(cell, name).any(), name
    assert cell.by.shape == (1, 5)


def test_gru_cell_forward():
    h_next, y = scenario_gru_cell()
    close(h_next, EXPECTED["gru_cell"][0])
    close(y, EXPECTED["gru_cell"][1])


# ---------- Task 3 ----------

def test_lstm_cell_shapes():
    LSTMCell = load_task(__file__, "3-lstm_cell.py", "LSTMCell")
    cell = LSTMCell(10, 15, 5)
    for name in ["Wf", "Wu", "Wc", "Wo"]:
        assert getattr(cell, name).shape == (25, 15), name
    assert cell.Wy.shape == (15, 5)
    for name in ["bf", "bu", "bc", "bo"]:
        assert getattr(cell, name).shape == (1, 15) and not getattr(cell, name).any(), name


def test_lstm_cell_forward():
    h_next, c_next, y = scenario_lstm_cell()
    close(h_next, EXPECTED["lstm_cell"][0])
    close(c_next, EXPECTED["lstm_cell"][1])
    close(y, EXPECTED["lstm_cell"][2])


# ---------- Task 4 ----------

def test_deep_rnn():
    H_all, Y = scenario_deep_rnn()
    assert H_all.shape == (T + 1, 3, M, H)
    assert Y.shape == (T, M, O)
    close(H_all, EXPECTED["deep_rnn"][0])
    close(Y, EXPECTED["deep_rnn"][1])


# ---------- Tasks 5-7 ----------

def test_bi_cell_shapes():
    Bi = load_task(__file__, "5-bi_forward.py", "BidirectionalCell")
    cell = Bi(10, 15, 5)
    assert cell.Whf.shape == (25, 15) and cell.Whb.shape == (25, 15)
    assert cell.Wy.shape == (30, 5)
    assert cell.bhf.shape == (1, 15) and cell.bhb.shape == (1, 15)
    assert cell.by.shape == (1, 5)


def test_bi_forward():
    close(scenario_bi_forward(), EXPECTED["bi_forward"])


def test_bi_backward():
    close(scenario_bi_backward(), EXPECTED["bi_backward"])


def test_bi_output():
    Y = scenario_bi_output()
    assert Y.shape == (T, M, O)
    assert np.allclose(Y.sum(axis=-1), 1)
    close(Y, EXPECTED["bi_output"])


# ---------- Task 8 ----------

def test_bi_rnn():
    H_all, Y = scenario_bi_rnn()
    assert H_all.shape == (T, M, 2 * H)
    assert Y.shape == (T, M, O)
    close(H_all, EXPECTED["bi_rnn"][0])
    close(Y, EXPECTED["bi_rnn"][1])
