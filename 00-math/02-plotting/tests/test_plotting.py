"""Tests for 00-math/02-plotting. Run: pytest 00-math/02-plotting

Each task function draws a figure and calls plt.show(). The tests switch
matplotlib to the non-interactive "Agg" backend, turn plt.show() into a no-op,
then inspect the figure your function left behind.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from helpers import load_task  # noqa: E402


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def run(filename, fn_name):
    fn = load_task(__file__, filename, fn_name)
    plt.close("all")
    fn()
    return plt.gcf()


def same_color(a, b):
    return np.allclose(mcolors.to_rgba(a), mcolors.to_rgba(b), atol=0.01)


def main_axes(fig):
    """Axes that aren't colorbars."""
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


# ---------- 0. Line Graph ----------

def test_0_line():
    fig = run("0-line.py", "line")
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    line = ax.lines[0]
    assert np.allclose(line.get_ydata(), np.arange(0, 11) ** 3)
    assert np.allclose(line.get_xdata(), np.arange(0, 11))
    assert same_color(line.get_color(), "red")
    assert line.get_linestyle() == "-"
    assert ax.get_xlim() == pytest.approx((0, 10))


# ---------- 1. Scatter ----------

def test_1_scatter():
    fig = run("1-scatter.py", "scatter")
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    pts = ax.collections[0].get_offsets()
    assert pts.shape == (2000, 2)
    assert pts[:, 0].mean() == pytest.approx(69, abs=1)
    assert pts[:, 1].mean() == pytest.approx(180, abs=1)
    assert same_color(ax.collections[0].get_facecolor()[0], "magenta")
    assert ax.get_xlabel() == "Height (in)"
    assert ax.get_ylabel() == "Weight (lbs)"
    assert ax.get_title() == "Men's Height vs Weight"


# ---------- 2. Change of scale ----------

def test_2_change_scale():
    fig = run("2-change_scale.py", "change_scale")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlim() == pytest.approx((0, 28650))
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    assert np.allclose(x, np.arange(0, 28651, 5730))
    assert y[0] == pytest.approx(1.0)
    assert y[1] == pytest.approx(0.5)  # one half-life later
    assert ax.get_xlabel() == "Time (years)"
    assert ax.get_ylabel() == "Fraction Remaining"
    assert ax.get_title() == "Exponential Decay of C-14"


# ---------- 3. Two is better than one ----------

def test_3_two():
    fig = run("3-two.py", "two")
    ax = fig.axes[0]
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert set(lines) == {"C-14", "Ra-226"}
    c14, ra = lines["C-14"], lines["Ra-226"]
    assert same_color(c14.get_color(), "red") and c14.get_linestyle() == "--"
    assert same_color(ra.get_color(), "green") and ra.get_linestyle() == "-"
    # half-lives: C-14 is 5730 years, Ra-226 is 1600 years
    x = np.asarray(c14.get_xdata())
    i = int(np.argmin(np.abs(x - 5000)))
    assert c14.get_ydata()[i] == pytest.approx(0.5 ** (x[i] / 5730), rel=1e-3)
    assert ra.get_ydata()[i] == pytest.approx(0.5 ** (x[i] / 1600), rel=1e-3)
    assert ax.get_xlim() == pytest.approx((0, 20000))
    assert ax.get_ylim() == pytest.approx((0, 1))
    assert ax.get_legend() is not None
    assert ax.get_xlabel() == "Time (years)"
    assert ax.get_ylabel() == "Fraction Remaining"
    assert ax.get_title() == "Exponential Decay of Radioactive Elements"


# ---------- 4. Frequency ----------

def test_4_frequency():
    fig = run("4-frequency.py", "frequency")
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == 10, "one bar per 10-point bin from 0 to 100"
    np.random.seed(5)
    grades = np.random.normal(68, 15, 50)
    in_range = np.sum((grades >= 0) & (grades <= 100))  # one grade is > 100
    assert sum(b.get_height() for b in bars) == in_range
    assert [round(b.get_x()) for b in bars] == list(range(0, 100, 10))
    assert same_color(bars[0].get_edgecolor(), "black")
    assert ax.get_xlim() == pytest.approx((0, 100))
    assert ax.get_ylim() == pytest.approx((0, 30))
    assert ax.get_xlabel() == "Grades"
    assert ax.get_ylabel() == "Number of Students"
    assert ax.get_title() == "Project A"


# ---------- 5. All in One ----------

def test_5_all_in_one():
    fig = run("5-all_in_one.py", "all_in_one")
    axes = main_axes(fig)
    assert len(axes) == 5
    assert fig._suptitle is not None and fig._suptitle.get_text() == "All in One"
    titles = [ax.get_title() for ax in axes]
    for t in ["Men's Height vs Weight", "Exponential Decay of C-14",
              "Exponential Decay of Radioactive Elements", "Project A"]:
        assert t in titles
    # the last plot spans both columns of the bottom row
    last = axes[-1].get_position()
    assert last.width > 1.5 * axes[0].get_position().width
    # small fonts everywhere
    for ax in axes[1:]:
        assert ax.title.get_fontsize() <= 9
        assert ax.xaxis.label.get_fontsize() <= 9


# ---------- 6. Stacking Bars ----------

def test_6_bars():
    fig = run("6-bars.py", "bars")
    ax = fig.axes[0]
    rects = ax.patches
    assert len(rects) == 12, "4 fruits x 3 people"
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]
    for k, color in enumerate(colors):
        for r in rects[3 * k: 3 * k + 3]:
            assert same_color(r.get_facecolor(), color)
            assert r.get_width() == pytest.approx(0.5)
    # bars are stacked: each fruit starts where the previous one ended
    for p in range(3):
        stack = [rects[3 * k + p] for k in range(4)]
        for lower, upper in zip(stack, stack[1:]):
            assert upper.get_y() == pytest.approx(lower.get_y() + lower.get_height())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["apples", "bananas", "oranges", "peaches"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Farrah", "Fred", "Felicia"]
    assert ax.get_ylim() == pytest.approx((0, 80))
    assert np.allclose(ax.get_yticks(), np.arange(0, 81, 10))
    assert ax.get_ylabel() == "Quantity of Fruit"
    assert ax.get_title() == "Number of Fruit per Person"


# ---------- 100. Gradient (bonus) ----------

def test_100_gradient():
    fig = run("100-gradient.py", "gradient")
    axes = main_axes(fig)
    assert len(axes) == 1
    ax = axes[0]
    sc = ax.collections[0]
    assert sc.get_offsets().shape == (2000, 2)
    assert sc.get_array() is not None and len(sc.get_array()) == 2000
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbars) == 1
    assert cbars[0].get_ylabel() == "elevation (m)"
    assert ax.get_xlabel() == "x coordinate (m)"
    assert ax.get_ylabel() == "y coordinate (m)"
    assert ax.get_title() == "Mountain Elevation"


# ---------- 101. PCA (bonus) ----------

def test_101_pca():
    pca = load_task(__file__, "101-pca.py", "pca")
    data_path = pathlib.Path(__file__).resolve().parent.parent / "pca.npz"
    plt.close("all")
    pca(str(data_path))
    fig = plt.gcf()
    ax = fig.axes[0]
    assert ax.name == "3d"
    assert len(ax.collections[0].get_array()) == 150
    assert ax.get_xlabel() == "U1"
    assert ax.get_ylabel() == "U2"
    assert ax.get_zlabel() == "U3"
    assert ax.get_title() == "PCA of Iris Dataset"
