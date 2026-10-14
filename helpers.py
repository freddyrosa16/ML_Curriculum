"""Shared test helper: load a task file (e.g. ``0-slice_me_up.py``) from a project folder.

Tests live in ``<project>/tests/``. Your solution files live in ``<project>/``.
A task you haven't written yet is skipped, not failed, so you can work one task at a time.

Set ``ANSWER_KEY=1`` to run the same tests against ``../answer-key/`` instead
(used to validate the tests themselves).
"""
import importlib.util
import os
import pathlib
import sys

import pytest

CURRICULUM_ROOT = pathlib.Path(__file__).resolve().parent
ANSWER_KEY_ROOT = CURRICULUM_ROOT.parent / "answer-key"


def project_dir(test_file):
    """Folder holding the solution files for the test file's project."""
    project = pathlib.Path(test_file).resolve().parent.parent
    if os.environ.get("ANSWER_KEY") == "1":
        project = ANSWER_KEY_ROOT / project.relative_to(CURRICULUM_ROOT)
    return project


_current_project = None


def _switch_project(project):
    """Put ``project`` on sys.path so solution files can import each other
    (e.g. ``__import__('0-neuron')``), and forget the previous project's
    task modules so same-named files in different projects don't collide."""
    global _current_project
    if _current_project == project:
        return
    if _current_project is not None:
        if str(_current_project) in sys.path:
            sys.path.remove(str(_current_project))
        old = str(_current_project)
        for name, mod in list(sys.modules.items()):
            if (getattr(mod, "__file__", None) or "").startswith(old):
                del sys.modules[name]
    sys.path.insert(0, str(project))
    _current_project = project


def load_task(test_file, filename, attr=None):
    """Import ``filename`` from the project and return the module, or ``attr`` from it.

    Usage inside a test:
        slice_me_up = load_task(__file__, "0-slice_me_up.py")
        add_arrays = load_task(__file__, "4-line_up.py", "add_arrays")
    """
    project = project_dir(test_file)
    path = project / filename
    if not path.exists():
        pytest.skip(f"{filename} not written yet")
    _switch_project(project)
    name = path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr) if attr else module
