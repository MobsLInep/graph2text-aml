"""The data package must not import torch.

Phases 1-6 and 10 are CPU-only and must stay installable without CUDA (D-004). If a loader
ever grows an unguarded ``import torch``, the default install stops working and CI stops
being representative. This test is deliberately not guarded by ``importorskip``: it must
run on precisely the machines that have no torch.
"""

from __future__ import annotations

import subprocess
import sys

CPU_MODULES = (
    "g2t_aml.data",
    "g2t_aml.data.canonical",
    "g2t_aml.data.download",
    "g2t_aml.data.stats",
    "g2t_aml.data.loaders.amlworld",
    "g2t_aml.data.loaders.elliptic2",
)


def test_importing_the_data_package_does_not_import_torch():
    """Run in a clean interpreter; an already-imported torch would mask the failure."""
    imports = "; ".join(f"import {name}" for name in CPU_MODULES)
    code = (
        f"import sys; {imports}; "
        "assert 'torch' not in sys.modules, "
        "'the data package imported torch: see DECISIONS.md D-004'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_pyg_adapter_explains_itself_when_the_extra_is_absent():
    """Without the graph extra the adapter must raise a pointed ImportError, not fail
    with an opaque NameError somewhere downstream."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # force the import to fail
        "from g2t_aml.data.pyg_adapter import _require_torch\n"
        "try:\n"
        "    _require_torch()\n"
        "except ImportError as exc:\n"
        "    assert 'make install-gpu' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_importing_the_encoder_package_does_not_import_torch():
    """Phase 7's package must stay importable in a CPU-only environment.

    ``g2t_aml.models.encoder`` re-exports the feature and dataset layers, which Phase 10's
    CPU-only evaluation may want in order to read the feature spec version or the split
    helpers without ever building a model. Those two modules therefore import torch inside
    the functions that need it, not at module scope, and this asserts the arrangement
    survives — an unguarded ``import torch`` at the top of ``dataset.py`` would break the
    default install and nothing else would notice.
    """
    code = (
        "import sys, g2t_aml.models.encoder; "
        "assert 'torch' not in sys.modules, "
        "'the encoder package imported torch at module scope: see DECISIONS.md D-004'; "
        "assert 'sklearn' not in sys.modules, "
        "'the encoder package imported scikit-learn at module scope'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_feature_builder_explains_itself_when_the_extra_is_absent():
    """Same contract as the PyG adapter: a pointed ImportError, not an opaque failure."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from g2t_aml.models.encoder.features import FeatureSpace, build_case_data\n"
        "try:\n"
        "    build_case_data(None, None)\n"
        "except ImportError as exc:\n"
        "    assert 'make install-gpu' in str(exc), str(exc)\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
