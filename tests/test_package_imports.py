from __future__ import annotations

import subprocess
import sys


def test_data_helper_import_does_not_eagerly_import_robosuite():
    script = (
        "import sys; "
        "import embodied_data_lab.local_reduced_views; "
        "assert 'robosuite' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
