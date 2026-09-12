import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
