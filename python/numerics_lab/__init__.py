"""numerics-lab -- equation-solving numerics: linear and nonlinear solves, time integration.

Depends only on numpy and scipy; the Rust core is ``numerics_lab._native``.
"""

from pkgutil import extend_path

# A `numerics_lab/` portion later on `sys.path` belongs to this package: the cluster serves each sha's
# `python/` first and the separately built kernel (`numerics_lab/_native`, under `$LAB_CI_NATIVE`) after
# it, and a submodule is found through `__path__` alone. tests/unit/test_a_later_portion_supplies_the_kernel.py.
__path__ = extend_path(__path__, __name__)
