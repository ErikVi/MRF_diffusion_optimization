"""Explicit coordinate order; tensor metric coordinates are known to be invalid."""

SCALAR_PARAMETERS = (
    "t1_ms",
    "t2_ms",
    "diffusivity_mm2_per_s",
    "equilibrium_magnetization",
)
LEGACY_TENSOR_METRIC_PARAMETERS = (
    "t1_ms",
    "t2_ms",
    "equilibrium_magnetization",
    "fractional_anisotropy",
    "legacy_mean_diffusivity",
)
CURRENT_TENSOR_OBJECTIVE_PARAMETERS = LEGACY_TENSOR_METRIC_PARAMETERS[:3]
