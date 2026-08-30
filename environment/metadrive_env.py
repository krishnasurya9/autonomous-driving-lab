"""Compatibility import for the 2D environment used by existing entry points."""

from environment.driving_2d_env import Driving2DEnv

# Keep the old symbol available for notebooks and scripts that import it directly.
CustomMetaDriveEnv = Driving2DEnv

__all__ = ["Driving2DEnv", "CustomMetaDriveEnv"]
