"""
Experiments framework for quantum paradox simulations.

This module provides a registry-like structure for different quantum experiments.
Currently supports the double-slit experiment, with extensibility for future
experiments (Mach-Zehnder, quantum eraser, etc.).
"""

from .double_slit import (
    DoubleSlitParams,
    compute_distribution,
    sample_hits,
    compute_metrics,
)

__all__ = [
    'DoubleSlitParams',
    'compute_distribution',
    'sample_hits',
    'compute_metrics',
]

