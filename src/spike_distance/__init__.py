"""
SPIKE-distance module for spike train analysis
"""

try:
    from .SpikeDistance import SpikeDistance
    __all__ = ['SpikeDistance']
except ImportError:
    # Handle case where PySpike library is not installed
    __all__ = []