"""
van Rossum distance module for spike train analysis
"""

try:
    from .VanRossumDistance import VanRossumDistance
    __all__ = ['VanRossumDistance']
except ImportError:
    # Handle case where elephant library is not installed
    __all__ = []