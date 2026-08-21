"""marlabs — orchestrated model-lifecycle pipelines over QM9.

Engine/config split: this package is the reusable engine (typed stages + the
promotion gate + the trigger layer). Individual pipelines are configuration over
these stages, not rewrites.
"""

__version__ = "0.1.0"
