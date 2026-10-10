"""Batch pipeline steps: extract → detect → crop → embed → score → audio/verbal → fuse → evaluate.

Each module is runnable with ``python -m src.pipeline.<step>`` and is configured
through environment variables (see each module docstring).
"""
