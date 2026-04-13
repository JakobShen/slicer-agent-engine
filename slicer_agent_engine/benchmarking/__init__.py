"""Benchmark utilities.

This package is intentionally small and pragmatic:

* Dataset adapters live in `slicer_agent_engine.benchmarking.datasets.*`
* Rule-based judges live in `slicer_agent_engine.benchmarking.judge`

The goal is to make it easy to add another dataset/task without touching the
OpenAI/Slicer "agent" runner.
"""
