#!/usr/bin/env python3
"""Compatibility entry point; use validate_llm.py for new deployments."""

from validate_llm import main


if __name__ == "__main__":
    raise SystemExit(main())
