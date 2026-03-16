
#!/usr/bin/env python3
"""Legacy compatibility shim.

This entrypoint now delegates to `scripts/open_ai_auto.py` so the project keeps
a single OpenAI auto-runner implementation instead of maintaining duplicate tool
schemas.
"""

from __future__ import annotations

from open_ai_auto import main


if __name__ == "__main__":
    main()
