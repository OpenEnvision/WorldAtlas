from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> int:
    from worldarena.visualization.showcase import main_with_mode

    return main_with_mode("method_compare")


if __name__ == "__main__":
    raise SystemExit(main())
