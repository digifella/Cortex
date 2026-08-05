#!/usr/bin/env python3
"""Export only the broker URL and API token needed by the protected SP4 client."""

from __future__ import annotations

import os
from pathlib import Path


SOURCE = Path(__file__).with_name("config.env")
TARGET = Path("/home/longboardfella/.config/hermes-delivery/sp4-client.env")


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def main() -> None:
    values = parse_env(SOURCE)
    token = values.get("HERMES_DELIVERY_API_TOKEN", "")
    host = values.get("HERMES_DELIVERY_HOST", "")
    port = values.get("HERMES_DELIVERY_PORT", "7341")
    if len(token) < 32:
        raise SystemExit("HERMES_DELIVERY_API_TOKEN is missing or too short")
    if not host:
        raise SystemExit("HERMES_DELIVERY_HOST is missing")

    TARGET.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".tmp")
    temporary.write_text(
        f"HERMES_DELIVERY_URL=http://{host}:{port}/deliver\n"
        f"HERMES_DELIVERY_API_TOKEN={token}\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, TARGET)
    os.chmod(TARGET, 0o600)
    print(f"Exported protected SP4 client configuration to {TARGET}")


if __name__ == "__main__":
    main()
