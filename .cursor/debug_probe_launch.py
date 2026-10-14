#!/usr/bin/env python3
"""Debug probe: launch.json types vs installed Cursor debug extensions."""
import json
import os
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent / "debug-208afe.log"
SESSION = "208afe"
ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / ".vscode" / "launch.json"
EXT_DIR = Path.home() / ".cursor" / "extensions"
BUILTIN_JS_DEBUG = Path(
    "/Applications/Cursor.app/Contents/Resources/app/extensions/ms-vscode.js-debug/package.json"
)


def log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    entry = {
        "sessionId": SESSION,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    # #endregion


def main() -> None:
    raw = LAUNCH.read_text(encoding="utf-8")
    # strip // comments for parse (VS Code allows them)
    lines = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    cfg = json.loads("\n".join(lines))
    types = [c.get("type") for c in cfg.get("configurations", [])]
    log(
        "A",
        "debug_probe_launch.py:types",
        "launch.json configuration types",
        {"types": types, "hasPwaNode": "pwa-node" in types, "hasJava": "java" in types},
    )

    user_exts = []
    if EXT_DIR.is_dir():
        user_exts = sorted(
            p.name for p in EXT_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    java_debug = [e for e in user_exts if "java-debug" in e or "vscode-java-pack" in e or e.startswith("redhat.java")]
    log(
        "B",
        "debug_probe_launch.py:extensions",
        "installed user extensions (java-related)",
        {
            "allUserExtensions": user_exts,
            "javaRelated": java_debug,
            "javaDebuggerPresent": len(java_debug) > 0,
        },
    )

    pwa_deprecated = None
    if BUILTIN_JS_DEBUG.is_file():
        js = json.loads(BUILTIN_JS_DEBUG.read_text(encoding="utf-8"))
        for dbg in js.get("contributes", {}).get("debuggers", []):
            if dbg.get("type") == "pwa-node":
                pwa_deprecated = dbg.get("deprecated")
    log(
        "C",
        "debug_probe_launch.py:js-debug",
        "builtin js-debug pwa-node deprecation text",
        {
            "pwaNodeDeprecatedMessage": pwa_deprecated,
            "matchesUserError": pwa_deprecated == "Please use type node instead",
        },
    )

    log(
        "D",
        "debug_probe_launch.py:conclusion-hint",
        "hypothesis snapshot",
        {
            "hypothesisA_falsePositiveFromMissingJavaDebug": (
                "java" in types and not java_debug and pwa_deprecated == "Please use type node instead"
            ),
            "hypothesisE_configActuallyPwaNode": "pwa-node" in types,
        },
    )
    print(f"Wrote probe logs to {LOG}")


if __name__ == "__main__":
    main()
