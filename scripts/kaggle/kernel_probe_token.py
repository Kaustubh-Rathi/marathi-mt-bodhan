"""Probe: which HF token candidates exist on this Kaggle account, and do they authenticate?"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.environ.get("MR_MT_REPO_DIR", "/kaggle/working/marathi-mt-bodhan")
REPO_URL = os.environ.get(
    "MR_MT_REPO_URL", "https://github.com/Kaustubh-Rathi/marathi-mt-bodhan.git"
)
if not (Path(REPO) / "scripts" / "kaggle" / "kaggle_env.py").is_file():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.path.insert(0, str(Path(REPO) / "src"))
sys.path.insert(0, str(Path(REPO) / "scripts" / "kaggle"))
import kaggle_env  # noqa: E402

kaggle_env._ensure_import_path()
from mr_mt.secrets import hf_token_can_access, iter_hf_tokens  # noqa: E402

# Probe every gated repo ANY stack needs, so one run verifies the token for
# both the primary (bodhan) and fallback (indictrans2) sessions.
required = list(
    dict.fromkeys(
        kaggle_env._required_gated("bodhan") + kaggle_env._required_gated("indictrans2")
    )
)
candidates = list(iter_hf_tokens())
print(
    "[probe] required gated repos:", ", ".join(rt + "/" + rid for rt, rid in required)
)
print("[probe] unique HF token candidates:", len(candidates))
if not candidates:
    print("[probe] NO token candidates found (no Secret, no mounted dataset, no env)")
for label, token in candidates:
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        body = json.load(urllib.request.urlopen(req, timeout=20))
        who = "OK as " + str(body.get("name"))
    except urllib.error.HTTPError as exc:
        who = "FAIL HTTP " + str(exc.code)
    except Exception as exc:  # noqa: BLE001
        who = "ERROR " + str(exc)
    parts = []
    for repo_type, repo_id in required:
        ok = hf_token_can_access(token, repo_type, repo_id)
        verdict = "ok" if ok else "BLOCKED" if ok is False else "unknown"
        parts.append(repo_type + "/" + repo_id + "=" + verdict)
    print("[probe] " + label + ": whoami=" + who + " | " + ", ".join(parts))

print("[probe] done")
