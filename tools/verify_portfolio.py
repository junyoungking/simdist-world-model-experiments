"""Static portfolio verification. Never imports or executes experiment modules."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODULES = {"portfolio_paths", "go2_probe_common", "curve_data", "explicit_data",
                 "state", "model_heads", "gap_config", "condition_config", "common",
                 "t3_fresh_replication", "target_env", "recovery_support"}
EXTERNAL_MODULES = {"simdist", "isaaclab", "isaaclab_tasks", "isaaclab_assets",
                    "isaaclab_rl", "isaacsim", "omni", "rsl_rl", "jax", "flax",
                    "optax", "orbax", "torch", "numpy", "scipy", "h5py", "hydra",
                    "omegaconf", "yaml", "matplotlib", "tqdm", "wandb"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def defined_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names.update(x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name))
    return names


def validate_png(path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG signature")
    pos = 8
    image_data = []
    dimensions = None
    saw_end = False
    while pos < len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        crc = struct.unpack(">I", data[pos + 8 + size:pos + 12 + size])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            raise ValueError("PNG CRC mismatch")
        if kind == b"IHDR":
            dimensions = list(struct.unpack(">II", payload[:8]))
        elif kind == b"IDAT":
            image_data.append(payload)
        elif kind == b"IEND":
            saw_end = True
        pos += size + 12
    if not dimensions or not saw_end or not zlib.decompress(b"".join(image_data)):
        raise ValueError("Incomplete PNG")
    return dimensions


def main():
    failures = []
    manifest = json.loads((ROOT / "docs/BUILD_MANIFEST.json").read_text())
    records = manifest["files"]
    for r in records:
        p = ROOT / r["destination"]
        if not p.is_file():
            failures.append({"kind": "missing_selected_file", "file": r["destination"]})
        elif r.get("public_sha256") and sha256(p) != r["public_sha256"]:
            failures.append({"kind": "selected_hash_changed", "file": r["destination"]})
    code_paths = sorted(ROOT.rglob("*.py"))
    trees = {}
    for p in code_paths:
        try:
            text = p.read_text()
            trees[p] = ast.parse(text, filename=str(p))
            compile(text, str(p), "exec")
        except SyntaxError as exc:
            failures.append({"kind": "python_syntax", "file": str(p.relative_to(ROOT)), "line": exc.lineno})
    lookup = {}
    for p in code_paths:
        lookup.setdefault(p.stem, []).append(p)
    external = set()
    local_edges = 0
    upstream_checked = 0
    simdist = Path(os.environ["SIMDIST_ROOT"]).resolve() if os.environ.get("SIMDIST_ROOT") else None
    for p, tree in trees.items():
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module or ""] if isinstance(n, ast.ImportFrom) else [])
            for mod in mods:
                head = mod.split(".")[0]
                if head in LOCAL_MODULES:
                    choices = lookup.get(head, [])
                    local = next((q for q in choices if q.parent == p.parent), choices[0] if choices else None)
                    if local is None:
                        failures.append({"kind": "missing_local_import", "file": str(p.relative_to(ROOT)), "module": mod})
                        continue
                    local_edges += 1
                    if isinstance(n, ast.ImportFrom):
                        exported = defined_names(trees[local])
                        for name in n.names:
                            if name.name != "*" and name.name not in exported:
                                failures.append({"kind": "missing_local_symbol", "file": str(p.relative_to(ROOT)), "module": mod, "symbol": name.name})
                elif head in EXTERNAL_MODULES:
                    external.add(head)
                    if head == "simdist" and simdist:
                        rel = mod.replace(".", "/")
                        exists = (simdist / (rel + ".py")).is_file() or (simdist / rel / "__init__.py").is_file()
                        if not exists:
                            failures.append({"kind": "missing_upstream_import", "file": str(p.relative_to(ROOT)), "module": mod})
                        upstream_checked += 1
                elif head and head not in sys.stdlib_module_names:
                    failures.append({"kind": "undeclared_import", "file": str(p.relative_to(ROOT)), "module": mod})
    shell_files = sorted(ROOT.rglob("*.sh"))
    for p in shell_files:
        check = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        if check.returncode:
            failures.append({"kind": "shell_syntax", "file": str(p.relative_to(ROOT))})
    links = 0
    for p in [ROOT / "README.md", ROOT / "docs/REPRODUCIBILITY.md"]:
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", p.read_text()):
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", link) or link.startswith("#"):
                continue
            target = link.split("#", 1)[0]
            links += 1
            if not (p.parent / target).is_file():
                failures.append({"kind": "missing_internal_link", "file": str(p.relative_to(ROOT)), "target": target})
    csv_rows = {}
    for r in records:
        p = ROOT / r["destination"]
        if p.suffix == ".csv":
            with p.open(newline="") as stream:
                reader = csv.DictReader(stream)
                items = list(reader)
                if not reader.fieldnames or any(None in x for x in items):
                    failures.append({"kind": "csv_shape", "file": r["destination"]})
                csv_rows[r["destination"]] = len(items)
    png_dimensions = {}
    for p in ROOT.rglob("*.png"):
        try:
            png_dimensions[str(p.relative_to(ROOT))] = validate_png(p)
        except (ValueError, struct.error, zlib.error):
            failures.append({"kind": "png_validation", "file": str(p.relative_to(ROOT))})
    private_patterns = [re.compile(re.escape("/home/" + "kojy" + "1009")),
                        re.compile(re.escape("/" + "mnt/")),
                        re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")]
    credential_patterns = [re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
                           re.compile(r"\b(?:ghp_|github_pat_|hf_|sk-proj-)[A-Za-z0-9_-]{16,}"),
                           re.compile(r'''(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?key|client[_-]?secret)["']?\s*[:=]\s*["'][^"']{8,}["']''')]
    private_flags = []
    credential_flags = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts or p.suffix == ".png":
            continue
        for line, text in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if any(pattern.search(text) for pattern in private_patterns):
                private_flags.append({"file": str(p.relative_to(ROOT)), "line": line})
            if any(pattern.search(text) for pattern in credential_patterns):
                credential_flags.append({"file": str(p.relative_to(ROOT)), "line": line})
    failures.extend({"kind": "private_path", **f} for f in private_flags)
    failures.extend({"kind": "credential_pattern", **f} for f in credential_flags)
    report = {"status": "STATIC_PASS" if not failures else "FAIL", "selected_files": len(records),
              "python_compiled": len(trees), "shell_syntax_checked": len(shell_files),
              "project_local_import_edges": local_edges,
              "external_dependencies": sorted(external),
              "external_runtime_status": "DECLARED_NOT_EXECUTED",
              "upstream_import_edges_checked": upstream_checked,
              "internal_links_checked": links, "selected_csv_files": len(csv_rows),
              "csv_rows": csv_rows, "png_dimensions": png_dimensions,
              "private_path_flags": private_flags, "credential_flags": credential_flags,
              "failures": failures}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
