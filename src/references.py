"""Install or check the pinned, read-only upstream repositories."""
import argparse
import json
import subprocess

from src.paths import REPO, REPOS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="clone missing repositories at their pinned revisions")
    parser.add_argument("--only", nargs="+", help="repository names; defaults to all locked references")
    args = parser.parse_args()
    locked = json.loads((REPO / "requirements-references.lock.json").read_text())
    for name in args.only or locked:
        spec, root = locked[name], REPOS / name
        if not root.exists() and args.install:
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "checkout", "--detach", spec["commit"]], check=True)
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        # Ignore generated caches, but reject any local source edits or untracked source.
        status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        changed = [line for line in status.splitlines() if "__pycache__/" not in line and not line.endswith(".pyc")]
        if commit != spec["commit"] or changed:
            raise ValueError(f"Reference {name} differs from its lock; preserve local work and use a clean pinned copy")
        print(name, commit)


if __name__ == "__main__":
    main()
