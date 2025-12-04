import subprocess

def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
    except Exception:
        return "unknown"

def is_dirty():
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT
        ).decode().strip()
        return len(status) > 0
    except Exception:
        return False

def get_git_diff():
    try:
        return subprocess.check_output(["git", "diff"]).decode()
    except Exception:
        return ""
