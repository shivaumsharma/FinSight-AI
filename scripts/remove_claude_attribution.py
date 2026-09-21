"""Rewrite all refs, removing Claude/Anthropic co-author trailers only."""

import os
import re
import subprocess


TRAILER = re.compile(
    rb"(?im)^\s*co-authored-by\s*:.*(?:claude|anthropic).*(?:\r?\n|$)"
)
IDENTITY = re.compile(rb"^(author|committer) (.+) <([^>]+)> (\d+ [+-]\d{4})$")


def git(*args, input_data=None, env=None):
    return subprocess.run(
        ["git", *args],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env=env,
    ).stdout


def rewrite_commit(sha, rewritten):
    if sha in rewritten:
        return rewritten[sha]

    raw = git("cat-file", "commit", sha)
    header, message = raw.split(b"\n\n", 1)
    lines = header.splitlines()
    tree = next(line.split(b" ", 1)[1] for line in lines if line.startswith(b"tree "))
    parents = [line.split(b" ", 1)[1] for line in lines if line.startswith(b"parent ")]
    new_parents = [rewrite_commit(parent, rewritten) for parent in parents]
    new_message = TRAILER.sub(b"", message)

    if parents == new_parents and message == new_message:
        rewritten[sha] = sha
        return sha

    commit_env = os.environ.copy()
    for kind in (b"author", b"committer"):
        identity = next(line for line in lines if line.startswith(kind + b" "))
        match = IDENTITY.match(identity)
        if not match:
            raise ValueError(f"Cannot parse {identity!r} in {sha.decode()}")
        prefix = kind.decode().upper()
        commit_env[f"GIT_{prefix}_NAME"] = match.group(2).decode(
            "utf-8", "surrogateescape"
        )
        commit_env[f"GIT_{prefix}_EMAIL"] = match.group(3).decode(
            "utf-8", "surrogateescape"
        )
        commit_env[f"GIT_{prefix}_DATE"] = match.group(4).decode("ascii")

    args = ["commit-tree", tree.decode("ascii")]
    for parent in new_parents:
        args.extend(["-p", parent.decode("ascii")])
    new_sha = git(*args, input_data=new_message, env=commit_env).strip()
    rewritten[sha] = new_sha
    return new_sha


refs = [
    tuple(line.split(b" ", 1))
    for line in git("for-each-ref", "--format=%(refname) %(objectname)").splitlines()
]
rewritten = {}
for sha in git("rev-list", "--all").splitlines():
    rewrite_commit(sha, rewritten)

for ref, old_sha in refs:
    new_sha = rewritten.get(old_sha, old_sha)
    if new_sha != old_sha:
        subprocess.run(
            ["git", "update-ref", ref.decode(), new_sha.decode(), old_sha.decode()],
            check=True,
        )

print(
    f"refs={len(refs)} "
    f"rewritten_commits={sum(old != new for old, new in rewritten.items())}"
)
