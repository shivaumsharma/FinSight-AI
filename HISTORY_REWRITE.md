# Claude attribution history rewrite

This repository's local refs were rewritten to remove only commit-message trailer
lines matching `Co-Authored-By:` with `Claude` or `Anthropic`. Trees, author and
committer identities, timestamps, parent order, commit order, and all other
message content were preserved. `.claude/` ignore rules and application or
design references were not changed.

To reproduce the rewrite in a fresh clone, save or copy
`scripts/remove_claude_attribution.py`, then run:

```powershell
git status --short
python scripts/remove_claude_attribution.py
git --no-pager log --all --format="%H%x09%B" |
  Select-String -CaseSensitive:$false -Pattern "(?im)^\s*co-authored-by\s*:.*(claude|anthropic)"
```

The final command must produce no output. Review the ref list before using the
script, and make a backup if the repository contains refs that must not be
rewritten. The script updates every ref returned by `git for-each-ref`, including
branches, tags, remote-tracking refs, and custom refs.

## Force-push implications

Rewritten commits have new object IDs. If these refs replace a hosted history,
push explicitly and with force-with-lease after coordinating with collaborators:

```powershell
git push --force-with-lease origin --all
git push --force-with-lease origin --tags
```

Do not push until all consumers have migrated to the rewritten history. Existing
clones, pull requests, cached objects, and hosting audit records may still retain
the old commits. Rewriting commits also cannot retroactively reassign GitHub
contribution graphs or erase provider-side metadata. Old objects can remain in
local reflogs and object storage until reflogs expire and garbage collection runs.
