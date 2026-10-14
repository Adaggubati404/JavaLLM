# JavaLLM: cleanup checklist

## Files to delete from the repo
- [ ] `.cursor/debug-208afe.log`: an editor debugging session log
- [ ] `.cursor/debug_probe_launch.py`: a one-off debug probe script, unrelated to the project
- [ ] `.vscode/settings.json`, or edit it: it hard-codes a macOS JDK path (`/Library/Java/JavaVirtualMachines/temurin-25.jdk/...`) that won't exist on other machines.
- [ ] In `.vscode/launch.json`, remove the `"env": { "JAVA_HOME": ... }` blocks for the same reason. Keep the launch configs themselves; the README's debug walkthrough uses them.

```bash
git rm -r --cached .cursor
git rm --cached .vscode/settings.json
```

## Lines to add to `.gitignore`
```gitignore
.cursor/
.vscode/settings.json
```
(`.vscode/extensions.json` and `.vscode/launch.json` stay tracked.)

## Add / fix
- [ ] `README.md` (replace with the new one). The old README said training runs "~500 steps", but `Main.java` has `TRAIN_STEPS = 100`; the new README uses 100.
- [ ] `LICENSE` (MIT)
- [ ] The About description from `descriptions.md`
- [ ] Fill in the TODO training-output and sample blocks with real output
