+++
title = 'Claude Code Fix'
date = 2025-07-13T14:03:25-04:00
draft = false
tags = ["claude", "code", "A.I.", "programming"]
+++

Tried installing clade code and it seemed to work the first time, but when I tried running with the `claude` command, I got this error:

```
npm error code EACCES
npm error syscall rename
npm error errno -13
npm error
npm error Your cache folder contains root-owned files, due to a bug in
npm error previous versions of npm which has since been addressed.
npm error
```

The fix was:

```
sudo chown -R $(whoami):$(id -gn) ~/.npm ~/.npm-global
npm install -g @anthropic-ai/claude-code
```

literally 2 lines.

