#!/bin/bash
# OPAL utility script for setting up git hooks (with modifications)

HOOK_NAMES="applypatch-msg pre-applypatch post-applypatch pre-commit prepare-commit-msg commit-msg post-commit pre-rebase post-checkout post-merge pre-receive update post-receive post-update pre-auto-gc"

GITBASEDIR=`git rev-parse --show-toplevel`
HOOK_DIR=$GITBASEDIR/.git/hooks

for hook in $HOOK_NAMES; do
    # If the hook already exists, is executable, and is not a symlink
    if [ ! -h $HOOK_DIR/$hook -a -x $HOOK_DIR/$hook ]; then
        mv $HOOK_DIR/$hook $HOOK_DIR/$hook.local
    fi
    # if the hook is defined in hooks/, then symlink it in the git hooks
    if [ -f $GITBASEDIR/hooks/$hook ]; then
	    ln -sf $GITBASEDIR/hooks/$hook $HOOK_DIR/$hook
    fi
done
