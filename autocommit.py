# Software Name : mislabeled-benchmark
# SPDX-FileCopyrightText: Copyright (c) Orange Innovation
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.md" file for more details
# or https://github.com/Orange-OpenSource/mislabeled-benchmark/blob/master/LICENSE.md

import subprocess
from filelock import FileLock


def autocommit():
    lock = FileLock("gitlock.lock")

    with lock:
        # check that source branch is actually the current branch
        working_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )

        # saves the current state of the directory into branch
        try:
            stash_msg = (
                subprocess.check_output(["git", "stash"]).decode("ascii").strip()
            )
        except subprocess.CalledProcessError as e:
            print("git stash stdout output:\n", e.output)
            raise

        if stash_msg == "No local changes to save":
            # saves commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        else:
            # switch to detached head
            subprocess.check_call(["git", "switch", "--detach"])

            # updates with commited (merge) and uncommited (stash) modifications
            subprocess.check_call(["git", "stash", "apply"])

            # commits changes
            try:
                subprocess.check_output(["git", "commit", "-am", "experiment"])
            except:
                # I am assuming that the "commit" command failed because there was nothing
                # new to commit
                pass

            # saves commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )

            # switches back to original branch
            subprocess.check_call(["git", "switch", working_branch])

            # reverts working directory (including uncommited changes)
            subprocess.check_call(["git", "stash", "pop"])

    return commit_hash
