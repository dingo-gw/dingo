"""
Utility functions for dealing with git (e.g., check if our repository
is in a clean state or if we need to save a diff file).
Based on https://github.com/timothygebhard/fm4ar/blob/main/fm4ar/utils/git_utils.py
"""

from pathlib import Path

import git


def get_repo() -> git.Repo | None:
    """
    Auxiliary function to get the git repository.
    """
    file_path = Path(__file__).resolve()
    root_dir = file_path.parents[3]
    try:
        return git.Repo(root_dir)
    except git.InvalidGitRepositoryError:
        return None


def get_git_hash() -> str:
    """
    Auxiliary function to get the current hash of the git HEAD.
    """
    repo = get_repo()
    if repo is None:
        return "Not a git repository."
    return str(repo.head.object.hexsha)


def is_dirty() -> bool:
    """
    Auxiliary function to check if the repository is in a dirty state.
    """
    repo = get_repo()
    if repo is not None:
        return bool(repo.is_dirty())
    return False


def get_diff() -> str:
    """
    Auxiliary function to get diff against the current HEAD.
    """
    repo = get_repo()
    if repo is None:
        return "Not a git repository."
    tree = repo.head.commit.tree
    return str(repo.git.diff(tree))


def document_git_status(out_dir: str, verbose: bool = True) -> None:
    """
    Document the current status of the git repository to stdout, and by
    creating files for the hash of the HEAD (and potentially for the
    diff against the HEAD) in the given `target_dir`.
    """
    out_dir = Path(out_dir)

    # Save the current git hash
    if verbose:
        print("Git hash of HEAD:", get_git_hash())
    file_path = out_dir / "git-hash.txt"
    with open(file_path, "w") as f:
        f.write(get_git_hash())

    # Check if the repository is clean or if we need to save a diff
    if is_dirty():
        if verbose:
            print("You have uncommitted changes! Saving diff...", end=" ")
        file_path = out_dir / "git-diff.txt"
        with open(file_path, "w") as f:
            f.write(get_diff())
        if verbose:
            print("Done!")
    else:
        if verbose:
            print("Repository is in a clean state! (No uncommitted changes)")
