"""Moved to dingo.core.SVD: the SVD basis is domain-agnostic (used by the core
NN build system for embedding initialization). Kept as a re-export so existing
imports keep working."""

from dingo.core.SVD import SVDBasis, ApplySVD  # noqa: F401
