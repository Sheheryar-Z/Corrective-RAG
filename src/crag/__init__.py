"""Corrective Retrieval-Augmented Generation (CRAG) with LangGraph."""

from .config import Settings
from .logic import Verdict, classify_retrieval, decompose, select_knowledge

__all__ = ["Settings", "Verdict", "classify_retrieval", "decompose", "select_knowledge"]
__version__ = "0.1.0"
