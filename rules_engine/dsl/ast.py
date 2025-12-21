# rules_engine/dsl/ast.py
from typing import Any, List
from dataclasses import dataclass


@dataclass
class DSLNode:
    """
    Base node for the internal DSL AST.
    """
    pass


@dataclass
class DSLCondition(DSLNode):
    """
    Leaf condition, e.g. 'NYHA_class >= 2'.
    """
    feature_name: str
    operator: str
    target_value: Any


@dataclass
class DSLLogicalOp(DSLNode):
    """
    Logical operator between conditions or subtrees, e.g. AND/OR.
    """
    op: str  # "AND", "OR", "NOT"
    children: List[DSLNode]