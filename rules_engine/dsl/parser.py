# rules_engine/dsl/parser.py
from typing import Union
from rules_engine.dsl.ast import DSLNode


class DSLParser:
    """
    Parses textual DSL expressions into AST for evaluation.
    """

    def parse(self, expression: str) -> DSLNode:
        """
        Parse DSL expression string into DSLNode AST.
        """
        # TODO: implement parsing logic
        raise NotImplementedError