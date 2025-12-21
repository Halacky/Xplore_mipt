# rules_engine/evaluators/criteria.py
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from rules_engine.dsl.ast import DSLNode
from rules_engine.dsl.parser import DSLParser
from nlp.extraction.patient_features import AtomicPatientFeature
from knowledge_base.schemas.trial import AtomicCriterion


@dataclass
class CriterionEvaluationResult:
    """
    Result of evaluating a single AtomicCriterion against patient features.
    """
    criterion_id: str
    passed: bool
    # mapping feature_name -> used value
    used_features: Dict[str, Any]
    # optional explanation text
    reasoning: str


class CriterionEvaluator:
    """
    Evaluates individual atomic criteria against atomic patient features.
    """

    def __init__(self, parser: DSLParser):
        self.parser = parser

    def evaluate(
        self,
        criterion: AtomicCriterion,
        features: Dict[str, AtomicPatientFeature],
    ) -> CriterionEvaluationResult:
        """
        Evaluate a single atomic criterion.

        features: dict indexed by feature name.
        """
        if not criterion.dsl_expression:
            # TODO: handle non-DSL criteria or raise error
            raise NotImplementedError

        ast: DSLNode = self.parser.parse(criterion.dsl_expression)
        passed, used = self._evaluate_ast(ast, features)
        reasoning = self._build_reasoning(criterion, passed, used)

        return CriterionEvaluationResult(
            criterion_id=criterion.id,
            passed=passed,
            used_features=used,
            reasoning=reasoning,
        )

    def _evaluate_ast(
        self,
        node: DSLNode,
        features: Dict[str, AtomicPatientFeature],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate AST nodes with given features. 
        Returns:
            (bool result, dict of used feature_name -> value)
        """
        # TODO: implement recursive AST evaluation
        raise NotImplementedError

    def _build_reasoning(
        self,
        criterion: AtomicCriterion,
        passed: bool,
        used_features: Dict[str, Any],
    ) -> str:
        """
        Build human-readable explanation text for logging/UI.
        """
        # TODO: implement explanation builder
        raise NotImplementedError