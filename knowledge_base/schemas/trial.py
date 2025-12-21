# knowledge_base/schemas/trial.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DocumentSpan:
    """
    Represents a span in a source document, used for explainability traces.
    """
    document_id: str
    start_char: int
    end_char: int
    text: str


@dataclass
class TrialProtocolDocument:
    """
    Raw trial protocol (or article) document.
    """
    document_id: str
    trial_id: str
    title: str
    full_text: str
    metadata: Dict[str, Any]


@dataclass
class AtomicCriterion:
    """
    Atomized inclusion/exclusion criterion.
    """
    id: str
    trial_id: str
    type: str  # "inclusion" or "exclusion"
    description: str
    # DSL expression or structured condition for rules engine
    dsl_expression: Optional[str]
    # link to original text
    source_spans: List[DocumentSpan]
    importance: str = "important"  # "critical" | "important" | "optional"

@dataclass
class CompositeCriterion:
    """
    Composite criterion decomposed into a tree of atomic sub-criteria.
    """
    id: str
    trial_id: str
    type: str  # "inclusion" or "exclusion"
    description: str
    logic_operator: str  # e.g. "AND", "OR", "NOT", "XOR"
    children: List["CompositeCriterion"]  # can include atomic as leaves
    atomic_leaf_ids: List[str]
    source_spans: List[DocumentSpan]