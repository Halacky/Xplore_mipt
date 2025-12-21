# /home/kirill/projects_2/folium/Xplore/nlp/llm/ensemble.py

from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
import concurrent.futures
import os
import json
from datetime import datetime
from .base import LLMEnsemble, LLMClient, LLMCallTrace
import threading
logger = logging.getLogger(__name__)

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(LOG_ROOT, exist_ok=True)

_file_lock = threading.Lock()
def _dump_trace_to_file(
    task_name: str,
    phase: str,
    trace: LLMCallTrace,
    suffix: str = "",
    call_id: str | None = None,
) -> None:
    """
    Dump a single LLMCallTrace to a JSON file on disk for later inspection.
    Files are organized as:
      logs/<task_name>/<YYYYMMDD_HHMMSS>_[callid]_ <phase>_<model>[_suffix].json
    """
    with _file_lock:
        try:
            task_dir = os.path.join(LOG_ROOT, task_name)
            os.makedirs(task_dir, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            safe_model = trace.model_name.replace(":", "_").replace("/", "_")

            # ВАЖНО: добавляем call_id, чтобы различать отдельные вызовы ансамбля
            if call_id:
                fname = f"{ts}_{call_id}_{phase}_{safe_model}"
            else:
                fname = f"{ts}_{phase}_{safe_model}"

            if suffix:
                fname += f"_{suffix}"
            fname += ".json"
            path = os.path.join(task_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": trace.model_name,
                        "provider": trace.provider,
                        "temperature": trace.temperature,
                        "seed": trace.seed,
                        "prompt": trace.prompt,
                        "raw_response": trace.raw_response,
                        "parsed_output": trace.parsed_output,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("Trace dumped to %s", path)
        except Exception as e:
            logger.warning("Failed to dump trace to file: %s", e)


class SimpleLLMEnsemble(LLMEnsemble):
    def run_task(self, task_prompt: str) -> Dict[str, Any]:
        all_traces: List[LLMCallTrace] = []
        for client in self.base_models:
            for _ in range(self.n_repeats):
                trace = client.generate(task_prompt)
                all_traces.append(trace)
        final_answer = all_traces[0].parsed_output if all_traces else ""
        return {
            "final_answer": final_answer,
            "traces": [t.__dict__ for t in all_traces],
        }


class EnsembleTaskRunner:
    """
    Deterministic ensemble runner with logging of all intermediate steps.
    """

    def __init__(
        self,
        base_models: List[LLMClient],
        judge_model: LLMClient,
        n_repeats: int = 3,
        per_model_aggregation_prompt_builder: Optional[
            Callable[[str, List[str]], str]
        ] = None,
        final_aggregation_prompt_builder: Optional[
            Callable[[str, List[str]], str]
        ] = None,
    ) -> None:
        self.base_models = base_models
        self.judge_model = judge_model
        self.n_repeats = n_repeats
        self.per_model_aggregation_prompt_builder = (
            per_model_aggregation_prompt_builder
        )
        self.final_aggregation_prompt_builder = final_aggregation_prompt_builder

    def run_task_generic(
        self,
        task_name: str,
        input_text: str,
        base_prompt_builder: Callable[[str], str],
    ) -> Dict[str, Any]:
        """
        Generic deterministic ensemble runner:
        - base_models x n_repeats runs on base_prompt
        - per-model judge aggregation
        - final judge aggregation
        """
        import uuid
        call_id = str(uuid.uuid4())[:8]
        logger.info(
            "=" * 80 +
            f"\nrun_task_generic CALLED: task_name={task_name}, call_id={call_id}\n" +
            f"base_models={len(self.base_models)}, n_repeats={self.n_repeats}\n" +
            "=" * 80
        )
        if self.per_model_aggregation_prompt_builder is None:
            raise ValueError("per_model_aggregation_prompt_builder is not set")
        if self.final_aggregation_prompt_builder is None:
            raise ValueError("final_aggregation_prompt_builder is not set")

        all_traces: List[LLMCallTrace] = []
        per_model_aggregated_outputs: List[str] = []
        per_model_aggregated_traces: List[LLMCallTrace] = []

        def _run_for_single_model(model_index: int, base_model: LLMClient) -> Tuple[List[LLMCallTrace], LLMCallTrace]:
            repeated_raw_outputs: List[str] = []
            repeated_traces: List[LLMCallTrace] = []

            base_prompt = base_prompt_builder(input_text)

            def _single_run(run_idx: int) -> LLMCallTrace:
                trace = base_model.generate(
                    base_prompt,
                    seed=None,
                    temperature=0.2,
                )
                _dump_trace_to_file(
                    task_name,
                    f"base_model_{model_index}",
                    trace,
                    suffix=f"run{run_idx+1}",
                    call_id=call_id,
                )
                return trace

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_repeats) as executor:
                futures = [executor.submit(_single_run, i) for i in range(self.n_repeats)]
                for f in concurrent.futures.as_completed(futures):
                    tr = f.result()
                    repeated_traces.append(tr)
                    repeated_raw_outputs.append(tr.raw_response)

            judge_prompt = self.per_model_aggregation_prompt_builder(
                input_text, repeated_raw_outputs
            )
            judge_trace = self.judge_model.generate(
                judge_prompt,
                seed=None,
                temperature=0.0,
            )
            _dump_trace_to_file(
                task_name,
                "per_model_judge",
                judge_trace,
                suffix=f"model{model_index}",
                call_id=call_id,
            )
            return repeated_traces, judge_trace

        for model_index, base_model in enumerate(self.base_models):
            repeated_traces, judge_trace = _run_for_single_model(model_index, base_model)
            all_traces.extend(repeated_traces)
            per_model_aggregated_outputs.append(judge_trace.raw_response)
            per_model_aggregated_traces.append(judge_trace)
            all_traces.append(judge_trace)

        final_prompt = self.final_aggregation_prompt_builder(
            input_text, per_model_aggregated_outputs
        )
        final_trace = self.judge_model.generate(
            final_prompt,
            seed=None,
            temperature=0.0,
        )
        _dump_trace_to_file(task_name, "final_judge", final_trace, call_id=call_id)
        all_traces.append(final_trace)

        return {
            "final_answer": final_trace.parsed_output,
            "final_trace": final_trace,
            "per_model_aggregates": per_model_aggregated_outputs,
            "per_model_aggregate_traces": [t.__dict__ for t in per_model_aggregated_traces],
            "all_traces": [t.__dict__ for t in all_traces],
        }
    
    def run_patient_feature_task(
        self,
        note_text: str,
        base_prompt_builder: Callable[[str], str],
    ) -> Dict[str, Any]:
        import uuid
        call_id = str(uuid.uuid4())[:8]
        logger.info(
            "EnsembleTaskRunner: starting patient feature task. "
            "call_id=%s base_models=%d, n_repeats=%d",
            call_id,
            len(self.base_models),
            self.n_repeats,
        )
        if self.per_model_aggregation_prompt_builder is None:
            raise ValueError("per_model_aggregation_prompt_builder is not set")
        if self.final_aggregation_prompt_builder is None:
            raise ValueError("final_aggregation_prompt_builder is not set")

        logger.info(
            "EnsembleTaskRunner: starting patient feature task. "
            "base_models=%d, n_repeats=%d",
            len(self.base_models),
            self.n_repeats,
        )

        task_name = "patient_features"

        all_traces: List[LLMCallTrace] = []
        per_model_aggregated_outputs: List[str] = []
        per_model_aggregated_traces: List[LLMCallTrace] = []

        # Optional: run base models in parallel as well
        def _run_for_single_model(model_index: int, base_model: LLMClient):
            logger.info("Base model #%d (%s): starting repeats", model_index, base_model.model_name)

            repeated_raw_outputs: List[str] = []
            repeated_traces: List[LLMCallTrace] = []

            base_prompt = base_prompt_builder(note_text)
            logger.debug(
                "Base model #%d prompt (truncated to 500 chars): %s",
                model_index,
                base_prompt[:500],
            )

            def _single_run(run_idx: int) -> LLMCallTrace:
                logger.info(
                    "Base model #%d: repeat %d/%d",
                    model_index,
                    run_idx + 1,
                    self.n_repeats,
                )
                trace = base_model.generate(
                    base_prompt,
                    seed=None,
                    temperature=0.2,
                )
                logger.debug(
                    "Base model #%d repeat %d raw_response (truncated to 500 chars): %s",
                    model_index,
                    run_idx + 1,
                    trace.raw_response[:500],
                )
                _dump_trace_to_file(
                    task_name,
                    f"base_model_{model_index}",
                    trace,
                    suffix=f"run{run_idx+1}",
                    call_id=call_id,
                )
                return trace

            # Repeats in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_repeats
            ) as executor:
                future_to_idx = {
                    executor.submit(_single_run, i): i
                    for i in range(self.n_repeats)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    trace = future.result()
                    repeated_traces.append(trace)
                    repeated_raw_outputs.append(trace.raw_response)

            # Build judge prompt for this model's repeated outputs
            judge_prompt = self.per_model_aggregation_prompt_builder(
                note_text, repeated_raw_outputs
            )

            logger.info("Per-model judge: aggregating outputs of base model #%d", model_index)
            logger.debug(
                "Per-model judge prompt for model #%d (truncated to 500 chars): %s",
                model_index,
                judge_prompt[:500],
            )

            judge_trace = self.judge_model.generate(
                judge_prompt,
                seed=None,
                temperature=0.0,
            )
            logger.debug(
                "Per-model judge raw_response for model #%d (truncated to 500 chars): %s",
                model_index,
                judge_trace.raw_response[:500],
            )
            _dump_trace_to_file(
                task_name,
                "per_model_judge",
                judge_trace,
                suffix=f"model{model_index}",
                call_id=call_id,
            )
            return repeated_traces, judge_trace

        for model_index, base_model in enumerate(self.base_models):
            repeated_traces, judge_trace = _run_for_single_model(model_index, base_model)
            all_traces.extend(repeated_traces)
            per_model_aggregated_outputs.append(judge_trace.raw_response)
            per_model_aggregated_traces.append(judge_trace)
            all_traces.append(judge_trace)

        # 2) Final aggregation across base models
        final_prompt = self.final_aggregation_prompt_builder(
            note_text, per_model_aggregated_outputs
        )

        logger.info("Final judge: aggregating all per-model aggregates")
        logger.debug(
            "Final judge prompt (truncated to 500 chars): %s",
            final_prompt[:500],
        )

        final_trace = self.judge_model.generate(
            final_prompt,
            seed=None,
            temperature=0.0,
        )
        logger.debug(
            "Final judge raw_response (truncated to 500 chars): %s",
            final_trace.raw_response[:500],
        )
        _dump_trace_to_file(task_name, "final_judge", final_trace, call_id=call_id)

        all_traces.append(final_trace)

        logger.info("EnsembleTaskRunner: patient feature task completed")

        return {
            "final_answer": final_trace.parsed_output,
            "final_trace": final_trace,
            "per_model_aggregates": per_model_aggregated_outputs,
            "per_model_aggregate_traces": [t.__dict__ for t in per_model_aggregated_traces],
            "all_traces": [t.__dict__ for t in all_traces],
        }