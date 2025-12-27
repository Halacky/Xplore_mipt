# Smart Patient Screening: An Explainable AI Approach to Clinical Trial Enrollment

## The Challenge We Solved
Clinical trial patient screening is traditionally a manual, time-consuming process prone to inconsistency. Medical specialists must read through extensive patient records, cross-reference complex inclusion/exclusion criteria, and make subjective eligibility decisions—often without clear documentation of their reasoning. This approach doesn't scale, lacks transparency, and creates bottlenecks in the drug development pipeline.

## Our Solution: A Deterministic, Explainable AI System
We developed MVP, a production-ready prototype that transforms patient screening through a unique architecture built on three core principles:
1. **Maximum Atomicity**.
   
    Instead of treating patient records and trial criteria as monolithic blocks of text, we decompose them into their smallest meaningful units:

    - Patient features: Individual clinical facts (LVEF=29%, NYHA Class II, eGFR=41 mL/min/1.73m²)
    - Trial criteria: Single, indivisible eligibility requirements

    Each atomic element is extracted with precise evidence spans—exact character positions linking back to the source text. This granular approach enables true traceability: every decision traces back to specific sentences in the original   documents.

2. **Ensemble-Based Determinism**

   Rather than relying on a single AI model (which can be unreliable), we employ an ensemble strategy:

    - 3 base models (mix of local Ollama and cloud-based APIs) each extract features/criteria independently, running 3 times each with controlled randomness
    - A judge model aggregates the 9+ outputs, assigning confidence scores based on cross-model consensus
    - This creates reproducible, auditable results while maintaining high accuracy
  
    The system runs multiple strategies in parallel:
  
    - LLM-based evaluation: Deep semantic reasoning about criterion satisfaction
    - Decision tree stub: Rule-based logic (ready for integration)
    - ML predictor stub: Statistical pattern recognition (extensible)

3. **Complete Explainability & Traceability**
    Every verdict comes with:

    - Matched criteria: Which requirements the patient satisfies, with supporting evidence
    - Non-matches: Conflicts that disqualify the patient, explicitly explained
    - Unknowns: Cases where data is insufficient or ambiguous
    - Evidence visualization: Interactive highlighting of text spans in both patient notes and trial protocols

Our frontend provides a forensic-level view: users can see exactly which phrase in the patient record corresponds to which criterion in the protocol, color-coded by match status.

## Technical Highlights

  - Hybrid architecture: Combines local GPU-accelerated models (Ollama) with cloud APIs for optimal speed/quality tradeoff
  - Advanced JSON repair: Deterministic + LLM-based fallback ensures robust parsing of model outputs
  - Batch processing: Parallel evaluation of CSV datasets with intelligent caching
  - PostgreSQL storage: Persistent results with full evaluation history
  - React frontend: Intuitive interface with collapsible sections, test data auto-fill, and real-time progress tracking

## Real-World Impact
This isn't just a proof-of-concept—it's a practical tool designed for clinical research coordinators. The system handles complex multi-condition criteria (e.g., stratified NT-proBNP thresholds based on LVEF ranges), distinguishes between critical vs. optional requirements, and scales to batch-process entire patient databases.
By automating the screening bottleneck while maintaining transparency, Xplore accelerates trial recruitment, reduces human error, and provides auditable documentation that satisfies regulatory requirements.


## How to deploy

Go to the monitoring directory:

```
cd Xplore_mipt/monitoring
```

after that, run the container build.
```
docker compose up -d
```

After launching, you will receive:

1. web UI interface available at http://localhost:3000/

2. Prometheus is available at http://localhost:9090/

3. Grafana at http://localhost:3001/


Also, before launching, create an .env file (an example of filling can be found in the .env.example file). Do not forget to replace the parameters with yours (the address of the provider providing hosting services for LLM models, the token and the names of the models according to your provider's api)

Work results:

https://github.com/user-attachments/assets/ced89b70-dad0-4caf-8055-d7eae68b60f0

Metrics:

<img width="761" height="480" alt="Screenshot from 2025-12-24 21-01-42" src="https://github.com/user-attachments/assets/9ca97bc3-7282-4ad2-8d5b-6597a815f338" />

We assessed the project very rigorously, so the metrics aren't particularly good. However, a preliminary analysis reveals a significant question regarding the true "expert_eligibility" values. In cases where there are discrepancies between the expert assessment and our model's assessment, common sense dictates that the explanation provided by our model is very sound, which calls into question the expert assessment. For example, the source data lack information about signed consent, yet the expert still states that the patient should be included in the study (even though signed consent is a mandatory inclusion criterion). There are also discrepancies with threshold values. For example, for patient emp-013, the LVEF is listed as 39%. The criteria specify a range of EF ≥ 36% to ≤ 40%. For an EF of 36–40%, the following requirements apply: Without (AF): NT-proBNP ≥ 2500 pg/mL, with AF: NT-proBNP ≥ 5000 pg/mL. In the patient: NT-proBNP = 860 pg/mL - which does not meet any of the thresholds. However, the labeling is still included.

Overall, the resulting solution is useful, if not for decision making, then at least for automating labeling and validating existing labeling.
