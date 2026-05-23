"""Dashboard-ops judges package — 8 deterministic + 5 LLM + 3 paired auditors.

Each module exposes ``run(spec, artifacts) -> JudgeResult`` so the
orchestrator can invoke them uniformly. The 16 evaluators are listed in
:mod:`services.judges.audit_runner` and weighted via WEIGHTS.
"""
