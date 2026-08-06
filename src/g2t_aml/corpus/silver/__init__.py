"""Silver: verified LLM rewrites of Bronze.

**Silver is verified synthetic supervision, not distillation**, and the difference is
mechanical rather than rhetorical. Three things make the distinction real:

1. **Two independent teachers.** A frontier API model and a strong open-weights model,
   assigned deterministically and balanced across typology and split. A corpus written by
   one model is that model's style, and a student trained on it has learned to imitate one
   system.
2. **Every record is verified against the fact record, not against the teacher.** A rewrite
   is accepted because :mod:`g2t_aml.facts.checkers` — the same instrument that produces
   the paper's faithfulness numbers, run in reverse — finds zero contradicted claims and an
   unverifiable rate inside the published budget. Teacher agreement is not evidence of
   anything and is never consulted.
3. **Evaluation is against human-authored Gold**, never against Silver. Measuring overlap
   against LLM-written references measures how well an 8B model imitates a frontier model,
   and a competent reviewer will say so.

**The discard rate is a result, not an operational nuisance.** A frontier model handed a
complete structured fact record and a correct draft still produces unrepairable factual
violations at some rate, and that rate — with its per-class breakdown — is the direct
motivation for a graph-conditioned architecture with a verifier in the loop. It is logged
structurally, per case, with reasons, and it goes in the paper.
"""
