# Self-Healing Validator Loop

## Validator Agent

The **Validator** agent checks whether the generator's answer is fully supported by the retrieved context. It uses a faithfulness check: unsupported claims, invented numbers, or facts not present in context should fail validation.

## Query Refinement on Failure

When validation fails and `retry_count` is below **MAX_RETRIES** (default **3**), the graph routes to the **heal** node, increments the retry counter, and returns control to the **Orchestrator**. The Orchestrator refines the query using validator feedback, then the pipeline re-retrieves, reranks, and regenerates.

## Retry Termination

If validation still fails after three retries, the graph ends and returns the last answer along with validation metadata. The frontend reasoning trace shows each retry via `retry_triggered` events.

## Orchestrator Role

The **Orchestrator** parses the user query, tracks retry count, and decides routing. On a healing loop it produces a `refined_query` aimed at retrieving better supporting evidence.
