"""Conversational product-retrieval eval (Route C — golden cases).

Multi-turn dialogue-driven retrieval eval: an LLM user simulator plays a
colloquial, emotional, info-dripping customer; the agent runs the real graph;
a case passes if ANY ``search`` call during the conversation retrieves a gold
product (any-of). Gold sets are enumerated from machine-checkable constraints
(category + price band), so they are complete and reproducible.

This package is offline tooling (not part of the CI gate).
"""
