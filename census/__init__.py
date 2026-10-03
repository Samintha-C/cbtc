"""Transcoder census: tag every sampled transcoder latent along independent axes
(activation profile, read type, write type, relevance) so you can measure what
kinds of things transcoder latents are, per layer.

Two stages:
  1. gpu_stage  (torch + circuit-tracer): sample latents, cache activations with
     per-token metadata, export weights, run ablations.
  2. offline    (numpy only): detectors -> tags -> LLM prompts -> report.
"""
