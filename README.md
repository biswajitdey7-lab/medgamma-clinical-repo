# Edge AI Medical Triage Assistant

## Overview

This project implements an offline medical triage assistant powered by a 4B parameter medical-domain instruction-tuned language model.

The system classifies patient cases into:

- Risk Level (Low / Medium / High)
- Urgency (Routine / Urgent / Emergency)
- Top 3 Probable Conditions
- Recommended Treatment Actions

The architecture is designed for edge deployment scenarios where internet connectivity may not be available.

---

## Model

**Primary Model:**  
google/medgemma-1.5-4b-it

- 4B parameter medical instruction-tuned model
- Loaded with 4-bit quantization for memory efficiency
- Runs fully offline after initial download
- No API usage
- No cloud dependency

---

## Problem Statement

In low-resource or remote healthcare settings, internet access may be unreliable or unavailable. Cloud-based medical AI systems become unusable in such scenarios.

This project demonstrates how a locally hosted medical LLM can:

- Perform structured triage
- Operate offline
- Be deployed on constrained hardware
- Provide standardized output formatting for clinical workflows

---

## System Architecture

Patient Input
↓
Structured Prompt Template
↓
4B Medical LLM (Local Inference)
↓
Output Parsing Layer (Regex Extraction)
↓
Formatted Triage Response

---

## How It Works

1. Patient information (age, sex, symptoms, country, travel history) is formatted into a structured prompt.
2. The local model generates a response.
3. The system extracts:
   - Risk Level
   - Urgency
   - Top 3 Conditions
   - Treatment recommendations
4. Output is displayed in a standardized triage format.

---

## Offline Capability

- The model is downloaded once via Hugging Face.
- After download, it runs entirely from local cache.
- No internet connection is required for inference.
- No external API calls are made.

This aligns with edge AI deployment principles.

---

## Hardware Notes

- Tested on CPU-only system (11th Gen Intel i5).
- 4-bit quantization enabled for memory efficiency.
- Inference latency observed to be high on CPU-only configuration.
- Demonstrates real-world tradeoff between model size and edge hardware capability.

For optimal performance, GPU acceleration is recommended.

---

## Installation

bash
-pip install -r requirements.txt

Run
-python main.py

## Disclaimer

This system is for research and demonstration purposes only.
It is not a replacement for professional medical diagnosis.
