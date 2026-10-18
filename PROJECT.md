---
name: CSCS Study App
repo: The-Pack-Man/CSCS-Streamlit-mobile
status: maintained
one_liner: Mobile-friendly Streamlit study app for the CSCS certification, with smart distractor generation and progress tracking.
problem: Off-the-shelf study tools did not cover CSCS content with useful practice questions.
user: Built for my own exam prep.
tags: [problem-selection]
stack: [Python, Streamlit]
run: streamlit run Streamlit_app.py
test: none

claims:
  - id: C1
    text: Generated plausible multiple-choice distractors using blended sequence and token-overlap similarity.
    evidence: [Streamlit_app.py]
  - id: C2
    text: Tracked daily progress and stats across terms, questions, and practice tests.
    evidence: [study_core.py]

metrics: []
decisions: []
limitations:
  - Not an AI/ML project. Use only as an example of building for a real personal need.
  - Personal progress files (stats.json, progress*.json*) are committed (left in place: the app reads them at startup).
---
