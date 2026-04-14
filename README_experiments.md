# Evaluator-Side Debate for Safety Classification

## Goal

Test whether multi-agent debate improves safety classification on BeaverTails, and whether any gains come from interaction rather than simple consensus.

## Research Questions

- [ ] RQ1: Does evaluator-side debate improve over a single evaluator?
- [ ] RQ2: Does debate outperform independent majority voting at matched multi-agent budget?
- [ ] RQ3: Does role-based debate outperform role-based voting?
- [ ] RQ4: Which BeaverTails harm categories benefit most or least from debate?

## Experiment Conditions

- [ ] Single
  One evaluator with one final judgment.
- [ ] Vote
  Multiple independent evaluators with majority vote.
- [ ] Role-Vote
  Safety roles answer independently, then majority vote decides.
- [ ] Role-Debate
  The same safety roles interact across rounds before the final aggregated judgment.

## Dataset Setup

- [ ] Use BeaverTails prompt-response pairs
- [ ] Confirm split
- [ ] Fix sample size
- [ ] Fix random seed(s)
- [ ] Confirm category labels are logged

## Metrics

- [ ] Overall accuracy
- [ ] Unsafe precision
- [ ] Unsafe recall
- [ ] False negative rate on unsafe responses
- [ ] Per-category accuracy
- [ ] Optional: seed stability across repeated runs

## Day 1: Setup

- [ ] Verify `safety_eval` data loader is correct
- [ ] Verify labels map to `(A) Safe` / `(B) Unsafe`
- [ ] Verify all four conditions are runnable
- [ ] Verify logs contain:
- [ ] gold label
- [ ] predicted label
- [ ] per-agent responses
- [ ] debate answer
- [ ] correctness
- [ ] category
- [ ] Run sanity check on 20-50 examples
- [ ] Inspect outputs manually for formatting and extraction issues

## Day 2: Main Runs

- [ ] Run `Single`
- [ ] Run `Vote`
- [ ] Run `Role-Vote`
- [ ] Run `Role-Debate`
- [ ] Keep the same data subset across all conditions
- [ ] Save all outputs to comparable files
- [ ] Record overall metrics in one results table
- [ ] Inspect disagreement cases:
- [ ] debate correct, vote wrong
- [ ] vote correct, debate wrong
- [ ] all methods wrong

## Day 3: Analysis

- [ ] Compare `Single` vs `Vote`
- [ ] Compare `Vote` vs `Role-Debate`
- [ ] Compare `Role-Vote` vs `Role-Debate`
- [ ] Compute per-category performance
- [ ] Summarize where debate helps
- [ ] Summarize where debate hurts
- [ ] Decide whether gains come from:
- [ ] interaction
- [ ] consensus
- [ ] role diversity

## Results Table

- [ ] Create final comparison table with:
- [ ] condition
- [ ] accuracy
- [ ] unsafe precision
- [ ] unsafe recall
- [ ] unsafe false negative rate
- [ ] Create category-wise summary table or plot

## Final Claim Check

- [ ] Can we say debate improves safety classification?
- [ ] Can we say debate beats consensus?
- [ ] Can we say role structure matters?
- [ ] Do the results support a clean mechanism story?

## Notes and Risks

- [ ] BeaverTails is a safety classification benchmark, not a dedicated judge-instability benchmark
- [ ] Majority vote is the key baseline for any debate claim
- [ ] If debate does not beat vote, the story may be "consensus matters more than interaction"
