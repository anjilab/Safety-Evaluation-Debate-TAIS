# Suggestions for the Debate Safety Research Proposal

## How to Cite the Attached Paper (Anonymous, 2025)

The position paper *"Uncertainty-Aware Policy-Preserving Abstractions with Abstention for One-Shot Decisions"* is relevant to your proposal in three specific places. The core connection: their paper argues that **decision systems operating near narrow margins should recognize their uncertainty rather than commit**, and your proposal studies a concrete method (debate) for improving decision confidence on exactly these narrow-margin cases.

### Citation Location 1: Introduction (Framing Inconsistency as a Margin Problem)
When you describe why LLM safety judges flip verdicts, cite this paper for the theoretical framing: verdict flips are concentrated on prompts where the judge's internal decision margin is narrow. Their Definition 2 (γ = M₁ − M₂, the gap between best and runner-up action utilities) maps to the judge's logit gap between "safe" and "unsafe" classifications. The connection: debate is an alternative to abstention—instead of refusing to judge when uncertain, provide adversarial arguments that sharpen the assessment.

### Citation Location 2: Related Work (Decision Margins)
Cite in a paragraph on "decision margins and abstention." Their argument that optimization-centric frameworks fail near decision boundaries provides theoretical grounding for why single-judge evaluation is unreliable on borderline cases—and motivates debate as a mechanism for widening the margin.

### Citation Location 3: Discussion (Connecting Results to Theory)
After presenting results, connect to their framework. If debate widens decision margins on borderline prompts, this vindicates the margin perspective—debate works by providing information that moves the judge away from the decision boundary. If debate narrows margins (under adversarial conditions), cite their abstention argument: maybe the right response is not to debate, but to flag these cases for human review.

**BibTeX entry:**
```
@inproceedings{tanwisuth2025arlet,
  title={Uncertainty-Aware Policy-Preserving Abstractions with Abstention for One-Shot Decisions},
  author={Anonymous},
  booktitle={Workshop on Aligning Reinforcement Learning Experimentalists and Theorists (ARLET)},
  year={2025}
}
```
*Note: Update with real author names and venue details once the paper is de-anonymized.*

---

## How to Cite Hossain et al. (2024) — "A Persuasive Approach to Combating Misinformation"

This paper (arXiv:2310.12065) uses Bayesian persuasion to model how platforms can combat misinformation through strategic information design. It is relevant to your attack surface analysis (Q2).

### Citation Location: Related Work (Information Design)
Cite when discussing why adversarial debate is a plausible attack vector. Hossain et al. show that a party with informational advantage can optimally design signals to manipulate a less-informed party's decisions. In your setting, the adversarial "safe" debater has the same structural advantage: it knows the content is harmful but strategically presents information to persuade the judge otherwise. Their characterization of the optimal signaling scheme as a linear program provides a theoretical upper bound on how effective such manipulation can be.

### Citation Location: Discussion (Attack Surface)
If your adversarial debater succeeds in degrading judge accuracy, cite Hossain et al. to argue that this is not a bug but a fundamental property of information-asymmetric systems. The platform/judge cannot naively trust strategic signals, just as social media platforms cannot naively trust user-generated content assessments.

**BibTeX entry:**
```
@article{hossain2024persuasion,
  title={A Persuasive Approach to Combating Misinformation},
  author={Hossain, Safwan and Mladenovic, Andjela and Chen, Yiling and Gidel, Gauthier},
  journal={arXiv preprint arXiv:2310.12065},
  year={2024}
}
```

---

## 9 Substantive Suggestions for the Proposal

### 1. Define "Borderline" Precisely (HIGH PRIORITY)

**Problem:** The proposal's central claim—"debate helps on borderline cases"—requires a formal definition of "borderline." Without it, the result is unfalsifiable: you can always cherry-pick the cases where debate helps and call those "borderline" post hoc.

**Fix:** Adopt the Seed Stability Index (SSI) from Dörner et al. (arXiv:2512.12066). Prompts with SSI < 0.8 across 5 seeds are borderline. This is quantitative, reproducible, and directly connected to the instability you're trying to fix. Compute SSI *before* running debate, so the borderline set is defined independently of debate performance.

### 2. Include a Compute-Matched Baseline (HIGH PRIORITY)

**Problem:** Debate uses ~5 LLM calls (2 debaters × 2 rounds + 1 judge). If you compare debate to a single judge call and find debate is better, the obvious objection is: "You just used 5x more compute."

**Fix:** Include majority-vote baselines at 3x and 5x the single-judge cost. The key comparison is: **Does debate (5 calls) beat majority vote (5 calls)?** If yes, the adversarial structure of debate adds value beyond mere ensembling. If no, debate is just an expensive ensemble.

### 3. The Attack Surface Is Your Most Novel Contribution (LEAN INTO IT)

**Problem:** "Debate improves consistency" is a modest finding—it might be true, but it's not surprising. The novel finding is in Q2: debate as an attack surface.

**Fix:** Make Experiment 3 (adversarial debater) the centerpiece of the paper. Design it as carefully as Experiment 2:
- **Persuasiveness curve**: Following Khan et al.'s Figure 4, vary the adversarial debater's strength (best-of-1, best-of-4, best-of-16, critique-and-refinement) and plot judge accuracy against debater win rate.
- **Truth asymmetry test**: Check whether the truth asymmetry from Khan et al. (persuasive debaters on the correct side improve accuracy, while those on the wrong side degrade it) holds in safety evaluation. If it does, debate has a structural defense. If it doesn't, safety evaluation is fundamentally different from factual QA—and that's an important finding.
- **Practical implication**: If the attack succeeds, recommend countermeasures (debater auditing, ensembled judges, adversarial training of judges against persuasive debaters).

### 4. Connect to Bayesian Persuasion Theory (USES HOSSAIN ET AL.)

**Problem:** Q2 (attack surface) is currently framed purely empirically. It would be stronger with theoretical grounding.

**Fix:** Frame the adversarial debater as a Bayesian persuader in the sense of Hossain et al. (2024) and Kamenica & Gentzkow (2011). The key insight: the adversarial debater has *informational advantage* (it has access to the prompt-response pair and argues strategically), while the judge relies on the debaters' signals. The optimal signaling scheme from Bayesian persuasion provides a theoretical upper bound on how much damage a strategic debater can do. If your empirical attack results are far below this upper bound, there may be room for more sophisticated attacks. If they're close, the current attack already approaches the theoretical limit.

### 5. Vary Judge Capability (Tests Scalable Oversight)

**Problem:** The proposal uses LLaMA 3.1 8B and 70B judges but doesn't explicitly frame this as a test of scalable oversight.

**Fix:** Make the judge capability comparison a central axis of the paper. Khan et al.'s core result is that debate helps *weaker* judges supervise *stronger* models. Your Hypothesis H5 captures this, but make it more prominent:
- Plot debate improvement (accuracy gain over single judge) as a function of judge capability.
- If the improvement is larger for 8B judges than 70B judges, this validates scalable oversight for safety.
- The practical implication is enormous: instead of deploying GPT-4 as a safety judge (expensive), you could deploy debate with GPT-3.5 (cheap) and get comparable accuracy.

### 6. Category-Level Analysis Is Essential

**Problem:** SORRY-Bench has 44 fine-grained categories, but the proposal only discusses aggregate results.

**Fix:** Break down ALL results by category (or at minimum, by broad harm type). Hypothesize which categories debate helps most:
- **Subtle harms** (coded language, dual-use queries, indirect discrimination): Debate should help most here, because the judge needs reasoning support to detect harm.
- **Obvious harms** (explicit violence, direct threats): Debate may not help because the harm is self-evident.
- **Normatively ambiguous** (cultural sensitivity, political speech): Debate may make things *worse* by introducing spurious arguments on both sides.

This analysis turns a single aggregate number into actionable deployment guidance: "Use debate for categories X, Y, Z; use single judge for categories A, B, C."

### 7. Address the "Ground Truth" Problem

**Problem:** Safety judgments are inherently more ambiguous than the factual QA tasks where debate was validated (Khan et al. used reading comprehension with objectively correct answers). Safety has genuine normative disagreement.

**Fix:** Measure inter-annotator agreement on SORRY-Bench human labels. Partition prompts into:
- **Clear ground truth**: High inter-annotator agreement (>90%). Here debate should straightforwardly improve accuracy.
- **Ambiguous ground truth**: Low inter-annotator agreement (<70%). Here "accuracy" is less meaningful. Instead, measure whether debate produces *more consistent* verdicts (lower flip rate) even if we can't determine which verdict is "correct."

This distinction prevents reviewers from dismissing your results by saying "you're just matching noisy labels."

### 8. Scope for NeurIPS Main Conference (9 Pages)

The proposal has 4 experiments plus preliminary results. Recommended structure for NeurIPS main:

- **Core (in main paper):** Experiments 2 (debate vs. baselines) and 3 (attack surface). These ARE the contribution.
- **Supporting (in main paper):** Experiment 1 (instability characterization) — this sets up the problem. Keep it concise, 1 page max.
- **Supplementary:** Experiment 4 (systematic misses), full category-level tables, additional judge models, debate round ablations.
- **Cut entirely:** Anything without clean results. Better to have 2 strong experiments than 4 mediocre ones.

The core narrative: **(1)** LLM safety judges are unreliable; **(2)** debate can help; **(3)** but debate also creates a new attack surface; **(4)** here's when to use it and when not to.

### 9. Frame Against "Just Use Ensembles" Criticism

The strongest criticism this paper will face is: "Why not just run 5 judges and take majority vote? Why does the adversarial structure of debate matter?"

You must provide a crisp answer. The possibilities are:

- **Debate is better than majority vote at matched compute**: This is the dream result. The adversarial structure forces the judge to consider arguments it wouldn't generate independently.
- **Debate is equal to majority vote on average but better on specific categories**: The adversarial structure helps on subtle/ambiguous cases where independent judges all make the same error.
- **Debate is worse than majority vote but reveals failure modes**: Even if debate doesn't improve accuracy, the adversarial debater analysis reveals how judges can be fooled—a valuable contribution to safety evaluation methodology.

Any of these is publishable, but you must lead with the strongest version of the result.

---

## Connection to Your Broader Research Program

Your existing work on Strategic Abstraction (SER, epistemic sufficiency/stability) and Coalitional Agency (well-founded arbitration under internal disagreement) provides a unique lens on this problem that distinguishes your contribution from pure empirical work.

**Strategic Abstraction connection:** The judge's verdict instability is a failure of strategic equivalence. Two seeds should be "strategically equivalent" (they don't change the optimal verdict) but empirically they're not—the judge's behavior depends on distinctions that *shouldn't* matter. Debate can be seen as a mechanism for enforcing strategic equivalence: by providing adversarial arguments, it forces the judge to base its verdict on the content rather than on seed-dependent features.

**Coalitional Agency connection:** The debate protocol is a pluralist arbitration problem. The judge must arbitrate between two "experts" (debaters) with divergent assessments. Your well-foundedness criterion (epistemic sufficiency + stability) maps to the judge's task: the verdict is well-founded when different weightings of the debaters' arguments lead to the same verdict (stability) and when compressing the arguments doesn't lose decision-relevant information (sufficiency). Borderline cases where the judge flips are precisely cases where the verdict is *not* well-founded.

You don't need to formalize these connections in the main paper, but mentioning them in the discussion or future work strengthens the paper's theoretical positioning and signals a coherent research agenda.
