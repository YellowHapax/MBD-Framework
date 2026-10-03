# Machine-Checked Proofs: How a Baseline Settles

**File:** [`BaselineConvergence.lean`](BaselineConvergence.lean)
**Checked by:** the Lean 4 proof assistant, using the Mathlib library (versions pinned in [`lean-toolchain`](lean-toolchain) and [`lakefile.toml`](lakefile.toml))
**Status:** every theorem is fully proven. Nothing is assumed or left as "trust me."

This report explains, without assuming any mathematics, what was proven, why it matters, and what it does *not* claim.

---

## 1. The rule being studied

Every paper in the Memory as Baseline Deviation (MBD) series is built on one update rule:

$$B(t+1) = B(t)\,(1 - \lambda) + I(t)\,\lambda$$

In words:

- **B** is the **baseline**: where a person (or system) currently "rests." Their default mood, expectation, or sense of normal.
- **I** is the **input**: what the world is actually delivering right now.
- **λ** (lambda) is the **update rate**: how much of each new experience gets absorbed. A λ of 0.2 means "move 20% of the way toward what just happened."

Each step, the new baseline is a blend: mostly the old baseline, plus a slice of the present. Psychologists know this rule as the Rescorla–Wagner learning rule. Engineers call it an exponential moving average. MBD treats it as the basic way experience becomes identity.

**Worked example.** Start at a baseline of 0. The world keeps delivering 10. Use λ = 0.2.

| Step | Baseline | Gap to the input |
|-----:|---------:|-----------------:|
| 0 | 0.00 | 10.00 |
| 1 | 2.00 | 8.00 |
| 2 | 3.60 | 6.40 |
| 3 | 4.88 | 5.12 |
| 10 | 8.93 | 1.07 |
| 20 | 9.88 | 0.12 |

The gap shrinks by the same proportion every step: it keeps 80% of itself, because 1 − 0.2 = 0.8. The first theorems make that pattern exact.

---

## 2. Why a machine-checked proof is different from a paper

A normal mathematical paper asks readers to trust that the author's reasoning is correct, and to check it themselves if they doubt it. Most readers can't, and most don't.

A **proof assistant** like Lean is a program that accepts a proof only when every single step follows from the rules of logic and from previously established facts. It does not skim, it does not get tired, and it does not care who wrote the proof. If the file compiles, the theorems are proven. If one step is wrong, it refuses.

That changes what it takes to dismiss the work. A PDF can be dismissed without reading it. A Lean file can only be refuted by running it and watching it fail, and anyone can run it for free.

**What you still have to trust.** You trust three things: Lean's small checking core, the Mathlib library's definitions of things like "real number" and "limit," and that each theorem's *formal statement* says what its English description claims. The third is the one humans must check. That's why every theorem in the file carries a plain-language comment directly above it, and why this report exists.

---

## 3. What was proven

### Result 1: the gap shrinks by a fixed proportion every step
*(Lean names: `error_step`, `abs_error_step`)*

After every update, the distance between the baseline and the input is multiplied by exactly (1 − λ). This is an exact identity, not an approximation, and it holds for any value of λ at all.

**Plain reading:** the update rule doesn't wander or surprise. It closes a fixed share of the remaining distance every time.

### Result 2: a formula for any point in the future
*(Lean name: `closed_form`)*

After *n* steps with a steady input, the remaining gap equals the starting gap multiplied by (1 − λ) a total of *n* times:

$$B(n) - I = (1-\lambda)^n \,\bigl(B(0) - I\bigr)$$

**Plain reading:** the starting point never disappears suddenly, but its influence fades steadily and predictably. With λ = 0.2, the influence of the starting point halves roughly every three steps.

### Result 3: the baseline always arrives
*(Lean names: `tendsto_of_abs_lt_one`, `tendsto_const_input`)*

If the input holds steady, the baseline eventually becomes the input, **no matter where it started**. This is proven for every update rate strictly between 0 and 2, and then stated separately for the usual range, from just above 0 up to 1.

**Plain reading:** under a stable world, where you began has no lasting authority. The theorem guarantees it.

**The overcorrection case.** Update rates between 1 and 2 mean "overshoot": each step jumps *past* the input, then back, then past again, by smaller and smaller amounts. Example: λ = 1.5, start at 0, input 10. The baseline goes 0 → 15 → 7.5 → 11.25 → 9.4 … and still settles at 10. A system that overcorrects every single time still arrives. At exactly λ = 2 it never settles: it bounces between 0 and 20 forever. That boundary is why the theorem stops just short of 2.

### Result 4: a stable world keeps you inside its range
*(Lean name: `stays_in_interval`)*

Suppose every input falls somewhere between a low value *m* and a high value *M*, and the baseline also starts in that range. Then, as long as λ is between 0 and 1, the baseline **never leaves that range**, however the inputs move around inside it.

**Plain reading:** this cuts both ways. A person whose inputs are confined to a narrow band will stay confined to it. No amount of wanting escapes it, because each new baseline is just an average of things inside the band. But the same law protects a range you choose for yourself. A baseline is only ever as safe as the stream of experience feeding it.

**Two conditions matter here:**
- The baseline must *start* inside the range. If it starts outside, this theorem says nothing directly. (What's true, by Result 1, is that it gets pulled toward the range.)
- The update rate must be at most 1. With overcorrection (λ above 1), the baseline *can* leave the range. In the example above, every input is 10 and the baseline starts at 0, yet it reaches 15. Overcorrecting systems arrive, but not without going outside the lines.

### Supporting result
*(Lean name: `Bv_const`)*

A consistency check: the general, time-varying version of the rule gives exactly the same answers as the simple version when the input happens to be constant. This confirms the two definitions in the file describe the same rule.

---

## 4. What this does *not* prove

Precision about the boundary matters as much as the result.

- **It does not prove that minds work this way.** It proves the *rule* behaves as MBD says it does. Whether real people follow the rule is an empirical question, tested by the predictions in the paper series.
- **It covers one dimension.** The baseline here is a single number. MBD describes personality as a list of many numbers (a vector). When each dimension updates independently with the same λ, the results carry over dimension by dimension, but that extension is not yet in the file.
- **λ is fixed.** Many MBD labs let the update rate change over time, for example ossification, where plasticity fades with age. This file does not cover that, and it matters: if λ fades fast enough, the starting point *keeps* a permanent share of influence. (Technically, the start fades away completely only if the update rates add up to infinity over time.)
- **Coupling (κ) and the other MBD terms are not included.** This file is the foundation those terms build on, not the whole framework.

---

## 5. What comes next

The natural next theorem, suggested during independent review: **if the input settles down over time, the baseline settles to the same place**, even if the input wandered on the way. In plain words: if a life converges, the baseline eventually believes it. After that, the candidates are a changing update rate, multiple dimensions, and the coupling term κ, one file at a time.

---

## 6. Check it yourself

You need [Lean 4 via elan](https://lean-lang.org/install/) (free). Then, from this folder:

```bash
lake exe cache get   # downloads the pre-built Mathlib library (a few GB)
lake build           # checks every proof
```

If `lake build` finishes without errors, every theorem above is proven. The same check runs automatically on every change to this folder (see `.github/workflows/lean-proofs.yml`). That check also lists the logical axioms each theorem depends on, and fails if any theorem relies on `sorry`, Lean's "trust me" placeholder.

---

## Credits

Author: Brandon Everett.
Independent review before publication: Kimi-3, and Gimel (Claude).

Part of the [MBD-Framework](../README.md). Cite via [`CITATION.cff`](../CITATION.cff).
