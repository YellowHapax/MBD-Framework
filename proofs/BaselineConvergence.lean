/-
  BaselineConvergence.lean

  Machine-checked facts about the exponential-smoothing baseline update

      B(t+1) = B(t) * (1 - λ) + I(t) * λ

  where B is the baseline, I is the input signal and λ is the update rate.

  Notation note: in Lean the symbol `λ` is a reserved keyword (it means
  "function"), so the update rate is written `lam` throughout this file.

  Reading guide for non-Lean readers:
  * `def` introduces a definition, `theorem` / `lemma` a statement plus proof.
  * Everything after `:= by` is the proof; Lean checks every step.
  * `ℕ` = natural numbers 0, 1, 2, ...;  `ℝ` = real numbers.
  * `|x|` is absolute value, `x ^ n` is x to the power n.
  * If this file compiles, every theorem below is proven: there is no
    `sorry` (Lean's "trust me" placeholder) anywhere.
-/
import Mathlib

open Filter Topology

namespace BaselineConvergence

/-! ## Part 1: constant input -/

/-- The baseline sequence when the input is a constant `I`.
    `B lam I b0 n` is the baseline after `n` update steps, starting from `b0`. -/
def B (lam I b0 : ℝ) : ℕ → ℝ
  | 0 => b0
  | n + 1 => B lam I b0 n * (1 - lam) + I * lam

/-- At step 0 the baseline is the starting value. -/
@[simp] lemma B_zero (lam I b0 : ℝ) : B lam I b0 0 = b0 := rfl

/-- One step of the update rule. -/
@[simp] lemma B_succ (lam I b0 : ℝ) (n : ℕ) :
    B lam I b0 (n + 1) = B lam I b0 n * (1 - lam) + I * lam := rfl

/-- **Error contraction (exact form).**
    The gap between the baseline and the input is multiplied by exactly
    `(1 - lam)` at every step:  B(n+1) - I = (1 - λ) * (B(n) - I).
    No assumption on λ is needed for this identity. -/
theorem error_step (lam I b0 : ℝ) (n : ℕ) :
    B lam I b0 (n + 1) - I = (1 - lam) * (B lam I b0 n - I) := by
  rw [B_succ]
  ring

/-- **Error contraction (absolute value form).**
    |B(n+1) - I| = |1 - λ| * |B(n) - I|. -/
theorem abs_error_step (lam I b0 : ℝ) (n : ℕ) :
    |B lam I b0 (n + 1) - I| = |1 - lam| * |B lam I b0 n - I| := by
  rw [error_step, abs_mul]

/-- **Theorem 1 (closed form).**
    After `n` steps the gap to the input is the initial gap times `(1 - λ)^n`:
        B(n) - I = (1 - λ)^n * (b0 - I).
    Proof: induction on `n`, using `error_step` at each step.
    This identity holds for every real λ; the restriction 0 < λ ≤ 1 is only
    needed later, for convergence. -/
theorem closed_form (lam I b0 : ℝ) (n : ℕ) :
    B lam I b0 n - I = (1 - lam) ^ n * (b0 - I) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [error_step, ih, pow_succ]
    ring

/-- **Theorem 2 (convergence, general form).**
    If |1 - λ| < 1 (equivalently 0 < λ < 2), the baseline converges to the
    input `I` as the number of steps goes to infinity, whatever `b0` is.
    `Tendsto f atTop (𝓝 I)` is Lean's way of writing  lim_{n→∞} f(n) = I.
    Proof: (1 - λ)^n → 0 (a standard Mathlib fact for |r| < 1), so by the
    closed form B(n) = (1 - λ)^n * (b0 - I) + I → 0 * (b0 - I) + I = I. -/
theorem tendsto_of_abs_lt_one (lam I b0 : ℝ) (h : |1 - lam| < 1) :
    Tendsto (B lam I b0) atTop (𝓝 I) := by
  -- (1 - λ)^n → 0
  have hpow : Tendsto (fun n : ℕ => (1 - lam) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_abs_lt_one h
  -- hence (1 - λ)^n * (b0 - I) + I → 0 * (b0 - I) + I
  have hlim : Tendsto (fun n : ℕ => (1 - lam) ^ n * (b0 - I) + I) atTop
      (𝓝 (0 * (b0 - I) + I)) :=
    (hpow.mul_const _).add_const _
  rw [zero_mul, zero_add] at hlim
  -- and that expression is exactly B(n), by the closed form
  refine hlim.congr (fun n => ?_)
  have := closed_form lam I b0 n
  linarith

/-- **Theorem 2 (convergence, for the usual parameter range 0 < λ ≤ 1).** -/
theorem tendsto_const_input (lam I b0 : ℝ) (h0 : 0 < lam) (h1 : lam ≤ 1) :
    Tendsto (B lam I b0) atTop (𝓝 I) := by
  apply tendsto_of_abs_lt_one
  rw [abs_lt]
  constructor <;> linarith

/-! ## Part 2: time-varying, bounded input -/

/-- The baseline sequence for a time-varying input `I : ℕ → ℝ`
    (I t is the input at time t):
        B(0) = b0,   B(t+1) = B(t) * (1 - λ) + I(t) * λ. -/
def Bv (lam : ℝ) (I : ℕ → ℝ) (b0 : ℝ) : ℕ → ℝ
  | 0 => b0
  | n + 1 => Bv lam I b0 n * (1 - lam) + I n * lam

/-- Sanity check: with a constant input, the time-varying sequence is the
    same as the constant-input sequence from Part 1. -/
theorem Bv_const (lam I b0 : ℝ) : Bv lam (fun _ => I) b0 = B lam I b0 := by
  funext n
  induction n with
  | zero => rfl
  | succ n ih => simp only [Bv, B, ih]

/-- **Theorem 3 (invariant interval).**
    Suppose 0 ≤ λ ≤ 1, every input value I(t) lies in the interval [m, M],
    and the starting baseline b0 also lies in [m, M]. Then the baseline B(t)
    stays in [m, M] forever.
    Reason: each new baseline is a weighted average of the old baseline and
    the current input, with non-negative weights (1 - λ) and λ that add to 1,
    and an average of two numbers in [m, M] is again in [m, M].
    Proof: induction on t. -/
theorem stays_in_interval (lam m M b0 : ℝ) (I : ℕ → ℝ)
    (hlam0 : 0 ≤ lam) (hlam1 : lam ≤ 1)
    (hI : ∀ t, I t ∈ Set.Icc m M) (hb : b0 ∈ Set.Icc m M) :
    ∀ t, Bv lam I b0 t ∈ Set.Icc m M := by
  intro t
  induction t with
  | zero => exact hb
  | succ n ih =>
    obtain ⟨hBlo, hBhi⟩ := ih          -- m ≤ B(n) ≤ M
    obtain ⟨hIlo, hIhi⟩ := hI n        -- m ≤ I(n) ≤ M
    have hw : 0 ≤ 1 - lam := by linarith
    simp only [Bv, Set.mem_Icc]
    constructor
    · -- lower bound: B(n)(1-λ) + I(n)λ ≥ m(1-λ) + mλ = m
      nlinarith [mul_le_mul_of_nonneg_right hBlo hw,
                 mul_le_mul_of_nonneg_right hIlo hlam0]
    · -- upper bound: B(n)(1-λ) + I(n)λ ≤ M(1-λ) + Mλ = M
      nlinarith [mul_le_mul_of_nonneg_right hBhi hw,
                 mul_le_mul_of_nonneg_right hIhi hlam0]

end BaselineConvergence
