/-
  BaselineTracking.lean

  Further machine-checked facts about the time-varying baseline sequence
  `Bv` defined in BaselineConvergence.lean:

      Bv(0)   = b0
      Bv(n+1) = Bv(n) * (1 - λ) + I(n) * λ

  Statements proposed by Gimel (Claude Opus 5), following Kimi-3's suggested
  tracking sequel; formalized and machine-checked in Lean 4 + Mathlib.

  As in BaselineConvergence.lean, the update rate λ is written `lam`
  (`λ` is a reserved word in Lean).

  Contents
  * Theorem 4  (convolution form)   `Bv_convolution`
  * Theorem 5  (partition of unity) `weights_sum_one`, `weights_nonneg`,
                                    `weights_nonneg_iff`, `weight_neg_of_one_lt`
  * Theorem 6  (tracking bound)     `tracking_bound`, `eventually_tracking`
  * Theorem 7  (mean lag)           `lag_weights_hasSum_one`, `mean_lag`

  Portability note (Mathlib renames): `eventually_tracking` uses the Mathlib
  lemma `tendsto_pow_atTop_nhds_zero_of_abs_lt_one` ("r^n → 0 when |r| < 1").
  In older Mathlib versions this lemma was called
  `tendsto_pow_atTop_nhds_0_of_abs_lt_1`. If the build fails at that line on a
  different Mathlib version, it is a lemma rename, not a problem with the math.
-/
import BaselineConvergence

open Filter Topology Finset

namespace BaselineConvergence

/-! ## Theorem 4: convolution form -/

/-- **Theorem 4 (convolution form).**
    The baseline after `n` steps is an explicit weighted sum of the starting
    value and all past inputs:

      Bv(n) = (1-λ)^n * b0 + λ * Σ_{k=0}^{n-1} (1-λ)^(n-1-k) * I(k).

    The input from `k` steps in the past gets weight λ(1-λ)^(n-1-k), so older
    inputs are discounted geometrically.
    (In Lean, `n - 1 - k` is natural-number subtraction, which cuts off at 0;
    since the sum only runs over k < n, it is always the ordinary
    difference here, so the statement is exactly the formula above.)
    Proof: induction on `n`. -/
theorem Bv_convolution (lam : ℝ) (I : ℕ → ℝ) (b0 : ℝ) (n : ℕ) :
    Bv lam I b0 n
      = (1 - lam) ^ n * b0 + lam * ∑ k ∈ range n, (1 - lam) ^ (n - 1 - k) * I k := by
  induction n with
  | zero => simp [Bv]
  | succ n ih =>
    -- multiplying the old weights by (1-λ) raises each exponent by one
    have hshift : (∑ k ∈ range n, (1 - lam) ^ (n - 1 - k) * I k) * (1 - lam)
        = ∑ k ∈ range n, (1 - lam) ^ (n + 1 - 1 - k) * I k := by
      rw [sum_mul]
      refine sum_congr rfl (fun k hk => ?_)
      have hk' : k < n := mem_range.mp hk
      have e : n + 1 - 1 - k = (n - 1 - k) + 1 := by omega
      rw [e, pow_succ]
      ring
    rw [Bv, ih, sum_range_succ, ← hshift, show n + 1 - 1 - n = 0 by omega, pow_zero, pow_succ]
    ring

/-! ## Theorem 5: the weights form a partition of unity -/

/-- **Theorem 5 (partition of unity).**
    The weights in the convolution form always add up to 1:

      (1-λ)^n + λ * Σ_{k<n} (1-λ)^(n-1-k) = 1      (for every real λ and every n).

    Proof: feed the constant input 1 with starting value 1 into the update.
    By the closed form from BaselineConvergence.lean the baseline then stays
    exactly 1, and by Theorem 4 it equals the sum of the weights. -/
theorem weights_sum_one (lam : ℝ) (n : ℕ) :
    (1 - lam) ^ n + lam * ∑ k ∈ range n, (1 - lam) ^ (n - 1 - k) = 1 := by
  have hconv := Bv_convolution lam (fun _ => 1) 1 n
  have hclosed := closed_form lam 1 1 n
  rw [Bv_const] at hconv
  simp only [mul_one, sub_self, mul_zero] at hconv hclosed
  linarith

/-- **Corollary (weights are non-negative when 0 ≤ λ ≤ 1).**
    Together with `weights_sum_one`, this says that for 0 ≤ λ ≤ 1 the baseline
    is a genuine weighted *average* of b0 and the past inputs. -/
theorem weights_nonneg (lam : ℝ) (h0 : 0 ≤ lam) (h1 : lam ≤ 1) (n : ℕ) :
    0 ≤ (1 - lam) ^ n ∧ ∀ k ∈ range n, 0 ≤ lam * (1 - lam) ^ (n - 1 - k) := by
  have hw : 0 ≤ 1 - lam := by linarith
  exact ⟨pow_nonneg hw n, fun k _ => mul_nonneg h0 (pow_nonneg hw _)⟩

/-- **Averaging vs. extrapolating (λ > 1).**
    If λ > 1 and n ≥ 2, the weight on the input two steps back, I(n-2), is
    λ(1-λ), which is negative. So with λ > 1 the update is no longer an
    average: it extrapolates (overshoots). -/
theorem weight_neg_of_one_lt (lam : ℝ) (hlam : 1 < lam) (n : ℕ) (hn : 2 ≤ n) :
    n - 2 ∈ range n ∧ lam * (1 - lam) ^ (n - 1 - (n - 2)) < 0 := by
  refine ⟨mem_range.mpr (by omega), ?_⟩
  rw [show n - 1 - (n - 2) = 1 by omega, pow_one]
  exact mul_neg_of_pos_of_neg (by linarith) (by linarith)

/-- **Theorem 5, full characterization.**
    For n ≥ 2: all the weights are non-negative **if and only if** 0 ≤ λ ≤ 1.
    (If λ < 0, the weight on the most recent input, λ itself, is negative;
    if λ > 1, the weight λ(1-λ) on the input two steps back is negative.) -/
theorem weights_nonneg_iff (lam : ℝ) (n : ℕ) (hn : 2 ≤ n) :
    (0 ≤ (1 - lam) ^ n ∧ ∀ k ∈ range n, 0 ≤ lam * (1 - lam) ^ (n - 1 - k))
      ↔ (0 ≤ lam ∧ lam ≤ 1) := by
  constructor
  · rintro ⟨-, hk⟩
    constructor
    · -- the weight on the most recent input I(n-1) is exactly λ
      have h := hk (n - 1) (mem_range.mpr (by omega))
      rwa [show n - 1 - (n - 1) = 0 by omega, pow_zero, mul_one] at h
    · -- if λ > 1, the weight on I(n-2) would be negative
      by_contra hgt
      obtain ⟨hmem, hneg⟩ := weight_neg_of_one_lt lam (not_le.mp hgt) n hn
      exact absurd (hk (n - 2) hmem) (not_le.mpr hneg)
  · rintro ⟨h0, h1⟩
    exact weights_nonneg lam h0 h1 n

/-! ## Theorem 6: tracking a noisy input -/

/-- **Theorem 6 (tracking bound).**
    Suppose 0 ≤ λ ≤ 1 and every input stays within ε of some level c:
    |I(k) - c| ≤ ε for all k. Then

      |Bv(n) - c| ≤ (1-λ)^n * |b0 - c| + (1 - (1-λ)^n) * ε.

    In words: the influence of the starting error |b0 - c| decays like (1-λ)^n,
    and the rest of the error is at most the input's own spread ε.
    Proof: induction, using |a + b| ≤ |a| + |b| (triangle inequality). -/
theorem tracking_bound (lam c ε b0 : ℝ) (I : ℕ → ℝ)
    (h0 : 0 ≤ lam) (h1 : lam ≤ 1) (hI : ∀ k, |I k - c| ≤ ε) (n : ℕ) :
    |Bv lam I b0 n - c| ≤ (1 - lam) ^ n * |b0 - c| + (1 - (1 - lam) ^ n) * ε := by
  have hw : 0 ≤ 1 - lam := by linarith
  induction n with
  | zero => simp [Bv]
  | succ n ih =>
    -- one step: new error = (1-λ) * old error + λ * (input error)
    have e : Bv lam I b0 (n + 1) - c
        = (1 - lam) * (Bv lam I b0 n - c) + lam * (I n - c) := by
      simp only [Bv]; ring
    rw [e]
    calc |(1 - lam) * (Bv lam I b0 n - c) + lam * (I n - c)|
        ≤ |(1 - lam) * (Bv lam I b0 n - c)| + |lam * (I n - c)| := abs_add_le _ _
      _ = (1 - lam) * |Bv lam I b0 n - c| + lam * |I n - c| := by
          rw [abs_mul, abs_mul, abs_of_nonneg hw, abs_of_nonneg h0]
      _ ≤ (1 - lam) * ((1 - lam) ^ n * |b0 - c| + (1 - (1 - lam) ^ n) * ε)
            + lam * ε :=
          add_le_add (mul_le_mul_of_nonneg_left ih hw)
                     (mul_le_mul_of_nonneg_left (hI n) h0)
      _ = (1 - lam) ^ (n + 1) * |b0 - c| + (1 - (1 - lam) ^ (n + 1)) * ε := by
          ring

/-- **Theorem 6, asymptotic form.**
    If 0 < λ ≤ 1 and every input is within ε of c, then for any extra margin
    δ > 0, from some time on the baseline is within ε + δ of c.
    ("Eventually" = for all sufficiently large n.)
    Proof: the bound above is at most (1-λ)^n * |b0 - c| + ε, and the first
    term tends to 0 because |1-λ| < 1. -/
theorem eventually_tracking (lam c ε b0 : ℝ) (I : ℕ → ℝ)
    (h0 : 0 < lam) (h1 : lam ≤ 1) (hI : ∀ k, |I k - c| ≤ ε)
    (δ : ℝ) (hδ : 0 < δ) :
    ∀ᶠ n in atTop, |Bv lam I b0 n - c| ≤ ε + δ := by
  have hw : 0 ≤ 1 - lam := by linarith
  have hε : 0 ≤ ε := le_trans (abs_nonneg _) (hI 0)
  have habs : |1 - lam| < 1 := by rw [abs_lt]; constructor <;> linarith
  -- (1-λ)^n → 0.  NOTE: older Mathlib name: `tendsto_pow_atTop_nhds_0_of_abs_lt_1`;
  -- a build failure on the next line is a lemma rename, not the math.
  have hpow : Tendsto (fun n : ℕ => (1 - lam) ^ n) atTop (𝓝 0) :=
    tendsto_pow_atTop_nhds_zero_of_abs_lt_one habs
  -- so (1-λ)^n * |b0 - c| → 0, hence is eventually below δ
  have htail : Tendsto (fun n : ℕ => (1 - lam) ^ n * |b0 - c|) atTop (𝓝 0) := by
    simpa using hpow.mul_const |b0 - c|
  filter_upwards [htail.eventually_lt_const hδ] with n hn
  have hb := tracking_bound lam c ε b0 I h0.le h1 hI n
  have hp : 0 ≤ (1 - lam) ^ n := pow_nonneg hw n
  nlinarith

/-! ## Theorem 7: mean lag -/

/-- **Lag weights sum to 1 (infinite version).**
    Looking back from the present, the input from `k` steps ago gets weight
    λ(1-λ)^k. For 0 < λ ≤ 1 these infinitely many weights add up to exactly 1.
    (`HasSum f s` means the series f(0) + f(1) + ... converges to s.) -/
theorem lag_weights_hasSum_one (lam : ℝ) (h0 : 0 < lam) (h1 : lam ≤ 1) :
    HasSum (fun k : ℕ => lam * (1 - lam) ^ k) 1 := by
  have h := (hasSum_geometric_of_lt_one (r := 1 - lam) (by linarith) (by linarith)).mul_left lam
  rwa [show 1 - (1 - lam) = lam by ring, mul_inv_cancel₀ h0.ne'] at h

/-- **Theorem 7 (mean lag).**
    For 0 < λ ≤ 1, the average age of the information in the baseline,
    i.e. Σ_k k * λ(1-λ)^k, equals (1-λ)/λ:

      Σ_{k=0}^∞ k * λ * (1-λ)^k = (1-λ)/λ.

    Example: λ = 0.1 gives a mean lag of 9 steps. The case λ = 1 is included:
    then the baseline just copies the latest input, and the sum is 0.
    Proof: Mathlib's `hasSum_coe_mul_geometric_of_norm_lt_one`
    (Σ k r^k = r/(1-r)^2 for |r| < 1) with r = 1-λ, multiplied by λ. -/
theorem mean_lag (lam : ℝ) (h0 : 0 < lam) (h1 : lam ≤ 1) :
    HasSum (fun k : ℕ => (k : ℝ) * lam * (1 - lam) ^ k) ((1 - lam) / lam) := by
  have hr : ‖1 - lam‖ < 1 := by
    rw [Real.norm_eq_abs, abs_lt]; constructor <;> linarith
  have h := (hasSum_coe_mul_geometric_of_norm_lt_one hr).mul_left lam
  rw [show 1 - (1 - lam) = lam by ring] at h
  convert h using 1
  · funext k; ring
  · field_simp

end BaselineConvergence
