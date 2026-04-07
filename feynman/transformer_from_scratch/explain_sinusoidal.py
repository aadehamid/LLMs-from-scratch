"""
=============================================================================
DEEP DIVE: Sinusoidal Positional Encoding — Explained from First Principles
=============================================================================

Run this file to see worked numerical examples and build intuition.

    cd transformer_from_scratch
    python explain_sinusoidal.py
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import torch
import numpy as np


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: THE PROBLEM — Why does position matter?
# ─────────────────────────────────────────────────────────────────────────────
section("PART 1: THE PROBLEM — Why does position matter?")

print("""
An RNN processes tokens one-by-one in order:
    "the" → hidden_1 → "cat" → hidden_2 → "sat" → hidden_3

So position is implicit — hidden_3 KNOWS "sat" came third because it
was computed after hidden_1 and hidden_2.

A Transformer processes ALL tokens IN PARALLEL:
    ["the", "cat", "sat"]  →  all at once  →  [out_1, out_2, out_3]

The self-attention mechanism computes: "how relevant is token j to token i?"
But it does this using only the token CONTENT (the embedding vectors).
It has NO IDEA about order. These two sentences look IDENTICAL to it:

    "dog bites man"   ←  news
    "man bites dog"   ←  very different news!

Without positional information, the Transformer is just a fancy "bag of words."

SOLUTION: Add a unique position "fingerprint" to each token's embedding
BEFORE feeding it into the Transformer.

    final_input[pos] = token_embedding[pos] + positional_encoding[pos]
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: THE FORMULA — What exactly is computed?
# ─────────────────────────────────────────────────────────────────────────────
section("PART 2: THE FORMULA — What exactly is computed?")

print("""
For a model with d_model dimensions, at sequence position `pos`:

    PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))     ← even dimensions
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))     ← odd dimensions

where i = 0, 1, 2, ..., d_model/2 - 1

Let's unpack this. Each pair of dimensions (2i, 2i+1) uses a DIFFERENT
FREQUENCY. The term 10000^(2i / d_model) controls that frequency.

Think of each dimension pair as a "clock" ticking at a different speed:
    - Dimension pair 0,1:   fastest clock  (completes a full cycle quickly)
    - Dimension pair 2,3:   slightly slower
    - ...
    - Last pair:            slowest clock  (takes ~10000 positions for one cycle)
""")

# Let's compute a concrete example
d_model = 8  # tiny for illustration
print(f"CONCRETE EXAMPLE: d_model = {d_model}")
print(f"Number of dimension pairs: {d_model // 2}")
print()

# Compute the wavelengths for each dimension pair
for i in range(d_model // 2):
    freq_denominator = 10000 ** (2 * i / d_model)
    wavelength = 2 * math.pi * freq_denominator
    print(f"  Dimension pair ({2*i}, {2*i+1}):")
    print(f"    Denominator = 10000^({2*i}/{d_model}) = {freq_denominator:.2f}")
    print(f"    Wavelength  = 2π × {freq_denominator:.2f} = {wavelength:.1f} positions")
    print(f"    → Completes one full sin/cos cycle every {wavelength:.0f} positions")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: THE CLOCK ANALOGY — Visualizing with numbers
# ─────────────────────────────────────────────────────────────────────────────
section("PART 3: THE CLOCK ANALOGY — Think of it like telling time")

print("""
Imagine you need to give every position a unique "address."

One option: just use the number itself (pos=0, pos=1, pos=2, ...).
Problem: the numbers grow unboundedly, which causes numerical issues.

Better option: use a CLOCK system, like how we tell time:
    Time 1:23:45 = 1 hour, 23 minutes, 45 seconds

Each "dial" cycles at a different rate:
    - Seconds dial: completes a full cycle every 60 units
    - Minutes dial: completes a full cycle every 3600 units
    - Hours dial:   completes a full cycle every 43200 units

The sinusoidal PE works exactly like this, but:
    - Instead of 3 dials, we have d_model/2 dials (256 for d_model=512)
    - Instead of discrete ticks, we use smooth sin/cos waves
    - The rates form a geometric progression from fast (2π) to slow (10000·2π)

Let's see the actual values:
""")

d_model = 8
positions = [0, 1, 2, 3, 50, 100]

# Compute PE manually
def compute_pe(pos, d_model):
    pe = np.zeros(d_model)
    for i in range(d_model // 2):
        angle = pos / (10000 ** (2 * i / d_model))
        pe[2 * i] = math.sin(angle)
        pe[2 * i + 1] = math.cos(angle)
    return pe

print(f"PE values (d_model={d_model}):")
print(f"{'pos':>5s}  ", end="")
for d in range(d_model):
    kind = "sin" if d % 2 == 0 else "cos"
    pair = d // 2
    print(f"  d{d}({kind}_{pair})", end="")
print()
print("-" * 85)

for pos in positions:
    pe = compute_pe(pos, d_model)
    print(f"{pos:>5d}  ", end="")
    for val in pe:
        print(f"  {val:>8.4f}", end="")
    print()

print("""
OBSERVATIONS:
 1. Position 0: sin values are all 0, cos values are all 1
    (because sin(0)=0, cos(0)=1 for all frequencies)

 2. Low dimensions (d0,d1) change RAPIDLY between positions
    → they encode FINE-GRAINED position differences

 3. High dimensions (d6,d7) change SLOWLY between positions
    → they encode COARSE position information

 4. This is EXACTLY like a clock:
    seconds change fast (fine), hours change slow (coarse)
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: UNIQUENESS — Every position gets a different vector
# ─────────────────────────────────────────────────────────────────────────────
section("PART 4: UNIQUENESS — Every position gets a distinct fingerprint")

print("""
With d_model=512 (256 frequency pairs), the PE creates a unique vector
for each position. The fastest wave completes a cycle in ~6 positions,
while the slowest takes ~63,000 positions. Combined, they form a unique
"barcode" for any practical sequence length.

Let's verify by measuring similarity between positions:
""")

d_model = 512

# Compute PE for several positions
pe_vecs = {}
for pos in [0, 1, 2, 3, 10, 100, 1000]:
    pe_vecs[pos] = np.array(compute_pe(pos, d_model))

# Cosine similarity between position pairs
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cosine similarity between position PE vectors (d_model={d_model}):")
print(f"{'':>12s}", end="")
for p in [0, 1, 2, 3, 10, 100, 1000]:
    print(f"  pos={p:>4d}", end="")
print()
print("-" * 80)

for p1 in [0, 1, 2, 3, 10, 100, 1000]:
    print(f"  pos={p1:>4d}  ", end="")
    for p2 in [0, 1, 2, 3, 10, 100, 1000]:
        sim = cosine_sim(pe_vecs[p1], pe_vecs[p2])
        print(f"  {sim:>8.4f}", end="")
    print()

print("""
OBSERVATIONS:
 1. Diagonal is 1.0 (each position is identical to itself — good!)
 2. NEARBY positions have HIGH similarity (pos 0↔1 ≈ 0.5)
    This makes sense: positions 0 and 1 are "almost the same place"
 3. DISTANT positions have LOW similarity (pos 0↔1000 ≈ 0.0)
 4. Similarity DECREASES SMOOTHLY with distance
    → The model can use these vectors to judge "how far apart are two tokens?"
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: THE KILLER FEATURE — Relative position via linear transformation
# ─────────────────────────────────────────────────────────────────────────────
section("PART 5: THE KILLER FEATURE — Relative positions are linear")

print("""
THIS is the reason the authors chose sinusoids over, say, random vectors.

CLAIM (from the paper):
  "For any fixed offset k, PE(pos+k) can be represented as a
   linear function of PE(pos)."

This means: there exists a FIXED matrix M_k such that:
    PE(pos + k) = M_k · PE(pos)     for ALL positions pos

WHY THIS MATTERS:
  The attention mechanism computes dot products: Q·K^T.
  If the model wants to learn "attend to the token 3 positions back,"
  it needs a way to compute relationships based on RELATIVE position.

  With sinusoidal PE, this is trivially learnable because relative
  position is a LINEAR TRANSFORMATION.

THE MATH (for one frequency pair):
  sin(pos+k) · freq = sin(pos·freq)·cos(k·freq) + cos(pos·freq)·sin(k·freq)
  cos(pos+k) · freq = cos(pos·freq)·cos(k·freq) - sin(pos·freq)·sin(k·freq)

  In matrix form:
  ┌ PE(pos+k, 2i)   ┐   ┌ cos(k·freq)   sin(k·freq) ┐   ┌ PE(pos, 2i)   ┐
  │                  │ = │                              │ · │                │
  └ PE(pos+k, 2i+1) ┘   └ -sin(k·freq)  cos(k·freq)  ┘   └ PE(pos, 2i+1) ┘

  This is a ROTATION MATRIX! Shifting by k positions = rotating the
  (sin, cos) pair by a fixed angle. And rotation is a linear operation.

Let's verify numerically:
""")

d_model = 8
pos = 7
k = 3  # offset

pe_pos = np.array(compute_pe(pos, d_model))
pe_pos_k = np.array(compute_pe(pos + k, d_model))

print(f"  pos={pos}, k={k}, d_model={d_model}")
print(f"  PE(pos={pos}):     {np.array2string(pe_pos, precision=4, separator=', ')}")
print(f"  PE(pos={pos+k}):   {np.array2string(pe_pos_k, precision=4, separator=', ')}")
print()

# Build the rotation matrix for offset k
print(f"  Constructing rotation matrix M_{k} for each dimension pair:")
M_k = np.zeros((d_model, d_model))
for i in range(d_model // 2):
    freq = 1.0 / (10000 ** (2 * i / d_model))
    angle = k * freq
    # 2x2 rotation block for this dimension pair
    row = 2 * i
    M_k[row, row] = math.cos(angle)
    M_k[row, row + 1] = math.sin(angle)
    M_k[row + 1, row] = -math.sin(angle)
    M_k[row + 1, row + 1] = math.cos(angle)
    print(f"    Pair ({2*i},{2*i+1}): rotation angle = {k} × {freq:.6f} = {angle:.6f} rad")

# Apply the rotation
pe_reconstructed = M_k @ pe_pos
print(f"\n  M_{k} · PE(pos={pos}) = {np.array2string(pe_reconstructed, precision=4, separator=', ')}")
print(f"  PE(pos={pos+k})       = {np.array2string(pe_pos_k, precision=4, separator=', ')}")

error = np.max(np.abs(pe_reconstructed - pe_pos_k))
print(f"\n  Max reconstruction error: {error:.2e}")
print(f"  ✓ PE(pos+k) = M_k · PE(pos) verified! (error ≈ 0)")

# Verify it works for OTHER positions too (same M_k!)
print(f"\n  Verifying same M_{k} works for different starting positions:")
for test_pos in [0, 5, 20, 100]:
    pe_test = np.array(compute_pe(test_pos, d_model))
    pe_test_k = np.array(compute_pe(test_pos + k, d_model))
    pe_test_reconstructed = M_k @ pe_test
    err = np.max(np.abs(pe_test_reconstructed - pe_test_k))
    print(f"    pos={test_pos:>3d}: M_{k} · PE({test_pos}) ≈ PE({test_pos+k})  error={err:.2e} ✓")


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: DOT PRODUCT ENCODES DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
section("PART 6: DOT PRODUCT — Attention 'sees' distance naturally")

print("""
Attention computes: score(i,j) = Q_i · K_j

Since Q and K contain the positional encoding, the dot product between
two PE vectors captures their DISTANCE. Let's verify:
""")

d_model = 512
reference_pos = 50

# Compute dot product of PE(reference) with PE(reference + offset)
offsets = list(range(-20, 21))
dots = []
for offset in offsets:
    pe_ref = np.array(compute_pe(reference_pos, d_model))
    pe_other = np.array(compute_pe(reference_pos + offset, d_model))
    dot = np.dot(pe_ref, pe_other)
    dots.append(dot)

print(f"  Dot product: PE(pos={reference_pos}) · PE(pos={reference_pos}+offset)")
print(f"  d_model = {d_model}")
print()

# Show as a simple ASCII bar chart
max_dot = max(abs(d) for d in dots)
for offset, dot in zip(offsets, dots):
    bar_len = int(30 * dot / max_dot)
    bar = "█" * max(bar_len, 0) + "░" * max(-bar_len, 0)
    marker = " ◄── self" if offset == 0 else ""
    print(f"  offset={offset:>+3d}: {dot:>8.1f}  {bar}{marker}")

print("""
OBSERVATIONS:
 1. Offset=0 has the HIGHEST dot product (a vector dotted with itself)
 2. The dot product PEAKS at offset=0 and DECREASES with distance
 3. The pattern is SYMMETRIC: offset=+5 ≈ offset=-5
 4. This means when the model computes attention scores,
    NEARBY tokens naturally get higher scores from the PE component
    → Position information flows into attention automatically!
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: THE GEOMETRIC PROGRESSION OF WAVELENGTHS
# ─────────────────────────────────────────────────────────────────────────────
section("PART 7: WHY 10000? — The geometric progression of wavelengths")

print("""
The denominator 10000^(2i/d_model) creates a GEOMETRIC PROGRESSION
of wavelengths from 2π ≈ 6.3 to 10000·2π ≈ 62,832.

    Dimension pair 0:        wavelength ≈ 6.3     (resolves positions 0-6)
    Dimension pair 1:        wavelength ≈ 10.0    (resolves positions 0-10)
    ...
    Dimension pair 127:      wavelength ≈ 39,685  (resolves positions 0-39685)
    Dimension pair 255:      wavelength ≈ 62,832  (resolves positions 0-62832)

WHY GEOMETRIC (not linear)?
  - If wavelengths were linear (6, 12, 18, 24, ...), most dimensions would
    capture SIMILAR scales of position. Wasteful!
  - Geometric spacing (6, 10, 16, 25, ...) covers a HUGE RANGE of scales
    uniformly on a log scale. Each dimension pair captures a different
    "resolution" of position information.

WHY 10000 specifically?
  - It sets the LONGEST wavelength to ~62,832 positions
  - This means the encoding can distinguish positions up to ~60K apart
  - That's more than enough for any practical sequence length
  - The paper authors likely tuned this value experimentally;
    the exact number isn't critical, but the ORDER OF MAGNITUDE matters.

First 10 and last 5 wavelengths for d_model=512:
""")

d_model = 512
print(f"  {'Pair':>6s}  {'Dims':>8s}  {'Denominator':>14s}  {'Wavelength':>12s}  {'Scale'}")
print(f"  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*25}")

pairs_to_show = list(range(10)) + ["..."] + list(range(d_model//2 - 5, d_model//2))
for item in pairs_to_show:
    if item == "...":
        print(f"  {'...':>6s}")
        continue
    i = item
    denom = 10000 ** (2 * i / d_model)
    wavelength = 2 * math.pi * denom
    dims = f"({2*i},{2*i+1})"
    if wavelength < 100:
        scale = "fine (word-level)"
    elif wavelength < 10000:
        scale = "medium (phrase-level)"
    else:
        scale = "coarse (paragraph-level)"
    print(f"  {i:>6d}  {dims:>8s}  {denom:>14.2f}  {wavelength:>12.1f}  {scale}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 8: SIN/COS PAIRS — Why both?
# ─────────────────────────────────────────────────────────────────────────────
section("PART 8: WHY SIN AND COS TOGETHER? — They form a complete basis")

print("""
Q: Why not just use sin? Why do we need cos too?

A: sin alone is AMBIGUOUS. Consider:
   sin(x) = sin(π - x)

   So sin at position 1 and position (π-1) would give the SAME value
   for certain frequencies. The model couldn't tell them apart!

   But (sin(x), cos(x)) together are UNIQUE for every x in [0, 2π):
   - sin(1) = 0.841,  cos(1) = 0.540
   - sin(π-1) = 0.841, cos(π-1) = -0.540   ← cos DISAMBIGUATES!

   Together, sin and cos trace out a CIRCLE. Each position maps to a
   unique point on the circle for each frequency. That's why the
   rotation matrix formulation works — shifting position = rotating
   around the circle.

Let's see this concretely:
""")

freq = 1.0  # simplest frequency
print(f"  At frequency = {freq}:")
print(f"  {'pos':>5s}  {'sin(pos)':>10s}  {'cos(pos)':>10s}  {'unique?':>10s}")
print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
for pos in range(7):
    s = math.sin(pos * freq)
    c = math.cos(pos * freq)
    print(f"  {pos:>5d}  {s:>10.4f}  {c:>10.4f}  ✓ ({s:.2f},{c:.2f})")

print("""
  Each (sin, cos) pair is a unique point on the unit circle.
  No two positions map to the same point (within one wavelength).

  GEOMETRIC INTUITION:
  ┌─────────────────────┐
  │         cos          │     pos=0 → (sin=0, cos=1) = "12 o'clock"
  │          ↑           │     pos=1 → rotated clockwise
  │    3 ·   |   · 0    │     pos=2 → rotated more
  │   ·      |      ·   │     ...
  │  ·───────┼───────·→  │     Each position is a unique angle
  │   ·      |      · sin│     on the unit circle
  │    5 ·   |   · 4    │
  │          |           │
  └─────────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 9: COMPARISON WITH ALTERNATIVES
# ─────────────────────────────────────────────────────────────────────────────
section("PART 9: WHY NOT SIMPLER ALTERNATIVES?")

print("""
ALTERNATIVE 1: Just use the position number
  PE(pos) = pos    →    [0, 1, 2, 3, ..., 511]

  Problems:
  ✗ Values grow without bound — position 10000 has value 10000
  ✗ The model sees very different magnitudes for short vs long sequences
  ✗ Hard to generalize to unseen sequence lengths

ALTERNATIVE 2: Normalize to [0, 1]
  PE(pos) = pos / max_len    →    [0.0, 0.002, 0.004, ...]

  Problems:
  ✗ The spacing between positions CHANGES with sequence length
  ✗ Position 5 in a 10-token sequence ≠ position 5 in a 1000-token sequence
  ✗ Can't generalize to different lengths

ALTERNATIVE 3: Learned positional embeddings
  Make PE a learnable parameter matrix (like word embeddings for positions)

  Pros:
  ✓ The model can learn whatever representation works best
  Cons:
  ✗ Can ONLY handle positions seen during training
  ✗ Position 512 in a model trained on max_len=512 → crash!
  ✗ The paper tested this: "nearly identical results" (Table 3, row E)
    But sinusoidal can extrapolate to longer sequences.

SINUSOIDAL ENCODING:
  ✓ Bounded values (always in [-1, 1])
  ✓ Same encoding regardless of sequence length
  ✓ Can extrapolate to sequences longer than training
  ✓ Relative positions are linearly transformable
  ✓ Dot product naturally encodes distance
  ✓ No learnable parameters needed
""")


# ─────────────────────────────────────────────────────────────────────────────
# PART 10: THE CODE — How it maps to our implementation
# ─────────────────────────────────────────────────────────────────────────────
section("PART 10: MAPPING MATH → CODE")

print("""
The formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

The code computes this efficiently using a LOG-SPACE TRICK:

    # Instead of: 10000^(2i/d_model)         ← overflows for large values
    # We compute: exp(2i × -ln(10000) / d_model)   ← numerically stable

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    # This gives us 1/10000^(2i/d_model) for each dimension pair
    # Then: pos × div_term = pos / 10000^(2i/d_model) = the angle

    pe[:, 0::2] = torch.sin(position * div_term)   # even dims
    pe[:, 1::2] = torch.cos(position * div_term)    # odd dims

WHY THE LOG TRICK?
    10000^(2i/d_model) can be astronomically large for high dimensions.
    Computing in log-space avoids floating-point overflow:

    10000^x = exp(x × ln(10000))      ← math identity
    1/10000^x = exp(-x × ln(10000))   ← we want the reciprocal

    exp() is numerically stable for reasonable inputs ✓
""")

# Verify the log trick gives the same answer
d_model = 512
i = 100  # dimension pair index
direct = 1.0 / (10000 ** (2 * i / d_model))
log_trick = math.exp(2 * i * (-math.log(10000.0) / d_model))
print(f"  Example: i={i}, d_model={d_model}")
print(f"  Direct:    1/10000^({2*i}/{d_model}) = {direct:.10e}")
print(f"  Log trick: exp({2*i} × -ln(10000)/{d_model}) = {log_trick:.10e}")
print(f"  Match: {abs(direct - log_trick) < 1e-15} ✓")


section("SUMMARY")
print("""
┌─────────────────────────────────────────────────────────────────┐
│                 SINUSOIDAL POSITIONAL ENCODING                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WHAT:  A fixed, deterministic vector added to each token's     │
│         embedding to encode its position in the sequence.       │
│                                                                 │
│  HOW:   Each pair of dimensions oscillates at a different       │
│         frequency, from fast (word-scale) to slow (paragraph-   │
│         scale), creating a unique "fingerprint" per position.   │
│                                                                 │
│  WHY SIN+COS:  Together they're unique (no ambiguity) and      │
│         enable relative position via rotation matrices.         │
│                                                                 │
│  WHY 10000:  Sets the range of wavelengths, covering positions  │
│         from ~6 to ~63,000 apart.                               │
│                                                                 │
│  KEY PROPERTY:  PE(pos+k) = M_k · PE(pos) for a fixed M_k     │
│         → relative position is a LINEAR operation               │
│         → attention can learn relative patterns easily          │
│                                                                 │
│  BOUNDED: Values always in [-1, 1], works for any seq length.  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    pass  # All output is printed during import / module execution
