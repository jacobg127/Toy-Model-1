# --- Reversible micro-dynamics -> arrow under coarse-graining (Permutation model) ---
# Figure-ready, with normalization, forward+rewind demo, and multi-scale comparison.
# Requirements: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ------------------------- Core functions -------------------------

def macro_entropy_from_blocks(bits: np.ndarray, B: int) -> float:
    """
    Coarse-grain into BxB blocks, count ones per block, and compute
    the Shannon entropy of the histogram over block-count values.
    Normalized by log(B^2 + 1) to lie in [0, 1].
    """
    H, W = bits.shape
    assert H % B == 0 and W % B == 0, "H and W must be multiples of B"
    counts = []
    for i in range(0, H, B):
        for j in range(0, W, B):
            counts.append(int(bits[i:i+B, j:j+B].sum()))
    hist = Counter(counts)  # over 0..B^2
    p = np.array(list(hist.values()), dtype=float)
    p /= p.sum()
    Hs = -np.sum(p * np.log(p + 1e-12))
    return Hs / np.log(B*B + 1)  # normalized to [0,1]

def step_perm(bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply a fixed bijection on sites (reversible). The same permutation each step.
    """
    flat = bits.ravel()
    flat = flat[perm]  # bijective map => information-preserving
    return flat.reshape(bits.shape)

def inverse_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv

# ------------------------- Simulation helpers -------------------------

def init_hotspot(H: int, W: int, side: int = 4) -> np.ndarray:
    """
    Low macro-entropy initial condition: small 'hot spot' of ones in a sea of zeros.
    """
    bits = np.zeros((H, W), dtype=np.uint8)
    hs, ws = H//2, W//2
    r = side//2
    bits[hs-r:hs+r, ws-r:ws+r] = 1
    return bits

def run_forward(bits0: np.ndarray, perm: np.ndarray, steps: int, B: int):
    """
    Run forward dynamics, returning (S_list, states_list).
    """
    bits = bits0.copy()
    S = []
    states = []
    for t in range(steps):
        S.append(macro_entropy_from_blocks(bits, B))
        states.append(bits.copy())
        bits = step_perm(bits, perm)
    return np.array(S), states, bits

def run_rewind(bits_end: np.ndarray, inv_perm: np.ndarray, steps: int, B: int):
    """
    Rewind using the exact inverse permutation; returns S_back list.
    """
    bits = bits_end.copy()
    S_back = []
    for t in range(steps):
        S_back.append(macro_entropy_from_blocks(bits, B))
        bits = step_perm(bits, inv_perm)
    return np.array(S_back)

def averaged_entropy_over_perms(H: int, W: int, B: int, steps: int, trials: int, seed: int = 0):
    """
    For robustness, average S(t) over several independent random permutations.
    """
    rng = np.random.default_rng(seed)
    S_accum = np.zeros(steps, dtype=float)
    for _ in range(trials):
        bits0 = init_hotspot(H, W, side=4)
        perm = np.arange(H*W)
        rng.shuffle(perm)
        S, _, _ = run_forward(bits0, perm, steps, B)
        S_accum += S
    return S_accum / trials

# ------------------------- Parameters -------------------------

H, W = 64, 64        # grid size
steps = 200          # forward steps
B_list = [2, 4, 8]   # coarse-grain block sizes (must divide H and W)
avg_trials = 5       # set to 1 to skip averaging; >1 for robustness

# ------------------------- Figure 1: Multi-scale, averaged arrow -------------------------

plt.figure(figsize=(7.5, 5.0))
for B in B_list:
    S_mean = averaged_entropy_over_perms(H, W, B, steps, avg_trials, seed=12345)
    plt.plot(S_mean, label=f"B={B}")
plt.xlabel("time step")
plt.ylabel("normalized coarse-grained entropy")
plt.title("Emergent arrow from reversible permutation dynamics (averaged over permutations)")
plt.legend()
plt.tight_layout()

# ------------------------- Figure 2: Explicit reversibility (forward vs rewind) -------------------------

# One concrete run (no averaging) to show exact retracing when rewound
rng = np.random.default_rng(2025)
bits0 = init_hotspot(H, W, side=4)
perm = np.arange(H*W); rng.shuffle(perm)
inv = inverse_perm(perm)

B_demo = 4  # choose one scale for the forward/rewind illustration
S_fwd, states, bits_end = run_forward(bits0, perm, steps, B_demo)
S_back = run_rewind(bits_end, inv, steps, B_demo)

plt.figure(figsize=(7.5, 5.0))
plt.plot(S_fwd, label="forward")
plt.plot(np.arange(steps, 2*steps), S_back, label="rewind")
plt.xlabel("time step")
plt.ylabel("normalized coarse-grained entropy")
plt.title(f"Arrow under coarse-graining, reversible micro-dynamics (B={B_demo})")
plt.legend()
plt.tight_layout()

# ------------------------- Optional: print figure caption to paste into paper -------------------------

caption = (
    "Figure: A 64×64 binary field evolves under a fixed bijective permutation (reversible). "
    "We coarse-grain into B×B blocks and compute the normalized Shannon entropy of the histogram "
    "of block sums. Starting from a localized hot-spot, the coarse-grained entropy increases "
    "toward its maximum (forward mixing). Applying the exact inverse permutation rewinds the state "
    "and the coarse-grained entropy retraces—demonstrating that irreversibility arises from information "
    "loss under coarse-graining rather than the micro-laws."
)
print(caption)

plt.show()
