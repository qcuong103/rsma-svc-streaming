import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# DCT / IDCT helpers (2-D, block-based)
# ---------------------------------------------------------------------------

def dct2d(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via separable 1-D transforms."""
    from scipy.fft import dctn
    return dctn(block.astype(float), norm="ortho")


def idct2d(block: np.ndarray) -> np.ndarray:
    """2-D inverse DCT."""
    from scipy.fft import idctn
    return idctn(block, norm="ortho")


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantize(coeffs: np.ndarray, qp: int) -> np.ndarray:
    """Uniform scalar quantisation (dead-zone at origin)."""
    step = 2 ** (qp / 6)           # exponential step size (H.264-like)
    return np.round(coeffs / step).astype(int)


def dequantize(qcoeffs: np.ndarray, qp: int) -> np.ndarray:
    """Reconstruct float coefficients from integer quant indices."""
    step = 2 ** (qp / 6)
    return qcoeffs.astype(float) * step


# ---------------------------------------------------------------------------
# Bit-plane decomposition (core of FGS)
# ---------------------------------------------------------------------------

def residual_to_bitplanes(residual: np.ndarray,
                          n_planes: int = 8)-> List[np.ndarray]:
    abs_r = np.abs(residual).astype(np.int32)
    sign  = np.sign(residual).astype(np.int8)   # -1, 0, +1
    planes = []
    for p in range(n_planes - 1, -1, -1):        # MSB first
        plane = ((abs_r >> p) & 1).astype(np.int8)
        planes.append(plane)
    planes[0] = planes[0] * sign                 # values in {-1, 0, +1}
    return planes


def bitplanes_to_residual(
    planes: List[np.ndarray],
    n_refine: Optional[int] = None
) -> np.ndarray:
    """
    Reconstruct residual from a (possibly truncated) list of bit-planes.

    Parameters
    ----------
    planes     : bit-planes, MSB first (output of residual_to_bitplanes)
    n_refine   : number of refinement planes to use (None = all).
                 Truncating here simulates network bandwidth constraint.
    """
    n_planes = len(planes)
    if n_refine is not None:
        n_refine = min(n_refine, n_planes)
        planes = planes[:n_refine]              # discard LSB planes

    # Recover sign from MSB plane
    sign_mask = np.sign(planes[0]).astype(np.int8)   # +1, -1, or 0
    abs_msb   = np.abs(planes[0]).astype(np.int32)

    # Bit-plane index k → shift = (n_planes - 1 - k)
    total_shift = n_planes - 1
    residual = abs_msb << total_shift

    for k, plane in enumerate(planes[1:], start=1):
        shift = total_shift - k
        residual += np.abs(plane).astype(np.int32) << shift

    return (sign_mask * residual).astype(float)


# ---------------------------------------------------------------------------
# Block-level FGS encoder / decoder
# ---------------------------------------------------------------------------

def fgs_encode_block(
    block: np.ndarray,
    base_qp: int = 28,
    n_bitplanes: int = 8
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    FGS-encode a single 8×8 pixel block.

    Returns
    -------
    base_qcoeffs : quantised DCT coefficients for the base layer
    fgs_planes   : bit-planes of the enhancement residual
    """
    coeffs      = dct2d(block)
    base_qc     = quantize(coeffs, base_qp)
    base_recon  = dequantize(base_qc, base_qp)
    residual    = coeffs - base_recon              # FGS residual
    # Scale residual to integer domain for bit-plane coding
    scale       = 2 ** (n_bitplanes - 1)
    r_int       = np.round(residual * scale).astype(np.int32)
    planes      = residual_to_bitplanes(r_int, n_bitplanes)
    return base_qc, planes


def fgs_decode_block(
    base_qcoeffs: np.ndarray,
    fgs_planes:   List[np.ndarray],
    base_qp:      int = 28,
    n_bitplanes:  int = 8,
    n_refine:     Optional[int] = None
) -> np.ndarray:
    """
    FGS-decode a single 8×8 block.

    Parameters
    ----------
    n_refine : how many enhancement bit-planes to apply (None = all).
               Set to 0 for base-layer-only reconstruction.
    """
    base_recon = dequantize(base_qcoeffs, base_qp)

    if n_refine == 0 or not fgs_planes:
        coeffs = base_recon
    else:
        scale    = 2 ** (n_bitplanes - 1)
        r_int    = bitplanes_to_residual(fgs_planes, n_refine)
        residual = r_int / scale
        coeffs   = base_recon + residual

    pixel_block = idct2d(coeffs)
    return np.clip(pixel_block, 0, 255)


# ---------------------------------------------------------------------------
# Frame-level FGS codec
# ---------------------------------------------------------------------------

BLOCK = 8   # block size


def _pad(frame: np.ndarray) -> np.ndarray:
    """Pad frame so height and width are multiples of BLOCK."""
    h, w = frame.shape
    ph = (BLOCK - h % BLOCK) % BLOCK
    pw = (BLOCK - w % BLOCK) % BLOCK
    return np.pad(frame, ((0, ph), (0, pw)), mode="edge")


class FGSEncoder:
    """
    Frame-level Fine Granular Scalability encoder.

    Usage
    -----
    enc = FGSEncoder(base_qp=28, n_bitplanes=8)
    bl_data, enh_data = enc.encode(frame)
    """

    def __init__(self, base_qp: int = 28, n_bitplanes: int = 8):
        self.base_qp     = base_qp
        self.n_bitplanes = n_bitplanes

    def encode(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, List[List[List[np.ndarray]]]]:
        """
        Encode a grayscale frame.

        Returns
        -------
        base_layer   : 2-D array of quantised DCT coefficients (base layer)
        enh_layer    : nested list [row][col] → list of bit-planes per block
        """
        frame_f  = frame.astype(float)
        padded   = _pad(frame_f)
        H, W     = padded.shape
        rows, cols = H // BLOCK, W // BLOCK

        base_layer: np.ndarray                       = np.zeros((H, W), dtype=np.int32)
        enh_layer:  List[List[List[np.ndarray]]]     = []

        for r in range(rows):
            row_planes = []
            for c in range(cols):
                y0, x0 = r * BLOCK, c * BLOCK
                block   = padded[y0:y0+BLOCK, x0:x0+BLOCK]

                bq, planes = fgs_encode_block(block, self.base_qp, self.n_bitplanes)
                base_layer[y0:y0+BLOCK, x0:x0+BLOCK] = bq
                row_planes.append(planes)

            enh_layer.append(row_planes)

        return base_layer, enh_layer


class FGSDecoder:
    """
    Frame-level Fine Granular Scalability decoder.

    Usage
    -----
    dec = FGSDecoder(base_qp=28, n_bitplanes=8)
    recon = dec.decode(bl_data, enh_data, orig_shape, n_refine=4)
    """

    def __init__(self, base_qp: int = 28, n_bitplanes: int = 8):
        self.base_qp     = base_qp
        self.n_bitplanes = n_bitplanes

    def decode(
        self,
        base_layer:  np.ndarray,
        enh_layer:   List[List[List[np.ndarray]]],
        orig_shape:  Tuple[int, int],
        n_refine:    Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct frame with optional enhancement truncation.

        Parameters
        ----------
        n_refine : bit-planes to apply from enhancement layer.
                   None = full quality; 0 = base only.
        """
        H, W     = base_layer.shape
        rows, cols = H // BLOCK, W // BLOCK
        recon    = np.zeros((H, W), dtype=float)

        for r in range(rows):
            for c in range(cols):
                y0, x0  = r * BLOCK, c * BLOCK
                bq      = base_layer[y0:y0+BLOCK, x0:x0+BLOCK]
                planes  = enh_layer[r][c]

                block_r = fgs_decode_block(
                    bq, planes,
                    base_qp=self.base_qp,
                    n_bitplanes=self.n_bitplanes,
                    n_refine=n_refine
                )
                recon[y0:y0+BLOCK, x0:x0+BLOCK] = block_r

        oh, ow = orig_shape
        return np.clip(recon[:oh, :ow], 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def estimate_bitrate(
    base_layer:  np.ndarray,
    enh_layer:   List[List[List[np.ndarray]]],
    n_refine:    Optional[int] = None
) -> int:
    # Base layer: count non-zero coefficients × 8 bits each
    base_bits = int(np.count_nonzero(base_layer)) * 8

    n_planes = len(enh_layer[0][0]) if enh_layer else 0
    use      = n_planes if n_refine is None else min(n_refine, n_planes)

    enh_bits = 0
    for row in enh_layer:
        for planes in row:
            for p in range(use):
                enh_bits += int(np.count_nonzero(planes[p]))

    return base_bits + enh_bits


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

def demo():
    """End-to-end FGS demo on a synthetic test frame."""
    print("=" * 60)
    print("SVC Fine Granular Scalability (FGS) — Demo")
    print("=" * 60)

    rng   = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    enc = FGSEncoder(base_qp=28, n_bitplanes=8)
    dec = FGSDecoder(base_qp=28, n_bitplanes=8)

    print(f"\nFrame size : {frame.shape}")
    print(f"Base QP    : {enc.base_qp}")
    print(f"Bit-planes : {enc.n_bitplanes}")
    print()

    base_layer, enh_layer = enc.encode(frame)

    header = f"{'Refinement planes':>20} {'PSNR (dB)':>12} {'Est. bits':>12}"
    print(header)
    print("-" * len(header))

    for n_ref in [0, 1, 2, 4, 6, 8, None]:
        label  = "all" if n_ref is None else str(n_ref)
        recon  = dec.decode(base_layer, enh_layer, frame.shape, n_refine=n_ref)
        q      = psnr(frame, recon)
        bits   = estimate_bitrate(base_layer, enh_layer, n_refine=n_ref)
        print(f"{label:>20} {q:>12.2f} {bits:>12,}")

    print()
    print("Quality increases monotonically as more bit-planes are received,")
    print("demonstrating fine granular scalability.")


if __name__ == "__main__":
    demo()
