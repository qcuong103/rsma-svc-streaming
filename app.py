import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time
import math
import warnings
from svc_fgs import FGSEncoder, FGSDecoder, estimate_bitrate

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CÀI ĐẶT TRANG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mô phỏng RSMA-SVC",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CSS TUỲ CHỈNH - Giao diện tối, kỹ thuật cao
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f;
}

/* Tiêu đề chính */
.main-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #00d4ff;
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
    letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Thẻ User */
.user-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}

.user-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #0080ff);
}

/* Badge trạng thái */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
}

.status-hd    { background: rgba(0,255,128,0.15); color: #00ff80; border: 1px solid #00ff80; }
.status-sd    { background: rgba(255,200,0,0.15);  color: #ffc800; border: 1px solid #ffc800; }
.status-out   { background: rgba(255,50,50,0.15);  color: #ff3232; border: 1px solid #ff3232; }

/* Metric nhỏ */
.metric-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;
}
.metric-box {
    background: rgba(0,212,255,0.06);
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 5px 10px;
    min-width: 90px;
}
.metric-label {
    font-size: 0.65rem;
    color: #5588aa;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.95rem;
    color: #00d4ff;
}

/* Thanh công suất */
.power-bar-bg {
    background: #0d1f35;
    border-radius: 4px;
    height: 16px;
    margin: 4px 0;
    overflow: hidden;
    border: 1px solid #1e3a5f;
}
.power-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
    font-size: 0.65rem;
    color: #fff;
    padding-left: 4px;
    line-height: 16px;
}
.bar-common  { background: linear-gradient(90deg, #0080ff, #00d4ff); }
.bar-private { background: linear-gradient(90deg, #8000ff, #cc00ff); }

/* Log box */
.log-box {
    background: #050d18;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #5588aa;
    max-height: 120px;
    overflow-y: auto;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e3a5f;
    margin: 12px 0;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# THÀNH PHẦN 1 — BỘ MÃ HÓA SVC TUỲ CHỈNH (OpenCV)
# ══════════════════════════════════════════════════════════
class AdvancedFGS_Codec:
    def __init__(self, base_qp=28, n_bitplanes=8):
        self.encoder = FGSEncoder(base_qp=base_qp, n_bitplanes=n_bitplanes)
        self.decoder = FGSDecoder(base_qp=base_qp, n_bitplanes=n_bitplanes)
        self.n_planes = n_bitplanes

    def encode(self, frame: np.ndarray):
        # Resize và chuyển xám để thuật toán FGS chạy tối ưu trên Python
        small_frame = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        bl_data, enh_data = self.encoder.encode(gray)
        bl_bits = estimate_bitrate(bl_data, enh_data, n_refine=0)

        el_bits_array = []
        for p in range(1, self.n_planes + 1):
            bits_upto_p = estimate_bitrate(bl_data, enh_data, n_refine=p)
            bits_upto_prev = estimate_bitrate(bl_data, enh_data, n_refine=p-1)
            el_bits_array.append((bits_upto_p - bits_upto_prev) // 8) # Đổi bit ra byte

        return bl_data, enh_data, (bl_bits // 8), el_bits_array, gray.shape

    def decode(self, bl_data, enh_data, layers_received: int, orig_shape):
        recon_gray = self.decoder.decode(bl_data, enh_data, orig_shape, n_refine=layers_received)
        recon_bgr = cv2.cvtColor(recon_gray, cv2.COLOR_GRAY2BGR)

        if layers_received == 0:
            status = "BL ONLY"
        elif layers_received == self.n_planes:
            status = "FULL QUALITY"
        else:
            status = f"FGS LEVEL {layers_received}/{self.n_planes}"

        return recon_bgr, status

    @staticmethod
    def compute_psnr(original: np.ndarray, decoded: np.ndarray) -> float:
        if original.shape != decoded.shape:
            decoded = cv2.resize(decoded, (original.shape[1], original.shape[0]))
        mse = np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2)
        if mse < 1e-10: return 100.0
        return 10 * math.log10(255.0 ** 2 / mse)

# ══════════════════════════════════════════════════════════
# THÀNH PHẦN 2 — TỐI ƯU RSMA MIN-MAX FAIRNESS (CVXPY)
# ══════════════════════════════════════════════════════════
def _capacity(h: float, p: float, noise_power: float, bw: float = 1.0) -> float:
    """Shannon capacity [Mbps] cho 1 user."""
    return bw * np.log2(1.0 + float(h) ** 2 * max(float(p), 0.0) / noise_power)


def _min_power_for_rate(h: float, R: float, noise_power: float, bw: float = 1.0) -> float:
    """Công suất tối thiểu (W) để đạt tốc độ R Mbps."""
    if R <= 0:
        return 0.0
    return noise_power * (2 ** (R / bw) - 1.0) / max(float(h) ** 2, 1e-30)


def optimize_rsma_minmax(
    H: np.ndarray,
    P_max: float,
    R_req_common: float,          # = BL_UE1 + BL_UE2 (tổng yêu cầu BL)
    noise_power: float = 1e-9,
    bl_mbps_per_user: list = None # [BL_UE1, BL_UE2] — ưu tiên cứng BL
):
    """
    Tối ưu RSMA Min-Max Fairness với ưu tiên cứng Base Layer (BL-first).

    Chiến lược BL-first:
      1. Dành trước công suất p_c_min để đảm bảo BL chắc chắn được phục vụ
         (hard constraint: C_common >= R_req_common).
      2. Phần công suất còn lại mới phân bổ cho EL theo min-max fairness.
      3. EL không bao giờ cạnh tranh tài nguyên với BL.
    """
    BW   = 1.0
    K    = len(H)
    nz   = max(noise_power, 1e-15)

    # ── BƯỚC 0: Kiểm tra khả thi vật lý ngay từ đầu ──────────────────────
    # Tính công suất tối thiểu lý thuyết cho BL (dùng user tốt nhất mỗi kênh,
    # vì đây là uplink từ BS → UE nhưng capacity tính theo SNR từng kênh)
    p_bl_min_per_user = [_min_power_for_rate(H[k], R_req_common / K, nz, BW) for k in range(K)]
    p_bl_min_total    = sum(p_bl_min_per_user)

    try:
        import cvxpy as cp

        p_c = cp.Variable(K, nonneg=True)   # công suất common (BL)
        p_p = cp.Variable(K, nonneg=True)   # công suất private (EL)

        constraints = [
            cp.sum(p_c) + cp.sum(p_p) <= P_max,
        ]

        cap_common_terms  = []
        cap_private_terms = []
        for k in range(K):
            h2     = float(H[k] ** 2)
            # Common stream (BL)
            sinr_c = h2 * p_c[k] / nz
            cap_c  = BW * cp.log(1 + sinr_c) / np.log(2)
            cap_common_terms.append(cap_c)
            # Private stream (EL)
            sinr_p = h2 * p_p[k] / nz
            cap_p  = BW * cp.log(1 + sinr_p) / np.log(2)
            cap_private_terms.append(cap_p)

        # ── RÀNG BUỘC CỨNG BL (BL-first): common capacity PHẢI >= tổng BL ──
        constraints.append(sum(cap_common_terms) >= R_req_common)

        # Ràng buộc mềm BL per-user (nếu được cung cấp): mỗi user ít nhất
        # nhận đủ BL từ common stream phần của họ
        if bl_mbps_per_user is not None:
            for k in range(K):
                h2     = float(H[k] ** 2)
                sinr_c = h2 * p_c[k] / nz
                cap_ck = BW * cp.log(1 + sinr_c) / np.log(2)
                constraints.append(cap_ck >= bl_mbps_per_user[k])

        # Mục tiêu: tối đa hoá min private capacity (EL fairness)
        t = cp.Variable()
        for k in range(K):
            constraints.append(cap_private_terms[k] >= t)

        prob = cp.Problem(cp.Maximize(t), constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"CVXPY solver: {prob.status}")

        pc_val = np.maximum(p_c.value, 0.0)
        pp_val = np.maximum(p_p.value, 0.0)

        c_common  = float(sum(_capacity(H[k], pc_val[k], nz, BW) for k in range(K)))
        c_private = [float(_capacity(H[k], pp_val[k], nz, BW)) for k in range(K)]

        return {
            "P_common":   pc_val,
            "P_private":  pp_val,
            "C_common":   c_common,
            "C_private":  c_private,
            "status":     "optimal",
            "bl_feasible": c_common >= R_req_common * 0.999,
        }

    except Exception as e:
        # ── FALLBACK: BL-first phân bổ tay ──────────────────────────────────
        # Dành P_BL trước cho BL, phần còn lại chia đều cho EL
        nz_fb = max(noise_power, 1e-15)

        # Tính p_c tối thiểu để đảm bảo BL
        p_bl_reserved = min(p_bl_min_total * 1.2, P_max * 0.70)  # tối đa 70% cho BL
        p_el_budget   = max(P_max - p_bl_reserved, 0.0)

        pc_fallback = np.array([p_bl_reserved / K] * K)
        pp_fallback = np.array([p_el_budget   / K] * K)

        c_common  = sum(_capacity(H[k], pc_fallback[k], nz_fb, BW) for k in range(K))
        c_private = [_capacity(H[k], pp_fallback[k], nz_fb, BW) for k in range(K)]

        return {
            "P_common":   pc_fallback,
            "P_private":  pp_fallback,
            "C_common":   float(c_common),
            "C_private":  [float(c) for c in c_private],
            "status":     f"fallback ({str(e)[:40]})",
            "bl_feasible": float(c_common) >= R_req_common * 0.999,
        }


# ══════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ — Kênh truyền & Frame giả lập
# ══════════════════════════════════════════════════════════
def path_loss(distance_m: float, freq_ghz: float = 2.4) -> float:
    fspl_db = (20 * np.log10(max(distance_m, 1)) + 20 * np.log10(freq_ghz * 1e9) + 20 * np.log10(4 * np.pi / 3e8))
    total_db = fspl_db + np.random.normal(0, 3)
    return max(10 ** (-total_db / 20), 1e-12)


def generate_test_frame(frame_idx: int, size=(320, 240), color_seed: int = 0) -> np.ndarray:
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    rng   = np.random.default_rng(color_seed)
    base_r, base_g, base_b = int(rng.integers(10, 60)), int(rng.integers(10, 60)), int(rng.integers(20, 80))
    t = frame_idx * 0.05
    for y in range(h):
        r = int(base_r + 15 * np.sin(t + y * 0.02))
        g = int(base_g + 10 * np.sin(t * 0.7 + y * 0.015))
        b = int(base_b + 20 * np.sin(t * 1.3 + y * 0.01))
        frame[y, :] = [max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r))]
    circle_color = (0, 200, 255) if color_seed == 0 else (0, 80, 255)
    cx = int(w // 2 + (w // 3) * np.sin(t))
    cy = int(h // 2 + (h // 4) * np.cos(t * 0.8))
    cv2.circle(frame, (cx, cy), 38, circle_color, -1)
    cv2.circle(frame, (cx, cy), 38, (255, 255, 255), 2)
    cv2.putText(frame, f"UE{color_seed+1} #{frame_idx:04d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════
# NGUỒN VIDEO ĐỘC LẬP CHO TỪNG UE
# ══════════════════════════════════════════════════════════
class VideoSource:
    def __init__(self, ue_idx: int):
        self.ue_idx = ue_idx
        self.cap = None
        self.filename = None
        self._tmppath = None

    def load_from_upload(self, uploaded_file) -> bool:
        if uploaded_file is None: return False
        if uploaded_file.name == self.filename and self.cap is not None: return True
        self._release()
        import tempfile, os
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush(); tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            os.unlink(tmp.name); return False
        self.cap = cap; self.filename = uploaded_file.name; self._tmppath = tmp.name
        return True

    def clear(self):
        self._release(); self.filename = None

    def _release(self):
        if self.cap is not None: self.cap.release(); self.cap = None
        if self._tmppath is not None:
            try: import os; os.unlink(self._tmppath)
            except: pass
            self._tmppath = None

    def next_frame(self, frame_idx: int, size=(320, 240)) -> np.ndarray:
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret: return cv2.resize(frame, size)
        return generate_test_frame(frame_idx, size, color_seed=self.ue_idx)

    @property
    def source_label(self) -> str:
        if self.filename:
            name = self.filename
            return f"📹 {name[:20]}{'…' if len(name) > 20 else ''}"
        return f"🔵 Tín hiệu giả lập UE{self.ue_idx + 1}"

def overlay_status(frame: np.ndarray, label: str, color) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(out, label, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 2: return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def add_noise_overlay(frame: np.ndarray, intensity=40) -> np.ndarray:
    noise = np.random.randint(0, intensity, frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise)


# ══════════════════════════════════════════════════════════
# GIAO DIỆN STREAMLIT — SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Cấu hình Hệ thống")
    st.markdown("---")

    st.markdown("### 🔋 Công suất phát")
    P_max_dbm = st.slider("P_max (dBm)", min_value=10, max_value=100, value=23, step=1)
    P_max_watt = 10 ** ((P_max_dbm - 30) / 10)

    st.markdown("### 📍 Khoảng cách người dùng")
    d1 = st.slider("User 1 – Gần (m)", min_value=10, max_value=500, value=80, step=10)
    d2 = st.slider("User 2 – Xa (m)", min_value=100, max_value=2000, value=600, step=50)

    st.markdown("### 🎬 Kênh truyền")
    noise_floor_dbm = st.slider("Nhiễu nền (dBm)", min_value=-120, max_value=-60, value=-100, step=5)
    noise_power = 10 ** ((noise_floor_dbm - 30) / 10)

    # Đã gỡ bỏ thanh trượt Tốc độ FPS.

    st.markdown("---")
    st.markdown("### 🎥 Tải video lên trạm BS")
    ue1_file = st.file_uploader("📤 Video UE1", type=["mp4", "avi", "mov"], key="upload_ue1")
    ue2_file = st.file_uploader("📤 Video UE2", type=["mp4", "avi", "mov"], key="upload_ue2")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑 Reset UE1", use_container_width=True):
            if "vsrc" in st.session_state: st.session_state.vsrc[0].clear()
    with col_c2:
        if st.button("🗑 Reset UE2", use_container_width=True):
            if "vsrc" in st.session_state: st.session_state.vsrc[1].clear()

    run_sim = st.toggle("▶ Chạy mô phỏng (Frame-by-Frame)", value=True)


# ══════════════════════════════════════════════════════════
# GIAO DIỆN STREAMLIT — MAIN
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">📡 MÔ PHỎNG RSMA FGS — FRAME-BY-FRAME</div>', unsafe_allow_html=True)

# Hàng 1: Khai báo Cấu trúc So sánh (Original vs Received)
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("#### 👤 User 1    (Gần trạm)")
    c1_orig, c1_recv = st.columns(2)
    with c1_orig:
        st.markdown("<div style='text-align:center; color:#88aaff; font-size:0.85rem;'>UE gửi (Original)</div>", unsafe_allow_html=True)
        ph_img1_orig = st.empty()
    with c1_recv:
        st.markdown("<div style='text-align:center; color:#88aaff; font-size:0.85rem;'>BS nhận (Decoded)</div>", unsafe_allow_html=True)
        ph_img1_recv = st.empty()
    ph_status1 = st.empty()

with col_v2:
    st.markdown("#### 👥 User 2   (Xa trạm)")
    c2_orig, c2_recv = st.columns(2)
    with c2_orig:
        st.markdown("<div style='text-align:center; color:#88aaff; font-size:0.85rem;'>UE gửi (Original)</div>", unsafe_allow_html=True)
        ph_img2_orig = st.empty()
    with c2_recv:
        st.markdown("<div style='text-align:center; color:#88aaff; font-size:0.85rem;'>BS nhận (Decoded)</div>", unsafe_allow_html=True)
        ph_img2_recv = st.empty()
    ph_status2 = st.empty()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Hàng 2: Phân bổ công suất + Log
col_chart, col_log = st.columns([3, 2])
with col_chart:
    st.markdown("#### ⚡ Phân bổ công suất theo thời gian")
    ph_chart = st.empty()
with col_log:
    st.markdown("#### 📋 Nhật ký sự kiện")
    ph_log = st.empty()

# Hàng 3: Thống kê tổng quan
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("#### 📊 Thống kê tổng quan")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
ph_stat1 = col_s1.empty()
ph_stat2 = col_s2.empty()
ph_stat3 = col_s3.empty()
ph_stat4 = col_s4.empty()


# ══════════════════════════════════════════════════════════
# KHỞI TẠO STATE & CONSTANTS
# ══════════════════════════════════════════════════════════
if "frame_idx"     not in st.session_state: st.session_state.frame_idx     = 0
if "power_history" not in st.session_state: st.session_state.power_history = []
if "log_events"    not in st.session_state: st.session_state.log_events    = []
if "outage_count"  not in st.session_state: st.session_state.outage_count  = 0
if "bl_only_count" not in st.session_state: st.session_state.bl_only_count = 0
if "total_frames"  not in st.session_state: st.session_state.total_frames  = 0

if "vsrc" not in st.session_state:
    st.session_state.vsrc = [VideoSource(0), VideoSource(1)]
vsrc = st.session_state.vsrc

for _k, _uploaded in enumerate([ue1_file, ue2_file]):
    if _uploaded is not None:
        if not vsrc[_k].load_from_upload(_uploaded):
            st.warning(f"⚠️ Không đọc được file video UE{_k+1}.")

codec = AdvancedFGS_Codec(base_qp=28, n_bitplanes=8)
FRAME_SIZE = (320, 240)
MAX_HISTORY = 30
MAX_LOG = 8

# GIẢ ĐỊNH TỐC ĐỘ FRAME CHUẨN ĐỂ TÍNH BĂNG THÔNG (Dù mô phỏng chạy chậm đến đâu)
VIDEO_FPS_ASSUMPTION = 30.0

def add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_events.insert(0, f"[{ts}] {msg}")
    if len(st.session_state.log_events) > MAX_LOG:
        st.session_state.log_events.pop()

def render_status_html(label: str, css_class: str, psnr: float, cap_mbps: float, p_w: float) -> str:
    return f"""
    <div class="user-card">
        <span class="status-badge {css_class}">{label}</span>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">PSNR</div>
                <div class="metric-value">{psnr:.1f} dB</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Capacity (Riêng)</div>
                <div class="metric-value">{cap_mbps:.2f} Mbps</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">P.Phát</div>
                <div class="metric-value">{p_w*1000:.1f} mW</div>
            </div>
        </div>
    </div>
    """


# ══════════════════════════════════════════════════════════
# VÒNG LẶP MÔ PHỎNG (SẼ CHẠY NHANH HẾT MỨC CPU CHO PHÉP)
# ══════════════════════════════════════════════════════════
if not run_sim:
    st.info("▶ Bật công tắc **'Chạy mô phỏng (Frame-by-Frame)'** để bắt đầu.")
    st.stop()

for _ in range(10000):
    if not run_sim: break

    idx = st.session_state.frame_idx
    st.session_state.total_frames += 1

    # ── 1. Lấy frame gốc của từng UE
    orig_frames = [vsrc[k].next_frame(idx, FRAME_SIZE) for k in range(2)]

    # ── 2. Mã hóa FGS độc lập (Nặng nhất, quyết định tốc độ vòng lặp)
    svc_results = [codec.encode(orig_frames[k]) for k in range(2)]
    bl_data  = [svc_results[k][0] for k in range(2)]
    el_data  = [svc_results[k][1] for k in range(2)]
    bl_bytes = [svc_results[k][2] for k in range(2)]
    el_bytes = [svc_results[k][3] for k in range(2)]

    # TÍNH TOÁN BĂNG THÔNG DỰA TRÊN TỐC ĐỘ FRAME CỐ ĐỊNH CHUẨN
    bl_mbps  = [(bl_bytes[k] * 8 * VIDEO_FPS_ASSUMPTION) / 1e6 for k in range(2)]
    el_mbps  = [
        [(layer_b * 8 * VIDEO_FPS_ASSUMPTION) / 1e6 for layer_b in el_bytes[k]]
        for k in range(2)
    ]

    # ── 3. Tính kênh truyền
    h1 = path_loss(d1); h2 = path_loss(d2)
    H  = np.array([h1, h2])

    # ── 4. Tối ưu RSMA & Điều khiển thu nhận (BL-First Priority)
    R_req_both = bl_mbps[0] + bl_mbps[1]

    # Truyền yêu cầu BL per-user để optimizer đảm bảo cứng từng user
    result = optimize_rsma_minmax(
        H, P_max_watt,
        R_req_both * 1.01,
        noise_power,
        bl_mbps_per_user=[bl_mbps[0] * 1.01, bl_mbps[1] * 1.01],
    )

    ue2_dropped_by_bs = False

    # Kiểm tra: nếu capacity common KHÔNG đủ cho cả 2 BL
    # → ưu tiên UE1 (BL-first): tái tối ưu chỉ phục vụ UE1
    if not result.get("bl_feasible", True) or result["C_common"] < R_req_both * 0.99:
        add_log(f"⚠️ Kênh không đủ cho cả 2 BL ({result['C_common']:.3f} < {R_req_both:.3f} Mbps). "
                f"BS ưu tiên BL UE1, ngắt UE2.")
        R_req_ue1_only = bl_mbps[0] * 1.01
        result = optimize_rsma_minmax(
            H, P_max_watt,
            R_req_ue1_only,
            noise_power,
            bl_mbps_per_user=[bl_mbps[0] * 1.01, 0.0],
        )
        ue2_dropped_by_bs = True

    c_common    = result["C_common"]
    c_priv      = result["C_private"]
    pc_vals     = result["P_common"]
    pp_vals     = result["P_private"]
    opt_status  = result["status"]
    bl_feasible = result.get("bl_feasible", c_common >= bl_mbps[0] * 0.99)

    # ── 5. Decode và Tính số lớp EL nhận được
    user_frames  = []
    user_labels  = []
    user_classes = []
    user_psnrs   = []

    # ── BL-first decode logic ────────────────────────────────────────────────
    # Sau khi optimizer đã đảm bảo BL, chỉ còn 2 trường hợp thực:
    #   (a) UE2 bị ngắt vì kênh không đủ cho cả 2 BL → ue2_dropped_by_bs
    #   (b) Cả 2 đều được phục vụ BL, EL dùng phần dư sau BL
    #
    # "TOTAL OUTAGE" chỉ xảy ra khi kênh không đủ ngay cả cho UE1 BL một mình.
    # Trường hợp này rất hiếm vì optimizer đã được ràng buộc cứng BL.

    bl1_feasible = bl_feasible   # Đã được kiểm tra trong optimizer (ràng buộc cứng BL)

    if not bl1_feasible:
        # Trường hợp ngoại lệ thực sự: kênh quá tệ, không cứu được
        st.session_state.outage_count += 1
        add_log(f"⛔ OUTAGE THỰC SỰ: C_common={c_common:.3f} < BL_UE1={bl_mbps[0]:.3f} Mbps "
                f"(kênh quá tệ, P_max có thể quá thấp)")
        for k in range(2):
            black = np.zeros((128, 128, 3), dtype=np.uint8)
            noisy = add_noise_overlay(black, 60)
            user_frames.append(overlay_status(noisy, "NO SIGNAL", (0, 0, 255)))
            user_labels.append("NO SIGNAL ⛔"); user_classes.append("status-out"); user_psnrs.append(0.0)
    else:
        for k in range(2):
            if k == 1 and ue2_dropped_by_bs:
                # UE2 bị ngắt có chủ đích để ưu tiên BL UE1 — hiển thị rõ lý do
                black = np.zeros((128, 128, 3), dtype=np.uint8)
                noisy = add_noise_overlay(black, 40)
                user_frames.append(overlay_status(noisy, "UE2: BL DEFERRED", (0, 100, 255)))
                user_labels.append("UE2 BL DEFERRED ⚠️"); user_classes.append("status-out"); user_psnrs.append(0.0)
                st.session_state.outage_count += 1
                continue

            # BL luôn được đảm bảo (do optimizer ràng buộc cứng)
            # → EL sử dụng phần dư: private_cap - 0 (BL đã nằm trong common stream)
            # private_cap hoàn toàn dành cho EL refinement
            private_cap      = c_priv[k]
            layers_to_decode = 0
            cap_accumulated  = 0.0

            for i, el_req in enumerate(el_mbps[k]):
                if private_cap >= cap_accumulated + el_req:
                    cap_accumulated  += el_req
                    layers_to_decode += 1
                else:
                    break   # FGS truncation — BL vẫn được giữ nguyên

            if layers_to_decode == 0:
                st.session_state.bl_only_count += 1
                add_log(f"📺 UE{k+1}: BL-Only (private_cap={private_cap:.3f} Mbps, "
                        f"EL[0]={el_mbps[k][0]:.3f} Mbps)")

            dec, status_text = codec.decode(
                bl_data[k], el_data[k],
                layers_received=layers_to_decode,
                orig_shape=(128, 128)
            )

            green_val = int((layers_to_decode / codec.n_planes) * 255)
            dec = overlay_status(dec, status_text, (0, green_val, 255 - green_val))

            user_frames.append(dec)
            user_labels.append(status_text)
            user_classes.append("status-hd" if layers_to_decode >= 6 else "status-sd")
            user_psnrs.append(codec.compute_psnr(orig_frames[k], dec))

    # ── 6. Hiển thị Giao diện Song song (Original vs Received)
    # User 1
    # Ghi chú: orig_frames là 320x240, nhưng FGS encode là 128x128.
    # Ta hiển thị cả 2, Streamlit sẽ tự scale cho vừa cột.
    ph_img1_orig.image(frame_to_rgb(orig_frames[0]), use_container_width=True)
    ph_img1_recv.image(frame_to_rgb(user_frames[0]), use_container_width=True)

    ph_status1.markdown(
        render_status_html(user_labels[0], user_classes[0], user_psnrs[0], c_priv[0], float(pc_vals[0]) + float(pp_vals[0])),
        unsafe_allow_html=True
    )

    # User 2
    ph_img2_orig.image(frame_to_rgb(orig_frames[1]), use_container_width=True)
    ph_img2_recv.image(frame_to_rgb(user_frames[1]), use_container_width=True)

    ph_status2.markdown(
        render_status_html(user_labels[1], user_classes[1], user_psnrs[1], c_priv[1], float(pc_vals[1]) + float(pp_vals[1])),
        unsafe_allow_html=True
    )

    # Biểu đồ và Log
    st.session_state.power_history.append({
        "Frame":      idx,
        "P_common_U1 (mW)":  float(pc_vals[0]) * 1000,
        "P_common_U2 (mW)":  float(pc_vals[1]) * 1000,
        "P_private_U1 (mW)": float(pp_vals[0]) * 1000,
        "P_private_U2 (mW)": float(pp_vals[1]) * 1000,
    })
    if len(st.session_state.power_history) > MAX_HISTORY: st.session_state.power_history.pop(0)

    ph_chart.bar_chart(pd.DataFrame(st.session_state.power_history).set_index("Frame"), height=220)

    log_html = '<div class="log-box">' + "<br>".join(
        st.session_state.log_events if st.session_state.log_events else ["Hệ thống đang chạy bình thường..."]
    ) + "</div>"
    ph_log.markdown(log_html, unsafe_allow_html=True)

    # Thống kê
    total = st.session_state.total_frames
    ph_stat1.metric("📦 Frames xử lý", f"{total}")
    ph_stat2.metric("⛔ Tỷ lệ mất kết nối", f"{st.session_state.outage_count / max(total, 1) * 100:.1f}%")
    ph_stat3.metric("📺 Tỷ lệ BL-Only", f"{st.session_state.bl_only_count / max(total, 1) * 100:.1f}%")
    ph_stat4.metric("⚡ Solver", opt_status[:12])

    st.session_state.frame_idx += 1