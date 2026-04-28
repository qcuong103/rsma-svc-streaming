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
        small_frame = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        bl_data, enh_data = self.encoder.encode(gray)
        bl_bits = estimate_bitrate(bl_data, enh_data, n_refine=0)

        el_bits_array = []
        for p in range(1, self.n_planes + 1):
            bits_upto_p = estimate_bitrate(bl_data, enh_data, n_refine=p)
            bits_upto_prev = estimate_bitrate(bl_data, enh_data, n_refine=p-1)
            el_bits_array.append((bits_upto_p - bits_upto_prev) // 8)

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
# THÀNH PHẦN 2 — TỐI ƯU MINMAX FAIRNESS BẰNG MATLAB SCA
# ══════════════════════════════════════════════════════════
def compute_rsma_rates_mm(alpha, H, P_max_watt, noise_power):
    """Tính toán Capacity theo chuẩn Uplink MAC SIC của MATLAB"""
    K = len(H)
    pc = alpha * P_max_watt
    pp = (1.0 - alpha) * P_max_watt
    Rc, Rp = np.zeros(K), np.zeros(K)

    # Stage 1: Common Stream (SIC theo thứ tự giảm dần)
    current_interf = np.sum((pc + pp) * H)
    sic_order_c = np.argsort(pc * H)[::-1]
    for k in sic_order_c:
        sig_k = pc[k] * H[k]
        interf_k = current_interf - sig_k
        Rc[k] = np.log2(1.0 + sig_k / max(interf_k + noise_power, 1e-12))
        current_interf -= sig_k

    # Stage 2: Private Stream (Cứu user yếu)
    sic_order_p = np.argsort(H * pp)[::-1]
    for idx, k in enumerate(sic_order_p):
        interf_p = sum(pp[sic_order_p[j]] * H[sic_order_p[j]] for j in range(idx + 1, K))
        Rp[k] = np.log2(1.0 + pp[k] * H[k] / max(interf_p + noise_power, 1e-12))

    return Rc, Rp

def solve_rsma_qoe_minmax_sca(H, P_max_watt, Q_bl_list, gamma, bl_reqs, noise_power):
    """Thuật toán SCA Gradient Descent tối ưu hàm QoE Logarit"""
    K = len(H)
    alpha = np.array([0.5, 0.5])
    best_alpha = np.copy(alpha)
    best_minQoE = -np.inf

    lr_init = 0.05
    # SCA Vòng lặp
    for iter_idx in range(1, 50):
        lr = lr_init / np.sqrt(iter_idx)
        Rc, Rp = compute_rsma_rates_mm(alpha, H, P_max_watt, noise_power)

        # Đánh giá theo Hàm QoE thực tiễn từ mắt người
        Q = np.zeros(K)
        for k in range(K):
            # Phạt trừ điểm cực kỳ nặng nếu Rc[k] không gánh nổi BL
            penalty = max(0.0, bl_reqs[k] - Rc[k]) * 200.0
            Q[k] = Q_bl_list[k] + gamma * np.log2(1.0 + Rp[k]) - penalty

        if np.min(Q) > best_minQoE:
            best_minQoE = np.min(Q)
            best_alpha = np.copy(alpha)

        k_min = np.argmin(Q)
        grad = np.zeros(K)

        for k in range(K):
            eps = 1e-4
            a_p = np.copy(alpha); a_p[k] = min(0.99, alpha[k] + eps)
            a_m = np.copy(alpha); a_m[k] = max(0.01, alpha[k] - eps)

            Rc_p, Rp_p = compute_rsma_rates_mm(a_p, H, P_max_watt, noise_power)
            Rc_m, Rp_m = compute_rsma_rates_mm(a_m, H, P_max_watt, noise_power)

            pen_p = max(0.0, bl_reqs[k_min] - Rc_p[k_min]) * 200.0
            Q_p = Q_bl_list[k_min] + gamma * np.log2(1.0 + Rp_p[k_min]) - pen_p

            pen_m = max(0.0, bl_reqs[k_min] - Rc_m[k_min]) * 200.0
            Q_m = Q_bl_list[k_min] + gamma * np.log2(1.0 + Rp_m[k_min]) - pen_m

            grad[k] = (Q_p - Q_m) / (2.0 * eps)

        alpha = np.clip(alpha + lr * grad, 0.01, 0.99)

    # Thủ thuật Power Back-off (Bảo vệ Base Layer như code MATLAB)
    Rc, Rp = compute_rsma_rates_mm(best_alpha, H, P_max_watt, noise_power)
    if np.any(Rc < bl_reqs):
        sorted_users = np.argsort(H)[::-1] # User mạnh ưu tiên hi sinh
        best_alpha[sorted_users[0]] = 0.99
        if K > 2: best_alpha[sorted_users[1]] = 0.95
        Rc, Rp = compute_rsma_rates_mm(best_alpha, H, P_max_watt, noise_power)

    pc_val = best_alpha * P_max_watt
    pp_val = (1.0 - best_alpha) * P_max_watt

    bl_feasible = all(Rc[k] >= bl_reqs[k] * 0.99 for k in range(K))

    return {
        "P_common": pc_val,
        "P_private": pp_val,
        "C_common": Rc,
        "C_private": Rp,
        "status": "SCA QoE Optimized",
        "bl_feasible": bl_feasible,
        "QoE": [Q_bl_list[k] + gamma * np.log2(1.0 + Rp[k]) for k in range(K)]
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

    st.markdown("### 🎬 Kênh truyền & QoE")
    noise_floor_dbm = st.slider("Nhiễu nền (dBm)", min_value=-120, max_value=-60, value=-100, step=5)
    noise_power = 10 ** ((noise_floor_dbm - 30) / 10)

    gamma_qoe = st.slider("Hệ số cảm nhận QoE (γ)", min_value=1.0, max_value=20.0, value=8.0, step=0.5)

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

col_chart, col_log = st.columns([3, 2])
with col_chart:
    st.markdown("#### ⚡ Phân bổ công suất theo thời gian")
    ph_chart = st.empty()
with col_log:
    st.markdown("#### 📋 Nhật ký sự kiện")
    ph_log = st.empty()

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
# VÒNG LẶP MÔ PHỎNG
# ══════════════════════════════════════════════════════════
if not run_sim:
    st.info("▶ Bật công tắc **'Chạy mô phỏng (Frame-by-Frame)'** để bắt đầu.")
    st.stop()

for _ in range(10000):
    if not run_sim: break

    idx = st.session_state.frame_idx
    st.session_state.total_frames += 1

    orig_frames = [vsrc[k].next_frame(idx, FRAME_SIZE) for k in range(2)]

    svc_results = [codec.encode(orig_frames[k]) for k in range(2)]
    bl_data  = [svc_results[k][0] for k in range(2)]
    el_data  = [svc_results[k][1] for k in range(2)]
    bl_bytes = [svc_results[k][2] for k in range(2)]
    el_bytes = [svc_results[k][3] for k in range(2)]

    bl_mbps  = [(bl_bytes[k] * 8 * VIDEO_FPS_ASSUMPTION) / 1e6 for k in range(2)]
    el_mbps  = [[(layer_b * 8 * VIDEO_FPS_ASSUMPTION) / 1e6 for layer_b in el_bytes[k]] for k in range(2)]

    Q_bl_list = []
    for k in range(2):
        bl_recon, _ = codec.decode(bl_data[k], el_data[k], layers_received=0, orig_shape=(128,128))
        Q_bl_list.append(codec.compute_psnr(cv2.resize(orig_frames[k], (128,128)), bl_recon))

    h1 = path_loss(d1); h2 = path_loss(d2)
    H  = np.array([h1, h2])

    result = solve_rsma_qoe_minmax_sca(H, P_max_watt, Q_bl_list, gamma_qoe, bl_mbps, noise_power)

    ue2_dropped_by_bs = False
    if not result["bl_feasible"]:
        add_log(f"⚠️ Kênh U2 quá tải. BS ưu tiên giữ U1, ngắt U2.")
        ue2_dropped_by_bs = True

    c_common    = result["C_common"]
    c_priv      = result["C_private"]
    pc_vals     = result["P_common"]
    pp_vals     = result["P_private"]
    opt_status  = result["status"]

    user_frames  = []
    user_labels  = []
    user_classes = []
    user_psnrs   = []

    for k in range(2):
        if k == 1 and ue2_dropped_by_bs:
            black = np.zeros((128, 128, 3), dtype=np.uint8)
            noisy = add_noise_overlay(black, 40)
            user_frames.append(overlay_status(noisy, "UE2: BL DEFERRED", (0, 100, 255)))
            user_labels.append("UE2 BL DEFERRED ⚠️")
            user_classes.append("status-out")
            user_psnrs.append(0.0)
            st.session_state.outage_count += 1
            continue

        private_cap      = c_priv[k]
        layers_to_decode = 0
        cap_accumulated  = 0.0

        for i, el_req in enumerate(el_mbps[k]):
            if private_cap >= cap_accumulated + el_req:
                cap_accumulated  += el_req
                layers_to_decode += 1
            else:
                break

        if layers_to_decode == 0:
            st.session_state.bl_only_count += 1
            if st.session_state.total_frames % 5 == 0:
                add_log(f"📺 UE{k+1}: Mạng yếu -> Chuyển về mức FGS 0/8 (BL-Only)")

        dec, status_text = codec.decode(bl_data[k], el_data[k], layers_received=layers_to_decode, orig_shape=(128, 128))
        green_val = int((layers_to_decode / codec.n_planes) * 255)
        dec = overlay_status(dec, status_text, (0, green_val, 255 - green_val))

        user_frames.append(dec)
        user_labels.append(status_text)
        user_classes.append("status-hd" if layers_to_decode >= 6 else "status-sd")
        user_psnrs.append(codec.compute_psnr(cv2.resize(orig_frames[k], (128, 128)), dec))

    # Giao diện
    ph_img1_orig.image(frame_to_rgb(orig_frames[0]), use_container_width=True)
    ph_img1_recv.image(frame_to_rgb(user_frames[0]), use_container_width=True)
    ph_status1.markdown(render_status_html(user_labels[0], user_classes[0], user_psnrs[0], c_priv[0], float(pc_vals[0]) + float(pp_vals[0])), unsafe_allow_html=True)

    ph_img2_orig.image(frame_to_rgb(orig_frames[1]), use_container_width=True)
    ph_img2_recv.image(frame_to_rgb(user_frames[1]), use_container_width=True)
    ph_status2.markdown(render_status_html(user_labels[1], user_classes[1], user_psnrs[1], c_priv[1], float(pc_vals[1]) + float(pp_vals[1])), unsafe_allow_html=True)

    st.session_state.power_history.append({
        "Frame":      idx,
        "P_common_U1 (mW)":  float(pc_vals[0]) * 1000,
        "P_common_U2 (mW)":  float(pc_vals[1]) * 1000,
        "P_private_U1 (mW)": float(pp_vals[0]) * 1000,
        "P_private_U2 (mW)": float(pp_vals[1]) * 1000,
    })
    if len(st.session_state.power_history) > MAX_HISTORY: st.session_state.power_history.pop(0)
    ph_chart.bar_chart(pd.DataFrame(st.session_state.power_history).set_index("Frame"), height=220)

    log_html = '<div class="log-box">' + "<br>".join(st.session_state.log_events if st.session_state.log_events else ["Hệ thống đang chạy bình thường..."]) + "</div>"
    ph_log.markdown(log_html, unsafe_allow_html=True)

    total = st.session_state.total_frames
    ph_stat1.metric("📦 Frames xử lý", f"{total}")
    ph_stat2.metric("⛔ Tỷ lệ rớt mạng UE2", f"{st.session_state.outage_count / max(total, 1) * 100:.1f}%")
    ph_stat3.metric("📺 Tỷ lệ FGS 0/8", f"{st.session_state.bl_only_count / max(total, 1) * 100:.1f}%")
    ph_stat4.metric("⚡ Solver", opt_status)

    st.session_state.frame_idx += 1