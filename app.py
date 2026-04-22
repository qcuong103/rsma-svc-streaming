"""
============================================================
 Mô phỏng thời gian thực: Truyền Video SVC qua mạng RSMA
 Thuật toán phân bổ công suất Min-Max Fairness
 Tác giả: Chuyên gia Kỹ sư Phần mềm & Nhà nghiên cứu Viễn thông
============================================================
Chạy: streamlit run app.py
Yêu cầu: pip install streamlit numpy cvxpy opencv-python pandas
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time
import math
import warnings

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

.user-card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    color: #00d4ff;
    margin-bottom: 4px;
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

/* Ẩn footer Streamlit */
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# THÀNH PHẦN 1 — BỘ MÃ HÓA SVC TUỲ CHỈNH (OpenCV)
# ══════════════════════════════════════════════════════════
class CustomSVCCodec:
    """
    Bộ mã hóa / giải mã video dạng SVC (Scalable Video Coding) đơn giản
    sử dụng OpenCV.
    - Base Layer (BL)   : ảnh nén 50% kích thước — thông tin thiết yếu.
    - Enhancement Layer (EL): phần dư (residual) — thông tin chi tiết.
    """

    def __init__(self, scale: float = 0.5):
        # Tỷ lệ thu nhỏ khi tạo Base Layer
        self.scale = scale

    def encode(self, frame: np.ndarray):
        """
        Mã hóa 1 frame thành 2 lớp SVC.
        Trả về: (bl_data, el_data, bl_bytes, el_bytes)
        """
        h, w = frame.shape[:2]
        # ── Base Layer: nén nhỏ 50%
        bl_small = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)),
                              interpolation=cv2.INTER_AREA)
        # Phóng lại kích thước gốc để tính residual
        bl_upscaled = cv2.resize(bl_small, (w, h),
                                 interpolation=cv2.INTER_LINEAR)

        # ── Enhancement Layer: phần dư giữa ảnh gốc và BL tái tạo
        frame_f = frame.astype(np.float32)
        bl_f    = bl_upscaled.astype(np.float32)
        residual = frame_f - bl_f                       # Có thể âm hoặc dương
        # Dịch về [0..255] để lưu trữ
        el_shifted = (residual + 255.0).clip(0, 510) / 2.0  # [0..255]
        el_uint8   = el_shifted.astype(np.uint8)

        # Ước tính kích thước byte (giả lập nén JPEG)
        _, bl_buf = cv2.imencode(".jpg", bl_small,  [cv2.IMWRITE_JPEG_QUALITY, 70])
        _, el_buf = cv2.imencode(".jpg", el_uint8,  [cv2.IMWRITE_JPEG_QUALITY, 50])

        return bl_small, el_uint8, len(bl_buf), len(el_buf)

    def decode(self, bl_small: np.ndarray, el_data=None,
               target_size=(320, 240)):
        """
        Giải mã frame SVC.
        - Chỉ có BL  → ảnh mờ (phóng to BL).
        - Có cả EL   → ảnh nét (BL + residual).
        Trả về (frame_decoded, quality_label)
        """
        h, w = target_size[1], target_size[0]
        bl_up = cv2.resize(bl_small, (w, h), interpolation=cv2.INTER_LINEAR)

        if el_data is None:
            # Chỉ Base Layer — ảnh mờ
            return bl_up, "BL_ONLY"

        # Khôi phục residual từ EL đã dịch
        el_resized = cv2.resize(el_data, (w, h), interpolation=cv2.INTER_LINEAR)
        residual   = el_resized.astype(np.float32) * 2.0 - 255.0
        reconstructed = (bl_up.astype(np.float32) + residual).clip(0, 255)
        return reconstructed.astype(np.uint8), "FULL_HD"

    @staticmethod
    def compute_psnr(original: np.ndarray, decoded: np.ndarray) -> float:
        """Tính PSNR (dB) giữa ảnh gốc và ảnh giải mã."""
        if original.shape != decoded.shape:
            decoded = cv2.resize(decoded, (original.shape[1], original.shape[0]))
        mse = np.mean((original.astype(np.float64) - decoded.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return 100.0
        return 10 * math.log10(255.0 ** 2 / mse)


# ══════════════════════════════════════════════════════════
# THÀNH PHẦN 2 — TỐI ƯU RSMA MIN-MAX FAIRNESS (CVXPY)
# ══════════════════════════════════════════════════════════
def optimize_rsma_minmax(H: np.ndarray, P_max: float,
                          R_req_common: float, noise_power: float = 1e-9):
    """
    Giải bài toán phân bổ công suất RSMA theo tiêu chí Min-Max Fairness.

    Mô hình RSMA đường lên (Uplink):
    - Mỗi người dùng k gửi: x_k = x_k^c (common) + x_k^p (private)
    - Bộ thu dùng SIC: giải mã Common trước, sau đó Private.

    Tham số:
        H            : mảng shape (K,) — độ lợi kênh của K người dùng
        P_max        : tổng công suất tối đa (Watt)
        R_req_common : tốc độ tối thiểu cần cho BL (Mbps)
        noise_power  : công suất nhiễu nền (Watt)

    Kết quả:
        dict chứa P_common, P_private_1, P_private_2, C_common,
                  C_private_1, C_private_2, status
    """
    try:
        import cvxpy as cp

        K = len(H)   # Số người dùng (= 2 trong bài toán này)

        # ── Biến tối ưu ──────────────────────────────────────────
        # p_c[k] : công suất Common của user k
        # p_p[k] : công suất Private của user k
        p_c = cp.Variable(K, nonneg=True)   # Công suất Common
        p_p = cp.Variable(K, nonneg=True)   # Công suất Private

        # ── Ràng buộc công suất ──────────────────────────────────
        constraints = []
        # Tổng công suất không vượt P_max
        constraints.append(cp.sum(p_c) + cp.sum(p_p) <= P_max)
        # Mỗi user không phát âm
        for k in range(K):
            constraints.append(p_c[k] >= 0)
            constraints.append(p_p[k] >= 0)

        # ── Tính SINR & Capacity (Tuyến tính hoá log để dùng CVXPY) ─
        # RSMA đường lên: Common được giải mã trước (SIC bậc 1)
        # SINR_common_k = H[k]^2 * p_c[k] / (H[k]^2 * p_p[k] + noise)
        # C_common = sum_k log2(1 + SINR_common_k)  — gần đúng tuyến tính

        # Vì log không lồi lõm thuận tiện trong CVXPY standard,
        # dùng xấp xỉ tuyến tính: C ≈ B * log2(1 + H^2*P / noise)
        # với B = 1 (chuẩn hoá băng thông = 1 Hz)

        # Để giữ bài toán convex, ta sử dụng biến thay thế:
        #   Xây dựng lower-bound capacity dạng log-sum bằng cp.log
        BW = 1e6  # 1 MHz băng thông (chuẩn hoá), đơn vị Mbps

        # SINR Common của mỗi user (xấp xỉ, nhiễu = noise_power)
        # Capacity Common tổng = sum log2(1 + H_k^2 * p_c[k] / noise)
        cap_common_terms = []
        cap_private_terms = []
        for k in range(K):
            h2 = float(H[k] ** 2)
            # Dùng cp.log(1 + ...) / log(2) * BW
            # Nhiễu tại Common: chỉ nhiễu Gaussian (SIC lý tưởng)
            sinr_c = h2 * p_c[k] / noise_power
            cap_c  = BW * cp.log(1 + sinr_c) / np.log(2)
            cap_common_terms.append(cap_c)

            # Nhiễu tại Private: sau khi trừ Common (SIC)
            sinr_p = h2 * p_p[k] / noise_power
            cap_p  = BW * cp.log(1 + sinr_p) / np.log(2)
            cap_private_terms.append(cap_p)

        # Tổng capacity Common phải ≥ R_req_common
        total_cap_common = sum(cap_common_terms)
        constraints.append(total_cap_common >= R_req_common)

        # ── Mục tiêu Min-Max Fairness ────────────────────────────
        # Tối đa hoá capacity Private nhỏ nhất trong 2 user
        t = cp.Variable()   # Biến phụ: min capacity private
        for k in range(K):
            constraints.append(cap_private_terms[k] >= t)

        objective = cp.Maximize(t)

        # ── Giải bài toán ────────────────────────────────────────
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"CVXPY: {prob.status}")

        # Lấy kết quả
        pc_val = np.maximum(p_c.value, 0)
        pp_val = np.maximum(p_p.value, 0)

        def cap(h, p):
            return BW * np.log2(1 + h**2 * p / noise_power)

        c_common   = sum(cap(H[k], pc_val[k]) for k in range(K))
        c_private  = [cap(H[k], pp_val[k]) for k in range(K)]

        return {
            "P_common":    pc_val,
            "P_private":   pp_val,
            "C_common":    float(c_common),
            "C_private":   [float(c) for c in c_private],
            "status":      "optimal",
        }

    except Exception as e:
        # Trả về kết quả mặc định khi không khả thi (fallback)
        pc_fallback = np.array([P_max * 0.4, P_max * 0.4])
        pp_fallback = np.array([P_max * 0.1, P_max * 0.1])
        BW = 1e6
        noise_power_fb = noise_power if noise_power else 1e-9

        def cap_fb(h, p):
            return BW * np.log2(1 + float(h)**2 * p / noise_power_fb)

        return {
            "P_common":    pc_fallback,
            "P_private":   pp_fallback,
            "C_common":    sum(cap_fb(H[k], pc_fallback[k]) for k in range(len(H))),
            "C_private":   [cap_fb(H[k], pp_fallback[k]) for k in range(len(H))],
            "status":      f"fallback ({str(e)[:40]})",
        }


# ══════════════════════════════════════════════════════════
# HÀM PHỤ TRỢ — Kênh truyền & Frame giả lập
# ══════════════════════════════════════════════════════════
def path_loss(distance_m: float, freq_ghz: float = 2.4) -> float:
    """
    Tính suy hao đường truyền theo mô hình Free-Space (FSPL).
    Kết quả: hệ số kênh h (biên độ).
    """
    # FSPL (dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
    fspl_db = (20 * np.log10(max(distance_m, 1)) +
               20 * np.log10(freq_ghz * 1e9) +
               20 * np.log10(4 * np.pi / 3e8))
    # Thêm fading Rayleigh nhỏ (±3 dB)
    fading_db = np.random.normal(0, 3)
    total_db = fspl_db + fading_db
    # Chuyển về hệ số tuyến tính
    h = 10 ** (-total_db / 20)
    return max(h, 1e-12)


def generate_test_frame(frame_idx: int, size=(320, 240)) -> np.ndarray:
    """
    Tạo frame thử nghiệm có chuyển động (thay thế camera/video thực).
    - Nền gradient thay đổi theo thời gian.
    - Hình tròn di chuyển biểu diễn nội dung video.
    """
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Nền gradient động
    t = frame_idx * 0.05
    for y in range(h):
        r = int(20 + 15 * np.sin(t + y * 0.02))
        g = int(30 + 10 * np.sin(t * 0.7 + y * 0.015))
        b = int(50 + 20 * np.sin(t * 1.3 + y * 0.01))
        frame[y, :] = [max(0, min(255, b)),
                       max(0, min(255, g)),
                       max(0, min(255, r))]

    # Vòng tròn chuyển động (biểu diễn nội dung)
    cx = int(w // 2 + (w // 3) * np.sin(t))
    cy = int(h // 2 + (h // 4) * np.cos(t * 0.8))
    cv2.circle(frame, (cx, cy), 40, (0, 200, 255), -1)
    cv2.circle(frame, (cx, cy), 40, (255, 255, 255), 2)

    # Nhãn số frame
    cv2.putText(frame, f"Frame #{frame_idx:04d}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def overlay_status(frame: np.ndarray, label: str, color) -> np.ndarray:
    """Thêm nhãn trạng thái lên góc trái dưới frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(out, label,
                (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
    return out


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Chuyển BGR (OpenCV) → RGB (Streamlit)."""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def add_noise_overlay(frame: np.ndarray, intensity=40) -> np.ndarray:
    """Thêm nhiễu ảnh để mô phỏng mất gói nghiêm trọng."""
    noise = np.random.randint(0, intensity, frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise)


# ══════════════════════════════════════════════════════════
# GIAO DIỆN STREAMLIT — SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Cấu hình Hệ thống")
    st.markdown("---")

    st.markdown("### 🔋 Công suất phát")
    P_max_dbm = st.slider(
        "P_max (dBm)", min_value=10, max_value=40, value=23, step=1,
        help="Tổng công suất phát tối đa của trạm gốc"
    )
    P_max_watt = 10 ** ((P_max_dbm - 30) / 10)  # dBm → Watt

    st.markdown("### 📍 Khoảng cách người dùng")
    d1 = st.slider(
        "User 1 – Gần (m)", min_value=10, max_value=500, value=80, step=10,
        help="Khoảng cách User 1 tới trạm gốc"
    )
    d2 = st.slider(
        "User 2 – Xa (m)", min_value=100, max_value=2000, value=600, step=50,
        help="Khoảng cách User 2 tới trạm gốc"
    )

    st.markdown("### 🎬 Tham số Video")
    target_fps = st.slider(
        "Tốc độ mô phỏng (fps)", min_value=1, max_value=10, value=3, step=1
    )
    noise_floor_dbm = st.slider(
        "Nhiễu nền (dBm)", min_value=-120, max_value=-60, value=-100, step=5
    )
    noise_power = 10 ** ((noise_floor_dbm - 30) / 10)

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin hệ thống")
    st.markdown(f"""
    <div style='font-size:0.78rem; color:#5588aa; font-family:monospace;'>
    • Mô hình: RSMA Uplink 2-User<br>
    • Codec: SVC (BL + EL)<br>
    • Tối ưu: Min-Max Fairness<br>
    • Solver: CVXPY / SCS<br>
    • Kênh: FSPL + Rayleigh fading<br>
    </div>
    """, unsafe_allow_html=True)

    run_sim = st.toggle("▶ Chạy mô phỏng", value=True)


# ══════════════════════════════════════════════════════════
# GIAO DIỆN STREAMLIT — MAIN
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">📡 MÔ PHỎNG RSMA-SVC — TRUYỀN VIDEO ĐỘ TRỄ THẤP</div>',
            unsafe_allow_html=True)

# Hàng 1: 2 cột Video
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown("#### 👤 User 1 — Gần trạm")
    ph_img1     = st.empty()
    ph_status1  = st.empty()
    ph_metrics1 = st.empty()

with col_v2:
    st.markdown("#### 👥 User 2 — Xa trạm")
    ph_img2     = st.empty()
    ph_status2  = st.empty()
    ph_metrics2 = st.empty()

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
# KHỞI TẠO STATE
# ══════════════════════════════════════════════════════════
if "frame_idx"     not in st.session_state: st.session_state.frame_idx     = 0
if "power_history" not in st.session_state: st.session_state.power_history = []
if "log_events"    not in st.session_state: st.session_state.log_events    = []
if "outage_count"  not in st.session_state: st.session_state.outage_count  = 0
if "bl_only_count" not in st.session_state: st.session_state.bl_only_count = 0
if "total_frames"  not in st.session_state: st.session_state.total_frames  = 0

codec = CustomSVCCodec(scale=0.5)
FRAME_SIZE = (320, 240)
MAX_HISTORY = 30   # Giữ 30 điểm lịch sử trên biểu đồ
MAX_LOG     = 8    # Số dòng log tối đa hiển thị


# ══════════════════════════════════════════════════════════
# VÒNG LẶP MÔ PHỎNG THỜI GIAN THỰC
# ══════════════════════════════════════════════════════════
def add_log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_events.insert(0, f"[{ts}] {msg}")
    if len(st.session_state.log_events) > MAX_LOG:
        st.session_state.log_events.pop()


def render_status_html(label: str, css_class: str, psnr: float,
                        cap_mbps: float, p_w: float) -> str:
    return f"""
    <div class="user-card">
        <span class="status-badge {css_class}">{label}</span>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">PSNR</div>
                <div class="metric-value">{psnr:.1f} dB</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Capacity</div>
                <div class="metric-value">{cap_mbps:.1f} Mbps</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Công suất</div>
                <div class="metric-value">{p_w*1000:.1f} mW</div>
            </div>
        </div>
    </div>
    """


def render_power_bar(pc1, pc2, pp1, pp2, total) -> str:
    """Tạo thanh công suất HTML trực quan."""
    def pct(v): return max(0, min(100, v / total * 100)) if total > 0 else 0
    return f"""
    <div style="font-size:0.75rem; color:#5588aa; margin-bottom:4px;">Common U1</div>
    <div class="power-bar-bg"><div class="power-bar-fill bar-common"
        style="width:{pct(pc1):.1f}%">{pct(pc1):.0f}%</div></div>
    <div style="font-size:0.75rem; color:#5588aa; margin-bottom:4px;">Common U2</div>
    <div class="power-bar-bg"><div class="power-bar-fill bar-common"
        style="width:{pct(pc2):.1f}%">{pct(pc2):.0f}%</div></div>
    <div style="font-size:0.75rem; color:#5588aa; margin-bottom:4px;">Private U1</div>
    <div class="power-bar-bg"><div class="power-bar-fill bar-private"
        style="width:{pct(pp1):.1f}%">{pct(pp1):.0f}%</div></div>
    <div style="font-size:0.75rem; color:#5588aa; margin-bottom:4px;">Private U2</div>
    <div class="power-bar-bg"><div class="power-bar-fill bar-private"
        style="width:{pct(pp2):.1f}%">{pct(pp2):.0f}%</div></div>
    """


if not run_sim:
    st.info("▶ Bật công tắc **'Chạy mô phỏng'** ở thanh bên để bắt đầu.")
    st.stop()

# ── Vòng lặp vô hạn (dừng khi tắt toggle) ──────────────
for _ in range(10000):

    if not run_sim:
        break

    idx = st.session_state.frame_idx
    st.session_state.total_frames += 1

    # ── 1. Tạo frame gốc ────────────────────────────────
    orig_frame = generate_test_frame(idx, FRAME_SIZE)

    # ── 2. Mã hóa SVC ───────────────────────────────────
    bl_data, el_data, bl_bytes, el_bytes = codec.encode(orig_frame)
    # Chuyển sang Mbps (giả sử mỗi frame gửi trong 1/fps giây)
    bl_mbps = (bl_bytes * 8) / 1e6 * target_fps
    el_mbps = (el_bytes * 8) / 1e6 * target_fps

    # ── 3. Tính kênh truyền ─────────────────────────────
    h1 = path_loss(d1)
    h2 = path_loss(d2)
    H  = np.array([h1, h2])

    # ── 4. Tối ưu RSMA Min-Max ──────────────────────────
    # R_req_common đảm bảo BL của CẢ 2 user đều qua được Common
    R_req = bl_mbps * 1.0   # factor 2 vì cả 2 user đều cần BL
    result = optimize_rsma_minmax(H, P_max_watt, R_req, noise_power)

    c_common   = result["C_common"]       # Mbps
    c_priv     = result["C_private"]      # [C_priv_1, C_priv_2]
    pc_vals    = result["P_common"]       # [pc1, pc2]
    pp_vals    = result["P_private"]      # [pp1, pp2]
    opt_status = result["status"]

    # ── 5. Quyết định chất lượng video mỗi User ─────────
    user_frames  = []
    user_labels  = []
    user_classes = []
    user_psnrs   = []

    for k in range(2):
        # Kiểm tra Outage: Common không đủ cho BL
        if c_common < bl_mbps:
            # Mất kết nối hoàn toàn — hiển thị màn hình đen có nhiễu
            black = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            noisy = add_noise_overlay(black, 60)
            noisy = overlay_status(noisy, "DISCONECT", (0, 0, 255))
            user_frames.append(noisy)
            user_labels.append("DISCONECT ⛔")
            user_classes.append("status-out")
            user_psnrs.append(0.0)
            if k == 0:
                st.session_state.outage_count += 1
                add_log(f"⛔ OUTAGE! C_common={c_common:.1f}<{bl_mbps:.1f} Mbps")

        elif c_priv[k] < el_mbps:
            # Chỉ có Base Layer — ảnh mờ
            dec, _ = codec.decode(bl_data, None, FRAME_SIZE)
            dec = overlay_status(dec, "CHỈ BASE LAYER", (0, 200, 255))
            psnr = codec.compute_psnr(orig_frame, dec)
            user_frames.append(dec)
            user_labels.append(f"BL ONLY 📺  (U{k+1})")
            user_classes.append("status-sd")
            user_psnrs.append(psnr)
            if k == 1:
                st.session_state.bl_only_count += 1
                add_log(f"⚠️ User2 BL-Only: C_priv={c_priv[1]:.1f}<{el_mbps:.1f} Mbps")
        else:
            # Full HD — Base Layer + Enhancement Layer
            dec, _ = codec.decode(bl_data, el_data, FRAME_SIZE)
            dec = overlay_status(dec, "FULL HD ✓", (0, 255, 128))
            psnr = codec.compute_psnr(orig_frame, dec)
            user_frames.append(dec)
            user_labels.append(f"FULL HD 🎬  (U{k+1})")
            user_classes.append("status-hd")
            user_psnrs.append(psnr)

    # ── 6. Cập nhật giao diện ───────────────────────────

    # Video User 1
    ph_img1.image(frame_to_rgb(user_frames[0]),
                  caption=f"User 1 | d={d1}m | h={h1:.2e}",
                  use_container_width=True)
    ph_status1.markdown(
        render_status_html(user_labels[0], user_classes[0],
                           user_psnrs[0], c_priv[0],
                           float(pc_vals[0]) + float(pp_vals[0])),
        unsafe_allow_html=True
    )

    # Video User 2
    ph_img2.image(frame_to_rgb(user_frames[1]),
                  caption=f"User 2 | d={d2}m | h={h2:.2e}",
                  use_container_width=True)
    ph_status2.markdown(
        render_status_html(user_labels[1], user_classes[1],
                           user_psnrs[1], c_priv[1],
                           float(pc_vals[1]) + float(pp_vals[1])),
        unsafe_allow_html=True
    )

    # Lịch sử công suất
    st.session_state.power_history.append({
        "Frame":      idx,
        "P_common_U1 (mW)":  float(pc_vals[0]) * 1000,
        "P_common_U2 (mW)":  float(pc_vals[1]) * 1000,
        "P_private_U1 (mW)": float(pp_vals[0]) * 1000,
        "P_private_U2 (mW)": float(pp_vals[1]) * 1000,
    })
    if len(st.session_state.power_history) > MAX_HISTORY:
        st.session_state.power_history.pop(0)

    # Biểu đồ Bar Chart công suất
    df_hist = pd.DataFrame(st.session_state.power_history).set_index("Frame")
    ph_chart.bar_chart(df_hist, height=220)

    # Log sự kiện
    log_html = '<div class="log-box">' + "<br>".join(
        st.session_state.log_events
        if st.session_state.log_events
        else ["Hệ thống đang chạy bình thường..."]
    ) + "</div>"
    ph_log.markdown(log_html, unsafe_allow_html=True)

    # Thống kê tổng quan
    total = st.session_state.total_frames
    outage_pct = st.session_state.outage_count / max(total, 1) * 100
    blonly_pct = st.session_state.bl_only_count / max(total, 1) * 100

    ph_stat1.metric("📦 Frames xử lý", f"{total}")
    ph_stat2.metric("⛔ Tỷ lệ mất kết nối", f"{outage_pct:.1f}%")
    ph_stat3.metric("📺 Tỷ lệ BL-Only (U2)", f"{blonly_pct:.1f}%")
    ph_stat4.metric("⚡ Solver", opt_status[:12])

    # Tăng frame index
    st.session_state.frame_idx += 1

    # Delay để điều chỉnh fps mô phỏng
    time.sleep(1.0 / target_fps)
