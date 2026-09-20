"""
COLDGUARD AI — Cold Chain Control Tower
Prototype quản trị rủi ro chuỗi lạnh hàng xuất khẩu (xoài · cửa khẩu Lào Cai)

Luồng xử lý (theo mô hình Cold Chain Priority trong tài liệu):
Data → ColdGuard AI → Risk Score → Risk Level → Priority (P1/P2/P3)
     → Recommendation → Dashboard → Feedback

Cấu trúc file:
  1. Cấu hình & hằng số        5. Biểu đồ (Plotly)
  2. Dữ liệu mẫu               6. Giao diện (CSS + component HTML)
  3. Đọc & làm sạch dữ liệu    7. Sidebar
  4. Mô hình ColdGuard AI      8. Các tab: Control Tower / Bản đồ / Case Study / Mô hình
"""
from __future__ import annotations

import html
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ColdGuard AI · Cold Chain Control Tower",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# 1. CẤU HÌNH & HẰNG SỐ
# ══════════════════════════════════════════════════════════════
CYAN, BLUE, GREEN, AMBER, RED = "#22D3EE", "#3B82F6", "#22C55E", "#F59E0B", "#EF4444"
TEXT, MUTED, LINE = "#E6EDF7", "#8CA0BD", "#1B2B45"
VN_TZ = timezone(timedelta(hours=7))

# Risk Level → Priority (theo bảng 3.1.4 và 3.1.5 của tài liệu)
LEVEL_META = {
    "High":   {"vi": "Cao",        "priority": "P1", "color": RED,   "handle": "Ưu tiên xử lý + kho lạnh"},
    "Medium": {"vi": "Trung bình", "priority": "P2", "color": AMBER, "handle": "Tăng giám sát + bố trí bảo quản"},
    "Low":    {"vi": "Thấp",       "priority": "P3", "color": GREEN, "handle": "Bình thường"},
}

# Trọng số các nhóm yếu tố. ĐÂY LÀ SỐ MINH HỌA lấy từ tài liệu (36/31/18/15).
# Khi có kết quả huấn luyện thật, thay bằng mức độ ảnh hưởng (feature importance) từ dữ liệu của nhóm.
WEIGHTS = {
    "Nhiệt độ": 0.36,
    "Thời gian chờ": 0.31,
    "Hạn bảo quản": 0.18,
    "Đặc tính hàng / độ ẩm": 0.15,
}
COOLING_PENALTY = 20  # điểm cộng thêm khi hệ thống lạnh báo bất thường
CAUSES = list(WEIGHTS) + ["Hệ thống lạnh"]
CAUSE_COLORS = {
    "Nhiệt độ": CYAN,
    "Thời gian chờ": BLUE,
    "Hạn bảo quản": "#A78BFA",
    "Đặc tính hàng / độ ẩm": "#34D399",
    "Hệ thống lạnh": "#F472B6",
}

# Recommendation System: nguyên nhân chính → hành động (mục 3.1.6)
RECO = {
    "Nhiệt độ": {
        "short": "Đưa vào kho lạnh",
        "steps": ["Ưu tiên xử lý container này trước.", "Đưa vào khu vực bảo quản lạnh.",
                  "Kiểm tra hệ thống lạnh / container.", "Tăng tần suất giám sát nhiệt độ."],
    },
    "Thời gian chờ": {
        "short": "Ưu tiên thông quan",
        "steps": ["Đẩy lên đầu hàng chờ xử lý.", "Ưu tiên hoàn tất các bước thủ tục.",
                  "Hạn chế tiếp tục chờ ngoài điều kiện bảo quản phù hợp.",
                  "Theo dõi thời gian bảo quản còn lại của hàng."],
    },
    "Hạn bảo quản": {
        "short": "Tăng giám sát",
        "steps": ["Theo dõi sát thời gian bảo quản còn lại.", "Tăng tần suất kiểm tra chất lượng lô hàng.",
                  "Ưu tiên giao / xuất sớm nếu điều kiện cho phép."],
    },
    "Đặc tính hàng / độ ẩm": {
        "short": "Chỉnh độ ẩm, thông gió",
        "steps": ["Điều chỉnh độ ẩm trong container về mức phù hợp.", "Kiểm tra thông gió và độ kín của cửa container.",
                  "Kiểm tra ngẫu nhiên mẫu hàng."],
    },
    "Hệ thống lạnh": {
        "short": "Kiểm tra thiết bị lạnh",
        "steps": ["Kiểm tra máy lạnh của container ngay.", "Chuyển hàng sang kho lạnh / container dự phòng.",
                  "Kiểm tra thiết bị theo dõi nhiệt độ."],
    },
}

# ══════════════════════════════════════════════════════════════
# 2. DỮ LIỆU MẪU (mô phỏng – thay bằng file của bạn ở sidebar)
# ══════════════════════════════════════════════════════════════
DEMO = pd.DataFrame(
    [
        ["MG001", 4.2, 82, 18, "Bình thường", 220],
        ["MG002", 5.1, 85, 24, "Bình thường", 210],
        ["MG003", 6.3, 88, 31, "Bình thường", 200],
        ["MG004", 4.8, 80, 20, "Bình thường", 215],
        ["MG005", 8.9, 94, 52, "Bất thường", 150],
        ["MG006", 5.6, 87, 50, "Bình thường", 170],
        ["MG007", 7.8, 90, 40, "Bình thường", 120],
        ["MG008", 6.0, 84, 22, "Bình thường", 24],
        ["MG009", 4.9, 83, 26, "Bình thường", 205],
        ["MG010", 5.3, 91, 28, "Bất thường", 190],
    ],
    columns=["Container", "Nhiệt độ (°C)", "Độ ẩm (%)", "Thời gian chờ (giờ)", "Làm lạnh",
             "Hạn bảo quản còn lại (giờ)"],
)

# ══════════════════════════════════════════════════════════════
# 3. ĐỌC & LÀM SẠCH DỮ LIỆU  (bước "Data → tiền xử lý" trong tài liệu)
# ══════════════════════════════════════════════════════════════
REQUIRED = {"container": "Container", "temp": "Nhiệt độ (°C)", "hum": "Độ ẩm (%)",
            "wait": "Thời gian chờ (giờ)", "cool": "Làm lạnh"}
ALIASES = {
    "container": ["container", "mã container", "ma container", "mã", "id"],
    "temp": ["nhiệt độ (°c)", "nhiệt độ", "nhiet do", "temperature", "temp"],
    "hum": ["độ ẩm (%)", "độ ẩm", "do am", "humidity"],
    "wait": ["thời gian chờ (giờ)", "thời gian chờ", "thoi gian cho", "wait_hours", "wait"],
    "cool": ["làm lạnh", "trạng thái làm lạnh", "lam lanh", "cooling", "reefer"],
    "shelf": ["hạn bảo quản còn lại (giờ)", "hạn bảo quản còn lại", "hạn còn lại (giờ)", "shelf_life"],
    "req": ["nhiệt độ yêu cầu (°c)", "nhiệt độ yêu cầu", "required_temp"],
    "lat": ["vĩ độ", "lat", "latitude"],
    "lon": ["kinh độ", "lon", "lng", "longitude"],
}


def to_num(s: pd.Series) -> pd.Series:
    """Chuyển về số; chấp nhận dấu phẩy thập phân kiểu Việt Nam (4,2 → 4.2)."""
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    picked = {k: next((lookup[a] for a in al if a in lookup), None) for k, al in ALIASES.items()}
    missing = [REQUIRED[k] for k in REQUIRED if picked[k] is None]
    if missing:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing))

    df = pd.DataFrame({k: (raw[c] if c is not None else np.nan) for k, c in picked.items()})
    df = df.dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError("File không có dòng dữ liệu nào.")

    df["container"] = df["container"].fillna("N/A").astype(str).str.strip()
    for k in ("temp", "hum", "wait", "shelf", "req", "lat", "lon"):
        df[k] = to_num(df[k])
    for k in ("temp", "hum", "wait"):  # xử lý dữ liệu thiếu: điền trung vị
        df[k] = df[k].fillna(df[k].median() if df[k].notna().any() else 0.0)

    cool = df["cool"].fillna("").astype(str).str.lower()
    df["cool_bad"] = cool.str.contains("bất|bat|abnormal|lỗi|fault|error|hỏng", regex=True)

    # Không có toạ độ thật → đặt vị trí MÔ PHỎNG quanh khu vực cửa khẩu Lào Cai
    rng = np.random.default_rng(11)
    n = len(df)
    cols = int(np.ceil(np.sqrt(n * 1.6)))  # xếp lưới như bãi container, cộng nhiễu nhẹ
    i = np.arange(n)
    lat_sim = 22.5000 - (i // cols) * 0.0045 + rng.uniform(-0.0008, 0.0008, n)
    lon_sim = 103.9600 + (i % cols) * 0.0060 + rng.uniform(-0.0010, 0.0010, n)
    df["sim_pos"] = df["lat"].isna() | df["lon"].isna()
    df["lat"] = np.where(df["sim_pos"], lat_sim, df["lat"])
    df["lon"] = np.where(df["sim_pos"], lon_sim, df["lon"])
    return df


def read_upload(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        blob = file.read()
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = blob.decode("cp1258", errors="replace")  # CSV xuất từ Excel tiếng Việt cũ
        first = text.splitlines()[0] if text.strip() else ""
        sep = ";" if first.count(";") > first.count(",") else ","  # Excel VN hay dùng dấu ;
        raw = pd.read_csv(io.StringIO(text), sep=sep)
    else:
        raw = pd.read_excel(file)
    return standardize(raw)


# ══════════════════════════════════════════════════════════════
# 4. MÔ HÌNH COLDGUARD AI  (Risk Score → Level → Priority → nguyên nhân → khuyến nghị)
# ══════════════════════════════════════════════════════════════
def compute(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Mô hình chấm điểm có trọng số (rule-based) mô phỏng cách ColdGuard AI cho ra Risk Score 0–100.
    Khi có dữ liệu lịch sử, thay riêng hàm này bằng model học máy (scikit-learn...) –
    phần giao diện phía dưới không cần sửa, miễn là vẫn trả về các cột: score, level, priority, cause...
    """
    out = df.copy()
    req = out["req"].fillna(cfg["req_temp"])
    out["dev"] = out["temp"] - req
    out["shelf_eff"] = out["shelf"].fillna(cfg["shelf_total"] - out["wait"]).clip(lower=0)

    sub = pd.DataFrame({  # điểm thành phần 0–100 cho từng nhóm yếu tố
        "Nhiệt độ": np.clip(out["dev"].abs() / cfg["temp_tol"], 0, 1) * 100,
        "Thời gian chờ": np.clip(out["wait"] / cfg["wait_max"], 0, 1) * 100,
        "Hạn bảo quản": np.clip(1 - out["shelf_eff"] / cfg["shelf_total"], 0, 1) * 100,
        "Đặc tính hàng / độ ẩm": np.clip((out["hum"] - cfg["hum_ideal"]).abs() / cfg["hum_tol"], 0, 1) * 100,
    })
    contrib = sub.mul(pd.Series(WEIGHTS))  # đóng góp vào Risk Score
    contrib["Hệ thống lạnh"] = np.where(out["cool_bad"], float(COOLING_PENALTY), 0.0)

    score = contrib.sum(axis=1).clip(0, 100).round(1)
    out["score"] = score
    out["level"] = np.select([score >= cfg["high_th"], score >= cfg["med_th"]], ["High", "Medium"], default="Low")
    out["priority"] = out["level"].map({k: v["priority"] for k, v in LEVEL_META.items()})
    out["main_factor"] = contrib.idxmax(axis=1)
    out["cause"] = np.where(out["level"] == "Low", "Ổn định", out["main_factor"])
    out["action"] = [RECO[f]["short"] if lv != "Low" else "Giám sát thường quy"
                     for f, lv in zip(out["main_factor"], out["level"])]
    out = pd.concat([out, contrib.add_prefix("c_")], axis=1)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def alert_text(r: pd.Series, cfg: dict) -> str:
    f = r["main_factor"]
    if f == "Nhiệt độ":
        return f"Nhiệt độ {r['temp']:.1f}°C, lệch {r['dev']:+.1f}°C so với mức yêu cầu."
    if f == "Thời gian chờ":
        return f"Thời gian chờ {r['wait']:.0f} giờ ({r['wait'] / cfg['wait_max'] * 100:.0f}% ngưỡng cảnh báo)."
    if f == "Hạn bảo quản":
        return f"Thời gian bảo quản còn lại thấp ({r['shelf_eff']:.0f} giờ)."
    if f == "Đặc tính hàng / độ ẩm":
        return f"Độ ẩm {r['hum']:.0f}% lệch khỏi mức lý tưởng ({cfg['hum_ideal']:.0f}%)."
    return "Hệ thống lạnh báo bất thường – cần kiểm tra container / thiết bị."

# ══════════════════════════════════════════════════════════════
# 5. BIỂU ĐỒ (Plotly – nền trong suốt, tông neon trên nền tối)
# ══════════════════════════════════════════════════════════════
def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=8, b=8), showlegend=legend,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(color=MUTED)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor=LINE, linecolor=LINE)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, theme=None, config={"displayModeBar": False})


def fig_risk_bar(df: pd.DataFrame, cfg: dict) -> go.Figure:
    d = df.sort_values("score", ascending=False)
    fig = go.Figure(go.Bar(
        x=d["container"], y=d["score"],
        marker=dict(color=[LEVEL_META[l]["color"] for l in d["level"]], line=dict(width=0)),
        text=d["score"].round(0).astype(int), textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color=TEXT),
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.1f}<extra></extra>",
    ))
    fig.add_hline(y=cfg["high_th"], line=dict(color=RED, dash="dot", width=1),
                  annotation_text=f"High ≥ {cfg['high_th']}", annotation_font_color=RED, annotation_position="top right")
    fig.add_hline(y=cfg["med_th"], line=dict(color=AMBER, dash="dot", width=1),
                  annotation_text=f"Medium ≥ {cfg['med_th']}", annotation_font_color=AMBER, annotation_position="top right")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, 115], title=None)
    return style(fig, 330, legend=False)


def fig_priority_donut(df: pd.DataFrame) -> go.Figure:
    order = ["High", "Medium", "Low"]
    counts = [int((df["level"] == l).sum()) for l in order]
    fig = go.Figure(go.Pie(
        labels=[f"{LEVEL_META[l]['priority']} · {LEVEL_META[l]['vi']}" for l in order], values=counts, hole=0.7,
        marker=dict(colors=[LEVEL_META[l]["color"] for l in order], line=dict(color="#060B14", width=3)),
        textinfo="value", textfont=dict(family="JetBrains Mono, monospace", size=15, color="#FFFFFF"),
        sort=False, direction="clockwise",
    ))
    fig.add_annotation(text=f"<b style='font-size:30px'>{len(df)}</b><br><span style='color:{MUTED}'>container</span>",
                       showarrow=False, font=dict(color=TEXT))
    return style(fig, 330)


def fig_contrib(df: pd.DataFrame) -> go.Figure:
    d = df.head(12).sort_values("score")
    fig = go.Figure()
    for cause in CAUSES:
        fig.add_trace(go.Bar(
            y=d["container"], x=d[f"c_{cause}"], name=cause, orientation="h",
            marker=dict(color=CAUSE_COLORS[cause]),
            hovertemplate=f"{cause}: %{{x:.1f}} điểm<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(type="category")
    fig.update_xaxes(title="Điểm rủi ro đóng góp theo nguyên nhân", range=[0, 100])
    return style(fig, max(320, 30 * len(d) + 110))


def fig_matrix(df: pd.DataFrame, cfg: dict) -> go.Figure:
    fig = go.Figure()
    for lv in ("Low", "Medium", "High"):
        d = df[df["level"] == lv]
        if d.empty:
            continue
        m = LEVEL_META[lv]
        fig.add_trace(go.Scatter(
            x=d["wait"], y=d["dev"], mode="markers+text", text=d["container"], textposition="top center",
            textfont=dict(color=MUTED, size=10), name=f"{m['priority']} · {m['vi']}",
            marker=dict(size=12 + d["score"] * 0.22, color=m["color"], opacity=0.85,
                        line=dict(color="rgba(255,255,255,.35)", width=1)),
            customdata=np.stack([d["container"], d["score"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>Chờ: %{x:.0f} giờ<br>Lệch nhiệt: %{y:+.1f}°C"
                          "<br>Risk: %{customdata[1]:.0f}<extra></extra>",
        ))
    fig.add_vline(x=cfg["wait_max"], line=dict(color=BLUE, dash="dot", width=1))
    for y in (cfg["temp_tol"], -cfg["temp_tol"]):
        fig.add_hline(y=y, line=dict(color=CYAN, dash="dot", width=1))
    fig.update_xaxes(title="Thời gian chờ (giờ)")
    fig.update_yaxes(title="Độ lệch nhiệt độ (°C)")
    return style(fig, 360)


def fig_gauge(score: float, level: str, cfg: dict, height: int = 250) -> go.Figure:
    color = LEVEL_META[level]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(score),
        number=dict(font=dict(size=46, color=color, family="JetBrains Mono, monospace")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=color, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, cfg["med_th"]], color="rgba(34,197,94,.16)"),
                   dict(range=[cfg["med_th"], cfg["high_th"]], color="rgba(245,158,11,.16)"),
                   dict(range=[cfg["high_th"], 100], color="rgba(239,68,68,.16)")],
            threshold=dict(line=dict(color=color, width=3), thickness=0.9, value=float(score)),
        ),
    ))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=0),
                      font=dict(family="Be Vietnam Pro, sans-serif", color=TEXT))
    return fig


def fig_single_contrib(r: dict) -> go.Figure:
    vals = [float(r[f"c_{c}"]) for c in CAUSES]
    fig = go.Figure(go.Bar(
        y=CAUSES, x=vals, orientation="h", marker=dict(color=[CAUSE_COLORS[c] for c in CAUSES]),
        text=[f"{v:.1f}" for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color=TEXT),
        hovertemplate="%{y}: %{x:.1f} điểm<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Điểm đóng góp vào Risk Score", range=[0, max(40, max(vals) * 1.25)])
    return style(fig, 260, legend=False)


def fig_weights() -> go.Figure:
    names, vals = list(WEIGHTS), [v * 100 for v in WEIGHTS.values()]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=[CAUSE_COLORS[n] for n in names]),
        text=[f"{v:.0f}%" for v in vals], textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color=TEXT),
        hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 50], title="Trọng số ảnh hưởng (minh họa)")
    return style(fig, 260, legend=False)


def fig_map(df: pd.DataFrame, selected: str) -> go.Figure:
    fig = go.Figure()
    sel = df[df["container"] == selected].head(1)
    if not sel.empty:  # vòng sáng quanh container đang theo dõi
        fig.add_trace(go.Scattermap(lat=sel["lat"], lon=sel["lon"], mode="markers", hoverinfo="skip",
                                    marker=dict(size=44, color=CYAN, opacity=0.25), showlegend=False))
    high = df[df["level"] == "High"]
    if not high.empty:  # hào quang cho P1
        fig.add_trace(go.Scattermap(lat=high["lat"], lon=high["lon"], mode="markers", hoverinfo="skip",
                                    marker=dict(size=34, color=RED, opacity=0.22), showlegend=False))
    for lv in ("Low", "Medium", "High"):
        d = df[df["level"] == lv]
        if d.empty:
            continue
        m = LEVEL_META[lv]
        hover = [f"<b>{c}</b><br>Risk {s:.0f} · {m['priority']}<br>{cause}" for c, s, cause in
                 zip(d["container"], d["score"], d["cause"])]
        fig.add_trace(go.Scattermap(
            lat=d["lat"], lon=d["lon"], mode="markers+text", text=d["container"], textposition="top right",
            textfont=dict(color=TEXT, size=11), name=f"{m['priority']} · {m['vi']}",
            marker=dict(size=12 + d["score"] * 0.14, color=m["color"], opacity=0.95),
            hovertext=hover, hoverinfo="text",
        ))
    span = max(df["lat"].max() - df["lat"].min(), df["lon"].max() - df["lon"].min(), 0.005)
    zoom = float(np.clip(np.log2(360 / (span * 3)), 3, 15))
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        map=dict(style="carto-darkmatter", zoom=zoom, center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean()))),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(8,17,31,.75)", font=dict(color=TEXT, size=12)),
        font=dict(family="Be Vietnam Pro, sans-serif", color=TEXT),
    )
    return fig


# ══════════════════════════════════════════════════════════════
# 6. GIAO DIỆN: CSS + COMPONENT HTML
# ══════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--card:rgba(13,24,41,.72);--line:rgba(56,189,248,.16);--text:#E6EDF7;--muted:#8CA0BD;--cyan:#22D3EE;--blue:#3B82F6;}
.stApp{font-family:'Be Vietnam Pro',sans-serif;
 background:radial-gradient(1200px 600px at 8% -10%,rgba(34,211,238,.10),transparent 60%),
 radial-gradient(900px 500px at 100% 0%,rgba(59,130,246,.13),transparent 55%),#060B14;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1400px;}
#MainMenu,footer{visibility:hidden;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#08111F,#060B14);border-right:1px solid var(--line);}
h1,h2,h3,h4,p,label,li{font-family:'Be Vietnam Pro',sans-serif;}
.mono{font-family:'JetBrains Mono',monospace;}
.muted{color:var(--muted);}
/* HERO */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:28px 32px;border:1px solid var(--line);
 background:linear-gradient(120deg,rgba(14,165,233,.24),rgba(30,58,138,.38) 55%,rgba(8,17,31,.92));
 box-shadow:0 20px 60px rgba(2,8,23,.55);}
.hero:before{content:"";position:absolute;inset:0;
 background-image:linear-gradient(rgba(148,163,184,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.08) 1px,transparent 1px);
 background-size:38px 38px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 45%);mask-image:linear-gradient(90deg,transparent,#000 45%);}
.hero>*{position:relative;}
.hero-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;}
.live{display:inline-flex;align-items:center;gap:8px;font-size:11px;letter-spacing:.16em;font-weight:700;color:#7DD3FC;
 background:rgba(8,17,31,.55);border:1px solid var(--line);padding:5px 12px;border-radius:999px;}
.live i{width:8px;height:8px;border-radius:50%;background:#22C55E;animation:pulse 1.8s infinite;}
.clock{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted);}
.hero h1{margin:14px 0 4px;font-size:44px;font-weight:800;letter-spacing:-.02em;color:#fff;line-height:1.1;}
.hero h1 b{background:linear-gradient(90deg,#22D3EE,#60A5FA);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{margin:0;color:#B6C6DE;font-size:15px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;align-items:center;}
.chips span{font-size:12px;font-weight:600;padding:5px 12px;border-radius:999px;background:rgba(8,17,31,.6);border:1px solid var(--line);color:#CFE3FA;}
.chips em{color:var(--cyan);font-style:normal;}
@keyframes pulse{0%{box-shadow:0 0 0 0 var(--ac,rgba(34,197,94,.65));}70%{box-shadow:0 0 0 10px rgba(0,0,0,0);}100%{box-shadow:0 0 0 0 rgba(0,0,0,0);}}
/* SECTION */
.sec{display:flex;align-items:center;gap:12px;margin:30px 0 12px;flex-wrap:wrap;}
.sec .tag{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);border:1px solid rgba(34,211,238,.4);padding:2px 9px;border-radius:999px;background:rgba(34,211,238,.07);}
.sec h3{margin:0;font-size:19px;font-weight:700;color:#fff;}
.sec .sub{font-size:13px;color:var(--muted);}
.pt{font-weight:700;font-size:14px;color:#fff;margin:2px 0 6px;}
.pt span{font-weight:400;color:var(--muted);margin-left:8px;font-size:12px;}
/* KPI */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:12px;margin:20px 0 4px;}
.kpi{position:relative;overflow:hidden;border-radius:16px;padding:16px 16px 14px;background:var(--card);border:1px solid var(--line);transition:transform .2s,border-color .2s;}
.kpi:hover{transform:translateY(-2px);border-color:rgba(34,211,238,.5);}
.kpi:after{content:"";position:absolute;left:0;top:0;height:3px;width:100%;background:var(--c,#22D3EE);}
.kpi .l{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:600;}
.kpi .v{font-family:'JetBrains Mono',monospace;font-size:32px;font-weight:700;color:var(--c,#22D3EE);line-height:1.15;margin-top:6px;}
.kpi .s{font-size:12px;color:var(--muted);margin-top:2px;}
/* PILL / TABLE */
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;border:1px solid;white-space:nowrap;}
.tbl{width:100%;border-collapse:separate;border-spacing:0 7px;}
.tbl th{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);text-align:left;padding:4px 14px;font-weight:600;}
.tbl td{padding:12px 14px;background:rgba(13,24,41,.8);font-size:14px;color:var(--text);}
.tbl td:first-child{border-radius:12px 0 0 12px;border-left:3px solid var(--rc);}
.tbl td:last-child{border-radius:0 12px 12px 0;}
.tbl small{display:block;color:var(--muted);font-size:11px;font-family:'JetBrains Mono',monospace;margin-top:2px;}
.rbar{display:inline-block;width:84px;height:6px;border-radius:99px;background:rgba(148,163,184,.18);vertical-align:middle;margin-right:10px;overflow:hidden;}
.rbar span{display:block;height:100%;border-radius:99px;}
.rbar.wide{display:block;width:100%;height:8px;margin:6px 0 2px;}
/* ALERTS */
.alert{display:flex;gap:12px;align-items:flex-start;padding:12px 14px;border-radius:12px;background:rgba(13,24,41,.8);border:1px solid var(--line);border-left:3px solid var(--ac);margin-bottom:8px;font-size:13.5px;color:var(--text);}
.alert .dot{width:9px;height:9px;border-radius:50%;background:var(--ac);margin-top:6px;flex:none;box-shadow:0 0 10px var(--ac);}
.alert b{font-family:'JetBrains Mono',monospace;}
.empty{padding:18px;border-radius:12px;border:1px dashed var(--line);color:var(--muted);text-align:center;}
/* RECO CARD */
.reco{border-radius:16px;padding:18px 20px;background:linear-gradient(135deg,rgba(13,24,41,.9),rgba(8,17,31,.9));border:1px solid var(--line);border-top:3px solid var(--ac);}
.reco .h{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.reco .id{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#fff;}
.reco .c{margin:10px 0 4px;color:var(--muted);font-size:13px;}
.reco .c b{color:var(--text);}
.reco ul{margin:6px 0 0;padding-left:18px;color:var(--text);font-size:14px;line-height:1.75;}
.reco .hd{margin-top:12px;padding:8px 12px;border-radius:10px;background:rgba(34,211,238,.07);border:1px solid rgba(34,211,238,.25);font-size:13px;color:#BAE6FD;}
/* TIMELINE */
.tl{display:flex;margin:14px 0 22px;}
.tl .st{flex:1;text-align:center;position:relative;font-size:10.5px;color:var(--muted);line-height:1.3;padding:0 2px;}
.tl .st i{display:block;margin:0 auto 8px;width:14px;height:14px;border-radius:50%;background:#14233B;border:2px solid #2B4166;position:relative;z-index:1;}
.tl .st:before{content:"";position:absolute;top:6px;left:-50%;width:100%;height:2px;background:#1B2B45;}
.tl .st:first-child:before{display:none;}
.tl .done i{background:var(--cyan);border-color:var(--cyan);}
.tl .done:before{background:var(--cyan);}
.tl .done{color:#BAE6FD;}
.tl .active i{background:var(--ac);border-color:var(--ac);animation:pulse 1.8s infinite;}
.tl .active:before{background:var(--cyan);}
.tl .active{color:#fff;font-weight:700;}
/* FLOW */
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(108px,1fr));gap:22px;margin:8px 0;}
.node{position:relative;padding:14px 12px 12px;border-radius:14px;background:var(--card);border:1px solid var(--line);}
.node:not(:last-child):after{content:"→";position:absolute;right:-18px;top:50%;transform:translateY(-50%);color:var(--cyan);font-size:18px;}
.node .n{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--cyan);}
.node .t{font-weight:700;color:#fff;font-size:13px;margin:4px 0 2px;}
.node .d{font-size:11.5px;color:var(--muted);line-height:1.45;}
/* STREAMLIT WIDGETS */
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 18px;border-radius:10px 10px 0 0;font-weight:600;color:var(--muted);}
.stTabs [aria-selected="true"]{color:#22D3EE !important;background:rgba(34,211,238,.08);}
.stTabs [data-baseweb="tab-highlight"]{background:#22D3EE;}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line) !important;border-radius:16px !important;background:rgba(13,24,41,.45);}
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{background:linear-gradient(90deg,#0EA5E9,#3B82F6);color:#fff;border:0;border-radius:12px;font-weight:700;padding:.55rem 1.3rem;box-shadow:0 8px 24px rgba(14,165,233,.22);}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{filter:brightness(1.12);color:#fff;border:0;}
.brand{display:flex;align-items:center;gap:10px;margin:4px 0 14px;}
.brand .lg{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;font-size:20px;background:linear-gradient(135deg,#0EA5E9,#3B82F6);box-shadow:0 0 22px rgba(34,211,238,.35);}
.brand b{font-size:16px;color:#fff;letter-spacing:.02em;display:block;line-height:1.1;}
.brand small{color:var(--muted);font-size:11px;}
.src{font-size:12.5px;padding:10px 12px;border-radius:10px;background:rgba(34,211,238,.07);border:1px solid rgba(34,211,238,.25);color:#BAE6FD;}
.foot{margin-top:38px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;}
"""
st.markdown("<style>" + " ".join(CSS.split()) + "</style>", unsafe_allow_html=True)

esc = html.escape


def md(s: str) -> None:
    """Render HTML: gộp thành 1 dòng để Markdown không hiểu nhầm dòng trống / thụt lề là code."""
    st.markdown(" ".join(s.split()), unsafe_allow_html=True)


def section(tag: str, title: str, sub: str = "") -> None:
    md(f'<div class="sec"><span class="tag">{tag}</span><h3>{title}</h3><span class="sub">{sub}</span></div>')


def ptitle(title: str, sub: str = "") -> None:
    md(f'<div class="pt">{title}<span>{sub}</span></div>')


def pill(level: str) -> str:
    m = LEVEL_META[level]
    c = m["color"]
    return f'<span class="pill" style="color:{c};border-color:{c}66;background:{c}1F">{m["priority"]} · {m["vi"]}</span>'


def kpi(label: str, value, sub: str, color: str) -> str:
    return f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def priority_table(df: pd.DataFrame, n: int = 8) -> str:
    rows = ""
    for _, r in df.head(n).iterrows():
        c = LEVEL_META[r["level"]]["color"]
        rows += (
            f'<tr style="--rc:{c}">'
            f'<td><b class="mono">{esc(r["container"])}</b><small>{r["temp"]:.1f}°C · {r["wait"]:.0f}h chờ</small></td>'
            f'<td><span class="rbar"><span style="width:{r["score"]:.0f}%;background:{c}"></span></span>'
            f'<b class="mono" style="color:{c}">{r["score"]:.0f}</b></td>'
            f'<td>{pill(r["level"])}</td><td>{esc(r["cause"])}</td><td>{esc(r["action"])}</td></tr>'
        )
    return ('<table class="tbl"><thead><tr><th>Container</th><th>Risk</th><th>Priority</th>'
            f'<th>Nguyên nhân chính</th><th>Hành động</th></tr></thead><tbody>{rows}</tbody></table>')


def alerts_html(df: pd.DataFrame, cfg: dict, n: int = 7) -> str:
    risky = df[df["level"] != "Low"].head(n)
    if risky.empty:
        return '<div class="empty">✅ Không có cảnh báo – toàn bộ container ở mức P3.</div>'
    out = ""
    for _, r in risky.iterrows():
        c = LEVEL_META[r["level"]]["color"]
        out += (f'<div class="alert" style="--ac:{c}"><span class="dot"></span>'
                f'<div><b style="color:{c}">{esc(r["container"])}</b> · {esc(alert_text(r, cfg))}</div></div>')
    return out


def reco_card(r) -> str:
    c = LEVEL_META[r["level"]]["color"]
    steps = RECO[r["main_factor"]]["steps"] if r["level"] != "Low" else [
        "Tiếp tục giám sát theo quy trình chuẩn.", "Ghi nhận nhiệt độ / độ ẩm định kỳ."]
    li = "".join(f"<li>{esc(s)}</li>" for s in steps)
    return (f'<div class="reco" style="--ac:{c}"><div class="h"><span class="id">{esc(str(r["container"]))}</span>'
            f'{pill(r["level"])}<span class="muted mono">Risk {float(r["score"]):.0f}</span></div>'
            f'<div class="c">Nguyên nhân chính: <b>{esc(str(r["cause"]))}</b></div><ul>{li}</ul>'
            f'<div class="hd">Xử lý theo mức ưu tiên: <b>{LEVEL_META[r["level"]]["handle"]}</b></div></div>')

# ══════════════════════════════════════════════════════════════
# 7. SIDEBAR: dữ liệu + cấu hình mô hình
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    md('<div class="brand"><div class="lg">❄️</div><div><b>COLDGUARD AI</b><small>Cold Chain Control Tower</small></div></div>')
    st.markdown("**📁 Dữ liệu**")
    upload = st.file_uploader("Tải file CSV hoặc Excel", type=["csv", "xlsx"], label_visibility="collapsed")
    st.download_button("⬇️ Tải file dữ liệu mẫu", DEMO.to_csv(index=False).encode("utf-8-sig"),
                       file_name="coldguard_du_lieu_mau.csv", mime="text/csv")

    with st.expander("⚙️ Cấu hình mô hình"):
        req_temp = st.number_input("Nhiệt độ yêu cầu (°C)", value=5.0, step=0.5)
        temp_tol = st.number_input("Độ lệch nhiệt tối đa (°C)", value=3.0, min_value=0.5, step=0.5,
                                   help="Lệch bằng mức này → điểm nhiệt độ đạt 100.")
        wait_max = st.number_input("Ngưỡng thời gian chờ (giờ)", value=48, min_value=1, step=1)
        shelf_total = st.number_input("Tổng hạn bảo quản (giờ)", value=240, min_value=1, step=10)
        hum_ideal = st.number_input("Độ ẩm lý tưởng (%)", value=85, step=1)
        hum_tol = st.number_input("Độ lệch ẩm tối đa (%)", value=15, min_value=1, step=1)
        med_th, high_th = st.slider("Ngưỡng Medium / High", 0, 100, (40, 70),
                                    help="Mặc định theo tài liệu: 0–39 Low · 40–69 Medium · 70–100 High")

cfg = dict(req_temp=req_temp, temp_tol=temp_tol, wait_max=wait_max, shelf_total=shelf_total,
           hum_ideal=hum_ideal, hum_tol=hum_tol, med_th=med_th, high_th=high_th)

source = "Dữ liệu mô phỏng"
try:
    if upload is not None:
        data = read_upload(upload)
        source = f"File: {upload.name}"
    else:
        data = standardize(DEMO)
except Exception as e:  # file sai định dạng → quay về dữ liệu mẫu và báo lỗi
    st.sidebar.error(f"Không đọc được file: {e}")
    data = standardize(DEMO)

scored = compute(data, cfg)

with st.sidebar:
    md(f'<div class="src">● {esc(source)} · {len(scored)} container</div>')
    st.markdown("**🎯 Container đang theo dõi**")
    selected = st.selectbox("Container", scored["container"].tolist(), label_visibility="collapsed")

sel_row = scored[scored["container"] == selected].iloc[0]

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
now = datetime.now(VN_TZ).strftime("%d/%m/%Y · %H:%M")
md(f"""
<div class="hero">
  <div class="hero-top">
    <span class="live"><i></i>LIVE · COLD CHAIN CONTROL TOWER</span>
    <span class="clock">CỬA KHẨU LÀO CAI · {now} (GMT+7)</span>
  </div>
  <h1>❄️ COLDGUARD <b>AI</b></h1>
  <p>AI-powered Cold Chain Risk Management — giám sát, dự báo rủi ro và ưu tiên xử lý container hàng xuất khẩu.</p>
  <div class="chips"><span>Data</span><em>→</em><span>ColdGuard AI</span><em>→</em><span>Risk Score</span><em>→</em>
  <span>Cold Chain Priority</span><em>→</em><span>Recommendation</span><em>→</em><span>Dashboard</span></div>
</div>
""")

tab1, tab2, tab3, tab4 = st.tabs(["🛰️ CONTROL TOWER", "🗺️ BẢN ĐỒ & HÀNH TRÌNH", "🥭 CASE STUDY", "🧠 MÔ HÌNH & FEEDBACK"])

# ══════════════════════════════════════════════════════════════
# TAB 1 – CONTROL TOWER (Dashboard 3 khu vực theo tài liệu 3.1.7)
# ══════════════════════════════════════════════════════════════
with tab1:
    n_p1, n_p2, n_p3 = (int((scored["level"] == l).sum()) for l in ("High", "Medium", "Low"))
    n_alert = n_p1 + n_p2
    md('<div class="kpis">' + "".join([
        kpi("Tổng container", len(scored), "đang được giám sát", CYAN),
        kpi("P1 · Ưu tiên cao", n_p1, "xử lý ngay", RED),
        kpi("P2 · Trung bình", n_p2, "tăng giám sát", AMBER),
        kpi("P3 · Bình thường", n_p3, "theo dõi thường quy", GREEN),
        kpi("Cảnh báo", n_alert, "P1 + P2", "#F472B6"),
        kpi("Risk Score TB", f"{scored['score'].mean():.1f}", "thang 0–100", BLUE),
    ]) + "</div>")

    section("01", "Tổng quan rủi ro", "Risk Score và cơ cấu mức ưu tiên")
    c1, c2 = st.columns([3, 2])
    with c1, st.container(border=True):
        ptitle("Risk Score theo container", "cao hơn = nguy cơ suy giảm chất lượng lớn hơn")
        show(fig_risk_bar(scored, cfg), "bar")
    with c2, st.container(border=True):
        ptitle("Cơ cấu Cold Chain Priority")
        show(fig_priority_donut(scored), "donut")

    section("02", "Container cần chú ý", "Xếp hạng theo Risk Score — kèm nguyên nhân chính và hành động")
    md(priority_table(scored))
    if n_p1:
        top = scored.iloc[0]
        st.warning(f"Container cần ưu tiên xử lý trước: **{top['container']}** — Risk Score {top['score']:.0f} · "
                   f"{top['priority']} · nguyên nhân chính: {top['cause']}.", icon="⚠️")

    section("03", "Cảnh báo & khuyến nghị", "Nhìn vào đây để biết container nào cần hành động ngay")
    c3, c4 = st.columns([2, 3])
    with c3:
        ptitle("Cảnh báo đang hoạt động")
        md(alerts_html(scored, cfg))
    with c4:
        ptitle("Khuyến nghị cho container đang theo dõi", "chọn ở thanh bên trái")
        md(reco_card(sel_row))

    section("04", "Phân tích nguyên nhân", "Risk Score cho biết “nguy hiểm đến đâu”; nguyên nhân cho biết “nguy hiểm vì cái gì”")
    c5, c6 = st.columns(2)
    with c5, st.container(border=True):
        ptitle("Đóng góp của từng nguyên nhân", "top 12 container")
        show(fig_contrib(scored), "contrib")
    with c6, st.container(border=True):
        ptitle("Ma trận rủi ro", "thời gian chờ × độ lệch nhiệt · kích thước = Risk Score")
        show(fig_matrix(scored, cfg), "matrix")

    with st.expander("📋 Dữ liệu chi tiết & tải kết quả"):
        view = scored[["container", "temp", "hum", "wait", "shelf_eff", "cool_bad", "score", "level", "priority", "cause", "action"]].rename(columns={
            "container": "Container", "temp": "Nhiệt độ (°C)", "hum": "Độ ẩm (%)", "wait": "Chờ (giờ)",
            "shelf_eff": "Hạn còn lại (giờ)", "cool_bad": "Lạnh bất thường", "score": "Risk Score",
            "level": "Risk Level", "priority": "Priority", "cause": "Nguyên nhân chính", "action": "Hành động"})
        st.dataframe(view, hide_index=True)
        st.download_button("⬇️ Tải kết quả (CSV)", view.to_csv(index=False).encode("utf-8-sig"),
                           file_name="coldguard_ket_qua.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════
# TAB 2 – BẢN ĐỒ & HÀNH TRÌNH
# ══════════════════════════════════════════════════════════════
with tab2:
    section("05", "Bản đồ container", "Quy mô: khu vực chờ tại cửa khẩu Lào Cai")
    m1, m2 = st.columns([3, 2])
    with m1, st.container(border=True):
        show(fig_map(scored, selected), "map")
        if scored["sim_pos"].any():
            st.caption("Vị trí container trên bản đồ là **mô phỏng**. Thêm cột “Vĩ độ” và “Kinh độ” vào file dữ liệu để hiển thị vị trí thật.")
    with m2:
        lc = LEVEL_META[sel_row["level"]]["color"]
        with st.container(border=True):
            md(f'<div class="h" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">'
               f'<span class="mono" style="font-size:22px;font-weight:700;color:#fff">{esc(selected)}</span>{pill(sel_row["level"])}</div>')
            show(fig_gauge(sel_row["score"], sel_row["level"], cfg), "gauge_sel")
            pct = min(sel_row["wait"] / cfg["wait_max"], 1) * 100
            md(f'<div class="muted" style="font-size:12px">Thời gian chờ: <b class="mono" style="color:#fff">{sel_row["wait"]:.0f}h</b> / ngưỡng {cfg["wait_max"]:.0f}h</div>'
               f'<div class="rbar wide"><span style="width:{pct:.0f}%;background:{lc}"></span></div>')
            k1, k2, k3 = st.columns(3)
            k1.metric("Nhiệt độ", f"{sel_row['temp']:.1f}°C", f"{sel_row['dev']:+.1f}°C", delta_color="inverse")
            k2.metric("Độ ẩm", f"{sel_row['hum']:.0f}%")
            k3.metric("Hạn còn lại", f"{sel_row['shelf_eff']:.0f}h")

    section("06", "Hành trình lô hàng", "Giai đoạn hiện tại suy ra từ dữ liệu thời gian chờ tại cửa khẩu")
    stages = ["Thu hoạch", "Sơ chế & đóng gói", "Kho lạnh", "Vận chuyển", "Chờ tại cửa khẩu", "Thông quan"]
    current = 4
    items = "".join(
        f'<div class="st {"done" if i < current else "active" if i == current else ""}"><i></i>{s}</div>'
        for i, s in enumerate(stages))
    md(f'<div class="tl" style="--ac:{LEVEL_META[sel_row["level"]]["color"]}">{items}</div>')
    md(reco_card(sel_row))

# ══════════════════════════════════════════════════════════════
# TAB 3 – CASE STUDY: xuất khẩu xoài qua cửa khẩu Lào Cai
# ══════════════════════════════════════════════════════════════
with tab3:
    section("07", "Case Study – Xuất khẩu xoài tại cửa khẩu Lào Cai", "Nhập thông số container để ColdGuard AI mô phỏng đánh giá rủi ro")
    with st.form("cs_form"):
        a1, a2, a3, a4 = st.columns(4)
        cs_id = a1.text_input("Mã container", "MG011")
        cs_cool = a2.selectbox("Trạng thái làm lạnh", ["Bình thường", "Bất thường"])
        cs_wait = a3.number_input("Thời gian chờ (giờ)", 0.0, 500.0, 48.0, 1.0)
        cs_shelf = a4.number_input("Hạn bảo quản còn lại (giờ)", 0.0, 1000.0, 120.0, 5.0)
        b1, b2, b3, _ = st.columns(4)
        cs_temp = b1.number_input("Nhiệt độ thực tế (°C)", -10.0, 40.0, 8.0, 0.5)
        cs_req = b2.number_input("Nhiệt độ yêu cầu (°C)", -10.0, 40.0, float(cfg["req_temp"]), 0.5)
        cs_hum = b3.number_input("Độ ẩm (%)", 0.0, 100.0, 90.0, 1.0)
        go_btn = st.form_submit_button("🔍 PHÂN TÍCH RỦI RO")

    if go_btn:
        one = pd.DataFrame([dict(container=cs_id.strip() or "N/A", temp=cs_temp, hum=cs_hum, wait=cs_wait,
                                 shelf=cs_shelf, req=cs_req, cool_bad=(cs_cool == "Bất thường"))])
        st.session_state["cs_result"] = compute(one, cfg).iloc[0].to_dict()

    res = st.session_state.get("cs_result")
    if res:
        r = pd.Series(res)
        g1, g2 = st.columns([2, 3])
        with g1, st.container(border=True):
            ptitle("Cold Chain Risk Score")
            show(fig_gauge(r["score"], r["level"], cfg, 270), "cs_gauge")
        with g2:
            md(reco_card(r))
        with st.container(border=True):
            ptitle("Nguyên nhân được phát hiện", "điểm đóng góp của từng yếu tố")
            show(fig_single_contrib(res), "cs_contrib")
    else:
        md('<div class="empty">Nhập thông số ở trên rồi bấm <b>PHÂN TÍCH RỦI RO</b> để xem kết quả.</div>')

# ══════════════════════════════════════════════════════════════
# TAB 4 – MÔ HÌNH & FEEDBACK (vòng phản hồi mục 3.1.8)
# ══════════════════════════════════════════════════════════════
with tab4:
    section("08", "Luồng vận hành tổng thể", "Cold Chain Priority là lớp ra quyết định; ColdGuard AI là lớp phân tích / dự báo")
    nodes = [("01", "Data", "Hàng hóa, nhiệt độ, thời gian, môi trường, tình trạng container"),
             ("02", "ColdGuard AI", "Học từ dữ liệu lịch sử để tìm mối quan hệ với chất lượng hàng"),
             ("03", "Risk Score", "Điểm rủi ro 0–100 cho từng container"),
             ("04", "Cold Chain Priority", "Chuyển Risk Level thành mức ưu tiên P1 / P2 / P3"),
             ("05", "Khuyến nghị", "Recommendation System: hành động theo mức độ và nguyên nhân chính"),
             ("06", "Dashboard", "Tổng quan · danh sách chú ý · cảnh báo"),
             ("07", "Feedback", "Ghi nhận kết quả thực tế, đưa lại cho AI học")]
    md('<div class="flow">' + "".join(
        f'<div class="node"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d in nodes) + "</div>")

    section("09", "Quy tắc chuyển đổi", "Risk Score → Risk Level → Priority")
    t1, t2 = st.columns([3, 2])
    with t1:
        rows = "".join(
            f'<tr style="--rc:{m["color"]}"><td class="mono">{rng}</td><td>{m["vi"]}</td><td>{pill(lv)}</td><td>{m["handle"]}</td></tr>'
            for lv, rng in (("Low", f"0–{med_th - 1}"), ("Medium", f"{med_th}–{high_th - 1}"), ("High", f"{high_th}–100"))
            for m in [LEVEL_META[lv]])
        md(f'<table class="tbl"><thead><tr><th>Risk Score</th><th>Mức rủi ro</th><th>Priority</th><th>Xử lý</th></tr></thead><tbody>{rows}</tbody></table>')
        st.info("Risk Score là **điểm rủi ro**, không phải xác suất. Ví dụ 87 nghĩa là rủi ro rất cao, không phải “87% hàng sẽ hỏng”.", icon="ℹ️")
    with t2, st.container(border=True):
        ptitle("Trọng số các nhóm yếu tố", "số minh họa từ tài liệu")
        show(fig_weights(), "weights")

    section("10", "Feedback – kết quả thực tế", "Ghi nhận sau xử lý để cải thiện mô hình theo thời gian")
    st.session_state.setdefault("feedback", [])
    with st.form("fb_form", clear_on_submit=True):
        f1, f2, f3 = st.columns(3)
        fb_id = f1.selectbox("Container", scored["container"].tolist(), index=scored["container"].tolist().index(selected))
        fb_out = f2.selectbox("Kết quả thực tế của hàng", ["Đạt chất lượng", "Suy giảm nhẹ", "Hư hỏng một phần", "Hư hỏng toàn bộ"])
        fb_act = f3.selectbox("Đã làm theo khuyến nghị?", ["Có", "Một phần", "Không"])
        fb_note = st.text_input("Ghi chú")
        if st.form_submit_button("💾 Ghi nhận kết quả"):
            row = scored[scored["container"] == fb_id].iloc[0]
            st.session_state["feedback"].append({
                "Thời gian": datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M"), "Container": fb_id,
                "Risk Score lúc đánh giá": row["score"], "Priority": row["priority"],
                "Nguyên nhân chính": row["cause"], "Làm theo khuyến nghị": fb_act,
                "Kết quả thực tế": fb_out, "Ghi chú": fb_note})
            st.toast("Đã ghi nhận phản hồi.", icon="✅")

    fb = pd.DataFrame(st.session_state["feedback"])
    if fb.empty:
        md('<div class="empty">Chưa có phản hồi nào. Dữ liệu này sẽ trở thành dữ liệu mới để ColdGuard AI học lại.</div>')
    else:
        st.dataframe(fb, hide_index=True)
        st.download_button("⬇️ Tải dữ liệu feedback (CSV)", fb.to_csv(index=False).encode("utf-8-sig"),
                           file_name="coldguard_feedback.csv", mime="text/csv")
        st.caption("Lưu ý: feedback nằm trong phiên làm việc hiện tại. Hãy tải CSV về, hoặc nối Google Sheets / database để lưu lâu dài.")

md('<div class="foot">COLDGUARD AI · Prototype mô phỏng quản trị rủi ro chuỗi lạnh · Dữ liệu phục vụ mục đích nghiên cứu và trình diễn.</div>')
