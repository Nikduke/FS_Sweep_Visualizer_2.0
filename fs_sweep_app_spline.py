import os
import hashlib
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


# ---- Page config ----
st.set_page_config(page_title="FS Sweep Visualizer (Spline)", layout="wide")

# ---- Layout constants ----
# Keep plot area height stable by:
# - disabling Plotly margin auto-expansion (legend won't steal plot space)
# - measuring legend height in the browser and relayouting total figure height
# - fixing the figure width for deterministic report exports
DEFAULT_FIGURE_WIDTH_PX = 1400  # Default figure width (px) when auto-width is disabled.
TOP_MARGIN_PX = 40  # Top margin (px); room for title/toolbar while keeping plot-area height stable.
BOTTOM_AXIS_PX = 60  # Bottom margin reserved for x-axis title/ticks (px); also defines plot-to-legend vertical gap.
LEFT_MARGIN_PX = 60  # Left margin (px); room for y-axis title and tick labels.
RIGHT_MARGIN_PX = 20  # Right margin (px); small breathing room to avoid clipping.
LEGEND_PADDING_PX = 18  # Extra padding (px) below legend to avoid clipping in exports.


def _clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(val)))


# ---- CSS ----
def _inject_bold_tick_css():
    st.markdown(
        """
        <style>
        .plotly .xtick text, .plotly .ytick text,
        .plotly .scene .xtick text, .plotly .scene .ytick text {
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---- Data loading ----
@st.cache_data(show_spinner=False)
def load_fs_sweep_xlsx(path_or_buf) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    xls = pd.ExcelFile(path_or_buf)
    for name in ["R1", "X1", "R0", "X0"]:
        if name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=name)
            # Frequency column normalization
            freq_col = None
            for c in df.columns:
                c_norm = str(c).strip().lower().replace(" ", "")
                if c_norm in ["frequency(hz)", "frequencyhz", "frequency_"]:
                    freq_col = c
                    break
                if str(c).strip().lower() in ["frequency (hz)", "frequency"]:
                    freq_col = c
                    break
            if freq_col is None:
                if "Frequency (Hz)" in df.columns:
                    freq_col = "Frequency (Hz)"
                else:
                    raise ValueError(f"Sheet '{name}' missing 'Frequency (Hz)' column")
            df = df.rename(columns={freq_col: "Frequency (Hz)"})
            df["Frequency (Hz)"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
            df = df.dropna(subset=["Frequency (Hz)"])
            dfs[name] = df
    return dfs


def list_case_columns(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    return [c for c in df.columns if c != "Frequency (Hz)"]


def split_case_location(name: str) -> Tuple[str, Optional[str]]:
    if "__" in str(name):
        base, loc = str(name).split("__", 1)
        loc = loc if loc else None
        return base, loc
    return str(name), None


def display_case_name(name: str) -> str:
    base, _ = split_case_location(name)
    return base


def split_case_parts(cases: List[str]) -> Tuple[List[List[str]], List[str]]:
    if not cases:
        return [], []
    temp_parts: List[Tuple[List[str], str]] = []
    max_parts = 0
    for name in cases:
        base_name, location = split_case_location(name)
        base_parts = str(base_name).split("_")
        max_parts = max(max_parts, len(base_parts))
        temp_parts.append((base_parts, location or ""))

    normalized: List[List[str]] = []
    for base_parts, location in temp_parts:
        padded = list(base_parts)
        if len(padded) < max_parts:
            padded.extend([""] * (max_parts - len(padded)))
        padded.append(location or "")
        normalized.append(padded)

    labels = [f"Case part {i+1}" for i in range(max_parts)] + ["Location"]
    return normalized, labels


def build_filters_for_case_parts(all_cases: List[str]) -> List[str]:
    st.sidebar.header("Case Filters")
    if not all_cases:
        return []
    parts_matrix, part_labels = split_case_parts(all_cases)
    if not part_labels:
        return all_cases

    reset_all = st.sidebar.button("Reset all filters", key="case_filters_reset_all")
    keep = np.ones(len(all_cases), dtype=bool)
    for i, label in enumerate(part_labels):
        col_key = f"case_part_{i+1}_ms"
        options = sorted({parts_matrix[j][i] for j in range(len(all_cases))})
        options_disp = [o if o != "" else "<empty>" for o in options]

        # init/sanitize
        if reset_all or col_key not in st.session_state:
            st.session_state[col_key] = list(options_disp)
        else:
            st.session_state[col_key] = [v for v in st.session_state[col_key] if v in options_disp]

        st.sidebar.markdown(label)
        c1, _c2 = st.sidebar.columns([1, 1])

        checkbox_keys: Dict[str, str] = {}
        for o in options_disp:
            h = hashlib.sha1(o.encode("utf-8")).hexdigest()[:12]
            checkbox_keys[o] = f"{col_key}__opt__{h}"

        if c1.button("Select all", key=f"{col_key}_all"):
            st.session_state[col_key] = list(options_disp)
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = True

        if _c2.button("Clear all", key=f"{col_key}_none"):
            st.session_state[col_key] = []
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = False

        if reset_all:
            for o in options_disp:
                st.session_state[checkbox_keys[o]] = True

        selected_disp: List[str] = []
        selected_set = set(st.session_state[col_key])
        cols = st.sidebar.columns(2)
        for idx, o in enumerate(options_disp):
            opt_key = checkbox_keys[o]
            if opt_key not in st.session_state:
                st.session_state[opt_key] = o in selected_set
            checked = cols[idx % 2].checkbox(o, key=opt_key)
            if checked:
                selected_disp.append(o)
        st.session_state[col_key] = selected_disp

        if i < len(part_labels) - 1:
            st.sidebar.markdown("---")

        selected_raw = ["" if s == "<empty>" else s for s in selected_disp]
        if 0 < len(selected_raw) < len(options):
            mask_i = np.array([parts_matrix[j][i] in selected_raw for j in range(len(all_cases))])
            keep &= mask_i
        if len(selected_raw) == 0:
            keep &= False
    return [c for c, k in zip(all_cases, keep) if k]


def compute_common_n_range(f_series: List[pd.Series], f_base: float) -> Tuple[float, float]:
    vals: List[float] = []
    for s in f_series:
        if s is None:
            continue
        v = pd.to_numeric(s, errors="coerce").dropna()
        if not v.empty:
            vals.extend([v.min() / f_base, v.max() / f_base])
    if not vals:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return (0.0, 1.0) if (not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi) else (lo, hi)


def add_harmonic_lines(fig: go.Figure, n_min: float, n_max: float, f_base: float, show_markers: bool, bin_width_hz: float):
    if not show_markers and (bin_width_hz is None or bin_width_hz <= 0):
        return
    shapes = []
    k_start = max(1, int(np.floor(n_min)))
    k_end = int(np.ceil(n_max))
    for k in range(k_start, k_end + 1):
        if show_markers:
            shapes.append(dict(type="line", xref="x", yref="paper", x0=k, x1=k, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.3)", width=1.5)))
        if bin_width_hz and bin_width_hz > 0:
            dn = (bin_width_hz / (2.0 * f_base))
            for edge in (k - dn, k + dn):
                shapes.append(dict(type="line", xref="x", yref="paper", x0=edge, x1=edge, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot")))
    fig.update_layout(shapes=fig.layout.shapes + tuple(shapes) if fig.layout.shapes else tuple(shapes))


def make_spline_traces(
    df: pd.DataFrame,
    cases: List[str],
    f_base: float,
    y_title: str,
    smooth: float,
    strip_location_suffix: bool,
    case_colors: Dict[str, str],
) -> Tuple[List[go.Scatter], Optional[pd.Series]]:
    if df is None:
        return [], None
    f = df["Frequency (Hz)"]
    n = f / f_base
    traces: List[go.Scatter] = []
    for idx, case in enumerate(cases):
        if case not in df.columns:
            continue
        y = pd.to_numeric(df[case], errors="coerce")
        cd = np.column_stack([f.values])
        color = case_colors.get(case)
        traces.append(
            go.Scatter(
                x=n,
                y=y,
                customdata=cd,
                mode="lines",
                name=display_case_name(case) if strip_location_suffix else str(case),
                line=dict(
                    shape="spline",
                    smoothing=float(smooth),
                    simplify=False,
                    color=color,
                ),
                hovertemplate=(
                    "Case=%{fullData.name}<br>n=%{x:.3f}<br>f=%{customdata[0]:.1f} Hz" + f"<br>{y_title}=%{{y}}<extra></extra>"
                ),
            )
        )
    return traces, f


def apply_common_layout(
    fig: go.Figure,
    plot_height: int,
    y_title: str,
    legend_entrywidth: int,
    use_auto_width: bool,
    figure_width_px: int,
):
    total_height = int(plot_height) + TOP_MARGIN_PX + BOTTOM_AXIS_PX + LEGEND_PADDING_PX
    legend_y = -(float(BOTTOM_AXIS_PX) + float(LEGEND_PADDING_PX)) / float(max(1, int(plot_height)))
    fig.update_layout(
        autosize=bool(use_auto_width),
        height=total_height,
        margin=dict(
            l=LEFT_MARGIN_PX,
            r=RIGHT_MARGIN_PX,
            t=TOP_MARGIN_PX,
            b=BOTTOM_AXIS_PX + LEGEND_PADDING_PX,
        ),
        margin_autoexpand=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            entrywidth=int(legend_entrywidth),
            entrywidthmode="pixels",
        ),
    )
    if not use_auto_width:
        fig.update_layout(width=int(figure_width_px), autosize=False)
    fig.update_xaxes(title_text="Harmonic number n = f / f_base", tick0=1, dtick=1)
    fig.update_yaxes(title_text=y_title)


def build_plot_spline(df: Optional[pd.DataFrame], cases: List[str], f_base: float, plot_height: int, y_title: str,
                      smooth: float, legend_entrywidth: int, strip_location_suffix: bool,
                      use_auto_width: bool, figure_width_px: int, case_colors: Dict[str, str]
                      ) -> Tuple[go.Figure, Optional[pd.Series]]:
    fig = go.Figure()
    traces, f_series = make_spline_traces(df, cases, f_base, y_title, smooth, strip_location_suffix, case_colors)
    for tr in traces:
        fig.add_trace(tr)
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, use_auto_width, figure_width_px)
    return fig, f_series


def build_x_over_r_spline(df_r: Optional[pd.DataFrame], df_x: Optional[pd.DataFrame], cases: List[str], f_base: float,
                          plot_height: int, seq_label: str, smooth: float, legend_entrywidth: int,
                          strip_location_suffix: bool, use_auto_width: bool, figure_width_px: int,
                          case_colors: Dict[str, str]
                          ) -> Tuple[go.Figure, Optional[pd.Series], int, int]:
    fig = go.Figure()
    xr_dropped = 0
    xr_total = 0
    f_series = None
    eps = 1e-9
    if df_r is not None and df_x is not None:
        both = [c for c in cases if c in df_r.columns and c in df_x.columns]
        f_series = df_r["Frequency (Hz)"]
        n = f_series / f_base
        for case in both:
            r = pd.to_numeric(df_r[case], errors="coerce")
            x = pd.to_numeric(df_x[case], errors="coerce")
            denom_ok = r.abs() >= eps
            y = pd.Series(np.where(denom_ok, x / r, np.nan))
            xr_dropped += int((~denom_ok | r.isna() | x.isna()).sum())
            xr_total += int(len(r))
            cd = np.column_stack([f_series.values])
            color = case_colors.get(case)
            fig.add_trace(
                go.Scatter(
                    x=n,
                    y=y,
                    customdata=cd,
                    mode="lines",
                    name=display_case_name(case) if strip_location_suffix else str(case),
                    line=dict(shape="spline", smoothing=float(smooth), simplify=False, color=color),
                    hovertemplate=(
                        "Case=%{fullData.name}<br>n=%{x:.3f}<br>f=%{customdata[0]:.1f} Hz<br>X/R=%{y}<extra></extra>"
                    ),
                )
            )
    y_title = "X1/R1 (unitless)" if seq_label == "Positive" else "X0/R0 (unitless)"
    apply_common_layout(fig, plot_height, y_title, legend_entrywidth, use_auto_width, figure_width_px)
    return fig, f_series, xr_dropped, xr_total


def render_plotly_with_auto_legend(figs: List[go.Figure], config: dict, key: str):
    figs_json = [f.to_plotly_json() for f in figs]
    payload = {
        "figs": figs_json,
        "config": config,
        "spacing": 24,
    }
    payload_json = json.dumps(payload, cls=PlotlyJSONEncoder)

    # Note: `components.html` renders inside an iframe. We adjust the iframe height by posting
    # `streamlit:setFrameHeight` messages after computing the final Plotly heights.
    html = f"""
    <div id="root" style="font-family: sans-serif; color: #666;">Loading plots…</div>
    <script type="application/json" id="payload">{payload_json}</script>
    <script>
      const payload = JSON.parse(document.getElementById("payload").textContent);
      const root = document.getElementById("root");
      const spacing = payload.spacing || 0;
      const cfg = payload.config || {{}};

      function ensurePlotlyLoaded() {{
        if (window.Plotly) return Promise.resolve(window.Plotly);
        return new Promise((resolve, reject) => {{
          const script = document.createElement("script");
          script.src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
          script.async = true;
          script.onload = () => resolve(window.Plotly);
          script.onerror = () => reject(new Error("Failed to load Plotly from CDN"));
          document.head.appendChild(script);
        }});
      }}

      function setFrameHeight(px) {{
        window.parent.postMessage({{
          isStreamlitMessage: true,
          type: "streamlit:setFrameHeight",
          height: px
        }}, "*");
      }}

      function measureLegendHeight(gd) {{
        const legend = gd.querySelector(".legend");
        if (!legend) return 0;
        const rect = legend.getBoundingClientRect();
        return Math.ceil(rect.height || 0);
      }}

      function computeDesired(gd) {{
        const legendH = measureLegendHeight(gd);
        if (gd.__baseHeight === undefined) {{
          gd.__baseHeight = (gd.layout && gd.layout.height) ? gd.layout.height : 0;
          gd.__baseBottom = (gd.layout && gd.layout.margin && gd.layout.margin.b) ? gd.layout.margin.b : 0;
        }}
        const bottom = gd.__baseBottom + legendH;
        const totalH = gd.__baseHeight + legendH;
        return {{ bottom, totalH }};
      }}

      function applyAutoLegendSize(gd) {{
        const {{ bottom, totalH }} = computeDesired(gd);
        const currentH = (gd._fullLayout && gd._fullLayout.height) ? gd._fullLayout.height : (gd.layout && gd.layout.height) || 0;
        const currentB = (gd._fullLayout && gd._fullLayout.margin) ? gd._fullLayout.margin.b : (gd.layout && gd.layout.margin && gd.layout.margin.b) || 0;
        if (Math.abs(currentH - totalH) <= 1 && Math.abs(currentB - bottom) <= 1) return Promise.resolve();
        return Plotly.relayout(gd, {{
          "height": totalH,
          "margin.b": bottom
        }});
      }}

      async function renderAll() {{
        let Plotly;
        try {{
          Plotly = await ensurePlotlyLoaded();
        }} catch (e) {{
          root.innerHTML =
            "<div style='color:#b00020'>Plotly could not be loaded inside the component iframe.</div>" +
            "<div style='color:#666'>If you are offline or your network blocks cdn.plot.ly, we need an offline loader fallback.</div>";
          return;
        }}

        root.innerHTML = "";
        const plots = [];
        for (let i = 0; i < payload.figs.length; i++) {{
          const container = document.createElement("div");
          container.id = "plot-" + i;
          container.style.marginBottom = (i === payload.figs.length - 1) ? "0px" : (spacing + "px");
          root.appendChild(container);

          const fig = payload.figs[i];
          await Plotly.newPlot(container, fig.data, fig.layout, cfg);
          plots.push(container);
        }}

        // Apply sizing twice (Plotly may refine layout after first relayout).
        for (const gd of plots) {{
          await applyAutoLegendSize(gd);
        }}
        for (const gd of plots) {{
          await applyAutoLegendSize(gd);
        }}

        // Resize iframe to fit all plots.
        let total = 0;
        for (const gd of plots) {{
          const rect = gd.getBoundingClientRect();
          total += Math.ceil(rect.height || 0);
        }}
        total += Math.max(0, plots.length - 1) * spacing;
        setFrameHeight(total + 4);
      }}

      renderAll();
    </script>
    """

    # Provide a reasonable fallback height + scrolling in case `streamlit:setFrameHeight` messaging is blocked.
    fallback_height = max(400, int(sum((f.layout.height or 0) for f in figs)) + (len(figs) - 1) * 24)
    components.html(html, height=fallback_height, scrolling=True)


def main():
    st.title("FS Sweep Visualizer (Spline)")
    _inject_bold_tick_css()

    # Data source
    default_path = "FS_sweep.xlsx"
    st.sidebar.header("Data Source")
    up = st.sidebar.file_uploader("Upload Excel", type=["xlsx"], help="If empty, loads 'FS_sweep.xlsx' from this folder.")
    try:
        if up is not None:
            data = load_fs_sweep_xlsx(up)
        elif os.path.exists(default_path):
            data = load_fs_sweep_xlsx(default_path)
            st.sidebar.info(f"Loaded local file: {default_path}")
        else:
            st.warning("Upload an Excel file or place 'FS_sweep.xlsx' here.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        st.stop()

    # Controls
    st.sidebar.header("Controls")
    seq_label = st.sidebar.radio("Sequence", ["Positive", "Zero"], index=0)
    seq = ("R1", "X1") if seq_label == "Positive" else ("R0", "X0")
    base_label = st.sidebar.radio("Base frequency", ["50 Hz", "60 Hz"], index=0)
    f_base = 50.0 if base_label.startswith("50") else 60.0
    plot_height = st.sidebar.slider("Plot area height (px)", min_value=100, max_value=1000, value=400, step=25)
    use_auto_width = st.sidebar.checkbox("Auto width (fit container)", value=True)
    figure_width_px = DEFAULT_FIGURE_WIDTH_PX
    if not use_auto_width:
        figure_width_px = st.sidebar.slider("Figure width (px)", min_value=800, max_value=2200, value=DEFAULT_FIGURE_WIDTH_PX, step=50)
    smooth = st.sidebar.slider("Spline smoothing", min_value=0.0, max_value=1.3, value=0.6, step=0.05)

    # Legend/Export controls
    st.sidebar.header("Legend & Export")
    auto_legend_entrywidth = st.sidebar.checkbox("Auto legend column width", value=True)
    legend_entrywidth = 180
    if not auto_legend_entrywidth:
        legend_entrywidth = st.sidebar.slider("Legend column width (px)", min_value=50, max_value=300, value=180, step=10)
    download_config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "plot",
            "scale": 4,
        }
    }

    # Cases / filters
    df_r = data.get(seq[0])
    df_x = data.get(seq[1])
    if df_r is None and df_x is None:
        st.error(f"Missing sheets for sequence '{seq_label}' ({seq[0]}/{seq[1]}).")
        st.stop()
    all_cases = sorted(list({*list_case_columns(df_r), *list_case_columns(df_x)}))
    filtered_cases = build_filters_for_case_parts(all_cases)
    if not filtered_cases:
        st.warning("No cases after filtering. Adjust filters.")
        st.stop()

    _parts_matrix, part_labels = split_case_parts(all_cases)
    strip_location_suffix = False
    if part_labels and part_labels[-1] == "Location":
        loc_key = f"case_part_{len(part_labels)}_ms"
        selected_disp = st.session_state.get(loc_key, [])
        selected_raw = ["" if s == "<empty>" else s for s in selected_disp]
        strip_location_suffix = len(selected_raw) == 1

    if auto_legend_entrywidth:
        display_names = [display_case_name(c) if strip_location_suffix else str(c) for c in filtered_cases]
        max_len = max((len(n) for n in display_names), default=12)
        approx_char_px = 7
        base_px = 44  # symbol + padding inside a legend item
        legend_entrywidth = _clamp_int(max_len * approx_char_px + base_px, 50, 300)

    palette = pc.qualitative.Plotly or ["#1f77b4"]
    case_colors: Dict[str, str] = {c: palette[i % len(palette)] for i, c in enumerate(filtered_cases)}

    # Harmonic decorations
    show_harmonics = st.sidebar.checkbox("Show harmonic lines", value=True)
    bin_width_hz = st.sidebar.number_input("Bin width (Hz)", min_value=0.0, value=0.0, step=1.0, help="0 disables tolerance bands")

    # Build plots
    r_title = "R1 (Ω)" if seq_label == "Positive" else "R0 (Ω)"
    x_title = "X1 (Ω)" if seq_label == "Positive" else "X0 (Ω)"
    fig_r, f_r = build_plot_spline(
        df_r,
        filtered_cases,
        f_base,
        plot_height,
        r_title,
        smooth,
        legend_entrywidth,
        strip_location_suffix,
        use_auto_width,
        figure_width_px,
        case_colors,
    )
    fig_x, f_x = build_plot_spline(
        df_x,
        filtered_cases,
        f_base,
        plot_height,
        x_title,
        smooth,
        legend_entrywidth,
        strip_location_suffix,
        use_auto_width,
        figure_width_px,
        case_colors,
    )
    fig_xr, f_xr, xr_dropped, xr_total = build_x_over_r_spline(
        df_r,
        df_x,
        filtered_cases,
        f_base,
        plot_height,
        seq_label,
        smooth,
        legend_entrywidth,
        strip_location_suffix,
        use_auto_width,
        figure_width_px,
        case_colors,
    )

    f_refs = [s for s in [f_r, f_x, f_xr] if s is not None]
    n_lo, n_hi = compute_common_n_range(f_refs, f_base)
    for fig in (fig_r, fig_x, fig_xr):
        fig.update_xaxes(range=[n_lo, n_hi])
        add_harmonic_lines(fig, n_lo, n_hi, f_base, show_harmonics, bin_width_hz)

    # Render
    st.subheader(f"Sequence: {seq_label} | Base: {int(f_base)} Hz")
    if xr_total > 0 and xr_dropped > 0:
        st.caption(f"X/R: dropped {xr_dropped} of {xr_total} points where |R| < 1e-9 or data missing.")

    render_plotly_with_auto_legend([fig_x, fig_r, fig_xr], download_config, key="main")


if __name__ == "__main__":
    main()
