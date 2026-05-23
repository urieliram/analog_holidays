"""Interactive UI for reviewing holiday and special-day labels."""
from __future__ import annotations

import calendar
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _APP_DIR.parents[1]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from analog_holidays.audit.charts import (  # noqa: E402
    LABEL_COLOR,
    LABEL_SYMBOL,
    plot_cluster_reference,
    plot_day_hourly,
)
from analog_holidays.audit.build_cache import main as rebuild_audit_cache  # noqa: E402
from analog_holidays.audit.data_loader import (  # noqa: E402
    CACHE_PATH,
    REGIONS,
    get_day_row,
    get_region_dates,
    load_audit_df,
)
from analog_holidays.audit.state_manager import (  # noqa: E402
    export_hourly_wide,
    load_audit_log,
    save_labels,
    update_label,
)
from analog_holidays.dataset_config import ACTIVE_CONFIG, format_region_label  # noqa: E402

st.set_page_config(
    page_title="Holiday Audit",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] > section[data-testid="stMain"] > div:first-child {
        padding-top: 0.5rem !important;
    }
    header[data-testid="stHeader"] {
        height: 0 !important; min-height: 0 !important;
        overflow: visible !important; background: transparent !important;
    }
    div[data-testid="stToolbar"] { display: none !important; }
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarExpandButton"],
    header[data-testid="stHeader"] button {
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 9999 !important;
    }
    div[data-testid="stTextInput"]:has(input[aria-label="_cal_jump"]) {
        display: none !important;
    }
    div[data-testid="stButton"] button {
        border-radius: 8px; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Loading audit cache...")
def _load_df() -> pd.DataFrame:
    return load_audit_df()


def _cache_to_csv(df: pd.DataFrame) -> bytes:
    export_df = export_hourly_wide(df)
    return export_df.to_csv(index=False).encode("utf-8-sig")


def _reload_cache_from_disk() -> None:
    st.cache_data.clear()
    st.session_state.df = _load_df()


def _run_destructive_cache_rebuild() -> tuple[bool, str]:
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            rebuild_audit_cache(force_rebuild=True)
    except Exception:
        buffer.write("\n")
        buffer.write(traceback.format_exc())
        return False, buffer.getvalue()

    log_text = buffer.getvalue()
    return "Cache saved" in log_text, log_text


@st.dialog("Reload base cache")
def _show_destructive_cache_dialog() -> None:
    st.error("This is a destructive action.")
    st.write(
        "A new cache will be rebuilt from the configured source and will overwrite "
        f"{CACHE_PATH.name}."
    )
    st.caption(
        "This action preserves audit_labels.parquet, but discards unsaved changes "
        "from the current session."
    )
    st.warning("Type destructive to enable reload.")

    confirm_text = st.text_input(
        "Confirmation",
        key="_destructive_cache_confirm",
        placeholder="destructive",
    )
    can_rebuild = confirm_text.strip().lower() == "destructive"

    col_cancel, col_run = st.columns(2)
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_show_destructive_cache_dialog"] = False
            st.rerun()
    with col_run:
        if st.button(
            "Reload cache",
            type="primary",
            use_container_width=True,
            disabled=not can_rebuild,
        ):
            with st.spinner("Rebuilding audit_cache.parquet..."):
                ok, log_text = _run_destructive_cache_rebuild()

            st.session_state["_show_destructive_cache_dialog"] = False
            st.session_state["_cache_reload_log"] = log_text

            if ok:
                _reload_cache_from_disk()
                st.session_state.unsaved = False
                st.session_state["_cache_reload_ok"] = True
                st.session_state["_cache_reload_error"] = ""
            else:
                st.session_state["_cache_reload_ok"] = False
                st.session_state["_cache_reload_error"] = (
                    "Could not rebuild audit_cache.parquet. Check the latest rebuild log."
                )

            st.rerun()


if not REGIONS:
    st.error(
        f"No configured regions found for dataset '{ACTIVE_CONFIG.key}'. "
        f"Check {ACTIVE_CONFIG.demand_path}."
    )
    st.stop()

if "df" not in st.session_state:
    st.session_state.df = _load_df()

if "unsaved" not in st.session_state:
    st.session_state.unsaved = False

if "_save_ok" not in st.session_state:
    st.session_state["_save_ok"] = False

if "date_idx" not in st.session_state:
    st.session_state.date_idx = 0

if "prev_region" not in st.session_state:
    st.session_state.prev_region = REGIONS[0]

if "_show_destructive_cache_dialog" not in st.session_state:
    st.session_state["_show_destructive_cache_dialog"] = False

if "_cache_reload_log" not in st.session_state:
    st.session_state["_cache_reload_log"] = ""

if st.session_state.get("_show_destructive_cache_dialog", False):
    _show_destructive_cache_dialog()

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;font-weight:bold;font-size:1rem'>🏖️ Holiday Feature</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    region = st.selectbox("Region", REGIONS, key="region_select")

    if st.session_state.prev_region != region:
        st.session_state.date_idx = 0
        st.session_state.prev_region = region

    dates = get_region_dates(st.session_state.df, region)
    if not dates:
        st.error("No data for this region.\nRun `build_cache.py` first.")
        st.stop()

    date_idx = min(st.session_state.date_idx, len(dates) - 1)
    st.session_state.date_idx = date_idx

    date_options = [str(pd.Timestamp(d).date()) for d in dates]

    _cur_ts = pd.Timestamp(dates[date_idx])
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = _cur_ts.to_period("M")

    _cal_jump = st.text_input(
        "_cal_jump",
        key="_cal_jump_input",
        label_visibility="hidden",
        value="",
    )
    _cal_jump_last = st.session_state.get("_cal_jump_last", "")
    if _cal_jump and _cal_jump != _cal_jump_last and _cal_jump in date_options:
        st.session_state["_cal_jump_last"] = _cal_jump
        st.session_state.date_idx = date_options.index(_cal_jump)
        st.session_state.cal_month = pd.Timestamp(_cal_jump).to_period("M")
        st.rerun()

    calendar_month = st.session_state.cal_month
    year_col_left, year_col_center, year_col_right = st.columns([1, 5, 1])
    with year_col_left:
        if st.button("◀", key="cal_prev_y"):
            st.session_state.cal_month = calendar_month - 12
            st.rerun()
    with year_col_center:
        st.markdown(
            f"<div style='text-align:center;font-size:1rem;font-weight:900;color:#FFFFFF'>"
            f"{pd.Period(str(calendar_month), 'M').to_timestamp().strftime('%Y')}</div>",
            unsafe_allow_html=True,
        )
    with year_col_right:
        if st.button("▶", key="cal_next_y"):
            st.session_state.cal_month = calendar_month + 12
            st.rerun()

    calendar_month = st.session_state.cal_month
    month_col_left, month_col_center, month_col_right = st.columns([1, 5, 1])
    with month_col_left:
        if st.button("◀", key="cal_prev_m"):
            st.session_state.cal_month = calendar_month - 1
            st.rerun()
    with month_col_center:
        month_label_ts = pd.Period(str(calendar_month), "M").to_timestamp()
        st.markdown(
            f"<div style='text-align:center;font-size:0.82rem;font-weight:bold'>"
            f"{calendar.month_name[month_label_ts.month]}</div>",
            unsafe_allow_html=True,
        )
    with month_col_right:
        if st.button("▶", key="cal_next_m"):
            st.session_state.cal_month = calendar_month + 1
            st.rerun()

    region_calendar_df = (
        st.session_state.df[st.session_state.df["unique_id"] == region]
        [["date", "label", "holiday_name"]].copy()
    )
    region_calendar_df["date_str"] = region_calendar_df["date"].apply(
        lambda d: str(pd.Timestamp(d).date())
    )
    date_info = {
        row["date_str"]: {
            "label": row["label"],
            "name": row.get("holiday_name") or "",
        }
        for _, row in region_calendar_df.iterrows()
    }

    calendar_year = int(str(calendar_month)[:4])
    calendar_month_number = int(str(calendar_month)[5:7])
    selected_date_str = str(_cur_ts.date())

    first_dow = calendar.monthrange(calendar_year, calendar_month_number)[0]
    days_in_month = calendar.monthrange(calendar_year, calendar_month_number)[1]

    calendar_grid = ""
    for hdr in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        calendar_grid += f'<div class="pml-cal-hdr">{hdr}</div>'
    for _ in range(first_dow):
        calendar_grid += '<div></div>'
    for day_number in range(1, days_in_month + 1):
        day_key = f"{calendar_year}-{calendar_month_number:02d}-{day_number:02d}"
        day_info = date_info.get(day_key)
        day_class = day_info["label"] if day_info else "nodata"
        selected_class = " selected" if day_key == selected_date_str else ""
        holiday_name = day_info["name"] if day_info else ""
        tooltip_html = f'<span class="tip">{holiday_name}</span>' if holiday_name else ""
        data_attr = f' data-ds="{day_key}"' if day_info else ""
        calendar_grid += (
            f'<div class="pml-cal-day {day_class}{selected_class}"{data_attr}>'
            f'{day_number}{tooltip_html}</div>'
        )

    _cal_html = f"""
<style>
  .pml-cal-grid {{
    display:grid; grid-template-columns:repeat(7,1fr); gap:2px;
    user-select:none; font-family:sans-serif; font-size:11px; margin-top:2px;
  }}
  .pml-cal-hdr {{ text-align:center; color:#999; font-weight:bold; padding:2px 0; }}
  .pml-cal-day {{
    text-align:center; padding:4px 1px; border-radius:4px;
    cursor:pointer; position:relative;
  }}
  .pml-cal-day.holiday     {{ background:#c3e6cb; color:#155724; }}
  .pml-cal-day.special_day {{ background:#f5c6cb; color:#721c24; }}
  .pml-cal-day.normal_day  {{ background:#cce5ff; color:#004085; }}
  .pml-cal-day.nodata      {{ color:#ccc; cursor:default; }}
  .pml-cal-day.selected    {{ outline:2px solid #333 !important; font-weight:bold; }}
  .pml-cal-day:hover .tip  {{ display:block; }}
  .tip {{
    display:none; position:absolute; left:50%; transform:translateX(-50%);
    bottom:115%; background:#333; color:#fff; padding:2px 7px;
    border-radius:4px; white-space:nowrap; z-index:9999; font-size:10px;
    pointer-events:none;
  }}
</style>
<div id="pml-cal-grid" class="pml-cal-grid">{calendar_grid}</div>
<script>
(function () {{
  function findInput() {{
    var scopes = [];
    var sb = document.querySelector('[data-testid="stSidebar"]');
    if (sb) scopes.push(sb);
    scopes.push(document);
    for (var s = 0; s < scopes.length; s++) {{
      var inputs = scopes[s].querySelectorAll('input');
      for (var i = 0; i < inputs.length; i++) {{
        var lbl = (inputs[i].getAttribute('aria-label') || '').toLowerCase();
        if (lbl.indexOf('cal_jump') >= 0) return inputs[i];
      }}
    }}
    return null;
  }}

  function triggerDate(ds) {{
    var inp = findInput();
    if (!inp) {{ console.warn('[pml-cal] input _cal_jump not found'); return; }}
    var tracker = inp._valueTracker;
    if (tracker) tracker.setValue('');
    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(inp, ds);
    inp.dispatchEvent(new Event('input',  {{bubbles: true, cancelable: true}}));
    inp.focus();
    ['keydown','keypress','keyup'].forEach(function(t) {{
      inp.dispatchEvent(new KeyboardEvent(t, {{
        key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
        bubbles: true, cancelable: true
      }}));
    }});
    console.log('[pml-cal] triggered date:', ds);
  }}

  function attachListener() {{
    var grid = document.getElementById('pml-cal-grid');
    if (!grid) {{ setTimeout(attachListener, 80); return; }}
    grid.addEventListener('click', function(e) {{
      var el = e.target;
      while (el && el !== grid) {{
        var ds = el.getAttribute('data-ds');
        if (ds) {{ triggerDate(ds); return; }}
        el = el.parentElement;
      }}
    }});
  }}

  attachListener();
}})();
</script>"""

    st.html(_cal_html, unsafe_allow_javascript=True)

    st.divider()

    st.caption("Label counts (this region)")
    rdf = st.session_state.df[st.session_state.df["unique_id"] == region]
    lbl_counts = rdf["label"].value_counts()
    for lbl in ("holiday", "special_day", "normal_day"):
        cnt = int(lbl_counts.get(lbl, 0))
        sym = LABEL_SYMBOL.get(lbl, "")
        col = LABEL_COLOR.get(lbl, "#888")
        st.markdown(
            f"{sym} <span style='color:{col}; font-weight:bold'>{lbl.replace('_',' ').title()}</span>: {cnt}",
            unsafe_allow_html=True,
        )

    st.divider()

    save_label = "💾 Save Changes" + (" ●" if st.session_state.unsaved else "")
    if st.button(save_label, use_container_width=True, type="primary"):
        save_labels(st.session_state.df)
        st.session_state.unsaved = False
        st.session_state["_save_ok"] = True
        st.cache_data.clear()
        st.rerun()

    if st.session_state.pop("_save_ok", False):
        st.success("✅ Saved successfully")

    if st.session_state.pop("_cache_reload_ok", False):
        st.success("✅ Cache rebuilt and reloaded from disk")

    cache_reload_error = st.session_state.pop("_cache_reload_error", "")
    if cache_reload_error:
        st.error(cache_reload_error)

    if st.session_state.unsaved:
        st.warning("⚠ Unsaved changes")

    if st.session_state.get("_cache_reload_log"):
        with st.expander("Latest cache rebuild"):
            st.text(st.session_state["_cache_reload_log"])

    with st.expander("📋 Audit Log"):
        log_df = load_audit_log()
        if log_df.empty:
            st.info("No changes recorded yet.")
        else:
            st.dataframe(
                log_df.sort_values("timestamp", ascending=False).head(50),
                use_container_width=True,
                hide_index=True,
            )

    st.download_button(
        "⬇ Export full cache to CSV",
        data=_cache_to_csv(st.session_state.df),
        file_name=f"holiday_audit_hourly_wide_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button(
        "⚠ Reload base cache",
        use_container_width=True,
        help="Destructive action: rebuilds and overwrites the base cache.",
    ):
        st.session_state["_destructive_cache_confirm"] = ""
        st.session_state["_show_destructive_cache_dialog"] = True
        st.rerun()

date_idx = min(st.session_state.date_idx, len(dates) - 1)
current_date = pd.Timestamp(dates[date_idx])
current_row = get_day_row(st.session_state.df, region, current_date)

if current_row is None:
    st.error(f"No data found for {region} on {current_date.date()}")
    st.stop()

current_label = str(current_row.get("label", "normal_day"))

region_rows = st.session_state.df[st.session_state.df["unique_id"] == region]
flagged_dates = sorted(
    pd.Timestamp(d) for d in
    region_rows[region_rows["label"].isin(["holiday", "special_day"])] ["date"].tolist()
)
dates_map = {pd.Timestamp(d): i for i, d in enumerate(dates)}

def _prev_flagged_idx() -> int:
    for flagged_date in reversed(flagged_dates):
        if flagged_date < current_date and flagged_date in dates_map:
            return dates_map[flagged_date]
    return date_idx

def _next_flagged_idx() -> int:
    for flagged_date in flagged_dates:
        if flagged_date > current_date and flagged_date in dates_map:
            return dates_map[flagged_date]
    return date_idx

flag_position = sum(1 for flagged_date in flagged_dates if flagged_date <= current_date)
flag_total = len(flagged_dates)

st.markdown("<div style='margin-top:-2.5rem'></div>", unsafe_allow_html=True)

dow_name = calendar.day_name[current_date.dayofweek]
date_label = (
        f"{calendar.day_name[current_date.dayofweek]}, "
        f"{calendar.month_name[current_date.month]} {current_date.day:02d}, {current_date.year}"
)
label_symbol = LABEL_SYMBOL.get(current_label, "")
label_text = current_label.replace("_", " ").title()
badge_bg = (
        "#2ca02c" if current_label == "holiday"
        else "#d62728" if current_label == "special_day"
        else "#1f77b4"
)

holiday_name = current_row.get("holiday_name") or ""
title_text = str(holiday_name).strip() if str(holiday_name).strip() else date_label

is_declared_holiday = bool(current_row.get("is_declared_holiday", False))
is_outlier = bool(current_row.get("is_outlier", False))
outlier_score = current_row.get("outlier_score")
outlier_score_text = f"{float(outlier_score):.4f}" if pd.notna(outlier_score) else "-"
year_value = int(current_row.get("year", 0))

day_label = f"{calendar.month_abbr[current_date.month]} {current_date.day}, {year_value}"
declared_icon = "✅" if is_declared_holiday else "❌"
declared_text = "Declared" if is_declared_holiday else "Not declared"
outlier_icon = "🔴" if is_outlier else "⚪"
outlier_text = "Outlier" if is_outlier else "Normal"

st.markdown(
    f"""<div style="margin-bottom:6px">
    <span style="font-size:1.4rem;font-weight:700">{title_text}</span>
    &nbsp;<span style="background:{badge_bg};color:white;padding:2px 10px;border-radius:12px;font-size:0.85rem;font-weight:bold">{label_symbol} {label_text}</span>
  &nbsp;
    <span style="background:#e8eaf6;color:#3949ab;padding:1px 8px;border-radius:10px;font-size:0.78rem;font-weight:600">📍 {format_region_label(region)}</span>
    <span style="background:#f3e5f5;color:#7b1fa2;padding:1px 8px;border-radius:10px;font-size:0.78rem">🚩 {flag_position}&thinsp;/&thinsp;{flag_total}</span>
    <span style="background:#fafafa;color:#555;border:1px solid #ddd;padding:1px 8px;border-radius:10px;font-size:0.78rem">📅 {day_label} · {dow_name}</span>
    <span style="background:#{'e8f5e9' if is_declared_holiday else 'fce4ec'};color:#{'2e7d32' if is_declared_holiday else 'c62828'};padding:1px 8px;border-radius:10px;font-size:0.78rem">{declared_icon} {declared_text}</span>
    <span style="background:#{'fce4ec' if is_outlier else 'f5f5f5'};color:#{'c62828' if is_outlier else '757575'};padding:1px 8px;border-radius:10px;font-size:0.78rem">{outlier_icon} {outlier_text} ({outlier_score_text})</span>
</div>""",
    unsafe_allow_html=True,
)

nav_l, center_col, nav_r = st.columns([1, 12, 1])

with center_col:
    region_display_df = st.session_state.df[
        st.session_state.df["unique_id"] == region
    ].sort_values("date")
    all_region_dates = list(region_display_df["date"].unique())
    try:
        current_index = next(
            i for i, day_value in enumerate(all_region_dates)
            if pd.Timestamp(day_value) == current_date
        )
    except StopIteration:
        current_index = 0
    context_window = all_region_dates[max(0, current_index - 4): current_index + 5]
    df_context = region_display_df[region_display_df["date"].isin(context_window)]

    fig_main = plot_day_hourly(
        current_row,
        title=f"{region}  |  {current_date.date()}",
        df_context=df_context,
    )
    st.plotly_chart(fig_main, use_container_width=True, key="main_chart")

st.html("""
<style>
#pml-sidebar-toggle {
    position: fixed;
    top: 6px;
    left: 6px;
    z-index: 99999;
    background: #6A1B9A;
    color: white;
    border: none;
    border-radius: 6px;
    width: 32px;
    height: 32px;
    font-size: 18px;
    cursor: pointer;
    line-height: 32px;
    text-align: center;
    padding: 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
#pml-sidebar-toggle:hover { background: #4a148c; }
</style>
<button id="pml-sidebar-toggle" title="Toggle sidebar">&#9776;</button>
<script>
(function() {
    document.getElementById('pml-sidebar-toggle').addEventListener('click', function() {
        var docs = [document];
        try { if (window.parent !== window) docs.push(window.parent.document); } catch(e) {}
        for (var di = 0; di < docs.length; di++) {
            var d = docs[di];
            var btn = d.querySelector('button[data-testid="stSidebarCollapseButton"]')
                   || d.querySelector('button[data-testid="stSidebarExpandButton"]')
                   || d.querySelector('[data-testid="stSidebar"] button');
            if (btn) { btn.click(); return; }
            var sb = d.querySelector('section[data-testid="stSidebar"]');
            if (sb) {
                sb.style.display = sb.style.display === 'none' ? '' : 'none';
                return;
            }
        }
    });

    function applyBtnColors() {
        document.querySelectorAll('button').forEach(function(btn) {
            var txt = btn.innerText || '';
            if (txt.indexOf('Holiday') > -1) {
                btn.style.setProperty('background-color','#2ca02c','important');
                btn.style.setProperty('color','white','important');
                btn.style.setProperty('border-color','#2ca02c','important');
            } else if (txt.indexOf('Special Day') > -1) {
                btn.style.setProperty('background-color','#d62728','important');
                btn.style.setProperty('color','white','important');
                btn.style.setProperty('border-color','#d62728','important');
            } else if (txt.indexOf('Normal Day') > -1) {
                btn.style.setProperty('background-color','#1f77b4','important');
                btn.style.setProperty('color','white','important');
                btn.style.setProperty('border-color','#1f77b4','important');
            }
        });
    }
    applyBtnColors();
    setTimeout(applyBtnColors, 300);
    var obs = new MutationObserver(applyBtnColors);
    obs.observe(document.body, {childList:true, subtree:true});
})();
</script>
""", unsafe_allow_javascript=True)

with nav_l:
    st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)
    if st.button("◀\nPREV", use_container_width=True, key="btn_prev"):
        st.session_state.date_idx = _prev_flagged_idx()
        _new_ts = pd.Timestamp(dates[st.session_state.date_idx])
        st.session_state.cal_month = _new_ts.to_period("M")
        st.rerun()

with nav_r:
    st.markdown("<div style='height:160px'></div>", unsafe_allow_html=True)
    if st.button("NEXT\n▶", use_container_width=True, key="btn_next"):
        st.session_state.date_idx = _next_flagged_idx()
        _new_ts = pd.Timestamp(dates[st.session_state.date_idx])
        st.session_state.cal_month = _new_ts.to_period("M")
        st.rerun()

st.markdown("<div style='margin-top:-18px'></div>", unsafe_allow_html=True)
fig_cluster = plot_cluster_reference(
    st.session_state.df, region,
    int(current_row["month"]), int(current_row["dow"]),
    selected_date=current_date,
)
st.plotly_chart(fig_cluster, use_container_width=True, key="cluster_chart")

_b1, _b2, _b3 = st.columns(3)
with _b1:
    is_active = current_label == "holiday"
    if st.button(
        "🟢 Holiday" + (" ✓" if is_active else ""),
        use_container_width=True,
        key="dec_holiday",
        type="primary" if is_active else "secondary",
    ):
        st.session_state.df = update_label(
            st.session_state.df, region, current_date, "holiday"
        )
        st.session_state.unsaved = True
        st.rerun()
with _b2:
    is_active = current_label == "special_day"
    if st.button(
        "🔴 Special Day" + (" ✓" if is_active else ""),
        use_container_width=True,
        key="dec_special",
        type="primary" if is_active else "secondary",
    ):
        st.session_state.df = update_label(
            st.session_state.df, region, current_date, "special_day"
        )
        st.session_state.unsaved = True
        st.rerun()
with _b3:
    is_active = current_label == "normal_day"
    if st.button(
        "🔵 Normal Day" + (" ✓" if is_active else ""),
        use_container_width=True,
        key="dec_normal",
        type="primary" if is_active else "secondary",
    ):
        st.session_state.df = update_label(
            st.session_state.df, region, current_date, "normal_day"
        )
        st.session_state.unsaved = True
        st.rerun()

is_holiday_feature = 1 if current_label == "holiday" else 0
is_special_feature = 1 if current_label == "special_day" else 0
_feat_df = pd.DataFrame(
    [
        {"feature": "special_day", **{f"{h:02d}h": is_special_feature for h in range(24)}},
        {"feature": "holiday", **{f"{h:02d}h": is_holiday_feature for h in range(24)}},
    ]
).set_index("feature")

def _color_feat(val):
    if val == 1:
        return "background-color:#d4edda; color:#155724; font-weight:bold"
    return "color:#aaa"

_feat_style = _feat_df.style.map(_color_feat).set_table_styles(
    [{"selector": "th, td", "props": [("font-size", "11px"), ("padding", "1px 5px"), ("text-align", "center")]}]
)
st.dataframe(_feat_style, use_container_width=True, height="content")

st.divider()
left_f, right_f = st.columns([3, 1])
with left_f:
    st.caption(
        "Precomputed by `identify_HOLIDAYS.py` · Metric: PEARSON · "
        "Outlier threshold: p97 · Exclude year: 2022"
    )
with right_f:
    total_edited = len(load_audit_log())
    st.caption(f"Total audit log entries: {total_edited}")
