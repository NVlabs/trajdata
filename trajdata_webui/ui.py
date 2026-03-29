"""
trajdata Web UI – full layout: header · sidebar · main content · footer.

Sections
--------
  Dashboard  – overview cards
  Dataset    – load & stats
  Visualize  – interactive trajectory viewer
  Augment    – before/after augmentation preview
  Simulate   – simulation runner + metrics table
  Export     – precomputed cache export
"""
from __future__ import annotations

import sys
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

# ── Bokeh ──────────────────────────────────────────────────────────────────
from bokeh.layouts import column, row
from bokeh.models import (
    Button, CheckboxGroup, ColumnDataSource, CustomJS,
    DataTable, Div, MultiLine, RangeSlider, Select,
    Slider, TableColumn, TextInput, Toggle, SVGIcon
)
from bokeh.plotting import figure

# ── trajdata ───────────────────────────────────────────────────────────────
from trajdata_webui.app_state import AppState
from trajdata_webui.backend.dataset_loader import (
    AVAILABLE_SPLITS, build_augmentations, load_dataset,
)
from trajdata_webui.backend.stats_computer import compute_stats
from trajdata_webui.backend.traj_renderer import batch_to_sources, sample_to_batch
from trajdata_webui.backend.aug_preview import compute_preview
from trajdata_webui.backend.sim_runner import run_simulation

_pool = ThreadPoolExecutor(max_workers=2)

# ── Color Palettes & UI Strings ─────────────────────────────────────────────
THEMES = {
    "dark": dict(
        bg        = "#030303",
        surface   = "#0c0c0e",
        surface2  = "#111114",
        surface3  = "#1a1a1f",
        border    = "#1f1f23",
        accent    = "#2563eb",  # Premium Blue
        accent2   = "#3b82f6",
        success   = "#10b981",
        warning   = "#f59e0b",
        danger    = "#ef4444",
        text      = "#f8fafc",
        muted     = "#94a3b8",
        plot_bg   = "#0c0c0e",
        grid      = "#1a1a1f",
    ),
    "light": dict(
        bg        = "#f8fafc",
        surface   = "#ffffff",
        surface2  = "#f1f5f9",
        surface3  = "#e2e8f0",
        border    = "#e2e8f0",
        accent    = "#2563eb",
        accent2   = "#3b82f6",
        success   = "#059669",
        warning   = "#d97706",
        danger    = "#dc2626",
        text      = "#0f172a",
        muted     = "#64748b",
        plot_bg   = "#ffffff",
        grid      = "#f1f5f9",
    )
}

STRINGS = {
    "en": {
        "dashboard": "Dashboard",
        "dataset": "Dataset",
        "visualize": "Visualize",
        "augment": "Augment",
        "simulate": "Simulate",
        "export": "Export",
        "run_demo": "Run Demo",
        "theme_light": "Light Mode",
        "theme_dark": "Dark Mode",
        "lang_switch": "Türkçe",
        "developed": "Developed by",
        "rights": "All rights reserved",
        # Hero
        "hero_title": "Project Overview",
        "hero_desc": "Welcome to the <b>trajdata</b> workspace. Monitor global dataset metrics below or use the <b>Quick Start</b> guide to begin your analysis.",
        # Quick Start
        "qs_title": "Quick Start & Workflow",
        "qs_1": "<b>Ingest Dataset</b>: Navigate to the <b>Dataset</b> tab to load raw trajectory splits. The system will pre-process agent distributions, sampling rates, and scenario counts for immediate analysis.",
        "qs_2": "<b>Trajectory Forensics</b>: Use the <b>Visualize</b> tab to deep-dive into agent behaviors. Scrub through temporal sequences to observe past history (dashed) vs. future ground-truth (solid) paths.",
        "qs_3": "<b>Data Augmentation</b>: In the <b>Augment</b> panel, apply complex kinematic transforms. Flip axes to simulate different driving cultures or scale velocities to expand the training manifold.",
        "qs_4": "<b>Validation & Simulation</b>: Verify model logic or data integrity using <b>Simulate</b>. Run closed-loop experiments to detect collisions and measure prediction errors (ADE/FDE) across scenes.",
        "qs_5": "<b>High-Performance Cache</b>: Once satisfied, use <b>Export</b> to precompute and save the workspace as a <b>Zarr Cache</b>. This ensures peak I/O performance during model training.",
        "qs_6": "<b>Interaction Demo</b>: If you are new to the platform, click <b>Run Demo</b> in the sidebar. This automated walkthrough will guide you through each functional unit of the trajdata system.",
        # Explainer: Visualize
        "viz_exp_title": "Understanding Trajectories",
        "viz_exp_hist_t": "History Paths (Dashed)",
        "viz_exp_hist_d": "Observations leading up to current T=0. Shows past speed, heading, and curvature.",
        "viz_exp_fut_t": "Future Ground-Truth (Solid)",
        "viz_exp_fut_d": "Actual path taken by the agent. Used to validate prediction accuracy.",
        "viz_exp_neigh_t": "Neighbor Agents (Muted)",
        "viz_exp_neigh_d": "Other actors in the scene. Critical for multi-agent interaction modeling.",
        "viz_exp_ctrl_t": "Visualization Control",
        "viz_exp_ctrl_d": "Use the slider above to scrub through temporal sequences in the loaded split.",
        # Explainer: Augment
        "aug_exp_title": "Augmentation Strategy",
        "aug_exp_spatial": "<b>Spatial Transforms:</b> Flipping across X/Y axes doubles the training manifold coverage (e.g., LHD to RHD scenarios).",
        "aug_exp_velocity": "<b>Velocity Scaling:</b> Modifies temporal relationships to simulate agents moving at different speeds.",
        "aug_exp_motion": "<b>Motion Labeling:</b> Automated heuristic labeling (Stationary, Walking, etc.) based on kinematic profiles.",
        # Explainer: Export
        "exp_note": "Pre-compute all batch elements to disk to skip on-the-fly cache building during model training. This ensures peak I/O throughput for large-scale experiments.",
        "exp_btn_label": " Export Workspace",
    },
    "tr": {
        "dashboard": "Panel",
        "dataset": "Veri Seti",
        "visualize": "Görselleştir",
        "augment": "Zenginleştir",
        "simulate": "Simüle Et",
        "export": "Dışa Aktar",
        "run_demo": "Demoyu Çalıştır",
        "theme_light": "Işık Modu",
        "theme_dark": "Karanlık Mod",
        "lang_switch": "English",
        "developed": "Geliştiren",
        "rights": "Tüm hakları saklıdır",
        # Hero
        "hero_title": "Proje Genel Bakışı",
        "hero_desc": "<b>trajdata</b> çalışma alanına hoş geldiniz. Küresel veri seti ölçümlerini aşağıdan izleyebilir veya analizinize başlamak için <b>Hızlı Başlangıç</b> kılavuzunu kullanabilirsiniz.",
        # Quick Start
        "qs_title": "Hızlı Başlangıç ve İş Akışı",
        "qs_1": "<b>Veri Seti Yükleme</b>: Ham yörünge verilerini içe aktarmak için <b>Veri Seti</b> sekmesini kullanın. Sistem; ajan dağılımını, örnekleme hızlarını ve senaryo sayılarını anında ön işleme tabi tutar.",
        "qs_2": "<b>Yörünge Analizi</b>: Ajan davranışlarını derinlemesine incelemek için <b>Görselleştir</b> sekmesini kullanın. Geçmiş gözlemler (kesikli) ve gelecek gerçekliği (düz) arasındaki ilişkiyi zaman içinde izleyin.",
        "qs_3": "<b>Veri Zenginleştirme</b>: <b>Zenginleştir</b> panelinde karmaşık kinematik dönüşümler uygulayın. Farklı sürüş kültürlerini simüle etmek için eksenleri aynalayın veya eğitim kapsamını genişletmek için hızları ölçeklendirin.",
        "qs_4": "<b>Doğrulama ve Simülasyon</b>: Model mantığını veya veri bütünlüğünü <b>Simüle Et</b> sekmesinde doğrulayın. Çarpışmaları tespit etmek ve tahmin hatalarını (ADE/FDE) ölçmek için kapalı döngü deneyler yapın.",
        "qs_5": "<b>Yüksek Performanslı Önbellek</b>: Ayarlarınızdan memnun kaldığınızda, çalışma alanını <b>Zarr Önbelleği</b> olarak kaydetmek için <b>Dışa Aktar</b> sekmesini kullanın. Bu, eğitim sırasında maksimum I/O hızı sağlar.",
        "qs_6": "<b>Etkileşimli Demo</b>: Platformu ilk kez kullanıyorsanız yan menüdeki <b>Demoyu Çalıştır</b> butonuna tıklayın. Bu otomatik tur, trajdata sisteminin her bir biriminde size rehberlik edecektir.",
        # Explainer: Export
        "exp_note": "Eğitim sırasında anlık önbellek oluşturma adımını atlamak için tüm toplu iş öğelerini diske önceden hesaplayarak kaydedin. Bu, büyük ölçekli deneyler için en yüksek I/O veri akışını sağlar.",
        "exp_btn_label": " Çalışma Alanını Dışa Aktar",
         # Explainer: Visualize
        "viz_exp_title": "Yörüngeleri Anlama",
        "viz_exp_hist_t": "Geçmiş Yollar (Kesikli)",
        "viz_exp_hist_d": "T=0 anına kadar olan gözlemler. Geçmiş hızı, rotayı ve eğriliği gösterir.",
        "viz_exp_fut_t": "Gelecek Gerçekliği (Düz)",
        "viz_exp_fut_d": "Ajanın izlediği gerçek yol. Tahmin doğruluğunu doğrulamak için kullanılır.",
        "viz_exp_neigh_t": "Komşu Ajanlar (Sönük)",
        "viz_exp_neigh_d": "Sahnedeki diğer aktörler. Çoklu ajan etkileşim modellemesi için kritiktir.",
        "viz_exp_ctrl_t": "Görselleştirme Kontrolü",
        "viz_exp_ctrl_d": "Yüklü veri seti içindeki zaman dizileri arasında geçiş yapmak için yukarıdaki sürgüyü kullanın.",
        # Explainer: Augment
        "aug_exp_title": "Zenginleştirme Stratejisi",
        "aug_exp_spatial": "<b>Uzamsal Dönüşümler:</b> X/Y eksenleri boyunca aynalama, eğitim kapsamını iki katına çıkarır (örn. LHD'den RHD senaryolarına).",
        "aug_exp_velocity": "<b>Hız Ölçeklendirme:</b> Farklı hızlarda hareket eden ajanları simüle etmek için zamansal ilişkileri değiştirir.",
        "aug_exp_motion": "<b>Hareket Etiketleme:</b> Kinematik profillere dayalı otomatik etiketleme (Sabit, Yürüyen vb.).",
    }
}

# Default Active Theme
C = THEMES["dark"]
S = STRINGS["en"]

_SIDEBAR_W = 248
_TRAJ_PH   = dict(xs=[[]], ys=[[]], line_color=["#252540"],
                  line_dash=["solid"], legend_label=[""])

_PANEL_TITLES = {
    "dashboard": ("Dashboard",   "Overview & quick start"),
    "dataset":   ("Dataset",     "Load & explore datasets"),
    "visualize": ("Visualize",   "Interactive trajectory viewer"),
    "augment":   ("Augment",     "Data augmentation preview"),
    "simulate":  ("Simulate",    "Run & evaluate simulations"),
    "export":    ("Export",      "Precompute & save caches"),
}

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _div(html: str, **kw) -> Div:
    return Div(text=html, **kw)


def _card(title: str, value: str, color: str = C["accent"]) -> Div:
    return _div(
        f"""<div style="background:{C['surface2']};border:1px solid {C['border']};
        border-left:3px solid {color};border-radius:10px;padding:14px 18px;min-width:160px">
        <div style="color:{C['muted']};font-size:11px;text-transform:uppercase;
        letter-spacing:.09em;margin-bottom:6px;font-weight:600">{title}</div>
        <div style="color:{C['text']};font-size:22px;font-weight:700;
        font-family:'Inter',-apple-system,'SF Pro Display',sans-serif">{value}</div>
        </div>""",
    )


def _section_title(text: str) -> Div:
    return _div(
        f"<div style='color:{C['text']};font-size:16px;font-weight:600;"
        f"font-family:Inter,-apple-system,SF Pro Display,sans-serif;"
        f"border-bottom:1px solid {C['border']};padding-bottom:10px;"
        f"margin-bottom:16px;letter-spacing:-.01em'>{text}</div>",
        width=800,
    )


def _title_html(title: str, subtitle: str) -> str:
    return (
        f"<div style='display:flex;flex-direction:column;justify-content:center;height:72px;padding-left:24px'>"
        f"<div style='color:{C['text']};font-size:15px;font-weight:600;"
        f"font-family:Inter,-apple-system,SF Pro Display,sans-serif;"
        f"letter-spacing:-.02em;line-height:1.2'>{title}</div>"
        f"<div style='color:{C['muted']};font-size:12px;margin-top:2px'>{subtitle}</div>"
        f"</div>"
    )


def _get_icon(name: str, color: str = "#ffffff") -> SVGIcon:
    path = f"/Users/hidirektor/PycharmProjects/trajdata/img/icon/icon_{name}.svg"
    try:
        import re
        with open(path, "r") as f:
            svg = f.read()
        
        # Color: Map named colors to proper hex for better SVG support
        c = "#ffffff" if color == "white" else ("#111114" if color == "black" else color)
        
        # Power-RE: Replace hex colors
        svg = re.sub(r'#([0-9a-fA-F]{3,6})', c, svg)
        # Named black replacements
        svg = svg.replace('stroke="black"', f'stroke="{c}"').replace('fill="black"', f'fill="{c}"')
        
        # Force-scale size (38px for maximum impact in 44px buttons)
        svg = re.sub(r'<svg([^>]*?)width="[^"]+"',  rf'<svg\1width="38px"',  svg, count=1)
        svg = re.sub(r'<svg([^>]*?)height="[^"]+"', rf'<svg\1height="38px"', svg, count=1)
        
        return SVGIcon(svg=svg)
    except Exception:
        return None


def _toast_html(msg: str, visible: bool = False) -> str:
    opp = "1" if visible else "0"
    tra = "translateY(0)" if visible else "translateY(-20px)"
    return f"""
<div style="
    position:fixed; top:24px; right:24px; z-index:9999;
    background:{C['surface']}; border:1px solid {C['border']};
    border-left:4px solid {C['accent']}; border-radius:10px;
    padding:16px 20px; box-shadow:0 12px 32px rgba(0,0,0,0.2);
    display:flex; align-items:center; gap:12px;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    opacity:{opp}; transform:{tra}; pointer-events:none;
    min-width:300px; font-family:'Inter',sans-serif;
">
    <div style="width:20px;height:20px;border-radius:50%;background:{C['accent']};
         display:flex;align-items:center;justify-content:center;color:white;font-size:10px">i</div>
    <div>
        <div style="color:{C['text']};font-size:13px;font-weight:600">Walkthrough Update</div>
        <div style="color:{C['muted']};font-size:12px;margin-top:2px">{msg}</div>
    </div>
</div>
"""


def _traj_figure(title: str, w: int = 520, h: int = 400) -> figure:
    p = figure(
        title=title, width=w, height=h,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        background_fill_color=C["plot_bg"],
        border_fill_color=C["surface"],
        outline_line_color=C["border"],
    )
    p.title.text_color         = C["text"]
    p.title.text_font_size     = "13px"
    p.xaxis.axis_label         = "x (m)"
    p.yaxis.axis_label         = "y (m)"
    p.axis.axis_label_text_color  = C["muted"]
    p.axis.major_label_text_color = C["muted"]
    p.axis.axis_line_color     = C["border"]
    p.axis.major_tick_line_color  = C["border"]
    p.grid.grid_line_color     = C["grid"]
    p.xaxis.axis_label_text_font_size = "11px"
    p.yaxis.axis_label_text_font_size = "11px"
    return p


def _add_traj_glyphs(fig: figure, hist_src: ColumnDataSource,
                     fut_src: ColumnDataSource) -> None:
    fig.add_glyph(hist_src, MultiLine(
        xs="xs", ys="ys", line_color="line_color",
        line_dash="line_dash", line_width=2, line_alpha=0.9,
    ))
    fig.add_glyph(fut_src, MultiLine(
        xs="xs", ys="ys", line_color="line_color",
        line_dash="line_dash", line_width=2, line_alpha=0.55,
    ))


def _nav_btn(label: str, icon_name: str, theme: str, on_click: callable, active: bool = False) -> Button:
    # Color logic: 
    #   Dark Mode -> Always white
    #   Light Mode -> Active? White : Black
    if theme == "dark":
        color = "white"
    else:
        color = "white" if active else "black"
    
    btn = Button(
        label=label,
        icon=_get_icon(icon_name, color),
        width=_SIDEBAR_W - 24,
        height=44,
        stylesheets=[f"""
            :host .bk-btn {{
                background: transparent;
                border: none !important;
                border-radius: 12px;
                color: {C['muted']};
                font-weight: 500;
                font-size: 15px;
                font-family: 'Inter', -apple-system, sans-serif;
                text-align: left;
                padding-left: 12px;
                display: flex;
                align-items: center;
                gap: 12px;
                transition: all .2s;
                cursor: pointer;
            }}
            :host .bk-btn:hover {{
                background: {C['surface3']};
                color: {C['text']};
            }}
            :host .bk-btn-primary {{
                background: {C['accent']} !important;
                color: white !important;
                font-weight: 600;
                box-shadow: 0 4px 12px {C['accent']}30 !important;
            }}
        """],
    )
    btn.on_click(on_click)
    if active:
        btn.button_type = "primary"
    return btn


# ═══════════════════════════════════════════════════════════════════════════
# Panels
# ═══════════════════════════════════════════════════════════════════════════

def _build_dashboard(state: AppState, cards_refs: dict) -> column:
    def _t(k): return STRINGS[state.lang].get(k, STRINGS["en"].get(k, k))
    
    quick_div = _div(
        f"""<div style='background:{C["surface2"]};border:1px solid {C["border"]};
        border-radius:12px;padding:24px 28px;margin-top:0;width:100%'>
        <div style='color:{C["text"]};font-weight:600;margin-bottom:16px;
        font-family:Inter,-apple-system,SF Pro Display,sans-serif;font-size:15px'>
        {_t('qs_title')}</div>
        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:32px'>
            <ol style='color:{C["muted"]};font-size:13px;line-height:2.4;margin:0;padding-left:18px'>
                <li>{_t('qs_1')}</li>
                <li>{_t('qs_2')}</li>
                <li>{_t('qs_3')}</li>
            </ol>
            <ol style='color:{C["muted"]};font-size:13px;line-height:2.4;margin:0;padding-left:18px' start="4">
                <li>{_t('qs_4')}</li>
                <li>{_t('qs_5')}</li>
                <li>{_t('qs_6')}</li>
            </ol>
        </div></div>""",
        sizing_mode="stretch_width",
    )

    return column(quick_div, sizing_mode="stretch_width")


def _build_stats_row(cards_refs: dict) -> row:
    # We rebuild cards ONLY if they don't exist in refs to avoid duplicate state
    if "samples" not in cards_refs:
        cards_refs["samples"] = _card("Samples",    "—",   C["accent"])
        cards_refs["scenes"]  = _card("Scenes",     "—",   C["success"])
        cards_refs["dt"]      = _card("dt (s)",     "—",   C["warning"])
        cards_refs["agents"]  = _card("Agent types","—",   C["danger"])
    
    return row(
        cards_refs["samples"], cards_refs["scenes"],
        cards_refs["dt"],      cards_refs["agents"],
        spacing=14, sizing_mode="stretch_width",
        styles={"justify-content": "space-between"}
    )

def _build_dataset_panel(doc, state: AppState, cards_refs: dict,
                         nav_fn) -> tuple:
    """Returns (panel_column, status_div, stats_source)."""
    split_sel   = Select(title="Split", value=state.dataset_split,
                         options=AVAILABLE_SPLITS, width=320)
    load_btn    = Button(label="Load Dataset", button_type="primary",
                         width=160, height=36, margin=(24, 0, 0, 12))
    status_div  = _div(f"<i style='color:{C['muted']}'>Select a split and load.</i>",
                       width=500)

    stats_src   = ColumnDataSource({"stat": [], "value": []})
    table_style = f"""
        :host {{ 
            background: {C['surface']} !important; 
            border: 1px solid {C['border']} !important;
            border-radius: 12px;
            overflow: hidden;
        }}
        .bk-data-table {{ 
            background: {C['surface']} !important;
            color: {C['text']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        .bk-cell-index {{ background: {C['surface2']} !important; color: {C['muted']} !important; }}
        .bk-header-column {{ 
            background: {C['surface3']} !important; 
            color: {C['text']} !important; 
            font-weight: 600 !important;
            border-bottom: 1px solid {C['border']} !important;
        }}
        .bk-header-column:hover {{ 
            background: {C['accent']} !important; 
            color: white !important;
        }}
        .slick-cell {{ border-right: 1px solid {C['border']} !important; border-bottom: 1px solid {C['border']} !important; }}
        .slick-row {{ background: {C['surface']} !important; }}
        .slick-row:hover {{ background: {C['accent']}15 !important; }}
        .slick-row.even {{ background: {C['surface2']} !important; }}
    """

    stats_tbl   = DataTable(
        source=stats_src,
        columns=[TableColumn(field="stat",  title="Statistic", width=240),
                 TableColumn(field="value", title="Value",     width=220)],
        sizing_mode="stretch_width", height=400, index_position=None,
        stylesheets=[table_style],
    )

    def _do_load():
        split = split_sel.value
        try:
            ds    = load_dataset(split, state.aug_config)
            stats = compute_stats(ds)
            def _update():
                state.dataset       = ds
                state.dataset_split = split
                state.current_sample_idx = 0
                rows = [
                    ("Split",           split),
                    ("Total samples",   f"{stats['total_samples']:,}"),
                    ("Scenes",          str(stats["num_scenes"])),
                    ("dt (s)",          str(stats["dt_s"])),
                    ("First scene",     str(stats["scene0_name"])),
                    ("Timesteps (s0)",  str(stats["scene0_timesteps"])),
                    ("Agents (s0)",     str(stats["scene0_agents"])),
                    ("Mean hist len",   str(stats["mean_hist_len"])),
                    ("Mean fut len",    str(stats["mean_fut_len"])),
                ] + [(f"Type: {k}", str(v)) for k, v in stats["agent_type_counts"].items()]
                stats_src.data = {"stat": [r[0] for r in rows],
                                   "value": [r[1] for r in rows]}
                # Update dashboard cards
                if "samples" in cards_refs:
                    cards_refs["samples"].text = _card("Samples",
                        f"{stats['total_samples']:,}", C["accent"]).text
                    cards_refs["scenes"].text  = _card("Scenes",
                        str(stats["num_scenes"]), C["success"]).text
                    cards_refs["dt"].text      = _card("dt (s)",
                        str(stats["dt_s"]), C["warning"]).text
                    types_str = "/".join(stats["agent_type_counts"].keys())
                    cards_refs["agents"].text  = _card("Agent types",
                        types_str, C["danger"]).text
                status_div.text = (
                    f"<span style='color:{C['success']}'>"
                    f"&#10003; Loaded <b>{split}</b> &mdash; {stats['total_samples']:,} samples</span>"
                )
            doc.add_next_tick_callback(_update)
        except Exception as e:
            doc.add_next_tick_callback(lambda: setattr(
                status_div, "text",
                f"<span style='color:{C['danger']}'>&#10007; {e}</span>"
            ))

    def on_load(_):
        status_div.text = f"<span style='color:{C['warning']}'>Loading&hellip;</span>"
        _pool.submit(_do_load)

    load_btn.on_click(on_load)

    panel = column(row(split_sel, load_btn, spacing=16, margin=(16,0,0,0)),
                   status_div, stats_tbl, sizing_mode="stretch_width")
    return panel, status_div, stats_src, split_sel, load_btn


def _build_viz_panel(doc, state: AppState, cards_refs: dict) -> tuple:
    hist_src = ColumnDataSource(_TRAJ_PH.copy())
    fut_src  = ColumnDataSource(_TRAJ_PH.copy())

    p = _traj_figure("Trajectory", w=600, h=450)
    _add_traj_glyphs(p, hist_src, fut_src)

    legend_div = _div(
        f"<div style='font-size:12px;color:{C['muted']};margin-top:6px'>"
        f"<span style='color:{C['accent']}'>- -</span> History &nbsp;&nbsp;"
        f"<span style='color:{C['success']}'>---</span> Future &nbsp;&nbsp;"
        f"<span style='color:{C['muted']}'>- -</span> Neighbor hist &nbsp;&nbsp;"
        f"<span style='color:{C['muted']}'>---</span> Neighbor fut</div>",
    )

    slider   = Slider(title="Sample", start=0, end=1, step=1, value=0, width=380)
    prev_btn = Button(label=" < ", width=50, height=34, button_type="default")
    next_btn = Button(label=" > ", width=50, height=34, button_type="default")
    opts     = CheckboxGroup(labels=["History", "Future", "Neighbors"],
                             active=[0, 1, 2], inline=True)
    info_div = _div(f"<i style='color:{C['muted']}'>Load a dataset first.</i>",
                    width=580)

    def _refresh(idx: int):
        if state.dataset is None:
            return
        n   = len(state.dataset)
        idx = max(0, min(idx, n - 1))
        state.current_sample_idx = idx
        slider.value = idx
        sh, sf, sn = 0 in opts.active, 1 in opts.active, 2 in opts.active
        try:
            b = sample_to_batch(state.dataset, idx)
            hd, fd = batch_to_sources(b, 0, sh, sf, sn)
            hist_src.data, fut_src.data = hd, fd
            info_div.text = (
                f"<span style='color:{C['muted']};font-size:12px'>"
                f"Sample <b style='color:{C['text']}'>{idx}</b>/{n-1} &nbsp;&middot;&nbsp; "
                f"Agent <code style='color:{C['accent']}'>{b.agent_name[0]}</code> &nbsp;&middot;&nbsp; "
                f"Hist {b.agent_hist_len[0].item()}ts &nbsp;&middot;&nbsp; "
                f"Fut {b.agent_fut_len[0].item()}ts &nbsp;&middot;&nbsp; "
                f"Neighbors {b.num_neigh[0].item()}</span>"
            )
        except Exception as e:
            info_div.text = f"<span style='color:{C['danger']}'>{e}</span>"

    slider.on_change("value",  lambda a, o, n: _refresh(n))
    prev_btn.on_click(lambda _: _refresh(state.current_sample_idx - 1))
    next_btn.on_click(lambda _: _refresh(state.current_sample_idx + 1))
    opts.on_change("active", lambda a, o, n: _refresh(state.current_sample_idx))

    def _periodic():
        if state.dataset is not None and slider.end == 1:
            slider.end = max(1, len(state.dataset) - 1)
            _refresh(0)
    doc.add_periodic_callback(_periodic, 1200)

    def _t(k): return STRINGS[state.lang].get(k, STRINGS["en"].get(k, k))
    # Detailed Explainers (Collapsible details tag)
    explain_div = _div(f"""
    <details style="background:{C['surface3']}; border-radius:12px; border:1px solid {C['border']}; margin-bottom:24px; cursor:pointer" open>
        <summary style="padding:16px 20px; color:{C['text']}; font-weight:600; font-size:14px; outline:none">
            {_t('viz_exp_title')}
        </summary>
        <div style="padding:0 20px 20px; display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:12px; line-height:1.6; cursor:default">
            <div>
                <b style="color:{C['accent']}">{_t('viz_exp_hist_t')}</b><br>
                <span style="color:{C['muted']}">{_t('viz_exp_hist_d')}</span>
            </div>
            <div>
                <b style="color:{C['success']}">{_t('viz_exp_fut_t')}</b><br>
                <span style="color:{C['muted']}">{_t('viz_exp_fut_d')}</span>
            </div>
            <div>
                <b style="color:{C['text']}">{_t('viz_exp_neigh_t')}</b><br>
                <span style="color:{C['muted']}">{_t('viz_exp_neigh_d')}</span>
            </div>
            <div>
                <b style="color:{C['warning']}">{_t('viz_exp_ctrl_t')}</b><br>
                <span style="color:{C['muted']}">{_t('viz_exp_ctrl_d')}</span>
            </div>
            <div style="grid-column: span 2; border-top:1px solid {C['border']}; padding-top:12px; margin-top:4px">
                <b style="color:{C['danger']}">Collision Detection / Çarpışma Tespiti</b><br>
                <span style="color:{C['muted']}">Overlap of paths indicates potential collisions. In Simulations, collisions are automatically highlighted and categorized by agent type.</span>
            </div>
        </div>
    </details>
    """, sizing_mode="stretch_width")

    controls = column(slider, row(prev_btn, next_btn, sizing_mode="stretch_width"), opts, info_div, legend_div,
                      sizing_mode="stretch_width", max_width=420)
    
    panel_layout = row(controls, p, spacing=24, sizing_mode="stretch_width")
    panel = column(explain_div, panel_layout, sizing_mode="stretch_width")
    return panel, _refresh, slider

def _build_aug_panel(doc, state: AppState, cards_refs: dict) -> tuple:
    bh_src = ColumnDataSource(_TRAJ_PH.copy())
    bf_src = ColumnDataSource(_TRAJ_PH.copy())
    ah_src = ColumnDataSource(_TRAJ_PH.copy())
    af_src = ColumnDataSource(_TRAJ_PH.copy())

    fig_o = _traj_figure("Original", w=420, h=380)
    fig_a = _traj_figure("Augmented", w=420, h=380)
    _add_traj_glyphs(fig_o, bh_src, bf_src)
    _add_traj_glyphs(fig_a, ah_src, af_src)

    slider       = Slider(title="Sample", start=0, end=1, step=1, value=0, width=360)
    mirror_tog   = Toggle(label="Mirror",      active=False, button_type="default",
                          width=130, height=34)
    mirror_ax    = Select(title="Axis", value="x", options=["x","y"], width=90)
    mirror_prob  = Slider(title="Prob", start=0.0, end=1.0, step=0.05, value=0.5, width=200)
    speed_tog    = Toggle(label="Speed Scale", active=False, button_type="default",
                          width=150, height=34)
    speed_range  = RangeSlider(title="Scale", start=0.3, end=2.5,
                                step=0.05, value=(0.8, 1.2), width=260)
    motion_tog   = Toggle(label="Motion Labels", active=False, button_type="default",
                          width=160, height=34)
    motion_div   = _div("", width=380)
    apply_btn    = Button(label="Apply to Dataset & Reload",
                          button_type="success", width=240, height=36)
    status_div   = _div("", width=500)

    def _read():
        state.aug_config.update({
            "mirror": mirror_tog.active, "mirror_axis": mirror_ax.value,
            "mirror_prob": mirror_prob.value, "speed_scale": speed_tog.active,
            "speed_min": speed_range.value[0], "speed_max": speed_range.value[1],
            "motion_labeler": motion_tog.active,
        })

    def _preview(idx: int):
        if state.dataset is None:
            return
        _read()
        idx = max(0, min(idx, len(state.dataset) - 1))
        try:
            bh, bf, ah, af = compute_preview(state.dataset, idx, state.aug_config)
            bh_src.data, bf_src.data = bh, bf
            ah_src.data, af_src.data = ah, af
            if state.aug_config["motion_labeler"]:
                from trajdata.data_structures.collation import agent_collate_fn
                elem = state.dataset[idx]
                b    = agent_collate_fn([elem], True, pad_format="outside")
                for aug in build_augmentations(state.aug_config):
                    try: aug.apply_agent(b)
                    except Exception: pass
                if "motion_type" in b.extras:
                    lm = {0:"STATIONARY",1:"WALKING",2:"RUNNING",3:"FAST"}
                    lbl = lm.get(b.extras["motion_type"][0].item(), "?")
                    motion_div.text = (
                        f"<span style='color:{C['accent']};font-size:13px'>"
                        f"Motion type: <b>{lbl}</b></span>"
                    )
        except Exception as e:
            status_div.text = f"<span style='color:{C['danger']}'>{e}</span>"

    slider.on_change("value", lambda a, o, n: _preview(n))
    for w in (mirror_tog, mirror_ax, mirror_prob, speed_tog, speed_range, motion_tog):
        prop = "active" if isinstance(w, Toggle) else "value"
        w.on_change(prop, lambda a, o, n: _preview(slider.value))

    def on_apply(_):
        _read()
        status_div.text = f"<span style='color:{C['warning']}'>Reloading&hellip;</span>"
        def _do():
            try:
                ds = load_dataset(state.dataset_split, state.aug_config)
                def _done():
                    state.dataset  = ds
                    slider.end     = max(1, len(ds) - 1)
                    status_div.text = (
                        f"<span style='color:{C['success']}'>&#10003; Reloaded with augmentations</span>"
                    )
                    _preview(0)
                doc.add_next_tick_callback(_done)
            except Exception as e:
                doc.add_next_tick_callback(lambda: setattr(
                    status_div, "text",
                    f"<span style='color:{C['danger']}'>{e}</span>"
                ))
        _pool.submit(_do)

    apply_btn.on_click(on_apply)

    def _periodic():
        if state.dataset is not None and slider.end == 1:
            slider.end = max(1, len(state.dataset) - 1)
            _preview(0)
    doc.add_periodic_callback(_periodic, 1200)

    def _sep(t): return _div(
        f"<div style='color:{C['muted']};font-size:10px;text-transform:uppercase;"
        f"letter-spacing:.09em;margin:10px 0 4px;font-weight:600'>{t}</div>", width=380)

    def _t(k): return STRINGS[state.lang].get(k, STRINGS["en"].get(k, k))
    # Detailed Augmentation Explanation
    aug_explain = _div(f"""
    <div style="background:{C['surface2']}; border-left:4px solid {C['success']}; padding:18px 22px; border-radius:12px; height:100%">
        <h4 style="color:{C['text']}; margin:0 0 10px; font-size:14px">{_t('aug_exp_title')}</h4>
        <p style="color:{C['muted']}; font-size:12px; margin:0; line-height:1.75">
            <b>{_t('viz_exp_hist_t')}</b>: {_t('aug_exp_spatial')}<br><br>
            <b>{_t('viz_exp_fut_t')}</b>: {_t('aug_exp_velocity')}<br><br>
            <b>{_t('viz_exp_neigh_t')}</b>: {_t('aug_exp_motion')}
        </p>
    </div>
    """, width=320)

    controls = column(
        slider,
        _sep("Mirror Augmentation"), row(mirror_tog, mirror_ax, sizing_mode="stretch_width"), mirror_prob,
        _sep("Speed Scale"),         row(speed_tog, speed_range, sizing_mode="stretch_width"),
        _sep("Motion Labeler"),      motion_tog, motion_div,
        apply_btn, status_div,
        sizing_mode="stretch_width", max_width=380,
    )
    figs = column(fig_o, fig_a, sizing_mode="stretch_width")
    panel = column(row(controls, figs, aug_explain, spacing=24, sizing_mode="stretch_width"),
                   sizing_mode="stretch_width")
    return panel, _preview, slider


def _build_sim_panel(doc, state: AppState, cards_refs: dict) -> tuple:
    scene_sl   = Slider(title="Scene", start=0, end=1, step=1, value=0, width=280)
    steps_sl   = Slider(title="Steps", start=5, end=200, step=5, value=30, width=280)
    met_check  = CheckboxGroup(labels=["ADE","FDE","Collision","OffRoad"],
                               active=[0,1,2], inline=True,
                               stylesheets=[f":host {{ color: {C['text']}; font-size: 13px; }}"])
    run_btn    = Button(label=" >  Run Simulation", button_type="success",
                        width=190, height=36)
    status_div = _div(f"<i style='color:{C['muted']}'>Load a dataset first.</i>",
                      width=520)
    means_div  = _div("", width=560)

    _MCOLS = ["agent","ade","fde","collision","offroad"]
    tbl_src = ColumnDataSource({c: [] for c in _MCOLS})
    table_style = f"""
        :host {{ 
            background: {C['surface']} !important; 
            border: 1px solid {C['border']} !important;
            border-radius: 12px;
            overflow: hidden;
        }}
        .bk-data-table {{ 
            background: {C['surface']} !important;
            color: {C['text']} !important;
            font-family: 'Inter', sans-serif !important;
        }}
        .bk-cell-index {{ background: {C['surface2']} !important; color: {C['muted']} !important; }}
        .bk-header-column {{ 
            background: {C['surface3']} !important; 
            color: {C['text']} !important; 
            font-weight: 600 !important;
            border-bottom: 1px solid {C['border']} !important;
        }}
        .bk-header-column:hover {{ 
            background: {C['accent']} !important; 
            color: white !important;
        }}
        .slick-cell {{ border-right: 1px solid {C['border']} !important; border-bottom: 1px solid {C['border']} !important; }}
        .slick-row {{ background: {C['surface']} !important; }}
        .slick-row:hover {{ background: {C['accent']}15 !important; }}
        .slick-row.even {{ background: {C['surface2']} !important; }}
    """

    tbl = DataTable(
        source=tbl_src,
        columns=[TableColumn(field=c, title=c.upper(), width=105) for c in _MCOLS],
        sizing_mode="stretch_width", height=320, index_position=None,
        stylesheets=[table_style],
    )

    def _run():
        names = ["ADE","FDE","Collision","OffRoad"]
        sel   = [names[i] for i in met_check.active]
        try:
            res = run_simulation(state.dataset, scene_sl.value, steps_sl.value, sel)
            def _update():
                rows = res["rows"]
                new  = {c: [] for c in _MCOLS}
                new["agent"] = [r["agent"] for r in rows]
                for m in ["ade","fde","collision","offroad"]:
                    new[m] = [round(r.get(m, float("nan")), 4) for r in rows]
                tbl_src.data = new
                parts = [f"<b style='color:{C['text']}'>{k.upper()}</b>"
                         f"<span style='color:{C['muted']}'>: {v:.4f}</span>"
                         for k, v in res.get("means", {}).items()]
                means_div.text = (
                    f"<div style='font-size:13px'>" + " &nbsp;&middot;&nbsp; ".join(parts) + "</div>"
                )
                status_div.text = (
                    f"<span style='color:{C['success']}'>&#10003; Done &mdash; "
                    f"{res['steps']} steps, {len(rows)} agents</span>"
                )
            doc.add_next_tick_callback(_update)
        except Exception as e:
            doc.add_next_tick_callback(lambda: setattr(
                status_div, "text",
                f"<span style='color:{C['danger']}'>{e}</span>"
            ))

    def on_run(_):
        if state.dataset is None:
            status_div.text = f"<span style='color:{C['warning']}'>Load a dataset first.</span>"
            return
        status_div.text = f"<span style='color:{C['warning']}'>Running&hellip;</span>"
        _pool.submit(_run)

    run_btn.on_click(on_run)

    def _periodic():
        if state.dataset is not None and scene_sl.end == 1:
            scene_sl.end = max(0, state.dataset.num_scenes() - 1)
    doc.add_periodic_callback(_periodic, 1200)

    controls = column(scene_sl, steps_sl,
                      _div(f"<div style='color:{C['muted']};font-size:10px;"
                           f"text-transform:uppercase;letter-spacing:.09em;"
                           f"font-weight:600;margin-top:8px'>Metrics</div>", width=300),
                      met_check, run_btn, status_div, width=320)
    results  = column(
        _div(f"<div style='color:{C['muted']};font-size:12px;margin-bottom:6px'>"
             f"Per-agent results</div>", width=580),
        tbl, means_div, sizing_mode="stretch_width",
    )
    panel = column(row(controls, results, spacing=24, sizing_mode="stretch_width"), sizing_mode="stretch_width")
    return panel, run_btn, status_div, scene_sl


def _build_export_panel(doc, state: AppState, cards_refs: dict) -> column:
    def _t(k): return STRINGS[state.lang].get(k, STRINGS["en"].get(k, k))
    
    note       = _div(
        f"<p style='color:{C['muted']};font-size:13px;max-width:100%;line-height:1.7'>"
        f"{_t('exp_note')}</p>",
        sizing_mode="stretch_width"
    )
    path_in    = TextInput(title="Output path", value="~/trajdata_export", sizing_mode="stretch_width", max_width=500)
    fmt_sel    = Select(title="Format", value="zarr",
                        options=["zarr","numpy"], width=160)
    bs_sl      = Slider(title="Batch size", start=8, end=256, step=8,
                        value=64, sizing_mode="stretch_width", max_width=340)
    exp_btn    = Button(label=_t('exp_btn_label'), button_type="primary",
                        width=220, height=36)
    status_div = _div(f"<i style='color:{C['muted']}'>Load a dataset first.</i>",
                       sizing_mode="stretch_width")
    result_div = _div("", sizing_mode="stretch_width")

    def _do():
        from trajdata.io import DataExporter
        out = str(Path(path_in.value).expanduser())
        try:
            DataExporter.export(state.dataset, out, format=fmt_sel.value,
                                batch_size=bs_sl.value, num_workers=0, verbose=True)
            def _done():
                _s = C["surface2"]; _ok = C["success"]; _t = C["text"]
                _ac = C["accent"];  _mu = C["muted"];  _fmt = fmt_sel.value
                result_div.text = (
                    f"<div style='background:{_s};border:1px solid {_ok};"
                    f"border-radius:10px;padding:14px;font-size:13px;color:{_t}'>"
                    f"<b>Export complete</b><br>"
                    f"Path: <code style='color:{_ac}'>{out}</code><br><br>"
                    f"<span style='color:{_mu}'>Load back with:</span><br>"
                    f"<code style='color:{_ac}'>"
                    f"PrecomputedDataset('{out}', format='{_fmt}')"
                    f"</code></div>"
                )
                status_div.text = f"<span style='color:{C['success']}'>Done.</span>"
            doc.add_next_tick_callback(_done)
        except Exception as e:
            doc.add_next_tick_callback(lambda: setattr(
                status_div, "text",
                f"<span style='color:{C['danger']}'>{e}</span>"
            ))

    def on_export(_):
        if state.dataset is None:
            status_div.text = f"<span style='color:{C['warning']}'>Load a dataset first.</span>"
            return
        status_div.text = f"<span style='color:{C['warning']}'>Exporting&hellip;</span>"
        result_div.text = ""
        _pool.submit(_do)

    exp_btn.on_click(on_export)

    return column(note, path_in, row(fmt_sel, bs_sl, align="end"),
                  exp_btn, status_div, result_div,
                  sizing_mode="stretch_width")


# ═══════════════════════════════════════════════════════════════════════════
# Run Demo
# ═══════════════════════════════════════════════════════════════════════════

def _run_demo(doc, state: AppState, refs: dict):
    """Automated demo: load → visualize → augment → simulate."""
    toast      = refs["toast"]
    demo_btn   = refs["demo_btn"]
    nav_fn     = refs["nav_fn"]
    viz_refresh= refs["viz_refresh"]
    aug_preview= refs["aug_preview"]
    sim_run_btn= refs["sim_run_btn"]
    sim_status = refs["sim_status"]

    demo_btn.disabled = True
    demo_btn.label    = "Running Demo..."

    def _log(msg: str):
        toast.text = _toast_html(msg, visible=True)
        # Auto-hide after 4s (or next log)
        doc.add_timeout_callback(lambda: _hide_if_same(msg), 4000)

    def _hide_if_same(msg: str):
        if msg in toast.text:
            toast.text = _toast_html(msg, visible=False)

    def _step1():
        nav_fn("dataset")
        _log(f"<span style='color:{C['accent']};font-weight:600'>Step 1/4</span>"
             f" &mdash; Loading <b>eupeds_eth-train</b>&hellip;")

        def _do():
            try:
                ds    = load_dataset("eupeds_eth-train", state.aug_config)
                stats = compute_stats(ds)
                def _done():
                    state.dataset = ds
                    state.dataset_split = "eupeds_eth-train"
                    refs["stats_src"].data = {
                        "stat":  ["Split","Samples","Scenes","dt (s)"],
                        "value": ["eupeds_eth-train",
                                   f"{stats['total_samples']:,}",
                                   str(stats["num_scenes"]),
                                   str(stats["dt_s"])],
                    }
                    refs["ds_status"].text = (
                        f"<span style='color:{C['success']}'>"
                        f"&#10003; {stats['total_samples']:,} samples loaded</span>"
                    )
                    _log(f"<span style='color:{C['success']}'>&#10003; Dataset loaded &mdash; "
                         f"{stats['total_samples']:,} samples, "
                         f"{stats['num_scenes']} scenes</span>")
                    doc.add_timeout_callback(_step2, 1500)
                doc.add_next_tick_callback(_done)
            except Exception as e:
                doc.add_next_tick_callback(lambda: _log(
                    f"<span style='color:{C['danger']}'>&#10007; {e}</span>"
                ))
        _pool.submit(_do)

    def _step2():
        nav_fn("visualize")
        _log(f"<span style='color:{C['accent']};font-weight:600'>Step 2/4</span>"
             f" &mdash; Visualizing sample 10&hellip;")
        if state.dataset is not None:
            refs["viz_slider"].end = max(1, len(state.dataset) - 1)
            viz_refresh(10)
        doc.add_timeout_callback(_step3, 2000)

    def _step3():
        nav_fn("augment")
        _log(f"<span style='color:{C['accent']};font-weight:600'>Step 3/4</span>"
             f" &mdash; Previewing Mirror augmentation&hellip;")
        state.aug_config["mirror"]      = True
        state.aug_config["mirror_axis"] = "x"
        state.aug_config["mirror_prob"] = 1.0
        if state.dataset is not None:
            refs["aug_slider"].end = max(1, len(state.dataset) - 1)
            aug_preview(10)
        doc.add_timeout_callback(_step4, 2500)

    def _step4():
        nav_fn("simulate")
        _log(f"<span style='color:{C['accent']};font-weight:600'>Step 4/4</span>"
             f" &mdash; Running simulation&hellip;")
        if state.dataset is None:
            doc.add_timeout_callback(_finish, 1000)
            return
        sim_status.text = f"<span style='color:{C['warning']}'>Running&hellip;</span>"
        def _do():
            try:
                res = run_simulation(state.dataset, 0, 20, ["ADE","FDE","Collision"])
                def _done():
                    parts = [f"<b>{k.upper()}</b>: {v:.4f}"
                             for k, v in res.get("means", {}).items()]
                    sim_status.text = (
                        f"<span style='color:{C['success']}'>&#10003; {res['steps']} steps</span>"
                    )
                    _log(
                        f"<span style='color:{C['success']}'>&#10003; Simulation done &mdash; "
                        + " &middot; ".join(parts) + "</span>"
                    )
                    doc.add_timeout_callback(_finish, 1500)
                doc.add_next_tick_callback(_done)
            except Exception as e:
                doc.add_next_tick_callback(lambda: doc.add_timeout_callback(_finish, 500))
        _pool.submit(_do)

    def _finish():
        nav_fn("dashboard")
        _log(f"<span style='color:{C['success']};font-weight:600'>"
             f"Demo complete! All features working.</span>")
        demo_btn.disabled = False
        demo_btn.label    = "Run Demo"

    doc.add_timeout_callback(_step1, 300)


# ═══════════════════════════════════════════════════════════════════════════
# Root builder
# ═══════════════════════════════════════════════════════════════════════════

def build_ui(doc, state=None):
    import bokeh
    if state is None:
        state = AppState()
    
    # ── Current Config ──────────────────────────────────────────────────
    global C, S
    C = THEMES.get(state.theme, THEMES["dark"])
    S = STRINGS.get(state.lang, STRINGS["en"])
    cards    = {}
    refs     = {}

    # ── Panels ─────────────────────────────────────────────────────────
    dash_panel = _build_dashboard(state, cards)

    ds_panel, ds_status, stats_src, _, _ = _build_dataset_panel(
        doc, state, cards, lambda s: None
    )
    refs["ds_status"] = ds_status
    refs["stats_src"] = stats_src

    viz_panel, viz_refresh, viz_slider = _build_viz_panel(doc, state, cards)
    refs["viz_refresh"] = viz_refresh
    refs["viz_slider"]  = viz_slider

    aug_panel, aug_preview, aug_slider = _build_aug_panel(doc, state, cards)
    refs["aug_preview"] = aug_preview
    refs["aug_slider"]  = aug_slider

    sim_panel, sim_run_btn, sim_status, _ = _build_sim_panel(doc, state, cards)
    refs["sim_run_btn"] = sim_run_btn
    refs["sim_status"]  = sim_status

    exp_panel = _build_export_panel(doc, state, cards)

    # Wrap panels in containers
    panels = {
        "dashboard": dash_panel,
        "dataset":   ds_panel,
        "visualize": viz_panel,
        "augment":   aug_panel,
        "simulate":  sim_panel,
        "export":    exp_panel,
    }
    for p in panels.values():
        p.visible = False

    # ── Sidebar nav buttons ──────────────────────────────────────────────
    # Local helper for localized string
    def _t(k): return S.get(k, STRINGS["en"].get(k, k))

    NAV = [
        ("dashboard", _t("dashboard")),
        ("dataset",   _t("dataset")),
        ("visualize", _t("visualize")),
        ("augment",   _t("augment")),
        ("simulate",  _t("simulate")),
        ("export",    _t("export")),
    ]

    # Header title div (updated on nav change)
    header_title_div = _div(
        _title_html("Dashboard", "Overview & quick start"),
        width=380,
        styles={"display": "flex", "align-items": "center"},
    )

    def nav_fn(key: str):
        old = state.active_panel
        if old == key:
            return
        panels[old].visible = False
        panels[key].visible = True
        nav_btns[old].button_type = "default"
        nav_btns[key].button_type = "primary"
        state.active_panel = key
        t, s = _PANEL_TITLES[key]
        header_title_div.text = _title_html(t, s)

    panels["dashboard"].visible = True

    refs["nav_fn"] = nav_fn

    # ── Demo button ──────────────────────────────────────────────────────
    demo_btn = Button(
        label=_t("run_demo"),
        icon=_get_icon("run_demo", "white"),
        button_type="default",
        sizing_mode="stretch_width",
        height=52,
        stylesheets=[f"""
            :host .bk-btn {{
                background: {C['accent']} !important;
                border: none !important;
                color: white !important;
                font-weight: 700 !important;
                font-size: 16px !important;
                border-radius: 14px !important;
                height: 52px !important;
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                gap: 12px !important;
                box-shadow: 0 6px 16px {C['accent']}40 !important;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}
            :host .bk-btn:hover {{
                transform: scale(1.02) translateY(-1px) !important;
                box-shadow: 0 8px 20px {C['accent']}60 !important;
                filter: brightness(1.1) !important;
            }}
            :host .bk-btn:active {{
                transform: scale(0.98) !important;
            }}
            :host .bk-btn-group {{ display: block; width: 100%; }}
        """],
    )
    refs["demo_btn"] = demo_btn
    
    # Toast notification div (fixed position)
    toast_div = _div(_toast_html("Demo started", visible=False), width=0, height=0)
    refs["toast"] = toast_div

    def on_demo(_):
        _run_demo(doc, state, refs)

    demo_btn.on_click(on_demo)

    # ── Sidebar HTML sections ────────────────────────────────────────────
    # Local helper for localized string
    def _t(k): return S.get(k, STRINGS["en"].get(k, k))

    sidebar_logo = _div(f"""
<div style="
    height:72px; padding:0;
    border-bottom:1px solid {C['border']};
    display:flex;align-items:center;justify-content:center;
    width:{_SIDEBAR_W}px !important;
">
    <span style="
        color:{C['text']};font-size:17px;font-weight:700;
        font-family:'Inter',sans-serif;letter-spacing:-.02em;
    ">trajdata</span>
</div>
""", width=_SIDEBAR_W)

    sidebar_run_btn_wrapper = column(
        demo_btn, 
        width=_SIDEBAR_W, 
        styles={
            "padding": "12px", 
            "border-bottom": f"1px solid {C['border']}",
            "background": C['surface'],
        }
    )

    # Navigation items (secondary)
    nav_btns = {}
    for key, label in [
        ("dashboard", _t("dashboard")),
        ("dataset",   _t("dataset")),
        ("visualize", _t("visualize")),
        ("augment",   _t("augment")),
        ("simulate",  _t("simulate")),
        ("export",    _t("export")),
    ]:
        is_active = (key == state.active_panel)
        btn = _nav_btn(label, key, state.theme, partial(nav_fn, key), is_active)
        nav_btns[key] = btn

    sidebar_nav_wrapper = column(
        *[nav_btns[k] for k in nav_btns],
        spacing=2,
        width=_SIDEBAR_W,
        styles={"padding": "8px 12px", "flex": "1", "overflow-y": "auto"}
    )


    # ── Sidebar bottom (Theme & Language) ──────────────────────────────
    def theme_fn():
        state.theme = "light" if state.theme == "dark" else "dark"
        doc.clear()
        build_ui(doc, state)

    def lang_fn():
        state.lang = "tr" if state.lang == "en" else "en"
        doc.clear()
        build_ui(doc, state)

    theme_btn = Button(
        label=_t("theme_light") if state.theme == "dark" else _t("theme_dark"),
        width=_SIDEBAR_W - 24,
        height=36,
        stylesheets=[f"""
            :host .bk-btn {{ background: transparent; border: none !important; 
                color: {C['muted']}; font-size: 13px; font-weight: 500; text-align: left; }}
            :host .bk-btn:hover {{ background: {C['surface3']}; color: {C['text']}; }}
        """],
    )
    theme_btn.on_click(theme_fn)

    lang_btn = Button(
        label=_t("lang_switch"),
        width=_SIDEBAR_W - 24,
        height=36,
        stylesheets=[f"""
            :host .bk-btn {{ background: transparent; border: none !important; 
                color: {C['muted']}; font-size: 13px; font-weight: 500; text-align: left; }}
            :host .bk-btn:hover {{ background: {C['surface3']}; color: {C['text']}; }}
        """],
    )
    lang_btn.on_click(lang_fn)

    sidebar_bottom = column(
        theme_btn,
        lang_btn,
        width=_SIDEBAR_W,
        styles={
            "padding": "12px",
            "border-top": f"1px solid {C['border']}",
            "margin-top": "auto"
        }
    )

    sidebar_content = column(
        sidebar_logo,
        sidebar_run_btn_wrapper,
        sidebar_nav_wrapper,
        sidebar_bottom,
        spacing=0,
        width=_SIDEBAR_W,
        styles={
            "background": C["surface"],
            "height": "100%",
            "overflow": "hidden"
        }
    )

    # ── Header ──────────────────────────────────────────────────────────
    # Header Title Area

    header_right = _div(f"""
<div style="display:flex;align-items:center;height:80px;padding:0 32px">
</div>
""", width=120, styles={"flex-shrink": "0"})

    spacer_div = _div("", sizing_mode="stretch_width")

    header = row(
        header_title_div,
        spacer_div,
        header_right,
        toast_div,
        sizing_mode="stretch_width",
        spacing=0,
        styles={
            "background":    C["surface"],
            "border-bottom": f"1px solid {C['border']}",
            "height":        "72px",
            "position":      "sticky",
            "top":           "0",
            "z-index":       "100",
            "align-items":   "center",
            "margin":        "0",
            "padding":       "0",
            "width":         f"calc(100vw - {_SIDEBAR_W}px)",
        },
    )

    # ── Footer ──────────────────────────────────────────────────────────
    footer = _div(f"""
<div style="
    background:{C['surface']};
    border-top:1px solid {C['border']};
    padding:14px 28px;
    display:flex;align-items:center;justify-content:space-between;
    font-family:'Inter',sans-serif;
    width:calc(100vw - {_SIDEBAR_W}px) !important;
    box-sizing:border-box;
">
  <div style="display:flex;align-items:center;gap:8px">
    <span style="
      background:{C['accent']};
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;font-weight:700;font-size:14px;
    ">trajdata</span>
    <span style="color:{C['muted']};font-size:12px">Web UI &nbsp;·&nbsp;</span>
    <span style="
      color:{C['muted']};font-size:12px;cursor:default;
      border-bottom:1px dashed {C['border']};
    " title="Build: trajdata-webui-1.0.0">
      build #1.0.0
    </span>
  </div>
  <span style="color:{C['muted']};font-size:12px">
    © 2024 trajdata contributors &nbsp;·&nbsp; {_t("rights")}
    &nbsp;·&nbsp;
    <a href="https://github.com/hidirektor/trajdata" target="_blank" style="color:{C['accent']};text-decoration:none;
       font-family:'Inter',sans-serif" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">
      GitHub ↗
    </a>
  </span>
</div>
""", sizing_mode="stretch_width", min_width=200, styles={"margin": "0", "width": "100%", "min-width": "100%"})

    # ── Body ─────────────────────────────────────────────────────────────
    main_area = column(
        *panels.values(),
        sizing_mode="stretch_width",
        styles={
            "padding": "32px 40px",
            "flex": "1",
            "overflow-y": "auto",
        },
    )

    sidebar_col = column(
        sidebar_content,
        width=_SIDEBAR_W,
        styles={
            "background":    C["surface"],
            "border-right":  f"1px solid {C['border']}",
            "height":       "100vh",
            "flex-shrink":   "0",
        },
    )

    content_col = column(
        header,
        main_area,
        footer,
        sizing_mode="stretch_both",
        spacing=0,
        styles={
            "flex": "1",
            "height": "100vh",
            "overflow": "hidden",
            "display": "flex",
            "flex-direction": "column",
        }
    )

    root = row(
        sidebar_col,
        content_col,
        spacing=0,
        sizing_mode="stretch_both",
        styles={
            "background": C["bg"],
            "margin": "0",
            "padding": "0",
            "overflow": "hidden",
            "width": "100vw",
            "height": "100vh",
        },
    )
    doc.theme = "dark_minimal" if state.theme == "dark" else None
    doc.add_root(root)
    doc.title = "trajdata Web UI"
    doc.template_variables["extra_head"] = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body, html {{
    margin: 0; padding: 0;
    background: {C['bg']};
    font-family: 'Inter', -apple-system, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {C['surface']}; }}
  ::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {C['accent']}; }}
  .bk-root {{ background: {C['bg']} !important; }}
  .bk-root > .bk-row {{ width: 100% !important; height: 100% !important; }}
  .bk-root .bk-column:last-child {{ flex: 1 !important; width: auto !important; }}
  .bk-root .bk-column:last-child > .bk {{ width: 100% !important; min-width: 100% !important; }}
  .bk-root .bk-column:last-child div.bk-clearfix {{ width: 100% !important; }}
</style>
"""
