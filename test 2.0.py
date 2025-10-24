# streamlit_app.py
# ------------------------------------------------------------
# Test de Razonamiento Matricial "estilo Raven" (60 preguntas)
# - Genera y CACHEA imágenes de los ítems y alternativas (A–H)
# - Flujo: Inicio → Test (1 por pantalla; clic en alternativa avanza) → Resultados
# - Informe profesional y descarga en PDF
# - Diseño UI con estilo moderno y componentes visuales
# ------------------------------------------------------------

import os
import io
import json
import math
import time
import random
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import plotly.graph_objects as go

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# --------------- CONFIGURACIÓN DE PÁGINA --------------------
st.set_page_config(
    page_title="Test Matricial (estilo Raven) – 60 preguntas",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------- ESTILOS GLOBALES (CSS) ---------------------
CUSTOM_CSS = """
<style>
    /* Fondo con gradiente sutil */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1f2937 100%);
        color: #e5e7eb !important;
    }
    /* Tarjetas */
    .glass {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        border-radius: 16px;
        padding: 1.25rem;
    }
    .hero-title{
        font-size: 40px; font-weight: 800; letter-spacing: 0.4px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle{ color:#cbd5e1; font-size:16px; }
    .pill { display:inline-block; padding:6px 12px; border: 1px solid rgba(255,255,255,0.2); border-radius: 999px; font-size:12px; color:#cbd5e1; }
    .kpi{ text-align:center; }
    .kpi h2{ margin:0; font-size: 28px; }
    .kpi p{ margin:0; color:#cbd5e1; }
    .opt-btn button{ width:100%; height:64px; font-weight:700; border-radius:12px !important; }
    .progress-wrap{ height: 10px; border-radius:8px; background:rgba(255,255,255,0.1); }
    .progress-bar{ height: 10px; border-radius:8px; background:linear-gradient(90deg,#22d3ee,#a78bfa); }
    .caption{ color:#94a3b8; font-size:12px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- UTILIDADES DE RUTA Y CACHE -----------------
BASE_DIR = os.path.abspath(os.getcwd())
ASSETS_DIR = os.path.join(BASE_DIR, "assets", "raven_items")
os.makedirs(ASSETS_DIR, exist_ok=True)

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def get_font(size=24):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()

# --------------- GENERACIÓN DE ÍTEMS (SIMULACIÓN) -----------
# Nota ética/IP: No reproducimos ítems oficiales del RPM. Generamos problemas
# matriciales “estilo Raven” mediante reglas procedurales originales.
# Cada ítem crea: imagen del enunciado (3x3 con casilla faltante) + 8 opciones (A–H).

CANVAS_SIZE = 512          # tamaño de la imagen principal
GRID = 3                   # 3x3
CELL = CANVAS_SIZE // GRID
BORDER = 12               # borde interno
OPTION_SIZE = 140          # tamaño de alternativa

SHAPES = ["circle", "square", "triangle", "diamond", "star"]


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, size: int, fill, rotate_deg: int = 0):
    """Dibuja una figura geométrica simple en el centro indicado."""
    if shape == "circle":
        bbox = [cx - size, cy - size, cx + size, cy + size]
        draw.ellipse(bbox, fill=fill)
    elif shape == "square":
        bbox = [cx - size, cy - size, cx + size, cy + size]
        draw.rectangle(bbox, fill=fill)
    elif shape == "triangle":
        pts = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size),
        ]
        draw.polygon(pts, fill=fill)
    elif shape == "diamond":
        pts = [
            (cx, cy - size),
            (cx - size, cy),
            (cx, cy + size),
            (cx + size, cy),
        ]
        draw.polygon(pts, fill=fill)
    elif shape == "star":
        # Estrella simple 5 puntas
        pts = []
        for i in range(10):
            angle = i * math.pi / 5 + math.radians(rotate_deg)
            r = size if i % 2 == 0 else size * 0.5
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=fill)


def generate_panel(shape, count, size, rotation, shade):
    """Genera una celda con N figuras y parámetros dados."""
    img = Image.new("RGB", (CELL - BORDER*2, CELL - BORDER*2), (245, 246, 248))
    d = ImageDraw.Draw(img)
    rng = random.Random(count * 1000 + size + rotation + shade)
    for _ in range(count):
        cx = rng.randint(30, img.width-30)
        cy = rng.randint(30, img.height-30)
        sz = max(10, int(size * rng.uniform(0.85, 1.15)))
        col = (40+shade, 40+shade, 40+shade)
        rot = int(rotation + rng.randint(-10,10))
        draw_shape(d, shape, cx, cy, sz, fill=col, rotate_deg=rot)
    return img


def rule_progressions(difficulty_seed):
    """Define progresiones a lo largo de filas/columnas para shape, count, size, rotation, shade."""
    rng = random.Random(difficulty_seed)
    shape_seq = rng.sample(SHAPES, 3)  # una forma por fila
    count_start = rng.randint(1, 3)
    size_start = rng.randint(14, 22)
    rotation_start = rng.choice([0, 15, 30, 45])
    shade_start = rng.randint(10, 120)

    count_step_row = rng.choice([1, 1, 2])
    size_step_col = rng.choice([2, 3, 4])
    rotation_step_col = rng.choice([15, 30])
    shade_step_row = rng.choice([10, 15, 20])

    grid_params = []
    for r in range(3):
        row = []
        for c in range(3):
            shape = shape_seq[r]
            count = count_start + r * count_step_row
            size = size_start + c * size_step_col
            rotation = (rotation_start + c * rotation_step_col) % 360
            shade = min(200, shade_start + r * shade_step_row)
            row.append((shape, count, size, rotation, shade))
        grid_params.append(row)
    return grid_params


def compose_matrix_image(params_grid, missing_pos=(2,2)):
    """Crea la imagen 3x3 con la posición missing en blanco (símbolo '?')."""
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (230, 232, 236))
    d = ImageDraw.Draw(img)

    # rejilla
    for r in range(GRID):
        for c in range(GRID):
            x0 = c * CELL
            y0 = r * CELL
            block = Image.new("RGB", (CELL, CELL), (250, 251, 253))
            bd = ImageDraw.Draw(block)
            bd.rectangle([0,0,CELL-1,CELL-1], outline=(210,214,220), width=2)

            if (r, c) != missing_pos:
                shape, count, size, rotation, shade = params_grid[r][c]
                panel = generate_panel(shape, count, size, rotation, shade)
                block.paste(panel, (BORDER, BORDER))
            else:
                # símbolo de pregunta elegante
                bd.text((CELL//2-10, CELL//2-20), "?", fill=(120, 124, 130), font=get_font(64))

            img.paste(block, (x0, y0))

    return img


def generate_distractors(correct_params, rng):
    """Genera variaciones plausibles como distractores."""
    variants = []
    shape, count, size, rotation, shade = correct_params
    for _ in range(7):
        v_shape = rng.choice([shape] + [s for s in SHAPES if s != shape])
        v_count = max(1, int(round(count + rng.choice([-1, 1, 0, 2, -2]))))
        v_size = max(8, int(round(size + rng.choice([-4, -2, 2, 4, 6]))))
        v_rotation = (rotation + rng.choice([-30, -15, 0, 15, 30, 45])) % 360
        v_shade = min(220, max(10, shade + rng.choice([-20, -10, 0, 10, 20, 30])))
        variants.append((v_shape, v_count, v_size, v_rotation, v_shade))
    return variants


def render_option_image(params):
    img = Image.new("RGB", (OPTION_SIZE, OPTION_SIZE), (245, 246, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, OPTION_SIZE-1, OPTION_SIZE-1], outline=(210,214,220), width=2)
    shape, count, size, rotation, shade = params
    cell = OPTION_SIZE
    inner = Image.new("RGB", (cell-16, cell-16), (245,246,248))
    draw_inner = ImageDraw.Draw(inner)
    rng = random.Random(sum([hash(x) for x in params]))
    for _ in range(count):
        cx = rng.randint(20, inner.width-20)
        cy = rng.randint(20, inner.height-20)
        sz = max(8, int(size * rng.uniform(0.85, 1.15)))
        col = (40+shade, 40+shade, 40+shade)
        rot = int(rotation + rng.randint(-10,10))
        draw_shape(draw_inner, shape, cx, cy, sz, fill=col, rotate_deg=rot)
    img.paste(inner, (8,8))
    return img


def build_item_bank(total_items=60, master_seed=20251024):
    """Crea o carga metadatos e imágenes de 60 ítems. Persiste en disco para reuso."""
    item_bank = []
    rng = random.Random(master_seed)

    for idx in range(1, total_items+1):
        folder = os.path.join(ASSETS_DIR, f"item_{idx:02d}")
        os.makedirs(folder, exist_ok=True)
        meta_path = os.path.join(folder, "meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            item_bank.append(meta)
            continue

        # Dificultad incremental modulada por idx
        difficulty_seed = master_seed + idx * 777
        params_grid = rule_progressions(difficulty_seed)
        missing = (2, 2)
        stem_img = compose_matrix_image(params_grid, missing)

        # Parámetros correctos esperados en (2,2)
        correct_params = params_grid[missing[0]][missing[1]]

        # Generar opciones
        rng_item = random.Random(difficulty_seed * 33)
        distractors = generate_distractors(correct_params, rng_item)
        options_params = distractors[:7]
        correct_index = rng_item.randint(0, 7)
        options_params.insert(correct_index, correct_params)

        # Guardar imágenes
        stem_path = os.path.join(folder, "stem.png")
        stem_img.save(stem_path)

        options_paths = []
        for j, p in enumerate(options_params):
            opt_img = render_option_image(p)
            pth = os.path.join(folder, f"opt_{j}.png")
            opt_img.save(pth)
            options_paths.append(pth)

        meta = {
            "id": idx,
            "folder": folder,
            "stem": stem_path,
            "options": options_paths,
            "correct": correct_index,
            "difficulty": round(0.2 + 0.8 * (idx/total_items), 3)  # 0.2 → 1.0
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        item_bank.append(meta)

    return item_bank

# --------------- ESTADO & MANAGER ---------------------------
class State:
    @staticmethod
    def init():
        if "stage" not in st.session_state:
            st.session_state.stage = "inicio"
        if "item_bank" not in st.session_state:
            st.session_state.item_bank = build_item_bank(60, master_seed=20251024)
        if "q_idx" not in st.session_state:
            st.session_state.q_idx = 0
        if "answers" not in st.session_state:
            st.session_state.answers = {}  # {idx: {"choice": int, "correct": bool, "rt": float}}
        if "user" not in st.session_state:
            st.session_state.user = {"nombre": "", "edad": "", "educacion": ""}
        if "timer" not in st.session_state:
            st.session_state.timer = time.time()
        if "start_ts" not in st.session_state:
            st.session_state.start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "scroll_key" not in st.session_state:
            st.session_state.scroll_key = 0

    @staticmethod
    def next_question():
        st.session_state.q_idx += 1
        st.session_state.timer = time.time()
        st.session_state.scroll_key += 1
        st.experimental_rerun()

    @staticmethod
    def to_results():
        st.session_state.stage = "resultados"
        st.experimental_rerun()

# --------------- WIDGETS DE UI ------------------------------

def top_progress():
    total = len(st.session_state.item_bank)
    idx = st.session_state.q_idx
    pct = int(100 * idx / total)
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.write(f"**Progreso:** {idx} / {total}")
    st.markdown("""
        <div class='progress-wrap'>
            <div class='progress-bar' style='width:%d%%'></div>
        </div>
    """ % pct, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def scroll_to_top():
    # Fuerza el scroll al top tras avanzar
    js_code = f"""
        <script>
        setTimeout(function(){{
            var topEl = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if(topEl) topEl.scrollTo({{top:0, behavior:'auto'}});
            window.parent.scrollTo({{top:0, behavior:'auto'}});
        }}, 120);
        </script>
    """
    st.components.v1.html(js_code, height=0, key=f"scroll_{st.session_state.scroll_key}")


# --------------- PÁGINAS ------------------------------------

def page_inicio():
    with st.container():
        colA, colB = st.columns([1.2, 1])
        with colA:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("<div class='pill'>Evaluación cognitiva</div>", unsafe_allow_html=True)
            st.markdown("<div class='hero-title'>Test de Razonamiento Matricial (estilo Raven)</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='hero-subtitle'>Medición de razonamiento abstracto mediante patrones visuales. **60 ítems** con dificultad creciente. Este instrumento es una simulación inspirada en Raven, no sustituye una evaluación clínica/laboral estandarizada.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            with st.container():
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.subheader("Datos del participante")
                n = st.text_input("Nombre y Apellido", value=st.session_state.user.get("nombre", ""))
                e = st.text_input("Edad (opcional)", value=st.session_state.user.get("edad", ""))
                ed = st.text_input("Educación / Cargo (opcional)", value=st.session_state.user.get("educacion", ""))
                st.session_state.user = {"nombre": n, "edad": e, "educacion": ed}

                st.markdown("—")
                st.checkbox("Confirmo que realizo esta prueba de forma voluntaria y en un contexto informativo.", key="consent", value=True)
                if st.button("🚀 Comenzar la evaluación", use_container_width=True, type="primary"):
                    st.session_state.stage = "test"
                    st.session_state.timer = time.time()
                    st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.subheader("Resumen del Test")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("<div class='kpi'><h2>60</h2><p>Preguntas</p></div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='kpi'><h2>20–35 min</h2><p>Duración típica</p></div>", unsafe_allow_html=True)
            with c3:
                st.markdown("<div class='kpi'><h2>3×3</h2><p>Matriz visual</p></div>", unsafe_allow_html=True)

            st.write("")
            st.caption("Las imágenes de los ítems se generan y cachean en disco para acelerar ejecuciones posteriores durante la vida de esta sesión/servidor.")
            st.markdown("</div>", unsafe_allow_html=True)



def page_test():
    scroll_to_top()
    top_progress()

    idx = st.session_state.q_idx
    total = len(st.session_state.item_bank)

    if idx >= total:
        State.to_results()
        return

    item = st.session_state.item_bank[idx]

    st.write("")
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(f"### Pregunta {idx+1} de {total}")
    st.caption("Observa el patrón en la matriz y selecciona la alternativa que complete la celda faltante.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(item["stem"], use_column_width=True, caption="Matriz 3×3 (celda inferior derecha faltante)")

    with col2:
        st.markdown("#### Alternativas")
        opt_cols = st.columns(2)
        labels = list("ABCDEFGH")
        buttons = []
        for i, pth in enumerate(item["options"]):
            with opt_cols[i % 2]:
                st.image(pth, use_column_width=True)
                clicked = st.button(f"Elegir {labels[i]}", key=f"btn_{idx}_{i}", use_container_width=True)
                buttons.append(clicked)

    st.markdown("</div>", unsafe_allow_html=True)

    # Manejo de respuesta
    for i, clicked in enumerate(buttons):
        if clicked:
            rt = time.time() - st.session_state.timer
            is_correct = (i == item["correct"])
            st.session_state.answers[idx] = {
                "choice": i,
                "correct": bool(is_correct),
                "rt": round(rt, 3),
                "difficulty": item.get("difficulty", 0.5)
            }
            if idx+1 >= total:
                State.to_results()
            else:
                State.next_question()
            return

    # Controles inferiores
    st.write("")
    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button("⏭️ Omitir esta pregunta", use_container_width=True):
            rt = time.time() - st.session_state.timer
            st.session_state.answers[idx] = {"choice": None, "correct": False, "rt": round(rt, 3), "difficulty": item.get("difficulty", 0.5)}
            State.next_question()
    with cB:
        if st.button("⏹️ Finalizar ahora", use_container_width=True):
            State.to_results()
    with cC:
        st.write("")


# --------------- CÁLCULOS DE RESULTADOS ---------------------

def compute_score(answers: dict):
    total = len(st.session_state.item_bank)
    answered = len(answers)
    correct = sum(1 for k,v in answers.items() if v.get("correct"))
    avg_rt = np.mean([v.get("rt", 0) for v in answers.values()]) if answered else 0

    # Ponderar por dificultad para una métrica adicional
    weighted = 0.0
    max_w = 0.0
    for idx, v in answers.items():
        diff = v.get("difficulty", 0.5)
        w = 0.5 + diff  # 0.7–1.5
        max_w += w
        if v.get("correct"):
            weighted += w
    weighted_pct = (weighted / max_w) if max_w > 0 else 0

    raw_pct = correct / total

    # Estimación de percentil heurístico (no clínico):
    # curva logística suave para visualización.
    perc = int(100 * (1 / (1 + math.exp(-10 * (raw_pct - 0.5)))))

    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "raw_pct": raw_pct,
        "avg_rt": avg_rt,
        "weighted_pct": weighted_pct,
        "percentile_est": perc
    }


def narrative_from_score(s):
    pct = s["raw_pct"]
    perc = s["percentile_est"]

    if pct >= 0.85:
        level = "Muy alto"
        summary = (
            "El desempeño sugiere una **capacidad sobresaliente** de razonamiento abstracto, detección de patrones y flexibilidad cognitiva."
            " La velocidad de resolución y la precisión indican una elevada eficiencia atencional y control ejecutivo."
        )
        recs = [
            "Buscar entornos de alta complejidad analítica.",
            "Participar en tareas de modelamiento, ciencia de datos o resolución de problemas.",
            "Mentorías o liderazgo técnico en proyectos.",
        ]
    elif pct >= 0.70:
        level = "Alto"
        summary = (
            "Resultados **superiores al promedio**. Muestra buena discriminación de reglas visuales, adaptabilidad y consistencia."
        )
        recs = [
            "Desafíos graduales con mayor carga analítica.",
            "Prácticas en razonamiento lógico bajo tiempo controlado.",
            "Roles que involucren análisis comparativo y toma de decisiones.",
        ]
    elif pct >= 0.50:
        level = "Medio"
        summary = (
            "Desempeño **promedio**: capacidad adecuada para identificar patrones, "
            "con oportunidades de mejora en rapidez y/o consistencia en ítems difíciles."
        )
        recs = [
            "Entrenar con rompecabezas visuales y ejercicios de series.",
            "Practicar bajo límite de tiempo moderado para mejorar ritmo.",
            "Revisar estrategias de verificación antes de responder.",
        ]
    elif pct >= 0.30:
        level = "Medio-bajo"
        summary = (
            "Resultados **bajo el promedio**: se observan aciertos en ítems de dificultad baja-media,"
            " con mayor desafío en reglas combinadas y distractores sutiles."
        )
        recs = [
            "Repasar principios de progresiones (cantidad, tamaño, rotación, sombreado).",
            "Ejercitar análisis por filas/columnas antes de integrar reglas.",
            "Aumentar exposición gradual a matrices con múltiples reglas.",
        ]
    else:
        level = "Bajo"
        summary = (
            "El puntaje indica **dificultades en la detección consistente de patrones** en matrices."
            " Puede haber interferencia por ansiedad de ejecución o manejo del tiempo."
        )
        recs = [
            "Practicar con guías paso a paso y ejemplos resueltos.",
            "Uso de técnicas de respiración/pausas breves para manejo del estrés.",
            "Repetición espaciada con feedback inmediato.",
        ]

    return level, summary, recs, perc


# --------------- PDF ----------------------------------------

def build_pdf_bytes(user, scores, details, start_ts):
    nombre = user.get("nombre") or "Participante"
    if not FPDF_AVAILABLE:
        # Fallback simple: generar un PDF mínimo en bytes manualmente no es trivial.
        # Indicamos que FPDF no está disponible.
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Encabezado
    pdf.set_fill_color(17, 24, 39)
    pdf.rect(0,0, 210, 30, 'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_xy(10,10)
    pdf.cell(0,10, 'Informe – Test Matricial (estilo Raven)', ln=1)

    pdf.set_text_color(0,0,0)

    pdf.ln(5)
    pdf.set_font('Arial','',12)
    pdf.cell(0,8, f"Participante: {nombre}", ln=1)
    if user.get("edad"):
        pdf.cell(0,8, f"Edad: {user.get('edad')}", ln=1)
    if user.get("educacion"):
        pdf.cell(0,8, f"Educación/Cargo: {user.get('educacion')}", ln=1)
    pdf.cell(0,8, f"Fecha/Hora de inicio: {start_ts}", ln=1)
    pdf.cell(0,8, f"Fecha/Hora de emisión: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)

    pdf.ln(4)
    pdf.set_font('Arial','B',13)
    pdf.cell(0,8, "Resumen de resultados", ln=1)

    pdf.set_font('Arial','',12)
    pdf.cell(0,8, f"Aciertos: {scores['correct']} de {scores['total']} ( {int(scores['raw_pct']*100)}% )", ln=1)
    pdf.cell(0,8, f"Tiempo medio por ítem: {scores['avg_rt']:.1f} s", ln=1)
    pdf.cell(0,8, f"Percentil estimado (referencial): {scores['percentile_est']}", ln=1)

    level, summary, recs, perc = narrative_from_score(scores)
    pdf.ln(4)
    pdf.set_font('Arial','B',13)
    pdf.cell(0,8, f"Perfil: {level}", ln=1)
    pdf.set_font('Arial','',12)
    pdf.multi_cell(0,7, summary)

    pdf.ln(2)
    pdf.set_font('Arial','B',12)
    pdf.cell(0,8, "Recomendaciones", ln=1)
    pdf.set_font('Arial','',12)
    for r in recs:
        pdf.multi_cell(0,7, f"• {r}")

    pdf.add_page()
    pdf.set_font('Arial','B',13)
    pdf.cell(0,8, "Detalle de respuestas", ln=1)
    pdf.set_font('Arial','',11)
    pdf.cell(0,8, "Ítem    Resp.     Correcto     RT(s)", ln=1)
    for i in range(scores['total']):
        v = details.get(i)
        if v is None:
            pdf.cell(0,6, f"{i+1:02d}      —         —           —", ln=1)
        else:
            ch = v.get('choice')
            ch_label = '-' if ch is None else chr(ord('A') + ch)
            corr = 'Sí' if v.get('correct') else 'No'
            rt = f"{v.get('rt',0):.1f}"
            pdf.cell(0,6, f"{i+1:02d}      {ch_label:>2}         {corr:^7}       {rt:>5}", ln=1)

    pdf.ln(4)
    pdf.set_font('Arial','I',9)
    pdf.multi_cell(0,5, (
        "Nota: Este instrumento reproduce el formato matricial de manera original para fines formativos/ocupacionales."
        " No reemplaza baterías estandarizadas ni interpretación clínica."
    ))

    out = pdf.output(dest='S').encode('latin-1', 'ignore')
    return out


# --------------- RESULTS PAGE --------------------------------

def page_resultados():
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Informe de resultados</div>", unsafe_allow_html=True)
    st.caption("Reporte interpretativo generado automáticamente. Percentiles estimados de forma referencial.")

    scores = compute_score(st.session_state.answers)
    level, summary, recs, perc = narrative_from_score(scores)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("<div class='kpi'><h2>"+str(scores['correct'])+f"/{scores['total']}</h2><p>Aciertos</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><h2>"+str(int(scores['raw_pct']*100))+"%</h2><p>Precisión</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><h2>"+str(scores['percentile_est'])+"</h2><p>Percentil (est.)</p></div>", unsafe_allow_html=True)

    # Gráfico gauge de percentil
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = scores['percentile_est'],
        title = {'text': "Percentil estimado"},
        gauge = {'axis': {'range': [None, 100]}, 'bar': {'thickness': 0.3}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Perfil: {level}")
    st.write(summary)

    st.markdown("**Recomendaciones prácticas**")
    for r in recs:
        st.markdown(f"- {r}")

    # Tabla simple de detalle (preview)
    with st.expander("Ver detalle de respuestas (preview)"):
        labels = list("ABCDEFGH")
        rows = []
        for i in range(scores['total']):
            v = st.session_state.answers.get(i)
            if v is None:
                rows.append([i+1, "—", "—", "—"]) 
            else:
                ch = v.get('choice')
                ch_label = '-' if ch is None else labels[ch]
                rows.append([i+1, ch_label, "Sí" if v.get('correct') else "No", v.get('rt')])
        st.dataframe(rows, use_container_width=True, hide_index=True, column_config={
            0: st.column_config.NumberColumn("Ítem"),
            1: st.column_config.TextColumn("Respuesta"),
            2: st.column_config.TextColumn("Correcto"),
            3: st.column_config.NumberColumn("RT (s)")
        })

    # Botones
    colx, coly = st.columns([1,1])
    with colx:
        if st.button("🔁 Rehacer prueba", use_container_width=True):
            # Reinicia estado manteniendo ítems cacheados en disco
            st.session_state.q_idx = 0
            st.session_state.answers = {}
            st.session_state.stage = "inicio"
            st.experimental_rerun()

    with coly:
        pdf_bytes = build_pdf_bytes(st.session_state.user, scores, st.session_state.answers, st.session_state.start_ts)
        if pdf_bytes:
            st.download_button(
                label="⬇️ Descargar informe en PDF",
                data=pdf_bytes,
                file_name=f"Informe_Matricial_{st.session_state.user.get('nombre','Participante').replace(' ','_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Para la descarga en PDF se requiere la librería `fpdf`. Instálala con `pip install fpdf`. Luego reinicia la app.")

    st.markdown("</div>", unsafe_allow_html=True)


# --------------- ROUTER -------------------------------------
State.init()

if st.session_state.stage == "inicio":
    page_inicio()
elif st.session_state.stage == "test":
    page_test()
else:
    page_resultados()
