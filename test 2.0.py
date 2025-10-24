import streamlit as st
import pandas as pd
import random
import time

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Test de Matrices Progresivas de Raven - 60 Ítems",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Datos del Test (Simulación de 60 Preguntas) ---
# El Test de Raven Estándar Progresivo (SPM) tiene 5 series (A, B, C, D, E) de 12 ítems cada una.
# La cantidad de opciones cambia: 6 opciones para A y B, 8 opciones para C, D, y E.
# Las respuestas correctas (correct_answer) son simuladas para propósitos de demostración.

def generate_raven_data():
    """Genera la estructura de datos simulada para 60 preguntas."""
    data = []
    
    # Simulación de Respuestas Correctas (1-6 para A/B, 1-8 para C/D/E)
    # Estas respuestas deben ser ajustadas a tus datos reales.
    simulated_answers = {
        'A': [4, 5, 1, 2, 6, 3, 6, 2, 1, 3, 5, 4],
        'B': [2, 6, 1, 2, 1, 3, 5, 6, 4, 3, 4, 5],
        'C': [8, 2, 4, 1, 5, 3, 6, 8, 4, 7, 2, 3],
        'D': [6, 4, 7, 1, 2, 5, 3, 8, 7, 6, 5, 1],
        'E': [3, 4, 5, 2, 6, 1, 7, 8, 2, 1, 3, 5]
    }
    
    for series_name, answers in simulated_answers.items():
        for i, correct_ans in enumerate(answers):
            q_num = (len(data) + 1)
            num_options = 6 if series_name in ['A', 'B'] else 8
            
            data.append({
                'id': q_num,
                'series': series_name,
                'item': i + 1,
                'title': f"Serie {series_name}, Ítem {i + 1}",
                'num_options': num_options,
                # La respuesta correcta es un índice basado en 1 (1, 2, 3...)
                'correct_answer': correct_ans
            })
    return pd.DataFrame(data)

RAVEN_DATA = generate_raven_data()
TOTAL_QUESTIONS = len(RAVEN_DATA)

# --- 2. Funciones de Simulación Visual (Placeholders SVG) ---

def generate_svg_matrix_placeholder(series, item, is_matrix=True, option_index=None):
    """
    Genera un SVG simple como placeholder visual para la matriz o las opciones.
    Esto simula la complejidad creciente del test.
    """
    width = "100%"
    height = "150px" if is_matrix else "100px"
    fill_color = "#3b82f6"  # Azul primario

    # Variación simple del diseño basado en la serie para simular dificultad
    if series == 'A':
        shape = f'<circle cx="50%" cy="50%" r="40%" fill="{fill_color}" opacity="0.6"/>'
        label = "Patrón Simple"
    elif series == 'B':
        shape = f'<rect x="20%" y="20%" width="60%" height="60%" fill="{fill_color}" opacity="0.7"/>'
        label = "Relación de Figuras"
    elif series == 'C':
        shape = f'<polygon points="50,10 90,90 10,90" fill="{fill_color}" opacity="0.8"/>'
        label = "Cambio Continuo"
    else: # D y E
        shape = f'<rect x="10%" y="10%" width="80%" height="80%" fill="none" stroke="{fill_color}" stroke-width="5"/>'
        shape += f'<line x1="10%" y1="90%" x2="90%" y2="10%" stroke="{fill_color}" stroke-width="5" opacity="0.9"/>'
        label = "Descomposición/Filtros"
    
    if option_index is not None:
        label = f"Opción {option_index}"

    svg_code = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="border: 1px solid #ddd; border-radius: 8px;">
        {shape}
        <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="8" fill="#1e3a8a" font-weight="bold">{label}</text>
    </svg>
    """
    return svg_code

def display_question(q_data):
    """Muestra la matriz del problema y las opciones de respuesta."""
    st.subheader(f"Pregunta {q_data['id']}: {q_data['title']}")

    # 1. Mostrar la Matriz (El problema)
    matrix_svg = generate_svg_matrix_placeholder(q_data['series'], q_data['item'], is_matrix=True)
    st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <p style="text-align: center; font-style: italic;">Matriz Incompleta (Placeholder)</p>
            {matrix_svg}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Seleccione la pieza que completa la matriz:")

    # 2. Mostrar las Opciones de Respuesta
    num_options = q_data['num_options']
    options = list(range(1, num_options + 1))
    
    # Crear una lista de tuplas (índice, SVG) para las opciones
    option_elements = []
    for i in options:
        option_svg = generate_svg_matrix_placeholder(q_data['series'], q_data['item'], is_matrix=False, option_index=i)
        option_elements.append((i, option_svg))

    # Definir columnas para layout responsivo (4 columnas en escritorio, 2 en móvil)
    cols = st.columns(min(num_options, 4))
    
    # Usar st.radio dentro de un contenedor para mantener la selección
    current_answer = st.session_state.user_answers.get(q_data['id'], None)
    
    # Renderizar las opciones en las columnas
    for i, (index, svg_code) in enumerate(option_elements):
        with cols[i % len(cols)]:
            # Creamos un botón de radio para cada opción
            key = f"q{q_data['id']}_opt_{index}"
            
            # Usamos un identificador único para el radio button y un label vacío
            # El SVG y el número son el contenido visual.
            is_selected = (current_answer == index)
            
            # Estilo personalizado para las opciones (CSS injection)
            st.markdown(f"""
                <div 
                    data-option-id="{index}" 
                    style="
                        border: 3px solid {'#10b981' if is_selected else '#e5e7eb'}; 
                        border-radius: 8px; 
                        padding: 10px; 
                        margin-bottom: 15px; 
                        cursor: pointer;
                        text-align: center;
                        transition: all 0.2s;
                    "
                    id="{key}-container"
                    onclick="
                        var widget = window.parent.document.querySelector('div[data-testid=\"stFormSubmitButton\"]').parentElement;
                        if(widget) {{
                            var hiddenInput = document.getElementById('{key}-hidden');
                            if (hiddenInput) {{
                                hiddenInput.value = '{index}';
                                hiddenInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                // Simular un clic para actualizar el estado de Streamlit (Hack)
                                // En un entorno real de Streamlit, usaríamos un callback de radio button
                            }}
                        }}
                    "
                >
                    <p style="font-weight: bold; margin-bottom: 5px;">{index}</p>
                    {svg_code}
                </div>
            """, unsafe_allow_html=True)

    # Botón de radio invisible que realmente registra el valor
    # Esto es necesario porque Streamlit no permite usar radio buttons con layout complejo fácilmente
    
    selected_value = st.radio(
        "Selección:",
        options=options,
        index=options.index(current_answer) if current_answer is not None else -1,
        format_func=lambda x: f"Opción {x}", # Formato visible
        key=f"radio_{q_data['id']}",
        label_visibility="hidden" # Ocultamos el label de Streamlit
    )

    # Guardar la respuesta seleccionada
    st.session_state.user_answers[q_data['id']] = selected_value


# --- 3. Funciones de Navegación y Lógica Principal ---

def init_session_state():
    """Inicializa las variables de estado si no existen."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'start'  # 'start', 'test', 'results'
        st.session_state.current_q_index = 0
        st.session_state.user_answers = {}  # {q_id: answer_index}
        st.session_state.start_time = None
        st.session_state.user_info = {}

def start_test():
    """Cambia el estado a la página del test."""
    st.session_state.current_page = 'test'
    st.session_state.current_q_index = 0
    st.session_state.user_answers = {}
    st.session_state.start_time = time.time()
    st.rerun()

def next_question():
    """Avanza a la siguiente pregunta o finaliza el test."""
    if st.session_state.current_q_index < TOTAL_QUESTIONS - 1:
        st.session_state.current_q_index += 1
    else:
        # Finalizar el test
        st.session_state.end_time = time.time()
        st.session_state.current_page = 'results'
    st.rerun()

def prev_question():
    """Retrocede a la pregunta anterior."""
    if st.session_state.current_q_index > 0:
        st.session_state.current_q_index -= 1
    st.rerun()

def calculate_results():
    """Calcula la puntuación bruta y la clasificación (simulada)."""
    raw_score = 0
    correct_answers_df = RAVEN_DATA[['id', 'correct_answer']].set_index('id')
    
    for q_id, user_ans in st.session_state.user_answers.items():
        if q_id in correct_answers_df.index:
            correct_ans = correct_answers_df.loc[q_id, 'correct_answer']
            if user_ans == correct_ans:
                raw_score += 1

    # --- SIMULACIÓN DE CLASIFICACIÓN (Percentiles) ---
    # En un entorno profesional, aquí se utilizaría una tabla de baremos (normas)
    # ajustada por edad/país para obtener el Percentil y el Rango.
    
    percentile_map = {
        (0, 15): ("Grado V", "Deficiente (Percentil < 5)"),
        (16, 25): ("Grado IV", "Inferior al promedio (Percentil 5-25)"),
        (26, 40): ("Grado III", "Promedio (Percentil 25-75)"),
        (41, 50): ("Grado II", "Superior al promedio (Percentil 75-95)"),
        (51, 60): ("Grado I", "Intelectualmente superior (Percentil > 95)"),
    }
    
    classification = "No determinado"
    grade = "N/A"
    
    for (lower, upper), (g, c) in percentile_map.items():
        if lower <= raw_score <= upper:
            grade = g
            classification = c
            break
            
    st.session_state.results = {
        'raw_score': raw_score,
        'total_questions': TOTAL_QUESTIONS,
        'grade': grade,
        'classification': classification,
        'time_taken': st.session_state.end_time - st.session_state.start_time if st.session_state.start_time else 0
    }
    
    return st.session_state.results

# --- 4. Vistas de la Aplicación ---

def render_start_page():
    """Página de inicio y recolección de datos."""
    st.title("Test de Matrices Progresivas de Raven (SPM)")
    st.markdown("""
        Esta aplicación simula de manera profesional el formato del Test de Raven, 
        evaluando la capacidad intelectual, específicamente la deducción de relaciones y el razonamiento no verbal.
        El test consta de 60 ítems distribuidos en 5 series (A, B, C, D, E).
        
        **Instrucciones Generales:**
        1.  El test se presenta en series de dificultad creciente.
        2.  Debe observar la matriz incompleta y elegir la pieza (opción) que la completa lógicamente.
        3.  No hay límite de tiempo estricto, pero se recomienda trabajar de manera constante.
    """)
    
    st.markdown("---")
    
    with st.form("user_info_form"):
        st.subheader("Datos del Evaluado")
        name = st.text_input("Nombre Completo o Identificador", key="input_name")
        age = st.number_input("Edad (años)", min_value=5, max_value=99, step=1, key="input_age")
        
        submitted = st.form_submit_button("Comenzar Test", type="primary")
        
        if submitted:
            if name and age:
                st.session_state.user_info['name'] = name
                st.session_state.user_info['age'] = age
                start_test()
            else:
                st.error("Por favor, complete todos los campos para comenzar.")


def render_test_page():
    """Página de ejecución del test."""
    
    q_index = st.session_state.current_q_index
    q_data = RAVEN_DATA.iloc[q_index]

    # Barra lateral de progreso
    with st.sidebar:
        st.header("Progreso del Test")
        st.metric(label="Ítem Actual", value=f"{q_index + 1} / {TOTAL_QUESTIONS}")
        
        # Mapa de progreso con Series
        st.markdown("---")
        st.subheader("Series Completadas")
        
        progress_info = {}
        for series in RAVEN_DATA['series'].unique():
            series_data = RAVEN_DATA[RAVEN_DATA['series'] == series]
            series_ids = series_data['id'].tolist()
            answered = sum(1 for q_id in series_ids if st.session_state.user_answers.get(q_id) is not None)
            progress_info[series] = answered
            
            # Determinar el color del badge
            color = "blue" if answered == len(series_data) else ("orange" if answered > 0 else "gray")
            st.markdown(f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; margin-right: 5px;">{series}: {answered}/{len(series_data)}</span>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Tiempo transcurrido (Placeholder)
        elapsed_time = time.time() - st.session_state.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.metric("Tiempo Transcurrido", f"{minutes:02d}:{seconds:02d}")


    # Contenido principal de la pregunta
    col_main = st.columns([1])[0]
    with col_main:
        display_question(q_data)

        # Controles de Navegación
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if q_index > 0:
                st.button("← Anterior", on_click=prev_question)
        
        with col3:
            if q_index < TOTAL_QUESTIONS - 1:
                st.button("Siguiente →", on_click=next_question, type="primary")
            else:
                st.button("Finalizar Test", on_click=next_question, type="success")


def render_results_page():
    """Página de resultados y reporte."""
    
    # 1. Calcular resultados finales (se llama solo una vez)
    if 'results' not in st.session_state:
        results = calculate_results()
    else:
        results = st.session_state.results

    st.title("✅ Test Finalizado: Informe de Resultados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Puntuación Bruta (PB)", f"{results['raw_score']} / {results['total_questions']}", delta_color="off")
    
    time_minutes = int(results['time_taken'] // 60)
    time_seconds = int(results['time_taken'] % 60)
    
    with col2:
        st.metric("Tiempo de Ejecución", f"{time_minutes} min {time_seconds} seg", delta_color="off")

    with col3:
        # Puesto que las normas no están implementadas, este campo es genérico
        st.metric("Clasificación (Simulada)", results['grade'], delta_color="off")
        
    st.markdown("---")
    
    # 2. Resumen Interpretativo (Profesional)
    st.subheader("Interpretación Cuantitativa y Cualitativa")
    st.info(f"""
        **Nombre/ID:** {st.session_state.user_info.get('name', 'N/A')} | **Edad:** {st.session_state.user_info.get('age', 'N/A')} años.
        
        El evaluado obtuvo una Puntuación Bruta de **{results['raw_score']}** de 60. 
        Este resultado se sitúa en la clasificación **{results['grade']} ({results['classification']})** en las normas de referencia simuladas.
        
        Esto sugiere una capacidad para la deducción de relaciones y el razonamiento 
        analógico **{results['classification'].lower().split('(')[0].strip()}** respecto al grupo normativo.
    """)
    
    st.markdown("---")

    # 3. Análisis de Errores por Serie (para feedback cualitativo)
    st.subheader("Análisis Detallado por Serie")
    
    series_results = []
    correct_answers_df = RAVEN_DATA[['id', 'series', 'correct_answer']].set_index('id')
    
    for series in RAVEN_DATA['series'].unique():
        series_data = RAVEN_DATA[RAVEN_DATA['series'] == series]
        series_score = 0
        total_series = len(series_data)
        
        for index, row in series_data.iterrows():
            q_id = row['id']
            user_ans = st.session_state.user_answers.get(q_id)
            if user_ans == row['correct_answer']:
                series_score += 1
                
        series_results.append({
            'Serie': series,
            'Aciertos': series_score,
            'Errores': total_series - series_score,
            'Total': total_series
        })
    
    df_results = pd.DataFrame(series_results)
    st.dataframe(df_results, hide_index=True, use_container_width=True)
    
    st.markdown("""
        <p style="font-size: small; color: gray;">
        *Nota: Un aumento súbito en la tasa de errores en una serie superior (ej. C, D o E) indica el punto 
        donde la dificultad excede la capacidad actual del evaluado para formar nuevos constructos lógicos.*
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("Volver a la Página de Inicio"):
        del st.session_state.results
        init_session_state()
        st.rerun()

# --- 5. Controlador Principal ---

def main():
    """Función principal que maneja el flujo de la aplicación."""
    init_session_state()

    if st.session_state.current_page == 'start':
        render_start_page()
    elif st.session_state.current_page == 'test':
        render_test_page()
    elif st.session_state.current_page == 'results':
        render_results_page()

if __name__ == "__main__":
    # Es necesario agregar un pequeño script JS para forzar la actualización de los botones de radio
    # cuando se hace clic en los contenedores SVG/HTML.
    # En Streamlit puro, la lógica de state debe ser manejada con cuidado.
    st.markdown("""
        <style>
        /* Estilos generales para un look más moderno */
        .stButton>button {
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1e40af; /* Azul oscuro corporativo */
        }
        </style>
    """, unsafe_allow_html=True)
    
    main()
