import streamlit as st
from PIL import Image, ImageDraw
import io

# Parámetros principales
total_questions = 60
questions_per_series = 12
series_names = ['A', 'B', 'C', 'D', 'E']

# Función para generar imagen de matriz con patrón
# (simplificado: tres figuras geométricas con una parte faltante)
def draw_matrix_example(seed=0):
    import random
    random.seed(seed)
    img_size = 220
    img = Image.new('RGB', (img_size,img_size), 'white')
    draw = ImageDraw.Draw(img)
    shape_type = random.choice(['ellipse','rectangle','polygon'])
    if shape_type=='ellipse':
        draw.ellipse([40,40,img_size-40,img_size-40], outline="black", width=6)
        draw.rectangle([img_size//2-20, img_size//2-20, img_size//2+20, img_size//2+20], fill="white") # elemento faltante
    elif shape_type=='rectangle':
        draw.rectangle([20,20,img_size-20,img_size-120], outline="black", width=6)
        draw.rectangle([img_size//2-30, img_size//2-10, img_size//2+30, img_size//2+30], fill='white')
    else: # polígono
        points = [(50,50),(150,50),(190,170),(50,190)]
        draw.polygon(points, outline="black", width=6)
        draw.rectangle([120,120,180,180], fill="white")
    return img

# Generar listado de preguntas
questions = []
for sidx, sname in enumerate(series_names):
    for qidx in range(questions_per_series):
        questions.append({
            'id':sidx*questions_per_series+qidx+1,
            'series': sname,
            'image': draw_matrix_example(seed=sidx*questions_per_series+qidx),
            'options': [f"Opción {x+1}" for x in range(6)], # puedes añadir lógica para opciones geométricas reales
            'correct': 0 # dummy
        })

# Streamlit app
st.title("Test de Matrices Progresivas de Raven")
st.write("Versión digital profesional: 60 preguntas en cinco series.")

st.sidebar.header("Navegación del Test")
question_id = st.sidebar.number_input("Pregunta #", min_value=1,max_value=total_questions,value=1,step=1)

curr_question = questions[question_id-1]
st.subheader(f"Serie {curr_question['series']} / Pregunta {curr_question['id']}")

buf = io.BytesIO()
curr_question['image'].save(buf, format='PNG')
st.image(buf, caption="Matriz incompleta", use_column_width=True)

st.write("Selecciona la respuesta correcto:")
selected = st.radio("Opciones", curr_question['options'])

# Guardar selección (simulado)
if st.button("Guardar respuesta"):
    st.success("Respuesta guardada correctamente. Puede avanzar a la siguiente pregunta.")

# Resultados
if st.sidebar.button("Ver informe final"):
    st.write("Informe simulado: puntuación total -- próximamente integración completa de análisis psychométrico.")
