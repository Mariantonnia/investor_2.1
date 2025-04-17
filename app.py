import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Cargar variables de entorno
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Configurar LLM
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Preguntas iniciales al inversor
preguntas_inversor = [
    "¿Cuál es tu objetivo principal al invertir?",
    "¿Cuál es tu horizonte temporal de inversión?",
    "¿Tienes experiencia previa invirtiendo en activos de mayor riesgo como acciones, criptomonedas o fondos alternativos?",
    "¿Estás dispuesto a sacrificar parte de la rentabilidad potencial a cambio de un impacto social o ambiental positivo?",
    "¿Qué opinas sobre el cambio climático?"
]

# Noticias para análisis
noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
]

# Plantillas de LLM
plantilla_evaluacion = """
Evalúa si esta respuesta del usuario es suficientemente detallada para un análisis ESG. 
Criterios:
- Claridad de la opinión
- Especificidad respecto a la noticia
- Mención de aspectos ESG (ambiental, social, gobernanza o riesgo)
- Identificación de preocupaciones o riesgos

Respuesta del usuario: {respuesta}

Si es vaga o superficial, responde "False".
Si contiene opinión sustancial y analizable, responde "True".

Solo responde "True" o "False".
"""
prompt_evaluacion = PromptTemplate(template=plantilla_evaluacion, input_variables=["respuesta"])
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

plantilla_reaccion = """
Reacción del inversor: {reaccion}
Genera ÚNICAMENTE una pregunta de seguimiento enfocada en profundizar en su opinión.
Ejemplo:  
"¿Consideras que la existencia de mecanismos robustos de control interno y transparencia podría mitigar tu preocupación por la gobernanza corporativa en esta empresa?"
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de reacciones: {analisis}
Genera un perfil del inversor basado en ESG (Ambiental, Social y Gobernanza) y aversión al riesgo.
Asigna puntuaciones de 0 a 100:

Formato:
Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Inicializar estados
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador_noticias = 0
    st.session_state.reacciones = []
    st.session_state.pregunta_general_idx = 0
    st.session_state.pregunta_pendiente = False
    st.session_state.mostrar_cuestionario = False
    st.session_state.cuestionario_enviado = False
    st.session_state.perfil_valores = {}
    st.session_state.esperando_respuesta = False

# Interfaz
st.title("Chatbot de Análisis de Inversor ESG")
st.markdown("""
**Primero interactuarás con un chatbot para evaluar tu perfil ESG.** 
**Al final, completarás un test tradicional de perfilado.**
""")

# Mostrar historial de conversación
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"], avatar="🤖" if mensaje["tipo"] == "bot" else None):
        st.write(mensaje["contenido"])

# Lógica principal de la conversación
if st.session_state.cuestionario_enviado:
    st.markdown("### ¡Gracias por completar tu perfil de inversor!")
elif st.session_state.mostrar_cuestionario:
    # Sección del cuestionario final
    st.header("Cuestionario Final de Perfilado")
    
    with st.form("formulario_final"):
        # [Todos los campos del cuestionario permanecen igual]
        
        enviar = st.form_submit_button("Enviar respuestas")
        if enviar:
            try:
                # [Código para enviar respuestas permanece igual]
                st.session_state.cuestionario_enviado = True
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error al guardar datos: {str(e)}")
elif st.session_state.pregunta_general_idx < len(preguntas_inversor):
    # Fase de preguntas generales
    if not st.session_state.esperando_respuesta:
        pregunta = preguntas_inversor[st.session_state.pregunta_general_idx]
        st.session_state.historial.append({"tipo": "bot", "contenido": pregunta})
        st.session_state.esperando_respuesta = True
        st.rerun()
    
    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.reacciones.append(user_input)
        st.session_state.pregunta_general_idx += 1
        st.session_state.esperando_respuesta = False
        st.rerun()
else:
    # Fase de análisis de noticias
    if st.session_state.contador_noticias < len(noticias):
        if not st.session_state.esperando_respuesta:
            noticia = noticias[st.session_state.contador_noticias]
            mensaje_noticia = f"**Noticia para analizar:** {noticia}\n\n¿Qué opinas sobre esta noticia?"
            st.session_state.historial.append({"tipo": "bot", "contenido": mensaje_noticia})
            st.session_state.esperando_respuesta = True
            st.rerun()
        
        user_input = st.chat_input("Escribe tu respuesta aquí...")
        if user_input:
            st.session_state.historial.append({"tipo": "user", "contenido": user_input})
            
            evaluacion = cadena_evaluacion.run(respuesta=user_input).strip().lower()
            if evaluacion == "false":
                pregunta_seguimiento = cadena_reaccion.run(reaccion=user_input).strip()
                st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
                st.session_state.pregunta_pendiente = True
            else:
                st.session_state.reacciones.append(user_input)
                st.session_state.contador_noticias += 1
                st.session_state.pregunta_pendiente = False
            
            st.session_state.esperando_respuesta = False
            st.rerun()
    else:
        # Generación del perfil final
        analisis_total = "\n".join(st.session_state.reacciones)
        perfil = cadena_perfil.run(analisis=analisis_total)

        puntuaciones = {
            "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
            "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
            "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
            "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
        }
        st.session_state.perfil_valores = puntuaciones

        st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})
        
        fig, ax = plt.subplots()
        ax.bar(puntuaciones.keys(), puntuaciones.values(), color="skyblue")
        ax.set_ylabel("Puntuación (0-100)")
        ax.set_title("Perfil del Inversor")
        st.pyplot(fig)
        
        st.session_state.mostrar_cuestionario = True
        st.rerun()
