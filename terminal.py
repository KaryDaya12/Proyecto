import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from sqlalchemy import create_engine
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------- Configuración de página ----------------------
st.set_page_config(page_title="Eficacia de Moodle", layout="wide", page_icon="📘")

# ---------------------- Inyectar estilos personalizados ----------------------
with open("style/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------- Mostrar logo y título ----------------------
logo = Image.open("Imagenes/logotecazuay.PNG")
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Proyecto Integrador - Eficacia de Moodle en la carrera de Big Data")
    st.markdown("#### Integrantes: Karina Chisaguano, Nube Gutierrez, Jimmy Sumba y Freddy Montalvan")
with col2:
    st.image(logo, width=250)

#Conexion a la bases de datos

db_url = "postgresql://dbproyecto_qvbd_user:hlqQwLvV3y35eDzferkuM6SN3ouQUTry@dpg-d1r4nqur433s739s11u0-a.oregon-postgres.render.com/dbproyecto_qvbd"
engine = create_engine(db_url)


# ---------------------- Tabs por materia ----------------------
tab0, tab1, tab2,tab3,tab4= st.tabs([
    " Inicio",
    "🗃️ Bases de Datos ll",
    "📦 Bases de Datos No Relacionales",
    "📚 Aprendizaje Profundo",
    "📊 Resultados"
])

with tab0:

    st.header("💡¿Sabías qué?")
   # Primer bloque
    st.markdown("""
    <div style="
        background-color: #1e3a5f;
        border-left: 6px solid #f1c40f;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        color: #ffffff;
        text-align: justify;
        line-height: 1.3;
    ">
        En el <strong>Instituto Superior Tecnológico del Azuay</strong>, Moodle es una herramienta clave para la gestión académica. 
        Sin embargo, su falta de <em>visualizaciones dinámicas</em> dificulta el análisis del rendimiento estudiantil.
        Este proyecto propone un <strong>tablero externo de visualización</strong>, conectado a Moodle mediante su API, que permite presentar datos clave de forma clara e interactiva, facilitando así la toma de decisiones basadas en datos.
    </div>
    """, unsafe_allow_html=True)

# Separación vertical (puedes aumentar a <br><br><br> si quieres más espacio)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Segundo bloque con columnas
    col1, col2, col3 = st.columns([1, 0.2, 3])

    with col1:
        st.markdown("""
        <div style="
            background-color: #5078a0; 
            padding: 20px; 
            font-weight: bold; 
            color: white;
            text-align: center;
            border-radius: 5px;
            font-size: 20px;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        ">
            Moodle
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <span style="font-size: 40px; color: #0033a0;">&#8594;</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">
            La plataforma Moodle es un sistema de enseñanza diseñado para crear y gestionar espacios de aprendizaje online adaptados a las necesidades de profesores, estudiantes y administradores.
        </div>
        """, unsafe_allow_html=True)
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background-color: #5078a0; 
            padding: 20px; 
            font-weight: bold; 
            color: white;
            text-align: center;
            border-radius: 5px;
            font-size: 20px;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        ">
            Problemática
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <span style="font-size: 40px; color: #0033a0;">&#8594;</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">
            El uso de Moodle en el Instituto es constante, pero no se analiza su eficacia real en el aprendizaje.   </div>
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">    
            Falta de aprovechamiento de los datos que genera la plataforma (actividades, tiempos, participación)</div>
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">
            No hay evidencia clara que relacione el uso de Moodle con el rendimiento académico.</div>
        """, unsafe_allow_html=True)
        
    
    with col1:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background-color: #5078a0; 
            padding: 20px; 
            font-weight: bold; 
            color: white;
            text-align: center;
            border-radius: 5px;
            font-size: 20px;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        ">
            Objetivo del Proyecto
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <span style="font-size: 40px; color: #0033a0;">&#8594;</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">
            Evaluar la eficacia de la plataforma Moodle.   </div>
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">    
            Recopilar y exportar datos relevantes desde Moodle </div>
        <div style="
            font-weight: bold; 
            font-size: 18px; 
            color: white;
        ">
            Desarrollar un tablero web utilizando Streamlit y la biblioteca Plotly.</div>
        """, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.header("Metodología CRISP-DM")
        metar = Image.open("Imagenes/metar.PNG")
        buffered = BytesIO()
        metar.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
        f"""
        
        <img src="data:image/png;base64,{img_b64}" width="800">
        
        """,
        unsafe_allow_html=True
            )
    st.header("Impacto del Proyecto")
   # Primer bloque
    st.markdown("""
    <div style="
        background-color: #1e3a5f;
        border-left: 6px solid #f1c40f;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        color: #ffffff;
        text-align: justify;
        line-height: 1.3;
    ">
        <div>
        <li>Seguimiento en tiempo real del desempeño estudiantil.</li>
        <li>Decisiones pedagógicas informadas y oportunas.</li>
        <li>Interfaz accesible para todo tipo de usuarios.</li>
        <li>Predicción de rendimiento con Deep Learning.</li>
        <li>Gestión de datos eficiente con MySQL y MongoDB.</li>
    </ul>
    </div>

    """, unsafe_allow_html=True)
    st.header("Proyección Futura")
   # Primer bloque
    st.markdown("""
    <div style="
        background-color: #1e3a5f;
        border-left: 6px solid #f1c40f;
        padding: 10px;
        border-radius: 10px;
        font-size: 20px;
        color: #ffffff;
        text-align: justify;
        line-height: 1.3;
    ">
        <div>
        <li>Integración de análisis predictivo avanzado.</li>
        <li>Adaptación a otras carreras del Instituto.</li>
        <li>Indicadores personalizados por área académica.</li>
        <li>Alertas automáticas para docentes y estudiantes.</li>
        <li>Compatibilidad con otras plataformas educativas.</li>
    </ul>
    </div>

    """, unsafe_allow_html=True)
    st.empty()
    

with tab1:
    st.header("🗃️ Base de Datos ll")
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Modelo Relacional:</p>",
    unsafe_allow_html=True)
    modelor = Image.open("Imagenes/modelor.PNG")
    buffered = BytesIO()
    modelor.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Creación de tablas:</p>",
    unsafe_allow_html=True)
    Tablas = Image.open("Imagenes/Tablas.PNG")
    buffered = BytesIO()
    Tablas.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="400">
    </div>
    """,
    unsafe_allow_html=True
            )
    
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Consulta generada:</p>",
    unsafe_allow_html=True)
    consulta = Image.open("Imagenes/consulta.PNG")
    buffered = BytesIO()
    consulta.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )
    st.empty()

with tab2:
    st.header("📦 Base de Datos No Relacional (NoSQL)")
    st.markdown("""Se utilizó MongoDB como base de datos No Relacional porque los datos de la API de Moodle, obtenidos en formato JSON con Insomnia, se adaptaban perfectamente a su modelo documental. Esta estructura flexible permitió almacenar información como usuarios, cursos y actividades sin un esquema rígido. Además, se aprovechó MongoDB Atlas para facilitar el acceso remoto, la escalabilidad y la integración con aplicaciones en Python. 
                Así, se logró una arquitectura eficiente para almacenar y consultar los datos extraídos para su análisis. """)
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Arquitectura Mongodb:</p>",
    unsafe_allow_html=True)
    arquitectura = Image.open("Imagenes/copia.PNG")
    buffered = BytesIO()
    arquitectura.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Seguridad:</p>",
    unsafe_allow_html=True)
    st.markdown("""Se implementaron credenciales de acceso y se configuraron roles y permisos para los usuarios. 
                Esto garantizó que solo personas autorizadas pudieran realizar operaciones de lectura y escritura en la base de datos.""")
    seguridad = Image.open("Imagenes/seguridad.PNG")
    buffered = BytesIO()
    seguridad.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )   
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Implementación de MongoDB Atlas, se siguieron los siguientes pasos:</p>",
    unsafe_allow_html=True)
    st.markdown("""Creación de una cuenta en MongoDB Atlas: Se registró en la plataforma y 
                se creó un nuevo proyecto para gestionar la base de datos.""")
    st.markdown("""Configuración de un clúster: Se configuró un clúster gratuito para alojar la base de datos,para trabajar en conjunto 
                se creó un usuario con permisos de lectura y escritura.
                """)
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Creación de colecciones:</p>",
    unsafe_allow_html=True)
    coleccion = Image.open("Imagenes/colecciones.PNG")
    buffered = BytesIO()
    coleccion.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="400">
    </div>
    """,
    unsafe_allow_html=True
            )   
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Inserción de datos:</p>",
    unsafe_allow_html=True)
    st.markdown("""Para la Inserción de datos, se utilizó los JSON descargados.""")
    captura = Image.open("Imagenes/captura.PNG")
    buffered = BytesIO()
    captura.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )
    st.markdown(
    "<p style='font-size:24px; font-weight:bold;'>Consulta:</p>",
    unsafe_allow_html=True)
    conm = Image.open("Imagenes/conm.PNG")
    buffered = BytesIO()
    conm.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_b64}" width="700">
    </div>
    """,
    unsafe_allow_html=True
            )
    st.empty()
    
with tab3:
    st.header("📚 Aplicación de Aprendizaje Profundo")

# Título principal
    st.markdown("## 🎓 Objetivo General")

    # Contenedor con estilo
    with st.container():
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
            <p style='font-size: 18px; color: #333;'>
                Desarrollar un <strong>modelo de Deep Learning</strong> que permita predecir si la plataforma 
                <strong>Moodle</strong> es eficaz, utilizando variables como:
            </p>
            <ul style='font-size: 17px; color: #444;'>
                <li>📘 Notas ponderadas a 10 puntos</li>
                <li>⏱️ Fecha de ultimo acceso a la plataforma </li>
                <li>⏱️ Tiempos de apertura y cierre de evaluaciones</li>
                <li>⏱️ Tiempos de apertura y cierre de tareas</li>
            </ul>
            <p style='font-size: 18px; color: #333;'>
                Todo esto con el fin de <strong>analizar la eficacia de Moodle</strong> en el acompañamiento del proceso académico.
            </p>
        </div>
        """, unsafe_allow_html=True)

        
        st.subheader("👀 Vista Previa de los Datos Originales Pruebas")
        query5 = """
        SELECT 
            e.id_est, 
            CONCAT(e.nombre_est, ' ', e.apellido_est) AS Nombre_estudiante,
            c.nombre_curso,
            pc.descripcion AS Periodo,
            p.nombre AS Nombre_prueba, 
            pre.nota AS Nota_Prueba,
            pre.fecha_entrega AS Fecha_Prueba,
            CONCAT(d.nombre_docente, ' ', d.apellido_docente) AS Nombre_docente, 
            p.calificacion_max AS Calmax_prueba,
            p.fecha_limite AS fecha_liprueba ,
            p.id As Prueba_num,
            p.curso_id as Curso,
            p.fecha_apertura as fecha_apeprueba,
            p.fecha_modificacion as fecha_modipru,
            ec.acceso_general as fecha_inicioestu,
            ec.ultimo_acceso as fecha_ultimoacc
        FROM estudiantes e
        INNER JOIN estudiantes_curso ec ON e.id_est = ec.id_estudiante
        INNER JOIN cursos c ON ec.id_curso = c.id_cursos
        INNER JOIN ciclo cl ON c.id_ciclo = cl.id_ciclo
        INNER JOIN periodo_carrera pc ON pc.id_periodoc = cl.id_periodo
        INNER JOIN carrera ca ON ca.id_carrera = pc.id_carrera
        INNER JOIN docentes d ON c.id_docente = d.id_docente
        LEFT JOIN pruebas_estudiantes pre ON e.id_est = pre.id_est
        LEFT JOIN pruebas p ON p.id = pre.id_prueba;

        """
        @st.cache_data
        def load_data():
            df = pd.read_sql(query5, engine)
            return df

        df = load_data()
        st.dataframe(df.head())
        
        st.subheader("👀 Vista Previa de los Datos Originales Tareas")
        querry="""
        SELECT 
        e.id_est, 
        CONCAT(e.nombre_est, ' ', e.apellido_est) AS Nombre_estudiante,
        c.nombre_curso,
        pc.descripcion AS Periodo,
        ta.nombre AS Nombre_tarea, 
        tae.nota AS Nota_Tarea,
        tae.fecha_entrega AS Fecha_tarea,
        CONCAT(d.nombre_docente, ' ', d.apellido_docente) AS Nombre_docente, 
        ta.calificacion_max AS Calmax_tarea,
        ta.fecha_limite as fecha_limtarea,
        ta.fecha_apertura as fecha_apetarea,
        ta.fecha_modificacion as fecha_moditarea,
        ec.acceso_general as fecha_inicioestu,
        ec.ultimo_acceso as fecha_ultimoacc
        FROM estudiantes e
        INNER JOIN estudiantes_curso ec ON e.id_est = ec.id_estudiante
        INNER JOIN cursos c ON ec.id_curso = c.id_cursos
        INNER JOIN ciclo cl ON c.id_ciclo = cl.id_ciclo
        INNER JOIN periodo_carrera pc ON pc.id_periodoc = cl.id_periodo
        INNER JOIN carrera ca ON ca.id_carrera = pc.id_carrera
        INNER JOIN docentes d ON c.id_docente = d.id_docente
        LEFT JOIN tareas_estudiantes tae ON tae.id_est = e.id_est
        LEFT JOIN tareas ta ON ta.id = tae.id_tarea;
            
        """
        def load_data():
            df1 = pd.read_sql(querry, engine)
            return df1

        df1 = load_data()
        st.dataframe(df1.head())
        # Ejecutar la consulta y cargar en DataFrame
        st.subheader("🧼 Limpieza de Datos")

        df['fecha_prueba'] = pd.to_datetime(df['fecha_prueba'], unit='s')
        df['fecha_liprueba'] = pd.to_datetime(df['fecha_liprueba'], unit='s')
        df['fecha_apeprueba'] = pd.to_datetime(df['fecha_apeprueba'], unit='s')
        df['fecha_modipru'] = pd.to_datetime(df['fecha_modipru'], unit='s')
        df['fecha_inicioestu'] = pd.to_datetime(df['fecha_inicioestu'], unit='s')
        df['fecha_ultimoacc'] = pd.to_datetime(df['fecha_ultimoacc'], unit='s')
        df['min_apertura_a_entrega'] = (df['fecha_liprueba'] - df['fecha_apeprueba']).dt.total_seconds() / 60
        df['min_ultimo_acc_a_lip'] = (df['fecha_liprueba'] - df['fecha_ultimoacc']).dt.total_seconds() / 60
        df['min_activo'] = (df['fecha_ultimoacc'] - df['fecha_inicioestu']).dt.total_seconds() / 60
        df['min_apertura_a_entrega'] = df['min_apertura_a_entrega'].round(2)
        df['buen_rendimiento'] = df['nota_prueba'].apply(lambda x: 1 if x >= 7 else 0)
        df1['fecha_tarea'] = pd.to_datetime(df1['fecha_tarea'], unit='s')
        df1['fecha_limtarea'] = pd.to_datetime(df1['fecha_limtarea'], unit='s')
        df1['fecha_apetarea'] = pd.to_datetime(df1['fecha_apetarea'], unit='s')
        df1['fecha_moditarea'] = pd.to_datetime(df1['fecha_moditarea'], unit='s')
        df1['fecha_inicioestu'] = pd.to_datetime(df1['fecha_inicioestu'], unit='s')
        df1['fecha_ultimoacc'] = pd.to_datetime(df1['fecha_ultimoacc'], unit='s')
        df1['min_apertura_a_entrega'] = (df1['fecha_limtarea'] - df1['fecha_apetarea']).dt.total_seconds() / 60
        df1['min_ultimo_acc_a_lip'] = (df1['fecha_limtarea'] - df1['fecha_ultimoacc']).dt.total_seconds() / 60
        df1['min_activo'] = (df1['fecha_ultimoacc'] - df1['fecha_inicioestu']).dt.total_seconds() / 60
        df1['min_apertura_a_entrega'] = df1['min_apertura_a_entrega'].round(2)
        df1['buen_rendimiento'] = df1['nota_tarea'].apply(lambda x: 1 if x >= 7 else 0)
        
        st.success("✅ Limpieza realizada con éxito.")
        
        st.dataframe(df.head(20))
        st.dataframe(df1.head(20))
        
        
        st.markdown("### Descarga tus archivos limpios 📥")

        # Crear dos columnas para los botones lado a lado
        col1, col2 = st.columns(2)

        with col1:
            csv_pruebas = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar CSV Limpio de Pruebas",
                data=csv_pruebas,
                file_name="Pruebas.csv",
                mime="text/csv",
                key="download-pruebas"
            )

        with col2:
            csv_tareas = df1.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar CSV Limpio de Tareas",
                data=csv_tareas,
                file_name="Tareas.csv",
                mime="text/csv",
                key="download-tareas"
            )

        # Opcional: agregar separación y explicación
        st.markdown(
            """
            <style>
            .stDownloadButton button {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 20px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("Modelo de Redes Neuronales")
        # Botón 1: Clustering
        if st.button("🔍 Ejecutar Modelo Predictivo de Pruebas"):
            # Variables predictoras
            features = ['min_apertura_a_entrega', 'min_ultimo_acc_a_lip', 'min_activo', 'nota_prueba']
            X = df[features]
            y = df['buen_rendimiento']  # binaria: 0 o 1

            # Escalar datos
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

            # Red neuronal
            model = Sequential()
            model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # salida binaria

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype("int32")  # convierte a 0 o 1
            from sklearn.metrics import classification_report
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=["Bajo rendimiento", "Buen rendimiento"],
                output_dict=True
            )

            # Lo conviertes a DataFrame
            report_df = pd.DataFrame(report_dict).transpose()

            # Título del reporte
            st.subheader("Reporte de Clasificación")

            # Mostrar como tabla con streamlit
            st.dataframe(report_df.style.format("{:.2f}"))
        # Botón 2: Prophet
        if st.button("📈 Ejecutar Modelo Predictivo de Tareas"):
            features = ['min_apertura_a_entrega', 'min_ultimo_acc_a_lip', 'min_activo', 'nota_tarea']
            X = df1[features]
            y = df1['buen_rendimiento']  # binaria: 0 o 1

            # Escalar datos
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

            # Red neuronal
            model = Sequential()
            model.add(Dense(16, input_dim=X.shape[1], activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))  # salida binaria

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype("int32")  # convierte a 0 o 1
            from sklearn.metrics import classification_report
            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=["Bajo rendimiento", "Buen rendimiento"],
                output_dict=True
            )

            # Lo conviertes a DataFrame
            report_df1 = pd.DataFrame(report_dict).transpose()

            # Título del reporte
            st.subheader("Reporte de Clasificación")

            # Mostrar como tabla con streamlit
            st.dataframe(report_df1.style.format("{:.2f}"))
            
            
with tab4:
    st.header("Análisis de Datos de Estudiantes - Big Data")
    st.markdown("Proporcionaremos los resultados del análisis realizado")

    @st.cache_data
    def obtener_periodos():
        query = "SELECT DISTINCT descripcion FROM periodo_carrera ORDER BY descripcion"
        return pd.read_sql(query, engine)['descripcion'].tolist()

    def obtener_ciclos(periodo):
        query = f"""
            SELECT DISTINCT c.nombre_ciclo
            FROM ciclo c
            JOIN periodo_carrera pc ON c.id_periodo = pc.id_periodoc
            WHERE pc.descripcion = '{periodo}'
            ORDER BY c.nombre_ciclo
        """
        return pd.read_sql(query, engine)['nombre_ciclo'].tolist()

    def obtener_cursos(periodo, ciclo):
        query = f"""
            SELECT DISTINCT cu.nombre_curso
            FROM cursos cu
            JOIN ciclo c ON cu.id_ciclo = c.id_ciclo
            JOIN periodo_carrera pc ON c.id_periodo = pc.id_periodoc
            WHERE pc.descripcion = '{periodo}' AND c.nombre_ciclo = '{ciclo}'
            ORDER BY cu.nombre_curso
        """
        return pd.read_sql(query, engine)['nombre_curso'].tolist()

    # Filtros dentro del tab4
    with st.expander("🔍 Filtros de visualización", expanded=True):
        colf1, colf2, colf3 = st.columns(3)

        with colf1:
            lista_periodos = obtener_periodos()
            filtro_periodo = st.selectbox("Filtrar periodos:", lista_periodos, index=0)

        with colf2:
            lista_ciclos = obtener_ciclos(filtro_periodo)
            filtro_ciclo = st.selectbox("Filtrar por ciclo:", options=["Todos"] + lista_ciclos, index=0)

        with colf3:
            lista_cursos = obtener_cursos(filtro_periodo, filtro_ciclo)
            filtro_curso = st.selectbox("Filtrar por curso:",  options=["Todos"] + lista_cursos, index=0)

    # Total estudiantes
    query4 = "SELECT COUNT(DISTINCT id_est) AS total_estudiantes FROM estudiantes;"
    df_total = pd.read_sql(query4, engine)
    total_estudiantes = df_total['total_estudiantes'][0]

    # Total docentes
    query7 = "SELECT COUNT(DISTINCT id_docente) AS total_docentes FROM docentes;"
    df_total1 = pd.read_sql(query7, engine)
    total_docentes = df_total1['total_docentes'][0]
    
    query9= """
    SELECT COUNT(DISTINCT id_estudiante) AS estudiantes_activos
    FROM estudiantes_curso
    WHERE TO_TIMESTAMP(ultimo_acceso) >= NOW() - INTERVAL '7 days';
    """
    df_activos = pd.read_sql(query9, engine)
    valor = df_activos['estudiantes_activos'][0]

    st.markdown("""
        <div style="display: flex; gap: 2rem; margin-top: 1rem;">
            <div style="background-color: #FFD700; padding: 1rem 2rem; border-radius: 12px; text-align: center;">
                <h4 style="margin: 0; color: #000;">Total Estudiantes</h4>
                <h2 style="margin: 0;">{}</h2>
            </div>
            <div style="background-color: #FFD700; padding: 1rem 2rem; border-radius: 12px; text-align: center;">
                <h4 style="margin: 0; color: #000;">Total Docentes</h4>
                <h2 style="margin: 0;">{}</h2>
            </div>
            <div style='background-color:#fff3cd; padding: 1rem 2rem; border-radius:12px; border-left: 6px solid #ffc107; text-align: center;'>
                <h4 style='color:#856404;'>Estudiantes con Acceso Reciente</h4>
                <h2 style='margin:0; color:#212529;'><b>{}</b></h2>
                <small>En los últimos 7 días</small>
            </div>
        </div>
    """.format(total_estudiantes, total_docentes , valor), unsafe_allow_html=True)

    
    # Estudiantes por materia y ciclo
    st.subheader("Estudiantes por Materia y Ciclo")
    query = f'''
        SELECT c.nombre_curso, ci.nombre_ciclo, COUNT(DISTINCT ec.id_estudiante) AS total_estudiantes
        FROM estudiantes_curso ec
        JOIN cursos c ON c.id_cursos = ec.id_curso
        JOIN ciclo ci ON ci.id_ciclo = c.id_ciclo
        JOIN periodo_carrera pc ON ci.id_periodo = pc.id_periodoc
        WHERE pc.descripcion = '{filtro_periodo}'
    '''
    if filtro_ciclo != "Todos":
        query += f" AND ci.nombre_ciclo = '{filtro_ciclo}'"
    if filtro_curso != "Todos":
        query += f" AND c.nombre_curso = '{filtro_curso}'"
    query += " GROUP BY c.nombre_curso, ci.nombre_ciclo;"
    df = pd.read_sql(query, engine)
    fig = px.bar(df, x='nombre_ciclo', y='total_estudiantes', color='nombre_curso', text='total_estudiantes')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(yaxis_title='Total de Estudiantes', xaxis_title='Nombre del Ciclo')
    st.plotly_chart(fig)

    # Accesos por fecha
    st.subheader("Accesos a Moodle por Fecha")
    # Consulta SQL con filtros
    query3 = f'''
        SELECT 
            TO_TIMESTAMP(ec.ultimo_acceso)::DATE AS fecha, 
            COUNT(DISTINCT ec.id_estudiante) AS total_estudiantes
        FROM estudiantes_curso ec
        JOIN cursos c ON ec.id_curso = c.id_cursos
        JOIN ciclo ci ON c.id_ciclo = ci.id_ciclo
        WHERE ec.ultimo_acceso IS NOT NULL
        AND ci.nombre_ciclo = '{filtro_ciclo}'
        AND c.nombre_curso = '{filtro_curso}'
        GROUP BY fecha
        ORDER BY fecha;
    '''


    df3 = pd.read_sql(query3, engine).iloc[1:]
    df3['fecha'] = pd.to_datetime(df3['fecha'])
    fig3 = px.line(df3, x='fecha', y='total_estudiantes', markers=True,
                title=f"Accesos diarios a Moodle - {filtro_ciclo} / {filtro_curso}")
    st.plotly_chart(fig3)
        # Promedios de notas por pruebas
    st.subheader("Promedios de Notas en Pruebas")
    query1 = '''
        SELECT cl.nombre_ciclo, AVG(pre.nota) AS promedio_prueba
        FROM estudiantes e
        JOIN estudiantes_curso ec ON e.id_est = ec.id_estudiante
        JOIN cursos c ON ec.id_curso = c.id_cursos
        JOIN ciclo cl ON c.id_ciclo = cl.id_ciclo
        JOIN periodo_carrera pc ON pc.id_periodoc = cl.id_periodo
        JOIN carrera ca ON ca.id_carrera = pc.id_carrera
        JOIN docentes d ON c.id_docente = d.id_docente
        JOIN pruebas_estudiantes pre ON e.id_est = pre.id_est
        JOIN pruebas p ON p.id = pre.id_prueba
        WHERE pc.descripcion = %(periodo)s
    '''
    params = {"periodo": filtro_periodo}
    if filtro_ciclo != "Todos":
        query1 += " AND cl.nombre_ciclo = %(ciclo)s"
        params["ciclo"] = filtro_ciclo
    query1 += " GROUP BY cl.nombre_ciclo;"
    df2 = pd.read_sql(query1, engine, params=params)
    fig1 = px.bar(df2, x='nombre_ciclo', y='promedio_prueba', text_auto='.2f', color='promedio_prueba')
    fig1.update_layout(title='Promedios de Notas en Pruebas por Ciclo', yaxis_title='Promedio de Notas', xaxis_title='Nombre del Ciclo')
    st.plotly_chart(fig1)
    
    
    # Promedios de notas por tareas
    st.subheader("Promedios de Notas en Tareas")
    query2 = '''
        SELECT cl.nombre_ciclo, AVG(pre.nota) AS promedio_tareas
        FROM estudiantes e
        JOIN estudiantes_curso ec ON e.id_est = ec.id_estudiante
        JOIN cursos c ON ec.id_curso = c.id_cursos
        JOIN ciclo cl ON c.id_ciclo = cl.id_ciclo
        JOIN periodo_carrera pc ON pc.id_periodoc = cl.id_periodo
        JOIN carrera ca ON ca.id_carrera = pc.id_carrera
        JOIN docentes d ON c.id_docente = d.id_docente
        JOIN tareas_estudiantes pre ON e.id_est = pre.id_est
        JOIN tareas p ON p.id = pre.id_tarea
        WHERE pc.descripcion = %(periodo)s
    '''
    params2 = {"periodo": filtro_periodo}
    if filtro_ciclo != "Todos":
        query2 += " AND cl.nombre_ciclo = %(ciclo)s"
        params2["ciclo"] = filtro_ciclo
    query2 += " GROUP BY cl.nombre_ciclo;"
    df7 = pd.read_sql(query2, engine, params=params)
    fig7 = px.bar(df7, x='nombre_ciclo', y='promedio_tareas', text_auto='.2f', color='promedio_tareas')
    fig7.update_layout(title='Promedios de Notas en Tareas por Ciclo', yaxis_title='Promedio de Notas', xaxis_title='Nombre del Ciclo')
    st.plotly_chart(fig7)


    # Tendencia mensual de accesos
    st.subheader("Tendencia Mensual de Accesos a Moodle")
    query6 = '''
        SELECT TO_CHAR(TO_TIMESTAMP(ultimo_acceso), 'YYYY-MM') AS mes,
               COUNT(DISTINCT id_estudiante) AS total_estudiantes
        FROM estudiantes_curso
        WHERE ultimo_acceso IS NOT NULL
        GROUP BY mes
        ORDER BY mes;
    '''
    df6 = pd.read_sql(query6, engine).iloc[1:]
    fig6 = px.line(df6, x='mes', y='total_estudiantes', markers=True,
                   title="Tendencia mensual de estudiantes que accedieron a Moodle")
    st.plotly_chart(fig6, use_container_width=True)
    
    import altair as alt

    st.subheader("Promedio de Notas por Materia")
    
    query8 = f"""
    SELECT c.nombre_curso, ROUND(AVG(te.nota), 2) AS promedio
    FROM tareas_estudiantes te
    JOIN tareas t ON te.id_tarea = t.id
    JOIN cursos c ON t.curso_id = c.id_cursos
    JOIN ciclo ci ON c.id_ciclo = ci.id_ciclo
    JOIN periodo_carrera pc ON ci.id_periodo = pc.id_periodoc
    WHERE pc.descripcion = '{filtro_periodo}'
    """
    
    if filtro_ciclo != "Todos":
        query8 += f" AND ci.nombre_ciclo = '{filtro_ciclo}'"
        
    query8 +=  " GROUP BY c.nombre_curso"
    query8 +=  " ORDER BY promedio DESC;"
    
    df_promedios = pd.read_sql(query8, engine)

    if not df_promedios.empty:
        chart = alt.Chart(df_promedios).mark_bar().encode(
            x=alt.X('promedio:Q', title='Promedio de Nota'),
            y=alt.Y('nombre_curso:N', sort='-x', title='Materia'),
            color=alt.Color('promedio:Q', scale=alt.Scale(scheme='yelloworangered')),
            tooltip=['nombre_curso', 'promedio']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No hay datos de notas para el ciclo seleccionado.")

    
    


st.markdown("---")
st.caption("© 2025 - Karina Chisaguano | Proyecto Integrador - Tec Azuay")

