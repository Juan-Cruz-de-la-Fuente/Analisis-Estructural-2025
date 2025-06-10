import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math
from datetime import datetime
import io
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64

# Configuración de la página con tema moderno
st.set_page_config(
    page_title="Análisis Estructural - Método de Matrices",
    page_icon="⚫",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar oculto inicialmente
)

# CSS personalizado mejorado para estilo web moderno
st.markdown("""
<style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS - CORREGIDAS */
    :root {
        --primary-black: #000000;
        --primary-white: #ffffff;
        --gray-100: #f8f9fa;
        --gray-200: #e9ecef;
        --gray-300: #dee2e6;
        --gray-400: #ced4da;
        --gray-500: #adb5bd;
        --gray-600: #6c757d;
        --gray-700: #495057;
        --gray-800: #343a40;
        --gray-900: #212529;
        --blue-500: #495057;
        --blue-600: #343a40;
        --green-500: #28a745;
        --green-600: #218838;
    }
    
    /* Ocultar sidebar inicialmente */
    .css-1d391kg {
        display: none;
    }
    
    /* Mostrar sidebar solo cuando hay modo seleccionado */
    .show-sidebar .css-1d391kg {
        display: block !important;
        background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-200) 100%);
        border-right: 2px solid var(--gray-300);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Fondo principal más web-like */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: var(--primary-black);
        min-height: 100vh;
    }
    
    /* Contenedor principal para modo inicial - CORREGIDO */
    .landing-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: -1rem;
        padding: 2rem;
    }
    
    /* Tarjetas de modo con efecto glassmorphism */
    .mode-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .mode-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(31, 38, 135, 0.5);
        background: rgba(255, 255, 255, 0.35);
    }
    
    /* Barra de progreso estilo web */
    .progress-bar {
        background: var(--primary-white);
        border-bottom: 3px solid var(--gray-300);
        padding: 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .progress-steps {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .progress-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .step-circle.completed {
        background: var(--green-500);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .step-circle.current {
        background: var(--blue-500);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        animation: pulse 2s infinite;
    }
    
    .step-circle.pending {
        background: var(--gray-300);
        color: var(--gray-600);
    }
    
    .step-line {
        width: 60px;
        height: 3px;
        background: var(--gray-300);
        transition: all 0.3s ease;
    }
    
    .step-line.completed {
        background: var(--green-500);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 4px 20px rgba(59, 130, 246, 0.6); }
        100% { box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
    }
    
    /* Títulos mejorados - CORREGIDOS */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: var(--primary-black);
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 3rem;
        margin-bottom: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid var(--gray-700);
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: var(--gray-800);
    }
    
    /* Botones mejorados */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, var(--blue-500) 0%, var(--blue-600) 100%);
        color: var(--primary-white);
        transition: all 0.3s ease;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, var(--blue-600) 0%, var(--blue-500) 100%);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, var(--gray-500) 0%, var(--gray-600) 100%);
        box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(135deg, var(--gray-600) 0%, var(--gray-500) 100%);
        box-shadow: 0 8px 25px rgba(107, 114, 128, 0.4);
    }
    
    /* Inputs mejorados */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        font-family: 'Inter', sans-serif;
        border: 2px solid var(--gray-300);
        border-radius: 10px;
        background-color: var(--primary-white);
        color: var(--primary-black);
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--blue-500);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Métricas mejoradas */
    .metric-container {
        background: linear-gradient(135deg, var(--primary-white) 0%, var(--gray-100) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid var(--gray-200);
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Tablas mejoradas */
    .dataframe {
        font-family: 'Inter', sans-serif;
        border: none;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Expanders mejorados */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--gray-100) 0%, var(--gray-200) 100%);
        border: 1px solid var(--gray-300);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, var(--gray-200) 0%, var(--gray-300) 100%);
    }
    
    /* Info boxes mejoradas */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid var(--blue-500);
        color: var(--primary-black);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid var(--green-500);
        color: var(--primary-black);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        color: var(--primary-black);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        color: var(--primary-black);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.1);
    }
    
    /* Footer estilo web - CORREGIDO */
    .footer-section {
        background: linear-gradient(135deg, var(--gray-900) 0%, var(--gray-800) 100%);
        color: var(--primary-white);
        padding: 3rem 0;
        margin-top: 4rem;
        border-radius: 20px 20px 0 0;
    }
    
    .footer-content {
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .footer-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .footer-survey {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Animaciones */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Base de datos de materiales aeroespaciales
MATERIALES_AEROESPACIALES = {
    "Aluminio 6061-T6": {
        "modulo_young": 68.9e9,  # Pa
        "densidad": 2700,  # kg/m³
        "descripcion": "Aleación de aluminio estructural común"
    },
    "Aluminio 7075-T6": {
        "modulo_young": 71.7e9,  # Pa
        "densidad": 2810,  # kg/m³
        "descripcion": "Aleación de aluminio de alta resistencia"
    },
    "Aluminio 2024-T3": {
        "modulo_young": 73.1e9,  # Pa
        "densidad": 2780,  # kg/m³
        "descripcion": "Aleación de aluminio para fuselajes"
    },
    "Titanio Ti-6Al-4V": {
        "modulo_young": 113.8e9,  # Pa
        "densidad": 4430,  # kg/m³
        "descripcion": "Aleación de titanio aeroespacial"
    },
    "Acero 4130": {
        "modulo_young": 205e9,  # Pa
        "densidad": 7850,  # kg/m³
        "descripcion": "Acero aleado para estructuras"
    },
    "Fibra de Carbono T300": {
        "modulo_young": 230e9,  # Pa
        "densidad": 1760,  # kg/m³
        "descripcion": "Compuesto de fibra de carbono"
    },
    "Magnesio AZ31B": {
        "modulo_young": 45e9,  # Pa
        "densidad": 1770,  # kg/m³
        "descripcion": "Aleación de magnesio ligera"
    }
}

# Inicializar estado de la sesión
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'modo' not in st.session_state:
    st.session_state.modo = None
if 'usuario_nombre' not in st.session_state:
    st.session_state.usuario_nombre = ""
if 'nodos' not in st.session_state:
    st.session_state.nodos = []
if 'elementos' not in st.session_state:
    st.session_state.elementos = []
if 'matrices_elementos' not in st.session_state:
    st.session_state.matrices_elementos = {}
if 'grados_libertad_info' not in st.session_state:
    st.session_state.grados_libertad_info = []
if 'nombres_fuerzas' not in st.session_state:
    st.session_state.nombres_fuerzas = {}
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'materiales_personalizados' not in st.session_state:
    st.session_state.materiales_personalizados = {}
if 'auto_calcular' not in st.session_state:
    st.session_state.auto_calcular = True
if 'nodo_seleccionado_interactivo' not in st.session_state:
    st.session_state.nodo_seleccionado_interactivo = None
if 'nodos_interactivos' not in st.session_state:
    st.session_state.nodos_interactivos = []
if 'elementos_interactivos' not in st.session_state:
    st.session_state.elementos_interactivos = []
if 'num_nodos' not in st.session_state:
    st.session_state.num_nodos = 2
if 'num_fijos' not in st.session_state:
    st.session_state.num_fijos = 1
if 'num_elementos' not in st.session_state:
    st.session_state.num_elementos = 1
if 'grupos_elementos' not in st.session_state:
    st.session_state.grupos_elementos = {}

def formatear_unidades(valor, tipo="presion"):
    """Formatear valores con prefijos apropiados"""
    if tipo == "presion":
        abs_valor = abs(valor)
        if abs_valor == 0:
            return "0 Pa"
        elif abs_valor < 10:
            return f"{valor:.3f} Pa"
        elif abs_valor < 1000:
            return f"{valor:.1f} Pa"
        elif abs_valor < 1e6:
            return f"{valor/1e3:.3f} kPa"
        elif abs_valor < 1e9:
            return f"{valor/1e6:.3f} MPa"
        else:
            return f"{valor/1e9:.3f} GPa"
    elif tipo == "fuerza":
        abs_valor = abs(valor)
        if abs_valor == 0:
            return "0 N"
        elif abs_valor < 1000:
            return f"{valor:.3f} N"
        elif abs_valor < 1e6:
            return f"{valor/1e3:.3f} kN"
        else:
            return f"{valor/1e6:.3f} MN"
    elif tipo == "desplazamiento":
        abs_valor = abs(valor)
        if abs_valor == 0:
            return "0 m"
        elif abs_valor < 1e-6:
            return f"{valor*1e9:.3f} nm"
        elif abs_valor < 1e-3:
            return f"{valor*1e6:.3f} μm"
        elif abs_valor < 1:
            return f"{valor*1e3:.3f} mm"
        else:
            return f"{valor:.6f} m"
    return f"{valor:.6e}"

def reset_app():
    """Reiniciar la aplicación"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def set_modo(modo):
    """Establecer el modo de análisis"""
    st.session_state.modo = modo
    st.session_state.step = 1
    st.rerun()

def next_step():
    """Avanzar al siguiente paso"""
    st.session_state.step += 1
    st.rerun()

def prev_step():
    """Retroceder al paso anterior"""
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()

def calcular_grados_libertad_globales(nodo_id):
    """Calcular grados de libertad globales para un nodo"""
    gl_por_nodo = 2
    return [(nodo_id - 1) * gl_por_nodo + 1, (nodo_id - 1) * gl_por_nodo + 2]

def calcular_longitud_elemento(nodo_inicio, nodo_fin):
    """Calcular la longitud del elemento"""
    dx = nodo_fin['x'] - nodo_inicio['x']
    dy = nodo_fin['y'] - nodo_inicio['y']
    return math.sqrt(dx**2 + dy**2)

def calcular_angulo_beta(nodo_inicio, nodo_fin):
    """Calcular el ángulo β entre la horizontal y la barra"""
    dx = nodo_fin['x'] - nodo_inicio['x']
    dy = nodo_fin['y'] - nodo_inicio['y']
    return math.atan2(dy, dx)

def calcular_area_seccion(tipo_seccion, parametros):
    """Calcular el área de la sección según su tipo"""
    if tipo_seccion == "circular_solida":
        radio = parametros.get("radio", 0)
        return math.pi * radio**2
    elif tipo_seccion == "circular_hueca":
        radio_ext = parametros.get("radio_ext", 0)
        radio_int = parametros.get("radio_int", 0)
        return math.pi * (radio_ext**2 - radio_int**2)
    elif tipo_seccion == "rectangular":
        lado1 = parametros.get("lado1", 0)
        lado2 = parametros.get("lado2", 0)
        return lado1 * lado2
    elif tipo_seccion == "cuadrada":
        lado = parametros.get("lado", 0)
        return lado**2
    else:
        return parametros.get("area", 0.01)

def generar_matriz_rigidez_barra(E, A, L, beta):
    """Generar matriz de rigidez para barra (4x4)"""
    c = math.cos(beta)
    s = math.sin(beta)
    factor = (E * A) / L
    
    matriz = factor * np.array([
        [c**2,      c*s,       -c**2,     -c*s],
        [c*s,       s**2,      -c*s,      -s**2],
        [-c**2,     -c*s,      c**2,      c*s],
        [-c*s,      -s**2,     c*s,       s**2]
    ])
    return matriz

def visualizar_estructura_moderna(mostrar_deformada=False, factor_escala=10):
    """Visualizar la estructura con estilo moderno minimalista"""
    if not st.session_state.nodos:
        st.warning("No hay nodos para visualizar")
        return None
    
    # Configurar matplotlib con estilo moderno
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_facecolor('white')
    
    # Obtener límites
    x_coords = [nodo['x'] for nodo in st.session_state.nodos]
    y_coords = [nodo['y'] for nodo in st.session_state.nodos]
    
    if len(set(x_coords)) == 1:
        x_min, x_max = (min(x_coords) - 1)*1.2, (max(x_coords) + 1)*1.2
    else:
        x_range = max(x_coords) - min(x_coords)
        x_min, x_max = (min(x_coords) - 0.1*x_range)*1.2, (max(x_coords) + 0.1*x_range)*1.2
    
    if len(set(y_coords)) == 1:
        y_min, y_max = (min(y_coords) - 1)*1.2, (max(y_coords) + 1)*1.2
    else:
        y_range = max(y_coords) - min(y_coords)
        y_min, y_max = (min(y_coords) - 0.1*y_range)*1.2, (max(y_coords) + 0.1*y_range)*1.2
    
    # Calcular posiciones deformadas si es necesario
    nodos_deformados = None
    if mostrar_deformada and st.session_state.resultados is not None:
        nodos_deformados = []
        for nodo in st.session_state.nodos:
            gl_indices = [gl - 1 for gl in nodo['grados_libertad_globales']]
            dx = st.session_state.resultados['desplazamientos'][gl_indices[0]] * factor_escala
            dy = st.session_state.resultados['desplazamientos'][gl_indices[1]] * factor_escala
            
            nodo_deformado = nodo.copy()
            nodo_deformado['x'] += dx
            nodo_deformado['y'] += dy
            nodos_deformados.append(nodo_deformado)
    
    # Dibujar estructura original si se muestra deformada
    if mostrar_deformada:
        for elemento in st.session_state.elementos:
            nodo_inicio = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_inicio']), None)
            nodo_fin = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_fin']), None)
            
            if nodo_inicio and nodo_fin:
                ax.plot([nodo_inicio['x'], nodo_fin['x']], 
                       [nodo_inicio['y'], nodo_fin['y']], 
                       color='#9CA3AF', linewidth=2, alpha=0.6, linestyle='--', 
                       label='Estructura Original' if elemento['id'] == 1 else "")
    
    # Dibujar elementos principales
    for elemento in st.session_state.elementos:
        nodo_inicio = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_inicio']), None)
        nodo_fin = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_fin']), None)
        
        if nodo_inicio and nodo_fin:
            if mostrar_deformada and nodos_deformados:
                nodo_inicio_def = next((n for n in nodos_deformados if n['id'] == elemento['nodo_inicio']), None)
                nodo_fin_def = next((n for n in nodos_deformados if n['id'] == elemento['nodo_fin']), None)
                
                if nodo_inicio_def and nodo_fin_def:
                    ax.plot([nodo_inicio_def['x'], nodo_fin_def['x']], 
                           [nodo_inicio_def['y'], nodo_fin_def['y']], 
                           color='#000000', linewidth=3, alpha=0.9,
                           label='Estructura Deformada' if elemento['id'] == 1 else "")
                    
                    # Etiqueta de barra deformada con estilo moderno
                    mid_x = (nodo_inicio_def['x'] + nodo_fin_def['x']) / 2
                    mid_y = (nodo_inicio_def['y'] + nodo_fin_def['y']) / 2
                    
                    ax.text(mid_x, mid_y, f'E{elemento["id"]}', 
                           ha='center', va='center', fontsize=9, fontweight='600',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                   edgecolor="black", linewidth=1.5, alpha=0.95),
                           fontfamily='sans-serif')
            else:
                ax.plot([nodo_inicio['x'], nodo_fin['x']], 
                       [nodo_inicio['y'], nodo_fin['y']], 
                       color='#000000', linewidth=3, alpha=0.9)
                
                # Etiqueta de barra original con estilo moderno
                mid_x = (nodo_inicio['x'] + nodo_fin['x']) / 2
                mid_y = (nodo_inicio['y'] + nodo_fin['y']) / 2
                
                ax.text(mid_x, mid_y, f'E{elemento["id"]}', 
                       ha='center', va='center', fontsize=9, fontweight='600',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                               edgecolor="black", linewidth=1.5, alpha=0.95),
                       fontfamily='sans-serif')
    
    # Dibujar nodos con estilo moderno (más pequeños)
    for i, nodo in enumerate(st.session_state.nodos):
        if nodo['tipo'] == 'fijo':
            color = '#DC2626'  # Rojo moderno
            edge_color = '#991B1B'
        else:
            color = '#2563EB'  # Azul moderno
            edge_color = '#1D4ED8'
        
        # Nodo más pequeño y moderno - CORREGIDO PARA SER COMO PUNTOS
        circle = plt.Circle((nodo['x'], nodo['y']), 0.03,  # Reducido de 0.08 a 0.03
                  color=color, alpha=0.9, zorder=10)
        ax.add_patch(circle)

        # Borde del nodo - CORREGIDO
        circle_edge = plt.Circle((nodo['x'], nodo['y']), 0.03,  # Reducido de 0.08 a 0.03
                       fill=False, edgecolor=edge_color, linewidth=1, zorder=11)
        ax.add_patch(circle_edge)

        # Número del nodo con estilo moderno - CORREGIDO
        ax.text(nodo['x'], nodo['y'], str(nodo['id']),
               ha='center', va='center', fontsize=6, fontweight='700',  # Reducido de 7 a 6
               color='white', zorder=12, fontfamily='sans-serif')

        # Coordenadas con estilo moderno - CORREGIDO
        ax.text(nodo['x'], nodo['y'] - 0.15, f'({nodo["x"]:.1f}, {nodo["y"]:.1f})',  # Reducido de -0.25 a -0.15
               ha='center', va='top', fontsize=5, color='#374151',  # Reducido de 6 a 5
               fontfamily='sans-serif', fontweight='500')

        # Dibujar nodos deformados si es necesario
        if mostrar_deformada and nodos_deformados:
            nodo_def = nodos_deformados[i]

            # Nodo deformado con color diferente - CORREGIDO
            circle_def = plt.Circle((nodo_def['x'], nodo_def['y']), 0.025,  # Reducido de 0.06 a 0.025
                      color='#28a745', alpha=0.8, zorder=10)
            ax.add_patch(circle_def)

            circle_def_edge = plt.Circle((nodo_def['x'], nodo_def['y']), 0.025,  # Reducido de 0.06 a 0.025
                           fill=False, edgecolor='#218838', linewidth=1, zorder=11)
            ax.add_patch(circle_def_edge)

            # Línea de conexión con estilo moderno
            ax.plot([nodo['x'], nodo_def['x']], [nodo['y'], nodo_def['y']],
                   color='#6B7280', linestyle=':', linewidth=1.5, alpha=0.7)

    # Configurar ejes con estilo moderno
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X [m]', fontsize=12, fontweight='600', fontfamily='sans-serif', color='#374151')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='600', fontfamily='sans-serif', color='#374151')

    # Título con estilo moderno
    if mostrar_deformada:
        ax.set_title(f'Estructura Deformada (Factor: {factor_escala}×)',
                    fontsize=16, fontweight='700', fontfamily='sans-serif',
                    color='#111827', pad=20)
    else:
        ax.set_title('Estructura Original',
                    fontsize=16, fontweight='700', fontfamily='sans-serif',
                    color='#111827', pad=20)

    # Grid moderno
    ax.grid(True, alpha=0.3, color='#E5E7EB', linewidth=0.8)
    ax.set_axisbelow(True)

    # Aspecto igual
    ax.set_aspect('equal', adjustable='box')

    # Leyenda moderna
    if mostrar_deformada:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC2626',
                      markersize=8, label='Nodos Fijos', markeredgecolor='#991B1B'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563EB',
                      markersize=8, label='Nodos Libres', markeredgecolor='#1D4ED8'),
            plt.Line2D([0], [0], color='#9CA3AF', linewidth=2, linestyle='--',
                      label='Estructura Original'),
            plt.Line2D([0], [0], color='#000000', linewidth=3,
                      label='Estructura Deformada'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#28a745',
                      markersize=6, label='Nodos Deformados', markeredgecolor='#218838')
        ]
    else:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC2626',
                      markersize=8, label='Nodos Fijos', markeredgecolor='#991B1B'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2563EB',
                      markersize=8, label='Nodos Libres', markeredgecolor='#1D4ED8'),
            plt.Line2D([0], [0], color='#000000', linewidth=3, label='Barras')
        ]

    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=True, shadow=True, fontsize=10,
             prop={'family': 'sans-serif', 'weight': '500'})

    # Remover spines superiores y derechos para look moderno
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')

    # Color de ticks
    ax.tick_params(colors='#6B7280', which='both')

    plt.tight_layout()
    return fig

def crear_grafico_interactivo_moderno():
    """Crear un gráfico interactivo con estilo moderno"""
    fig = go.Figure()

    # Configurar aspecto moderno del gráfico
    fig.update_layout(
        title=dict(
            text="Editor Interactivo de Estructura",
            font=dict(family="Inter, sans-serif", size=20, color="#111827", weight=700),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text="X [m]", font=dict(family="Inter, sans-serif", size=14, color="#374151", weight=600)),
            showgrid=True,
            gridcolor="#E5E7EB",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#9CA3AF",
            zerolinewidth=2,
            range=[-10, 10],
            tickfont=dict(family="Inter, sans-serif", size=12, color="#6B7280"),
            linecolor="#E5E7EB",
            mirror=True
        ),
        yaxis=dict(
            title=dict(text="Y [m]", font=dict(family="Inter, sans-serif", size=14, color="#374151", weight=600)),
            showgrid=True,
            gridcolor="#E5E7EB",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="#9CA3AF",
            zerolinewidth=2,
            range=[-10, 10],
            scaleanchor="x",
            scaleratio=1,
            tickfont=dict(family="Inter, sans-serif", size=12, color="#6B7280"),
            linecolor="#E5E7EB",
            mirror=True
        ),
        showlegend=True,
        legend=dict(
            font=dict(family="Inter, sans-serif", size=12, color="#374151", weight=500),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E5E7EB',
            borderwidth=1,
            x=1,
            y=1,
            xanchor='right',
            yanchor='top'
        ),
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif")
    )

    # Añadir nodos existentes con estilo moderno
    nodos_fijos_x = []
    nodos_fijos_y = []
    nodos_fijos_text = []

    nodos_libres_x = []
    nodos_libres_y = []
    nodos_libres_text = []

    for nodo in st.session_state.nodos_interactivos:
        if nodo['tipo'] == 'fijo':
            nodos_fijos_x.append(nodo['x'])
            nodos_fijos_y.append(nodo['y'])
            nodos_fijos_text.append(f"Nodo {nodo['id']}<br>({nodo['x']:.1f}, {nodo['y']:.1f})<br>Tipo: Fijo")
        else:
            nodos_libres_x.append(nodo['x'])
            nodos_libres_y.append(nodo['y'])
            nodos_libres_text.append(f"Nodo {nodo['id']}<br>({nodo['x']:.1f}, {nodo['y']:.1f})<br>Tipo: Libre")

    # Añadir nodos fijos con estilo moderno
    if nodos_fijos_x:
        fig.add_trace(go.Scatter(
            x=nodos_fijos_x,
            y=nodos_fijos_y,
            mode='markers+text',
            marker=dict(
                size=12,
                color='#DC2626',
                line=dict(width=2, color='#991B1B'),
                symbol='circle'
            ),
            text=[f"{nodo['id']}" for nodo in st.session_state.nodos_interactivos if nodo['tipo'] == 'fijo'],
            textposition="middle center",
            textfont=dict(size=10, color='white', family="Inter, sans-serif", weight=700),
            hoverinfo='text',
            hovertext=nodos_fijos_text,
            name='Nodos Fijos',
            hovertemplate='<b>%{hovertext}</b><extra></extra>'
        ))

    # Añadir nodos libres con estilo moderno
    if nodos_libres_x:
        fig.add_trace(go.Scatter(
            x=nodos_libres_x,
            y=nodos_libres_y,
            mode='markers+text',
            marker=dict(
                size=12,
                color='#2563EB',
                line=dict(width=2, color='#1D4ED8'),
                symbol='circle'
            ),
            text=[f"{nodo['id']}" for nodo in st.session_state.nodos_interactivos if nodo['tipo'] == 'libre'],
            textposition="middle center",
            textfont=dict(size=10, color='white', family="Inter, sans-serif", weight=700),
            hoverinfo='text',
            hovertext=nodos_libres_text,
            name='Nodos Libres',
            hovertemplate='<b>%{hovertext}</b><extra></extra>'
        ))

    # Añadir elementos (barras) con estilo moderno
    for elemento in st.session_state.elementos_interactivos:
        nodo_inicio = next((n for n in st.session_state.nodos_interactivos if n['id'] == elemento['nodo_inicio']), None)
        nodo_fin = next((n for n in st.session_state.nodos_interactivos if n['id'] == elemento['nodo_fin']), None)

        if nodo_inicio and nodo_fin:
            # Calcular punto medio para etiqueta
            mid_x = (nodo_inicio['x'] + nodo_fin['x']) / 2
            mid_y = (nodo_inicio['y'] + nodo_fin['y']) / 2
            longitud = calcular_longitud_elemento(nodo_inicio, nodo_fin)

            # Añadir etiqueta de barra con estilo moderno
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=[f"E{elemento['id']}"],
                textposition="middle center",
                textfont=dict(size=10, color='#111827', family="Inter, sans-serif", weight=600),
                hoverinfo='skip',
                showlegend=False
            ))

            # Añadir barra con estilo moderno (más fina)
            fig.add_trace(go.Scatter(
                x=[nodo_inicio['x'], nodo_fin['x']],
                y=[nodo_inicio['y'], nodo_fin['y']],
                mode='lines',
                line=dict(width=4, color='#000000'),
                name=f"Barra {elemento['id']}",
                hoverinfo='text',
                hovertext=f"<b>Barra {elemento['id']}</b><br>Nodo {nodo_inicio['id']} → Nodo {nodo_fin['id']}<br>Longitud: {longitud:.3f} m",
                showlegend=True,
                hovertemplate='%{hovertext}<extra></extra>'
            ))

    # Configurar interactividad
    fig.update_layout(
        dragmode='pan',
        clickmode='event+select',
        hovermode='closest'
    )

    return fig

def mostrar_matriz_formateada_moderna(matriz, titulo="Matriz", es_simbolica=True):
    """Mostrar matriz en formato tabla con estilo moderno"""
    if matriz is None or len(matriz) == 0:
        st.warning("⚠️ Matriz vacía")
        return

    st.markdown(f"### {titulo}")

    if es_simbolica:
        df = pd.DataFrame(matriz)
        df.index = [f"Fila {i+1}" for i in range(len(matriz))]
        df.columns = [f"Col {i+1}" for i in range(len(matriz[0]))]
    else:
        # Formatear valores con unidades apropiadas
        matriz_formateada = []
        for fila in matriz:
            fila_formateada = []
            for valor in fila:
                if "rigidez" in titulo.lower() or "K" in titulo:
                    fila_formateada.append(formatear_unidades(valor, "presion"))
                else:
                    fila_formateada.append(f"{valor:.6e}")
            matriz_formateada.append(fila_formateada)

        df = pd.DataFrame(matriz_formateada)
        df.index = [f"GL{i+1}" for i in range(len(matriz))]
        df.columns = [f"GL{i+1}" for i in range(len(matriz[0]))]

    # Mostrar dataframe con estilo
    st.dataframe(df, use_container_width=True)

    # Botón para descargar tabla con estilo moderno
    csv_data = df.to_csv(sep='\t')
    st.download_button(
        label="📋 Descargar Tabla",
        data=csv_data,
        file_name=f"{titulo.replace(' ', '_')}.csv",
        mime="text/csv",
        type="secondary"
    )

def crear_tabla_nodos():
    """Crear tabla de nodos con coordenadas y grados de libertad"""
    if not st.session_state.nodos:
        return pd.DataFrame()

    nodos_data = []
    for nodo in st.session_state.nodos:
        nodos_data.append({
            'ID': nodo['id'],
            'Tipo': nodo['tipo'].title(),
            'X [m]': f"{nodo['x']:.3f}",
            'Y [m]': f"{nodo['y']:.3f}",
            'GL X': nodo['grados_libertad_globales'][0],
            'GL Y': nodo['grados_libertad_globales'][1],
            'Coordenadas': f"({nodo['x']:.3f}, {nodo['y']:.3f})"
        })

    return pd.DataFrame(nodos_data)

def crear_tabla_conectividad():
    """
    Crear tabla de conectividad de elementos.
    VERSIÓN CORREGIDA Y ROBUSTA: Maneja claves faltantes.
    """
    if not st.session_state.elementos:
        return pd.DataFrame()

    conectividad_data = []
    for elem in st.session_state.elementos:
        # ---- MANEJO SEGURO DE CLAVES ----
        # Para 'tipo_seccion'
        tipo_seccion_val = elem.get('tipo_seccion')
        if tipo_seccion_val:
            seccion_str = tipo_seccion_val.replace('_', ' ').title()
        else:
            seccion_str = "No Definida"

        # Para las otras claves que podrían faltar
        conectividad_data.append({
            'Elemento': elem.get('id', 'N/A'),
            'Tipo': elem.get('tipo', 'Barra'),
            'Nodo Inicio': elem.get('nodo_inicio', 'N/A'),
            'Nodo Fin': elem.get('nodo_fin', 'N/A'),
            'Material': elem.get('material', 'No Definido'),
            'Sección': seccion_str, # Usamos la variable segura que creamos
            'Área [m²]': f"{elem.get('area', 0.0):.6f}",
            'Longitud [m]': f"{elem.get('longitud', 0.0):.3f}",
            'Ángulo β [rad]': f"{elem.get('beta', 0.0):.4f}",
            'GL Globales': str(elem.get('grados_libertad_global', '[]'))
        })

    return pd.DataFrame(conectividad_data)

# Función mejorada para generar Excel - ARREGLADA CON PANDAS NATIVO
def generar_excel_completo():
    """Generar archivo Excel con todas las tablas usando pandas nativo"""
    if not st.session_state.resultados:
        st.error("No hay resultados para exportar")
        return None
    
    try:
        output = io.BytesIO()
        
        # Crear un diccionario con todas las hojas
        hojas_excel = {}
        
        # Hoja 1: Información del proyecto
        info_data = {
            'Parámetro': ['Usuario', 'Fecha', 'Hora', 'Modo', 'Nodos', 'Elementos', 'Grados de Libertad'],
            'Valor': [
                st.session_state.usuario_nombre,
                datetime.now().strftime('%d/%m/%Y'),
                datetime.now().strftime('%H:%M:%S'),
                st.session_state.modo.capitalize(),
                len(st.session_state.nodos),
                len(st.session_state.elementos),
                len(st.session_state.grados_libertad_info)
            ]
        }
        hojas_excel['Información'] = pd.DataFrame(info_data)
        
        # Hoja 2: Tabla de nodos
        df_nodos = crear_tabla_nodos()
        if not df_nodos.empty:
            hojas_excel['Nodos'] = df_nodos
        
        # Hoja 3: Tabla de conectividad
        df_conectividad = crear_tabla_conectividad()
        if not df_conectividad.empty:
            hojas_excel['Conectividad'] = df_conectividad
        
        # Hoja 4: Matriz de rigidez global
        resultado = st.session_state.resultados
        K_data = []
        for i, fila in enumerate(resultado['K_global']):
            fila_dict = {f'GL{j+1}': formatear_unidades(val, "presion") for j, val in enumerate(fila)}
            fila_dict['GL'] = f'GL{i+1}'
            K_data.append(fila_dict)
        
        df_K = pd.DataFrame(K_data)
        cols = ['GL'] + [f'GL{i+1}' for i in range(len(resultado['K_global']))]
        df_K = df_K[cols]
        hojas_excel['Matriz_Rigidez'] = df_K
        
        # Hoja 5: Resultados de fuerzas y desplazamientos
        resultados_data = []
        for i, (info, fuerza, desplazamiento) in enumerate(zip(
            st.session_state.grados_libertad_info, 
            resultado['fuerzas'], 
            resultado['desplazamientos']
        )):
            nombre = st.session_state.nombres_fuerzas.get(i+1, f"F{i+1}")
            tipo_f = "Dato" if info['fuerza_conocida'] else "Calculado"
            tipo_u = "Dato" if info['desplazamiento_conocido'] else "Calculado"
            
            resultados_data.append({
                'GL': f"GL{i+1}",
                'Nodo': f"N{info['nodo']}",
                'Dirección': info['direccion'],
                'Nombre_Fuerza': nombre,
                'Fuerza': formatear_unidades(fuerza, "fuerza"),
                'Tipo_Fuerza': tipo_f,
                'Desplazamiento': formatear_unidades(desplazamiento, "desplazamiento"),
                'Tipo_Desplazamiento': tipo_u,
                'Fuerza_Valor_Numerico': fuerza,
                'Desplazamiento_Valor_Numerico': desplazamiento
            })
        
        hojas_excel['Resultados'] = pd.DataFrame(resultados_data)
        
        # Escribir todas las hojas al archivo Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for nombre_hoja, df in hojas_excel.items():
                df.to_excel(writer, sheet_name=nombre_hoja, index=False)
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error al generar Excel: {str(e)}")
        return None

def generar_pdf_completo(factor_escala=10):
    """Generar archivo PDF con reporte completo mejorado"""
    if not st.session_state.resultados:
        st.error("No hay resultados para exportar")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Título principal
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Centrado
        )
        story.append(Paragraph("Análisis Estructural - Método de Matrices", title_style))
        story.append(Spacer(1, 20))
        
        # Información del proyecto
        info_data = [
            ['Parámetro', 'Valor'],
            ['Usuario', st.session_state.usuario_nombre],
            ['Fecha', datetime.now().strftime('%d/%m/%Y')],
            ['Hora', datetime.now().strftime('%H:%M:%S')],
            ['Modo', st.session_state.modo.capitalize()],
            ['Nodos', str(len(st.session_state.nodos))],
            ['Elementos', str(len(st.session_state.elementos))],
            ['Grados de Libertad', str(len(st.session_state.grados_libertad_info))]
        ]
        
        info_table = Table(info_data)
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("Información del Proyecto", styles['Heading2']))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Tabla de nodos
        if st.session_state.nodos:
            story.append(Paragraph("Tabla de Nodos", styles['Heading2']))
            nodos_data = [['ID', 'Tipo', 'X [m]', 'Y [m]', 'GL X', 'GL Y']]
            for nodo in st.session_state.nodos:
                nodos_data.append([
                    str(nodo['id']),
                    nodo['tipo'].title(),
                    f"{nodo['x']:.3f}",
                    f"{nodo['y']:.3f}",
                    str(nodo['grados_libertad_globales'][0]),
                    str(nodo['grados_libertad_globales'][1])
                ])
            
            nodos_table = Table(nodos_data)
            nodos_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(nodos_table)
            story.append(Spacer(1, 20))
        
        # Generar gráfico ORIGINAL para PDF
        if st.session_state.nodos:
            story.append(Paragraph("Estructura Original", styles['Heading2']))
            
            # Crear gráfico original (sin deformación)
            fig_original = visualizar_estructura_moderna(mostrar_deformada=False, factor_escala=1)
            if fig_original:
                img_buffer_original = io.BytesIO()
                fig_original.savefig(img_buffer_original, format='png', dpi=300, bbox_inches='tight')
                img_buffer_original.seek(0)
                
                img_original = Image(img_buffer_original, width=400, height=300)
                story.append(img_original)
                plt.close(fig_original)
                story.append(Spacer(1, 20))
        
        # Generar gráfico DEFORMADO para PDF
        if st.session_state.nodos and st.session_state.resultados:
            story.append(Paragraph("Estructura Deformada", styles['Heading2']))
            
            # Crear gráfico deformado
            fig_deformado = visualizar_estructura_moderna(mostrar_deformada=True, factor_escala=factor_escala)
            if fig_deformado:
                img_buffer_deformado = io.BytesIO()
                fig_deformado.savefig(img_buffer_deformado, format='png', dpi=300, bbox_inches='tight')
                img_buffer_deformado.seek(0)
                
                img_deformado = Image(img_buffer_deformado, width=400, height=300)
                story.append(img_deformado)
                plt.close(fig_deformado)
                story.append(Spacer(1, 20))
        
        # Resultados
        resultado = st.session_state.resultados
        story.append(Paragraph("Resultados del Análisis", styles['Heading2']))
        
        resultados_data = [['GL', 'Nodo', 'Dir', 'Fuerza', 'Desplazamiento']]
        for i, (info, fuerza, desplazamiento) in enumerate(zip(
            st.session_state.grados_libertad_info, 
            resultado['fuerzas'], 
            resultado['desplazamientos']
        )):
            resultados_data.append([
                f"GL{i+1}",
                f"N{info['nodo']}",
                info['direccion'],
                formatear_unidades(fuerza, "fuerza"),
                formatear_unidades(desplazamiento, "desplazamiento")
            ])
        
        resultados_table = Table(resultados_data)
        resultados_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(resultados_table)
        
        # Página nueva para tablas horizontales
        story.append(PageBreak())
        
        # Tabla de conectividad en horizontal
        if st.session_state.elementos:
            story.append(Paragraph("Tabla de Conectividad de Elementos", styles['Heading2']))
            conectividad_data = [['Elemento', 'Tipo', 'Nodo Inicio', 'Nodo Fin', 'Material', 'Área [m²]', 'Longitud [m]', 'Ángulo β [rad]']]
            for elem in st.session_state.elementos:
                conectividad_data.append([
                    str(elem['id']),
                    elem['tipo'],
                    str(elem['nodo_inicio']),
                    str(elem['nodo_fin']),
                    elem['material'][:15] + '...' if len(elem['material']) > 15 else elem['material'],
                    f"{elem['area']:.6f}",
                    f"{elem['longitud']:.3f}",
                    f"{elem['beta']:.4f}"
                ])
            
            conectividad_table = Table(conectividad_data)
            conectividad_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            story.append(conectividad_table)
            story.append(Spacer(1, 20))
        
        # Matriz de rigidez global en horizontal
        story.append(PageBreak())
        story.append(Paragraph("Matriz de Rigidez Global", styles['Heading2']))
        
        # Crear tabla de matriz K más compacta
        K_data = [['GL'] + [f'GL{i+1}' for i in range(len(resultado['K_global']))]]
        for i, fila in enumerate(resultado['K_global']):
            fila_formateada = [f'GL{i+1}']
            for val in fila:
                # Usar notación científica más compacta
                fila_formateada.append(f"{val:.2e}")
            K_data.append(fila_formateada)
        
        K_table = Table(K_data)
        K_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (1, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(K_table)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error al generar PDF: {str(e)}")
        return None

def resolver_sistema():
    """Resolver el sistema de ecuaciones automáticamente"""
    if not st.session_state.elementos or not st.session_state.grados_libertad_info:
        return None
    
    try:
        # Ensamblar matriz K global numérica
        max_gl = len(st.session_state.grados_libertad_info)
        K_global = np.zeros((max_gl, max_gl))
        
        # Ensamblar usando las matrices numéricas de cada elemento
        for elemento in st.session_state.elementos:
            if elemento['id'] in st.session_state.matrices_elementos:
                matriz_num = np.array(st.session_state.matrices_elementos[elemento['id']]['numerica'])
                
                for i, gl_i in enumerate(elemento['grados_libertad_global']):
                    for j, gl_j in enumerate(elemento['grados_libertad_global']):
                        K_global[gl_i-1, gl_j-1] += matriz_num[i][j]
        
        # Vectores de fuerzas y desplazamientos
        F = np.zeros(max_gl)
        U = np.zeros(max_gl)
        
        for i, info in enumerate(st.session_state.grados_libertad_info):
            if info['fuerza_conocida']:
                F[i] = info['valor_fuerza']
            if info['desplazamiento_conocido']:
                U[i] = info['valor_desplazamiento']
        
        # Identificar incógnitas
        incognitas_u = [i for i, info in enumerate(st.session_state.grados_libertad_info) 
                        if not info['desplazamiento_conocido']]
        conocidos_u = [i for i, info in enumerate(st.session_state.grados_libertad_info) 
                       if info['desplazamiento_conocido']]
        
        # Resolver para desplazamientos desconocidos
        if incognitas_u:
            K_uu = K_global[np.ix_(incognitas_u, incognitas_u)]
            K_uk = K_global[np.ix_(incognitas_u, conocidos_u)] if conocidos_u else np.zeros((len(incognitas_u), 0))
            
            F_u = F[incognitas_u]
            U_k = U[conocidos_u] if conocidos_u else np.array([])
            
            F_efectivo = F_u - (K_uk @ U_k if conocidos_u else 0)
            
            if np.linalg.det(K_uu) != 0:
                U_u = np.linalg.solve(K_uu, F_efectivo)
                
                for i, idx in enumerate(incognitas_u):
                    U[idx] = U_u[i]
            else:
                return None
        
        # Calcular fuerzas resultantes
        F_calculado = K_global @ U
        
        # Corregir el problema con fuerzas conocidas
        for i, info in enumerate(st.session_state.grados_libertad_info):
            if info['fuerza_conocida']:
                F_calculado[i] = info['valor_fuerza']
        
        return {
            'K_global': K_global,
            'desplazamientos': U,
            'fuerzas': F_calculado,
            'determinante': np.linalg.det(K_global),
            'exito': True
        }
        
    except Exception as e:
        return None

# Auto-calcular cuando hay cambios
def auto_calcular():
    """Calcular automáticamente cuando hay datos suficientes"""
    if (st.session_state.elementos and 
        st.session_state.grados_libertad_info and 
        st.session_state.auto_calcular):
        resultado = resolver_sistema()
        if resultado and resultado['exito']:
            st.session_state.resultados = resultado

# Funciones auxiliares para manejo de nodos y elementos
def eliminar_nodo(nodo_id):
    st.session_state.nodos = [n for n in st.session_state.nodos if n['id'] != nodo_id]
    for i, nodo in enumerate(st.session_state.nodos):
        nodo['id'] = i + 1
        nodo['grados_libertad_globales'] = calcular_grados_libertad_globales(i + 1)
    st.session_state.elementos = [e for e in st.session_state.elementos 
                                 if e['nodo_inicio'] != nodo_id and e['nodo_fin'] != nodo_id]
    st.rerun()

def eliminar_elemento(elemento_id):
    st.session_state.elementos = [e for e in st.session_state.elementos if e['id'] != elemento_id]
    if elemento_id in st.session_state.matrices_elementos:
        del st.session_state.matrices_elementos[elemento_id]
    for i, elemento in enumerate(st.session_state.elementos):
        elemento['id'] = i + 1
    st.rerun()

def eliminar_nodo_interactivo(nodo_id):
    st.session_state.nodos_interactivos = [n for n in st.session_state.nodos_interactivos if n['id'] != nodo_id]
    for i, nodo in enumerate(st.session_state.nodos_interactivos):
        nodo['id'] = i + 1
        nodo['grados_libertad_globales'] = calcular_grados_libertad_globales(i + 1)
    st.session_state.elementos_interactivos = [e for e in st.session_state.elementos_interactivos 
                                             if e['nodo_inicio'] != nodo_id and e['nodo_fin'] != nodo_id]
    st.rerun()

def eliminar_elemento_interactivo(elemento_id):
    st.session_state.elementos_interactivos = [e for e in st.session_state.elementos_interactivos if e['id'] != elemento_id]
    for i, elemento in enumerate(st.session_state.elementos_interactivos):
        elemento['id'] = i + 1
    st.rerun()

def agregar_nodo_interactivo(x, y, tipo='libre'):
    nodo_id = len(st.session_state.nodos_interactivos) + 1
    gl_globales = calcular_grados_libertad_globales(nodo_id)
    
    nuevo_nodo = {
        'id': nodo_id,
        'x': x,
        'y': y,
        'tipo': tipo,
        'grados_libertad_globales': gl_globales
    }
    
    st.session_state.nodos_interactivos.append(nuevo_nodo)
    return nodo_id

def agregar_elemento_interactivo(nodo_inicio_id, nodo_fin_id):
    if nodo_inicio_id == nodo_fin_id:
        return None
    
    for elem in st.session_state.elementos_interactivos:
        if (elem['nodo_inicio'] == nodo_inicio_id and elem['nodo_fin'] == nodo_fin_id) or \
           (elem['nodo_inicio'] == nodo_fin_id and elem['nodo_fin'] == nodo_inicio_id):
            return None
    
    elemento_id = len(st.session_state.elementos_interactivos) + 1
    
    nodo_inicio = next((n for n in st.session_state.nodos_interactivos if n['id'] == nodo_inicio_id), None)
    nodo_fin = next((n for n in st.session_state.nodos_interactivos if n['id'] == nodo_fin_id), None)
    
    if not nodo_inicio or not nodo_fin:
        return None
    
    gl_globales = nodo_inicio['grados_libertad_globales'] + nodo_fin['grados_libertad_globales']
    
    nuevo_elemento = {
        'id': elemento_id,
        'nodo_inicio': nodo_inicio_id,
        'nodo_fin': nodo_fin_id,
        'grados_libertad_global': gl_globales,
        'tipo': 'Barra',
        'material': None,
        'tipo_seccion': None,
        'parametros_seccion': {}
    }
    
    st.session_state.elementos_interactivos.append(nuevo_elemento)
    return elemento_id

def transferir_datos_interactivos():
    st.session_state.nodos = st.session_state.nodos_interactivos.copy()
    st.session_state.num_nodos = len(st.session_state.nodos)
    st.session_state.num_fijos = sum(1 for n in st.session_state.nodos if n['tipo'] == 'fijo')
    st.session_state.num_libres = st.session_state.num_nodos - st.session_state.num_fijos
    st.session_state.elementos = []
    st.session_state.matrices_elementos = {}
    st.session_state.num_elementos = len(st.session_state.elementos_interactivos)
    
    # Pre-poblar st.session_state.elementos desde interactivos para que grupos funcione
    for elem_interactivo in st.session_state.elementos_interactivos:
        st.session_state.elementos.append(elem_interactivo.copy())

    st.session_state.step = 6
    st.rerun()


# ----------------------------------------------------------------------
# ------------ FUNCIÓN DE APLICAR GRUPOS CORREGIDA ---------------------
# ----------------------------------------------------------------------
def aplicar_propiedades_grupo(nombre_grupo, material, tipo_seccion, parametros_seccion):
    """
    Aplica material y sección a todos los elementos de un grupo.
    ESTA VERSIÓN CORREGIDA NO ALTERA LA CONECTIVIDAD DE LOS NODOS.
    """
    if nombre_grupo not in st.session_state.grupos_elementos:
        st.error(f"Error: El grupo '{nombre_grupo}' no fue encontrado.")
        return False
    
    elementos_grupo_ids = st.session_state.grupos_elementos[nombre_grupo]
    todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
    
    if material not in todos_materiales:
        st.error(f"Error: Material '{material}' no encontrado.")
        return False
    
    props_material = todos_materiales[material]
    area_calculada = calcular_area_seccion(tipo_seccion, parametros_seccion)
    
    elementos_actualizados = 0
    
    # Iteramos sobre los IDs de elementos que pertenecen al grupo
    for elemento_id in elementos_grupo_ids:
        # Buscamos el elemento correspondiente en la lista principal de elementos
        elemento_a_actualizar = next((e for e in st.session_state.elementos if e['id'] == elemento_id), None)
        
        if elemento_a_actualizar is None:
            # Si el elemento no existe en la lista principal, es un error de flujo. No lo creamos aquí.
            st.warning(f"Elemento {elemento_id} del grupo '{nombre_grupo}' no encontrado en la lista principal. Se omitirá.")
            continue

        # Buscamos los nodos para recalcular longitud y ángulo
        nodo_inicio_obj = next((n for n in st.session_state.nodos if n['id'] == elemento_a_actualizar['nodo_inicio']), None)
        nodo_fin_obj = next((n for n in st.session_state.nodos if n['id'] == elemento_a_actualizar['nodo_fin']), None)
        
        if nodo_inicio_obj and nodo_fin_obj:
            # --- SECCIÓN DE ACTUALIZACIÓN DE PROPIEDADES ---
            # NO SE MODIFICA LA CONECTIVIDAD (nodo_inicio, nodo_fin)

            # Actualizar propiedades del material y sección
            elemento_a_actualizar['material'] = material
            elemento_a_actualizar['tipo_seccion'] = tipo_seccion
            elemento_a_actualizar['parametros_seccion'] = parametros_seccion
            elemento_a_actualizar['area'] = area_calculada
            
            # Recalcular propiedades geométricas basadas en su conectividad ORIGINAL
            longitud = calcular_longitud_elemento(nodo_inicio_obj, nodo_fin_obj)
            beta = calcular_angulo_beta(nodo_inicio_obj, nodo_fin_obj)
            
            elemento_a_actualizar['longitud'] = longitud
            elemento_a_actualizar['beta'] = beta
            
            # Regenerar y guardar la matriz de rigidez del elemento
            E = props_material['modulo_young']
            A = area_calculada
            L = longitud
            
            matriz_numerica = generar_matriz_rigidez_barra(E, A, L, beta)
            st.session_state.matrices_elementos[elemento_id] = {
                # Mantenemos la simbólica simple para consistencia, aunque no se use en el cálculo final
                'simbolica': [], 
                'numerica': matriz_numerica.tolist()
            }
            
            elementos_actualizados += 1
        else:
            st.warning(f"No se encontraron los nodos para el elemento {elemento_id}. Se omitirá la actualización.")
            
    # Finalmente, actualizamos la lista de elementos en el session_state si es necesario
    # (Aunque al modificar el diccionario 'elemento_a_actualizar' ya se modifica por referencia)
    st.session_state.elementos = st.session_state.elementos

    return elementos_actualizados > 0

# ----------------------------------------------------------------------
# ------------ NUEVA FUNCIÓN PARA ANÁLISIS DE TENSIONES ----------------
# ----------------------------------------------------------------------
def calcular_y_mostrar_tensiones(resultado):
    """Calcula y muestra las tensiones y deformaciones en cada elemento."""
    st.markdown("### ⚡ Análisis de Tensiones y Deformaciones")
    
    tensiones_data = []
    todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}

    for elemento in st.session_state.elementos:
        # Obtener desplazamientos de los nodos del elemento
        gl_globales = elemento['grados_libertad_global']
        u_locales = np.array([
            resultado['desplazamientos'][gl_globales[0]-1],
            resultado['desplazamientos'][gl_globales[1]-1],
            resultado['desplazamientos'][gl_globales[2]-1],
            resultado['desplazamientos'][gl_globales[3]-1]
        ])

        # Matriz de transformación T
        beta = elemento['beta']
        c = math.cos(beta)
        s = math.sin(beta)
        T = np.array([[-c, -s, c, s]])

        # Deformación axial (ε)
        L = elemento['longitud']
        if L == 0:
            deformacion_unitaria = 0
        else:
            # delta = T * u
            delta = T @ u_locales
            deformacion_unitaria = delta[0] / L
        
        # Tensión (σ = E * ε)
        E = todos_materiales[elemento['material']]['modulo_young']
        tension = E * deformacion_unitaria
        
        tensiones_data.append({
            'Elemento': elemento['id'],
            'Material': elemento['material'],
            'Deformación Unitaria (ε)': f"{deformacion_unitaria:.6e}",
            'Tensión (σ)': formatear_unidades(tension, 'presion')
        })

    if tensiones_data:
        df_tensiones = pd.DataFrame(tensiones_data)
        st.dataframe(df_tensiones, use_container_width=True)
    else:
        st.warning("No se pudieron calcular las tensiones.")


def mostrar_barra_progreso():
    """Mostrar barra de progreso estilo web moderno"""
    if st.session_state.step == 0:
        return
    
    pasos = [
        "Información del Usuario",
        "Número de Nodos", 
        "Clasificación de Nodos",
        "Coordenadas de Nodos",
        "Número de Elementos",
        "Definición de Elementos", 
        "Configuración de Incógnitas",
        "Resultados"
    ]
    
    st.markdown("""
    <div class='progress-bar'>
        <div style='max-width: 1200px; margin: 0 auto; padding: 0 2rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                <h1 style='margin: 0; font-size: 1.8rem;'>Análisis Estructural - Método de Matrices</h1>
                <div style='display: flex; gap: 1rem;'>
    """, unsafe_allow_html=True)
    
    # Botones de control en la barra
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Reiniciar", key="reset_top"):
            reset_app()
    with col2:
        if st.session_state.step > 1:
            if st.button("← Anterior", key="prev_top"):
                prev_step()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Mostrar pasos con círculos
    progress_html = "<div class='progress-steps'>"
    
    for i, paso in enumerate(pasos, 1):
        if i < st.session_state.step:
            circle_class = "completed"
        elif i == st.session_state.step:
            circle_class = "current"
        else:
            circle_class = "pending"
        
        progress_html += f"""
        <div class='progress-step'>
            <div class='step-circle {circle_class}'>
                {'✓' if i < st.session_state.step else i}
            </div>
        """
        
        if i < len(pasos):
            line_class = "completed" if i < st.session_state.step else ""
            progress_html += f"<div class='step-line {line_class}'></div>"
        
        progress_html += "</div>"
    
    progress_html += "</div>"
    
    # Mostrar paso actual
    progress_html += f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <div style='font-size: 1.1rem; font-weight: 600; color: var(--gray-800);'>
                Paso {st.session_state.step}: {pasos[st.session_state.step - 1]}
            </div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def mostrar_sidebar_mejorado():
    """Mostrar sidebar mejorado solo cuando hay modo seleccionado"""
    if st.session_state.modo is None:
        return
    
    # Agregar clase CSS para mostrar sidebar
    st.markdown('<div class="show-sidebar">', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Progreso del Análisis")
        
        # Indicador de paso actual
        pasos = [
            "1. Información del Usuario",
            "2. Número de Nodos",
            "3. Clasificación de Nodos", 
            "4. Coordenadas de Nodos",
            "5. Número de Elementos",
            "6. Definición de Elementos",
            "7. Configuración de Incógnitas",
            "8. Resultados"
        ]
        
        for i, paso in enumerate(pasos, 1):
            if i == st.session_state.step:
                st.markdown(f"**→ {paso}**")
            elif i < st.session_state.step:
                st.markdown(f"✓ {paso}")
            else:
                st.markdown(f"⏳ {paso}")
        
        st.divider()
        
        # Información del proyecto
        st.markdown("### Información del Proyecto")
        
        if st.session_state.usuario_nombre:
            st.markdown(f"**Usuario:** {st.session_state.usuario_nombre}")
        
        if st.session_state.modo:
            st.markdown(f"**Modo:** {st.session_state.modo.capitalize()}")
        
        if st.session_state.step >= 4 and st.session_state.nodos:
            st.markdown(f"**Nodos:** {len(st.session_state.nodos)}")
        
        if st.session_state.step >= 6 and st.session_state.elementos:
            st.markdown(f"**Elementos:** {len(st.session_state.elementos)}")
        
        if st.session_state.step >= 7 and st.session_state.grados_libertad_info:
            st.markdown(f"**Grados de Libertad:** {len(st.session_state.grados_libertad_info)}")
        
        st.markdown(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
        st.markdown(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")

import streamlit as st

# Solo la función que da problemas
def mostrar_footer_encuesta():
    """Mostrar footer con encuesta y agradecimientos"""
    st.markdown("""
    <div style='background-color: #212529; padding: 2rem;'>
        <div class='footer-section'>
            <div class='footer-content'>
                <div class='footer-survey'>
                    <h3 style='font-size: 2rem; font-weight: 700; margin-bottom: 1rem; color: white;'>📝 Evaluación del Sistema</h3>
                    <p style='font-size: 1.1rem; margin-bottom: 1.5rem; line-height: 1.6; color: white;'>
                        <strong>¡Su opinión es importante!</strong><br>
                        Ayúdenos a mejorar este sistema de análisis estructural completando nuestra breve encuesta.
                    </p>
                    <a href='https://forms.gle/31KgSu263hf8dH5UA' target='_blank' 
                       style='display: inline-block; background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%); 
                              color: white; padding: 1rem 2rem; border-radius: 12px; text-decoration: none; 
                              font-weight: 600; font-size: 1.1rem; transition: all 0.3s ease;
                              box-shadow: 0 4px 15px rgba(74, 85, 104, 0.3);'>
                        📋 Evaluar Sistema de Análisis Estructural
                    </a>
                    <p style='margin-top: 1rem; font-size: 0.9rem; color: rgba(255,255,255,0.8);'>
                        Su retroalimentación nos ayuda a desarrollar mejores herramientas para la comunidad de ingeniería.
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# INTERFAZ PRINCIPAL MODIFICADA
# Mostrar barra de progreso si no estamos en step 0
mostrar_barra_progreso()

# Mostrar sidebar solo si hay modo seleccionado
mostrar_sidebar_mejorado()

# Contenido principal según el paso
if st.session_state.step == 0:
    # Página de inicio estilo landing page - CORREGIDA
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); 
                min-height: 80vh; display: flex; align-items: center; justify-content: center; 
                margin: -1rem; padding: 2rem; border-radius: 15px;'>
        <div style='max-width: 1200px; text-align: center;'>
            <h1 style='font-size: 4rem; margin-bottom: 1rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                Análisis Estructural
            </h1>
            <p style='font-size: 1.5rem; color: rgba(255,255,255,0.9); margin-bottom: 3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);'>
                Método de Matrices • Diseño Moderno
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tarjetas de modo con Streamlit nativo
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; border: 2px solid #e2e8f0; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;'>
            <h3 style='color: #1a202c; font-size: 1.8rem; margin-bottom: 1rem; text-align: center;'>Modo Manual</h3>
            <p style='color: #4a5568; line-height: 1.6; margin-bottom: 1.5rem; text-align: center;'>
                Control preciso con coordenadas exactas.<br><br>
                • Coordenadas exactas de cada nodo<br>
                • Conexiones entre nodos para formar barras<br>
                • Propiedades de materiales y secciones
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("SELECCIONAR MANUAL", key="manual_mode", type="primary", use_container_width=True):
            set_modo("manual")
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; border: 2px solid #e2e8f0; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;'>
            <h3 style='color: #1a202c; font-size: 1.8rem; margin-bottom: 1rem; text-align: center;'>Modo Interactivo</h3>
            <p style='color: #4a5568; line-height: 1.6; margin-bottom: 1.5rem; text-align: center;'>
                Diseño visual rápido e intuitivo.<br><br>
                • Colocar nodos haciendo clic en un gráfico<br>
                • Conectar nodos visualmente para crear barras<br>
                • Definir propiedades mediante formularios simples
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("SELECCIONAR INTERACTIVO", key="interactive_mode", type="primary", use_container_width=True):
            set_modo("interactivo")
    
    # Características del sistema
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px; border: 2px solid #e2e8f0; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 2rem 0;'>
        <h3 style='color: #1a202c; font-size: 1.5rem; margin-bottom: 1.5rem; text-align: center;'>
            🎯 Características del Sistema Avanzado
        </h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; color: #4a5568;'>
            <div style='text-align: left;'>
                <p>✓ Análisis de <strong>barras</strong> con matrices 4×4</p>
                <p>✓ <strong>Base de datos de materiales</strong> aeroespaciales</p>
                <p>✓ <strong>Formateo inteligente de unidades</strong></p>
                <p>✓ Configuración automática de grados de libertad</p>
                <p>✓ <strong>Exportación PDF y Excel</strong> completa</p>
            </div>
            <div style='text-align: left;'>
                <p>✓ <strong>Tablas copiables</strong> para Word/Excel</p>
                <p>✓ Visualización gráfica optimizada</p>
                <p>✓ <strong>Cálculo automático</strong> en tiempo real</p>
                <p>✓ <strong>Edición y eliminación</strong> de elementos</p>
                <p>✓ <strong>Secciones personalizables</strong> avanzadas</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.step == 1:
    st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
    st.markdown("## Información del Usuario")
    st.markdown("Bienvenido al sistema de análisis estructural avanzado. Por favor, ingrese su información.")
    
    usuario_nombre = st.text_input("👤 Nombre completo:", 
                                  value=st.session_state.usuario_nombre,
                                  placeholder="Ej: Juan Pérez")
    
    if usuario_nombre:
        st.session_state.usuario_nombre = usuario_nombre
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 2rem; border-radius: 15px; border: 1px solid #dee2e6; margin: 2rem 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #000000; margin-bottom: 1.5rem; font-weight: 600;'>🎯 Características del Sistema Avanzado</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; color: #495057;'>
                <div>
                    <p>✓ Análisis de <strong>barras</strong> con matrices 4×4</p>
                    <p>✓ <strong>Base de datos de materiales</strong> aeroespaciales</p>
                    <p>✓ <strong>Formateo inteligente de unidades</strong></p>
                    <p>✓ Configuración automática de grados de libertad</p>
                    <p>✓ <strong>Exportación PDF y Excel</strong> completa</p>
                </div>
                <div>
                    <p>✓ <strong>Tablas copiables</strong> para Word/Excel</p>
                    <p>✓ Visualización gráfica optimizada</p>
                    <p>✓ <strong>Cálculo automático</strong> en tiempo real</p>
                    <p>✓ <strong>Edición y eliminación</strong> de elementos</p>
                    <p>✓ <strong>Secciones personalizables</strong> avanzadas</p>
                </div>
            </div>
            <div>
            <h4 style='color: #000000; margin: 1.5rem 0 1rem 0; font-weight: 600;'>🧪 Materiales Aeroespaciales Incluidos</h4>
            <p style='color: #6c757d; line-height: 1.6;'>
                Aluminio 6061-T6, 7075-T6, 2024-T3 • Titanio Ti-6Al-4V • Acero 4130 • 
                Fibra de Carbono T300 • Magnesio AZ31B • + Materiales personalizados
            </div>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continuar →", type="primary", use_container_width=True):
            if st.session_state.modo == "manual":
                next_step()
            else:
                st.session_state.step = 2
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.modo == "manual" and st.session_state.step == 2:
    st.markdown("## Número de Nodos")
    st.markdown("Ingrese el número total de nodos en el sistema")
    
    num_nodos = st.number_input("Número de Nodos", min_value=1, max_value=20, 
                               value=st.session_state.num_nodos)
    
    if st.button("Continuar →", type="primary"):
        st.session_state.num_nodos = num_nodos
        next_step()

elif st.session_state.modo == "manual" and st.session_state.step == 3:
    st.markdown("## Clasificación de Nodos")
    st.markdown("Defina cuántos nodos son fijos y cuántos son libres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_fijos = st.number_input("Nodos Fijos", min_value=0, max_value=st.session_state.num_nodos, 
                                   value=st.session_state.num_fijos)
    
    with col2:
        num_libres = st.session_state.num_nodos - num_fijos
        st.metric("Nodos Libres", num_libres)
    
    if st.button("Continuar →", type="primary"):
        st.session_state.num_fijos = num_fijos
        st.session_state.num_libres = num_libres
        next_step()

elif st.session_state.modo == "manual" and st.session_state.step == 4:
    st.markdown("## Coordenadas de Nodos")
    st.markdown("Ingrese las coordenadas para cada nodo")
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpiar Todos los Nodos", type="secondary"):
            st.session_state.nodos = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Cambiar Número de Nodos", type="secondary"):
            st.session_state.step = 2
            st.rerun()
    
    # Determinar tipos de nodos
    tipos_nodos = ['fijo'] * st.session_state.num_fijos + ['libre'] * st.session_state.num_libres
    
    # Mostrar formulario para todos los nodos
    with st.form("coordenadas_nodos"):
        st.markdown("### Coordenadas de Todos los Nodos")
        
        nodos_temp = []
        
        for i in range(st.session_state.num_nodos):
            nodo_id = i + 1
            tipo_actual = tipos_nodos[i]
            
            # Buscar nodo existente o usar valores por defecto
            nodo_existente = next((n for n in st.session_state.nodos if n['id'] == nodo_id), None)
            x_default = nodo_existente['x'] if nodo_existente else 0.0
            y_default = nodo_existente['y'] if nodo_existente else 0.0
            
            st.markdown(f"**Nodo {nodo_id} ({tipo_actual.title()})**")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                x = st.number_input(f"X{nodo_id}", value=x_default, format="%.2f", key=f"x_{nodo_id}")
            
            with col2:
                y = st.number_input(f"Y{nodo_id}", value=y_default, format="%.2f", key=f"y_{nodo_id}")
            
            with col3:
                gl_globales = calcular_grados_libertad_globales(nodo_id)
                st.markdown(f"GL Globales: {gl_globales}")
                st.markdown(f"GL{gl_globales[0]} → X, GL{gl_globales[1]} → Y")
            
            nodos_temp.append({
                'id': nodo_id,
                'x': x,
                'y': y,
                'tipo': tipo_actual,
                'grados_libertad_globales': gl_globales
            })
        
        if st.form_submit_button("💾 Guardar Todos los Nodos", type="primary"):
            st.session_state.nodos = nodos_temp
            st.success(f"✅ Se guardaron {len(nodos_temp)} nodos correctamente")
    
    # Mostrar nodos guardados
    if st.session_state.nodos:
        st.markdown("### 📋 Nodos Configurados")
        df_nodos = pd.DataFrame(st.session_state.nodos)
        st.dataframe(df_nodos, use_container_width=True)
        
        if len(st.session_state.nodos) == st.session_state.num_nodos:
            if st.button("Continuar →", type="primary"):
                next_step()

elif st.session_state.modo == "interactivo" and st.session_state.step == 2:
    st.markdown("## Editor Interactivo de Estructura")
    st.markdown("Utilice el gráfico para crear su estructura. Haga clic para añadir nodos y conectarlos para formar barras.")
    
    # Columnas para controles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tipo_nodo = st.radio("Tipo de nodo a añadir:", ["libre", "fijo"])
    
    with col2:
        if st.button("🗑️ Limpiar Todo", type="secondary"):
            st.session_state.nodos_interactivos = []
            st.session_state.elementos_interactivos = []
            st.session_state.nodo_seleccionado_interactivo = None
            st.rerun()
    
    with col3:
        if st.button("✅ Finalizar Diseño", type="primary"):
            if len(st.session_state.nodos_interactivos) >= 2 and len(st.session_state.elementos_interactivos) >= 1:
                transferir_datos_interactivos()
            else:
                st.error("Necesita al menos 2 nodos y 1 barra para continuar")
    
    # Crear gráfico interactivo moderno
    fig = crear_grafico_interactivo_moderno()
    st.plotly_chart(fig, use_container_width=True)

    # Controles para añadir nodos manualmente
    st.markdown("### Añadir Nodos Manualmente")
    col1, col2, col3 = st.columns(3)

    with col1:
        x_nuevo = st.number_input("Coordenada X", value=0.0, format="%.2f", key="x_nuevo")

    with col2:
        y_nuevo = st.number_input("Coordenada Y", value=0.0, format="%.2f", key="y_nuevo")

    with col3:
        if st.button("➕ Añadir Nodo", type="primary"):
            nodo_id = agregar_nodo_interactivo(x_nuevo, y_nuevo, tipo_nodo)
            st.success(f"Nodo {nodo_id} ({tipo_nodo}) añadido en ({x_nuevo:.2f}, {y_nuevo:.2f})")
            st.rerun()

    # Controles para crear barras
    if len(st.session_state.nodos_interactivos) >= 2:
        st.markdown("### Crear Barras")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nodos_disponibles = [n['id'] for n in st.session_state.nodos_interactivos]
            nodo_inicio_sel = st.selectbox("Nodo Inicio", nodos_disponibles, key="nodo_inicio_sel")
        
        with col2:
            nodos_fin_disponibles = [n for n in nodos_disponibles if n != nodo_inicio_sel]
            nodo_fin_sel = st.selectbox("Nodo Fin", nodos_fin_disponibles, key="nodo_fin_sel")
        
        with col3:
            if st.button("🔗 Crear Barra", type="primary"):
                elemento_id = agregar_elemento_interactivo(nodo_inicio_sel, nodo_fin_sel)
                if elemento_id:
                    st.success(f"Barra {elemento_id} creada entre nodos {nodo_inicio_sel} y {nodo_fin_sel}")
                    st.rerun()
                else:
                    st.warning("No se pudo crear la barra (puede que ya exista)")
    
    # Mostrar nodos y elementos en tablas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Nodos")
        if st.session_state.nodos_interactivos:
            for nodo in st.session_state.nodos_interactivos:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"Nodo {nodo['id']} ({nodo['tipo']}): ({nodo['x']:.2f}, {nodo['y']:.2f})")
                with col_b:
                    if st.button(f"🗑️", key=f"del_nodo_{nodo['id']}"):
                        eliminar_nodo_interactivo(nodo['id'])
    
    with col2:
        st.markdown("### Barras")
        if st.session_state.elementos_interactivos:
            for elem in st.session_state.elementos_interactivos:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"Barra {elem['id']}: Nodo {elem['nodo_inicio']} → Nodo {elem['nodo_fin']}")
                with col_b:
                    if st.button(f"🗑️", key=f"del_elem_{elem['id']}"):
                        eliminar_elemento_interactivo(elem['id'])

elif st.session_state.step == 5:
    st.markdown("## Número de Elementos")
    st.markdown("Ingrese el número total de elementos (barras) en el sistema")
    
    num_elementos = st.number_input("Número de Elementos", min_value=1, max_value=50, 
                                   value=st.session_state.num_elementos)
    
    if st.button("Continuar →", type="primary"):
        st.session_state.num_elementos = num_elementos
        next_step()

elif st.session_state.step == 6:
    st.markdown("## Definición de Elementos")
    st.markdown("Configure cada elemento (barra) del sistema")
    
    # NUEVA FUNCIONALIDAD: Gestión de grupos de elementos AL INICIO - CORREGIDA
    st.markdown("### 🔗 Grupos de Elementos")
    st.markdown("Cree grupos de elementos para asignar materiales y secciones de forma eficiente a múltiples elementos.")
    
    with st.expander("Crear y Gestionar Grupos de Elementos", expanded=True):
        # Crear nuevo grupo
        st.markdown("#### Crear Nuevo Grupo")
        col1, col2 = st.columns(2)
        
        with col1:
            # Generar nombre por defecto
            num_grupos = len(st.session_state.grupos_elementos)
            nombre_default = f"Grupo de elementos {num_grupos + 1}"
            nombre_grupo = st.text_input("Nombre del Grupo", value=nombre_default, key="nombre_grupo_elementos")
        
        with col2:
            if st.session_state.modo == "interactivo":
                elementos_disponibles = [f"Elemento {e['id']}" for e in st.session_state.elementos]
            else:
                elementos_disponibles = [f"Elemento {i+1}" for i in range(st.session_state.num_elementos)]
            
            elementos_seleccionados = st.multiselect("Seleccionar Elementos", elementos_disponibles, key="elementos_grupo")
        
        if st.button("Crear Grupo de Elementos", key="crear_grupo_elementos") and nombre_grupo and elementos_seleccionados:
            ids_elementos = [int(e.split()[1]) for e in elementos_seleccionados]
            st.session_state.grupos_elementos[nombre_grupo] = ids_elementos
            st.success(f"Grupo '{nombre_grupo}' creado con {len(ids_elementos)} elementos")
            st.rerun()
        
        # Mostrar grupos existentes y aplicar propiedades
        if st.session_state.grupos_elementos:
            st.markdown("#### Grupos Existentes y Aplicación de Propiedades")
            
            for nombre, elementos in st.session_state.grupos_elementos.items():
                with st.container():
                    st.markdown(f"**{nombre}:** Elementos {elementos}")
                    
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    with col1:
                        todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
                        nombres_materiales = list(todos_materiales.keys())
                        material_grupo = st.selectbox(f"Material", nombres_materiales, key=f"material_grupo_{nombre}")
                    
                    with col2:
                        tipo_seccion_grupo = st.selectbox(
                            "Tipo de Sección",
                            ["circular_solida", "circular_hueca", "rectangular", "cuadrada"],
                            format_func=lambda x: {
                                "circular_solida": "Circular Sólida",
                                "circular_hueca": "Circular Hueca",
                                "rectangular": "Rectangular",
                                "cuadrada": "Cuadrada"
                            }[x],
                            key=f"tipo_seccion_grupo_{nombre}"
                        )
                    
                    with col3:
                        parametros_grupo = {}
                        if tipo_seccion_grupo == "circular_solida":
                            radio = st.number_input("Radio (m)", value=0.01, min_value=0.001, format="%.4f", key=f"radio_grupo_{nombre}")
                            parametros_grupo['radio'] = radio
                            area_grupo = math.pi * radio**2
                            st.caption(f"Área: {area_grupo:.6f} m²")
                        elif tipo_seccion_grupo == "circular_hueca":
                            radio_ext = st.number_input("Radio Ext (m)", value=0.02, min_value=0.001, format="%.4f", key=f"radio_ext_grupo_{nombre}")
                            radio_int = st.number_input("Radio Int (m)", value=0.01, min_value=0.0, max_value=radio_ext*0.99, format="%.4f", key=f"radio_int_grupo_{nombre}")
                            parametros_grupo['radio_ext'] = radio_ext
                            parametros_grupo['radio_int'] = radio_int
                            area_grupo = math.pi * (radio_ext**2 - radio_int**2)
                            st.caption(f"Área: {area_grupo:.6f} m²")
                        elif tipo_seccion_grupo == "rectangular":
                            lado1 = st.number_input("Lado 1 (m)", value=0.02, min_value=0.001, format="%.4f", key=f"lado1_grupo_{nombre}")
                            lado2 = st.number_input("Lado 2 (m)", value=0.01, min_value=0.001, format="%.4f", key=f"lado2_grupo_{nombre}")
                            parametros_grupo['lado1'] = lado1
                            parametros_grupo['lado2'] = lado2
                            area_grupo = lado1 * lado2
                            st.caption(f"Área: {area_grupo:.6f} m²")
                        elif tipo_seccion_grupo == "cuadrada":
                            lado = st.number_input("Lado (m)", value=0.02, min_value=0.001, format="%.4f", key=f"lado_grupo_{nombre}")
                            parametros_grupo['lado'] = lado
                            area_grupo = lado**2
                            st.caption(f"Área: {area_grupo:.6f} m²")
                    
                    with col4:
                        if st.button("Aplicar al Grupo", type="primary", key=f"aplicar_grupo_{nombre}"):
                            if aplicar_propiedades_grupo(nombre, material_grupo, tipo_seccion_grupo, parametros_grupo):
                                st.success(f"✅ Propiedades aplicadas al grupo '{nombre}' exitosamente")
                                st.rerun()
                            else:
                                st.error(f"❌ Error al aplicar propiedades al grupo '{nombre}'")
                        
                        if st.button("🗑️ Eliminar", type="secondary", key=f"eliminar_grupo_{nombre}"):
                            del st.session_state.grupos_elementos[nombre]
                            st.success(f"Grupo '{nombre}' eliminado")
                            st.rerun()
                    
                    st.divider()
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Limpiar Todos los Elementos", type="secondary"):
            st.session_state.elementos = []
            st.session_state.matrices_elementos = {}
            st.rerun()
    with col2:
        if st.button("🔄 Cambiar Número de Elementos", type="secondary"):
            st.session_state.step = 5
            st.rerun()
    
    with st.expander("📚 Base de Datos de Materiales"):
        for nombre, props in MATERIALES_AEROESPACIALES.items():
            st.markdown(f"**{nombre}**: E = {formatear_unidades(props['modulo_young'], 'presion')}, ρ = {props['densidad']} kg/m³")
            st.caption(props['descripcion'])
    
    with st.expander("➕ Agregar Material Personalizado"):
        with st.form("material_personalizado"):
            col1, col2, col3 = st.columns(3)
            with col1:
                nombre_material = st.text_input("Nombre del Material")
            with col2:
                modulo_young = st.number_input("Módulo de Young (Pa)", value=200e9, format="%.2e")
            with col3:
                densidad = st.number_input("Densidad (kg/m³)", value=7850.0)
            descripcion = st.text_area("Descripción")
            if st.form_submit_button("Agregar Material"):
                if nombre_material:
                    st.session_state.materiales_personalizados[nombre_material] = {'modulo_young': modulo_young, 'densidad': densidad, 'descripcion': descripcion}
                    st.success(f"Material '{nombre_material}' agregado exitosamente")
    
    st.markdown("### 🔧 Configuración Individual de Elementos")
    
    if st.session_state.modo == "interactivo":
        elementos_base = st.session_state.elementos
    else:
        elementos_base = [{'id': i+1} for i in range(st.session_state.num_elementos)]
    
    for i, elem_base in enumerate(elementos_base):
        elemento_id = elem_base['id']
        with st.expander(f"🔧 Elemento {elemento_id} (Barra)", expanded=False):
            elemento_existente = next((e for e in st.session_state.elementos if e['id'] == elemento_id), None)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.modo == "interactivo":
                    nodo_inicio = elem_base['nodo_inicio']
                    nodo_fin = elem_base['nodo_fin']
                    st.markdown(f"**Nodo Inicio:** {nodo_inicio}")
                    st.markdown(f"**Nodo Fin:** {nodo_fin}")
                else:
                    nodos_disponibles = [n['id'] for n in st.session_state.nodos]
                    nodo_inicio_default = elemento_existente['nodo_inicio'] if elemento_existente else nodos_disponibles[0]
                    nodo_fin_default = elemento_existente['nodo_fin'] if elemento_existente else nodos_disponibles[-1]
                    nodo_inicio = st.selectbox(f"Nodo Inicio", nodos_disponibles, index=nodos_disponibles.index(nodo_inicio_default) if nodo_inicio_default in nodos_disponibles else 0, key=f"inicio_{elemento_id}")
                    nodo_fin = st.selectbox(f"Nodo Fin", nodos_disponibles, index=nodos_disponibles.index(nodo_fin_default) if nodo_fin_default in nodos_disponibles else -1, key=f"fin_{elemento_id}")
            
            with col2:
                todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
                nombres_materiales = list(todos_materiales.keys())
                material_default = elemento_existente['material'] if elemento_existente and elemento_existente.get('material') else nombres_materiales[0]
                material_idx = nombres_materiales.index(material_default) if material_default in nombres_materiales else 0
                material_seleccionado = st.selectbox(f"Material", nombres_materiales, index=material_idx, key=f"material_{elemento_id}")
                props_material = todos_materiales[material_seleccionado]
                st.markdown(f"E = {formatear_unidades(props_material['modulo_young'], 'presion')}")
                st.markdown(f"ρ = {props_material['densidad']} kg/m³")
            
            st.markdown("#### Tipo de Sección")
            tipo_seccion_default = elemento_existente.get('tipo_seccion', 'circular_solida') if elemento_existente else 'circular_solida'
            tipo_seccion = st.radio("Seleccione el tipo de sección:", ["circular_solida", "circular_hueca", "rectangular", "cuadrada"], format_func=lambda x: {"circular_solida": "Circular Sólida", "circular_hueca": "Circular Hueca (Cilindro)", "rectangular": "Rectangular", "cuadrada": "Cuadrada"}[x], index=["circular_solida", "circular_hueca", "rectangular", "cuadrada"].index(tipo_seccion_default) if tipo_seccion_default in ["circular_solida", "circular_hueca", "rectangular", "cuadrada"] else 0, key=f"tipo_seccion_{elemento_id}")
            
            parametros_seccion = {}
            if tipo_seccion == "circular_solida":
                col1, col2 = st.columns(2)
                with col1:
                    radio_default = elemento_existente.get('parametros_seccion', {}).get('radio', 0.01) if elemento_existente else 0.01
                    radio = st.number_input(f"Radio (m)", value=radio_default, min_value=0.001, format="%.4f", key=f"radio_{elemento_id}")
                    parametros_seccion['radio'] = radio
                with col2:
                    area = math.pi * radio**2
                    st.metric("Área calculada", f"{area:.6f} m²")
            
            elif tipo_seccion == "circular_hueca":
                col1, col2, col3 = st.columns(3)
                with col1:
                    radio_ext_default = elemento_existente.get('parametros_seccion', {}).get('radio_ext', 0.02) if elemento_existente else 0.02
                    radio_ext = st.number_input(f"Radio Exterior (m)", value=radio_ext_default, min_value=0.001, format="%.4f", key=f"radio_ext_{elemento_id}")
                    parametros_seccion['radio_ext'] = radio_ext
                with col2:
                    radio_int_default = elemento_existente.get('parametros_seccion', {}).get('radio_int', 0.01) if elemento_existente else 0.01
                    radio_int = st.number_input(f"Radio Interior (m)", value=radio_int_default, min_value=0.0, max_value=radio_ext*0.99, format="%.4f", key=f"radio_int_{elemento_id}")
                    parametros_seccion['radio_int'] = radio_int
                with col3:
                    area = math.pi * (radio_ext**2 - radio_int**2)
                    st.metric("Área calculada", f"{area:.6f} m²")
            
            elif tipo_seccion == "rectangular":
                col1, col2, col3 = st.columns(3)
                with col1:
                    lado1_default = elemento_existente.get('parametros_seccion', {}).get('lado1', 0.02) if elemento_existente else 0.02
                    lado1 = st.number_input(f"Lado 1 (m)", value=lado1_default, min_value=0.001, format="%.4f", key=f"lado1_{elemento_id}")
                    parametros_seccion['lado1'] = lado1
                with col2:
                    lado2_default = elemento_existente.get('parametros_seccion', {}).get('lado2', 0.01) if elemento_existente else 0.01
                    lado2 = st.number_input(f"Lado 2 (m)", value=lado2_default, min_value=0.001, format="%.4f", key=f"lado2_{elemento_id}")
                    parametros_seccion['lado2'] = lado2
                with col3:
                    area = lado1 * lado2
                    st.metric("Área calculada", f"{area:.6f} m²")
            
            elif tipo_seccion == "cuadrada":
                col1, col2 = st.columns(2)
                with col1:
                    lado_default = elemento_existente.get('parametros_seccion', {}).get('lado', 0.02) if elemento_existente else 0.02
                    lado = st.number_input(f"Lado (m)", value=lado_default, min_value=0.001, format="%.4f", key=f"lado_{elemento_id}")
                    parametros_seccion['lado'] = lado
                with col2:
                    area = lado**2
                    st.metric("Área calculada", f"{area:.6f} m²")
            
            area_final = calcular_area_seccion(tipo_seccion, parametros_seccion)
            
            if st.button(f"💾 Guardar Elemento {elemento_id}", key=f"guardar_{elemento_id}"):
                nodo_inicio_obj = next((n for n in st.session_state.nodos if n['id'] == nodo_inicio), None)
                nodo_fin_obj = next((n for n in st.session_state.nodos if n['id'] == nodo_fin), None)
                if nodo_inicio_obj and nodo_fin_obj:
                    longitud = calcular_longitud_elemento(nodo_inicio_obj, nodo_fin_obj)
                    beta = calcular_angulo_beta(nodo_inicio_obj, nodo_fin_obj)
                    gl_globales = nodo_inicio_obj['grados_libertad_globales'] + nodo_fin_obj['grados_libertad_globales']
                    nuevo_elemento = {'id': elemento_id, 'nodo_inicio': nodo_inicio, 'nodo_fin': nodo_fin, 'tipo': 'Barra', 'material': material_seleccionado, 'tipo_seccion': tipo_seccion, 'parametros_seccion': parametros_seccion, 'area': area_final, 'longitud': longitud, 'beta': beta, 'grados_libertad_global': gl_globales}
                    elemento_idx = next((i for i, e in enumerate(st.session_state.elementos) if e['id'] == elemento_id), None)
                    if elemento_idx is not None:
                        st.session_state.elementos[elemento_idx] = nuevo_elemento
                    else:
                        st.session_state.elementos.append(nuevo_elemento)
                    
                    E = props_material['modulo_young']
                    A = area_final
                    L = longitud
                    matriz_numerica = generar_matriz_rigidez_barra(E, A, L, beta)
                    st.session_state.matrices_elementos[elemento_id] = {'simbolica': [], 'numerica': matriz_numerica.tolist()}
                    st.success(f"✅ Elemento {elemento_id} guardado correctamente")
                    st.rerun()
    
    if st.session_state.elementos:
        st.markdown("### 📋 Elementos Configurados")
        df_elementos = crear_tabla_conectividad()
        st.dataframe(df_elementos, use_container_width=True)
        if len(st.session_state.elementos) >= st.session_state.num_elementos:
            if st.button("Continuar →", type="primary"):
                max_gl = max([max(nodo['grados_libertad_globales']) for nodo in st.session_state.nodos])
                st.session_state.grados_libertad_info = []
                for i in range(max_gl):
                    gl_num = i + 1
                    nodo_propietario = None
                    direccion = None
                    for nodo in st.session_state.nodos:
                        if gl_num in nodo['grados_libertad_globales']:
                            nodo_propietario = nodo
                            direccion = 'X' if nodo['grados_libertad_globales'].index(gl_num) == 0 else 'Y'
                            break
                    es_fijo = nodo_propietario['tipo'] == 'fijo' if nodo_propietario else False
                    info_gl = {'numero': gl_num, 'nodo': nodo_propietario['id'] if nodo_propietario else None, 'direccion': direccion, 'desplazamiento_conocido': es_fijo, 'valor_desplazamiento': 0.0 if es_fijo else None, 'fuerza_conocida': not es_fijo, 'valor_fuerza': 0.0 if not es_fijo else None}
                    st.session_state.grados_libertad_info.append(info_gl)
                next_step()

elif st.session_state.step == 7:
    st.markdown("## Configuración de Incógnitas y Datos")
    st.markdown("Defina qué fuerzas y desplazamientos son conocidos (datos) o desconocidos (incógnitas)")
    st.info("💡 **Cálculo Automático:** Los resultados se calculan automáticamente cuando complete esta configuración.")
    st.markdown("### ⚙️ Configuración de Grados de Libertad")
    for i, info in enumerate(st.session_state.grados_libertad_info):
        with st.container():
            st.markdown(f"#### Grado de Libertad {info['numero']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Nodo:** {info['nodo']}")
                st.markdown(f"**Dirección:** {info['direccion']}")
                nombre_fuerza = st.text_input(f"Nombre de la fuerza:", value=st.session_state.nombres_fuerzas.get(info['numero'], f"F{info['numero']}"), key=f"nombre_fuerza_{info['numero']}")
                st.session_state.nombres_fuerzas[info['numero']] = nombre_fuerza
            with col2:
                st.markdown("**Desplazamiento**")
                if info['desplazamiento_conocido']:
                    st.markdown("🔒 **CONOCIDO** (Nodo fijo)")
                    valor_desplazamiento = st.number_input(f"Valor del desplazamiento [m]:", value=info['valor_desplazamiento'], format="%.6f", key=f"desp_{info['numero']}")
                    st.session_state.grados_libertad_info[i]['valor_desplazamiento'] = valor_desplazamiento
                    st.markdown(f"Valor: {formatear_unidades(valor_desplazamiento, 'desplazamiento')}")
                else:
                    st.markdown("❓ **INCÓGNITA** (Se calculará)")
                    st.markdown("Valor: *Se calculará automáticamente*")
            with col3:
                st.markdown("**Fuerza**")
                if info['fuerza_conocida']:
                    st.markdown("📝 **DATO** (Debe especificar)")
                    valor_fuerza = st.number_input(f"Valor de la fuerza [N]:", value=info['valor_fuerza'], format="%.3f", key=f"fuerza_{info['numero']}")
                    st.session_state.grados_libertad_info[i]['valor_fuerza'] = valor_fuerza
                    st.markdown(f"Valor: {formatear_unidades(valor_fuerza, 'fuerza')}")
                else:
                    st.markdown("❓ **INCÓGNITA** (Se calculará)")
                    st.markdown("Valor: *Se calculará automáticamente*")
            st.divider()
    
    if st.button("🧮 Calcular Sistema", type="primary", use_container_width=True):
        resultado = resolver_sistema()
        if resultado and resultado['exito']:
            st.session_state.resultados = resultado
            st.success("✅ Sistema resuelto exitosamente")
            next_step()
        else:
            st.error("❌ Error al resolver el sistema. Verifique los datos ingresados.")

# ----------------------------------------------------------------------
# ------------ BLOQUE DEL PASO 8 REORDENADO Y MEJORADO -----------------
# ----------------------------------------------------------------------
elif st.session_state.step == 8:
    st.markdown("## 🎉 Resultados del Análisis")
    st.markdown("El análisis estructural se ha completado exitosamente.")

    if st.session_state.resultados:
        resultado = st.session_state.resultados
        
        # --- 1. MÉTRICAS PRINCIPALES ---
        st.markdown("### 📈 Métricas Principales")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodos", len(st.session_state.nodos))
        with col2:
            st.metric("Elementos", len(st.session_state.elementos))
        with col3:
            st.metric("Grados de Libertad", len(st.session_state.grados_libertad_info))
        with col4:
            det_K = resultado.get('determinante', 0)
            st.metric("Det(K)", f"{det_K:.2e}")
        
        st.divider()

        # --- 2. TODAS LAS TABLAS DE DATOS ---
        st.markdown("### 📋 Tablas de Resultados y Análisis")

        # Tabla principal de resultados (Fuerzas y Desplazamientos)
        st.markdown("#### Resultados Globales")
        resultados_data = []
        for i, (info, fuerza, desplazamiento) in enumerate(zip(
            st.session_state.grados_libertad_info, 
            resultado['fuerzas'], 
            resultado['desplazamientos']
        )):
            nombre = st.session_state.nombres_fuerzas.get(i+1, f"F{i+1}")
            tipo_f = "Dato" if info['fuerza_conocida'] else "Calculado"
            tipo_u = "Dato" if info['desplazamiento_conocido'] else "Calculado"
            
            resultados_data.append({
                'GL': f"GL{i+1}", 'Nodo': f"N{info['nodo']}", 'Dirección': info['direccion'],
                'Nombre Fuerza': nombre, 'Fuerza': formatear_unidades(fuerza, "fuerza"), 'Tipo F': tipo_f,
                'Desplazamiento': formatear_unidades(desplazamiento, "desplazamiento"), 'Tipo U': tipo_u
            })
        
        df_resultados = pd.DataFrame(resultados_data)
        st.dataframe(df_resultados, use_container_width=True)

        # Tabla de Nodos
        st.markdown("#### 📍 Tabla de Nodos")
        df_nodos = crear_tabla_nodos()
        if not df_nodos.empty:
            st.dataframe(df_nodos, use_container_width=True)

        # Tabla de Conectividad
        st.markdown("#### 🔗 Tabla de Conectividad")
        df_conectividad = crear_tabla_conectividad()
        if not df_conectividad.empty:
            st.dataframe(df_conectividad, use_container_width=True)

        # Análisis de Tensiones
        calcular_y_mostrar_tensiones(resultado)

        # Matriz de Rigidez Global K
        st.markdown("#### 🔧 Matriz de Rigidez Global K")
        mostrar_matriz_formateada_moderna(resultado['K_global'], "Matriz de Rigidez Global K", es_simbolica=False)

        st.divider()

        # --- 3. GRÁFICOS DE LA ESTRUCTURA ---
        st.markdown("### 📊 Visualización de la Estructura")
        factor_escala_on_screen = st.slider("Factor de escala para visualización en pantalla:", 1, 100, 1, key="factor_escala_pantalla")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Estructura Deformada")
            fig_deformada = visualizar_estructura_moderna(mostrar_deformada=True, factor_escala=factor_escala_on_screen)
            if fig_deformada:
                st.pyplot(fig_deformada)
        with col2:
            st.markdown("#### Estructura Original")
            fig_original = visualizar_estructura_moderna(mostrar_deformada=False)
            if fig_original:
                st.pyplot(fig_original)

        st.divider()

        # --- 4. SECCIÓN DE EXPORTACIÓN ---
        st.markdown("### 📤 Exportación de Resultados")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Opciones para PDF")
            # NUEVO: Input para el factor de escala del PDF
            factor_escala_pdf = st.number_input("Factor de Escala para PDF:", min_value=1, max_value=5000, value=100, step=10)
            
            pdf_data = generar_pdf_completo(factor_escala_pdf)
            if pdf_data:
                st.download_button(
                    label="📄 Descargar Reporte PDF",
                    data=pdf_data,
                    file_name=f"analisis_estructural_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
        with col2:
            st.markdown("##### Opciones para Excel")
            st.write(" ") # Espaciador para alinear el botón
            excel_data = generar_excel_completo()
            if excel_data:
                st.download_button(
                    label="📊 Descargar Datos Excel",
                    data=excel_data,
                    file_name=f"datos_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
    
    # Footer
    mostrar_footer_encuesta()

else:
    st.error("Paso no reconocido o error en la carga de datos.")