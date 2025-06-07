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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64

# Configuración de la página
st.set_page_config(
    page_title="Análisis Estructural - Método de Matrices",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.session_state.step = 0  # Ahora empezamos en 0 para la selección de modo
if 'modo' not in st.session_state:
    st.session_state.modo = None  # 'manual' o 'interactivo'
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
    st.session_state.step = 1  # Avanzar al primer paso real
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
    else:
        return parametros.get("area", 0.01)  # Valor por defecto

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

def visualizar_estructura(mostrar_deformada=False, factor_escala=10):
    """Visualizar la estructura con matplotlib"""
    if not st.session_state.nodos:
        st.warning("No hay nodos para visualizar")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Obtener límites
    x_coords = [nodo['x'] for nodo in st.session_state.nodos]
    y_coords = [nodo['y'] for nodo in st.session_state.nodos]
    
    if len(set(x_coords)) == 1:
        x_min, x_max = (min(x_coords) - 1)*1.5, (max(x_coords) + 1)*1.5
    else:
        x_range = max(x_coords) - min(x_coords)
        x_min, x_max = (min(x_coords) - 0.1*x_range)*1.5, (max(x_coords) + 0.1*x_range)*1.5
    
    if len(set(y_coords)) == 1:
        y_min, y_max = (min(y_coords) - 1)*1.5, (max(y_coords) + 1)*1.5
    else:
        y_range = max(y_coords) - min(y_coords)
        y_min, y_max = (min(y_coords) - 0.1*y_range)*1.5, (max(y_coords) + 0.1*y_range)*1.5
    
    # Calcular posiciones deformadas si es necesario
    nodos_deformados = None
    if mostrar_deformada and st.session_state.resultados is not None:
        nodos_deformados = []
        for nodo in st.session_state.nodos:
            # Obtener índices de GL para este nodo
            gl_indices = [gl - 1 for gl in nodo['grados_libertad_globales']]
            
            # Obtener desplazamientos para este nodo
            dx = st.session_state.resultados['desplazamientos'][gl_indices[0]] * factor_escala
            dy = st.session_state.resultados['desplazamientos'][gl_indices[1]] * factor_escala
            
            # Crear nodo deformado
            nodo_deformado = nodo.copy()
            nodo_deformado['x'] += dx
            nodo_deformado['y'] += dy
            nodos_deformados.append(nodo_deformado)
    
    # Dibujar elementos
    for elemento in st.session_state.elementos:
        nodo_inicio = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_inicio']), None)
        nodo_fin = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_fin']), None)
        
        if nodo_inicio and nodo_fin:
            # Dibujar elemento original
            color = 'blue'  # Todas son barras ahora
            linewidth = 6
            
            ax.plot([nodo_inicio['x'], nodo_fin['x']], 
                   [nodo_inicio['y'], nodo_fin['y']], 
                   color=color, linewidth=linewidth, alpha=0.7)
            
            mid_x = (nodo_inicio['x'] + nodo_fin['x']) / 2
            mid_y = (nodo_inicio['y'] + nodo_fin['y']) / 2
            
            etiqueta = f'E{elemento["id"]}\n(Barra)'
            
            ax.text(mid_x, mid_y, etiqueta, 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Dibujar elemento deformado si es necesario
            if mostrar_deformada and nodos_deformados:
                nodo_inicio_def = next((n for n in nodos_deformados if n['id'] == elemento['nodo_inicio']), None)
                nodo_fin_def = next((n for n in nodos_deformados if n['id'] == elemento['nodo_fin']), None)
                
                if nodo_inicio_def and nodo_fin_def:
                    ax.plot([nodo_inicio_def['x'], nodo_fin_def['x']], 
                           [nodo_inicio_def['y'], nodo_fin_def['y']], 
                           color='red', linewidth=linewidth, alpha=0.7, linestyle='--')
    
    # Dibujar nodos con radios actualizados
    for i, nodo in enumerate(st.session_state.nodos):
        # Usar los nuevos valores de radio
        if nodo['tipo'] == 'fijo':
            radio_exterior = 0.25  # Actualizado
            radio_interior = 0.2   # Actualizado
            color_exterior = 'darkred'
            color_interior = 'red'
        else:
            radio_exterior = 0.25  # Actualizado
            radio_interior = 0.2   # Actualizado
            color_exterior = 'darkblue'
            color_interior = 'blue'
        
        circle_ext = plt.Circle((nodo['x'], nodo['y']), radio_exterior, 
                              color=color_exterior, alpha=0.8)
        ax.add_patch(circle_ext)
        
        circle_int = plt.Circle((nodo['x'], nodo['y']), radio_interior, 
                              color=color_interior, alpha=0.9)
        ax.add_patch(circle_int)
        
        ax.text(nodo['x'], nodo['y'], str(nodo['id']), 
               ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        ax.text(nodo['x'], nodo['y'] - 0.05, f'({nodo["x"]}, {nodo["y"]})', 
               ha='center', va='top', fontsize=7)
        
        gl_text = f'GL: {nodo["grados_libertad_globales"]}'
        ax.text(nodo['x'], nodo['y'] + 0.05, gl_text, 
               ha='center', va='bottom', fontsize=7, style='italic')
        
        # Dibujar nodos deformados si es necesario
        if mostrar_deformada and nodos_deformados:
            nodo_def = nodos_deformados[i]
            
            circle_ext_def = plt.Circle((nodo_def['x'], nodo_def['y']), radio_exterior * 0.8, 
                                      color='darkgreen', alpha=0.6)
            ax.add_patch(circle_ext_def)
            
            circle_int_def = plt.Circle((nodo_def['x'], nodo_def['y']), radio_interior * 0.8, 
                                      color='green', alpha=0.7)
            ax.add_patch(circle_int_def)
            
            # Dibujar línea que conecta nodo original con deformado
            ax.plot([nodo['x'], nodo_def['x']], [nodo['y'], nodo_def['y']], 
                   color='green', linestyle=':', linewidth=1)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    if mostrar_deformada:
        ax.set_title(f'Estructura Deformada (Factor: {factor_escala}x)', fontsize=14, fontweight='bold')
        # Leyenda para deformada
        legend_elements = [
            patches.Patch(facecolor='darkred', label='Nodos Fijos'),
            patches.Patch(facecolor='darkblue', label='Nodos Libres'),
            patches.Patch(facecolor='blue', label='Barras (4 GL)'),
            patches.Patch(facecolor='red', label='Barras Deformadas'),
            patches.Patch(facecolor='green', label='Nodos Deformados')
        ]
    else:
        ax.set_title('Visualización de la Estructura Original', fontsize=14, fontweight='bold')
        # Leyenda para estructura original
        legend_elements = [
            patches.Patch(facecolor='darkred', label='Nodos Fijos'),
            patches.Patch(facecolor='darkblue', label='Nodos Libres'),
            patches.Patch(facecolor='blue', label='Barras (4 GL)')
        ]
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig

def crear_grafico_interactivo():
    """Crear un gráfico interactivo con Plotly para añadir nodos y elementos"""
    # Crear figura base
    fig = go.Figure()
    
    # Configurar aspecto del gráfico
    fig.update_layout(
        title="Editor Interactivo de Estructura",
        xaxis=dict(
            title="X",
            showgrid=True,
            zeroline=True,
            range=[-10, 10],
            tickfont=dict(color='black'),
        ),
        yaxis=dict(
            title="Y",
            showgrid=True,
            zeroline=True,
            range=[-10, 10],
            scaleanchor="x",
            scaleratio=1,tickfont=dict(color='black'),
        ),
        showlegend=True,
        legend=dict(                    # <-- AGREGAR ESTA LÍNEA
        font=dict(color='black'),   # <-- Y ESTA LÍNEA
        bgcolor='white',            # <-- Y ESTA LÍNEA
        bordercolor='black',        # <-- Y ESTA LÍNEA
        borderwidth=1               # <-- Y ESTA LÍNEA
    ),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Añadir nodos existentes
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
            nodos_fijos_text.append(f"Nodo {nodo['id']}<br>({nodo['x']}, {nodo['y']})")
        else:
            nodos_libres_x.append(nodo['x'])
            nodos_libres_y.append(nodo['y'])
            nodos_libres_text.append(f"Nodo {nodo['id']}<br>({nodo['x']}, {nodo['y']})")
    
    # Añadir nodos fijos
    if nodos_fijos_x:
        fig.add_trace(go.Scatter(
            x=nodos_fijos_x,
            y=nodos_fijos_y,
            mode='markers+text',
            marker=dict(
                size=15,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text=[f"NODO {nodo['id']}" for nodo in st.session_state.nodos_interactivos if nodo['tipo'] == 'fijo'],
            textposition="top center",
            textfont=dict(size=12, color='darkred'),
            hoverinfo='text',
            hovertext=nodos_fijos_text,
            name='Nodos Fijos'
        ))
    
    # Añadir nodos libres
    if nodos_libres_x:
        fig.add_trace(go.Scatter(
            x=nodos_libres_x,
            y=nodos_libres_y,
            mode='markers+text',
            marker=dict(
                size=15,
                color='blue',
                line=dict(width=2, color='darkblue')
            ),
            text=[f"NODO {nodo['id']}" for nodo in st.session_state.nodos_interactivos if nodo['tipo'] == 'libre'],
            textposition="top center",
            textfont=dict(size=12, color='darkblue'),
            hoverinfo='text',
            hovertext=nodos_libres_text,
            name='Nodos Libres'
        ))
    
    # Añadir elementos (barras)
    for elemento in st.session_state.elementos_interactivos:
        nodo_inicio = next((n for n in st.session_state.nodos_interactivos if n['id'] == elemento['nodo_inicio']), None)
        nodo_fin = next((n for n in st.session_state.nodos_interactivos if n['id'] == elemento['nodo_fin']), None)
        
        if nodo_inicio and nodo_fin:
            # Calculate midpoint for text position
            mid_x = (nodo_inicio['x'] + nodo_fin['x']) / 2
            mid_y = (nodo_inicio['y'] + nodo_fin['y']) / 2
        
            # Añadir rectángulo blanco de fondo para el texto
            #fig.add_shape(
                #type="rect",
               # x0=0.1, y0=0.1,
                #x1=0.1, y1=0.1,
                #fillcolor="black",
                #line=dict(color="black", width=1),
           # )
        
            # Add text label for the element
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='text',
                text=[f"BARRA {elemento['id']}"],
                textposition="middle center",
                textfont=dict(size=10, color='red'),
                hoverinfo='skip',
                showlegend=False
            ))
        
            # Añadir barra completamente negra
            fig.add_trace(go.Scatter(
                x=[nodo_inicio['x'], nodo_fin['x']],
                y=[nodo_inicio['y'], nodo_fin['y']],
                mode='lines',
                line=dict(width=8, color='black'),
                name=f"Barra {elemento['id']}",
                text=f"Barra {elemento['id']}: Nodo {nodo_inicio['id']} → Nodo {nodo_fin['id']}<br>Longitud: {calcular_longitud_elemento(nodo_inicio, nodo_fin):.3f} m",
                hoverinfo='text',
                showlegend=True
            ))
    
    # Configurar interactividad
    fig.update_layout(
        dragmode='pan',
        clickmode='event+select',
        hovermode='closest'
    )
    
    return fig

def mostrar_matriz_formateada(matriz, titulo="Matriz", es_simbolica=True):
    """Mostrar matriz en formato tabla con unidades formateadas"""
    if matriz is None or len(matriz) == 0:
        st.warning("Matriz vacía")
        return

    st.subheader(titulo)
    
    if es_simbolica:
        df = pd.DataFrame(matriz)
        df.index = [f"Fila {i+1}" for i in range(len(matriz))]
        df.columns = [f"Col {i+1}" for i in range(len(matriz[0]))]
        st.dataframe(df, use_container_width=True)
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
        st.dataframe(df, use_container_width=True)
        
        # Botón para copiar tabla
        csv_data = df.to_csv(sep='\t')
        st.download_button(
            label="📋 Copiar Tabla",
            data=csv_data,
            file_name=f"{titulo.replace(' ', '_')}.csv",
            mime="text/csv"
        )

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

def generar_pdf_resultados(factor_escala_pdf=10):
    """Generar PDF con los resultados del análisis incluyendo gráficos"""
    if not st.session_state.resultados:
        st.error("No hay resultados para exportar")
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Centrado
        )
        story.append(Paragraph("Análisis Estructural - Resultados", title_style))
        story.append(Spacer(1, 20))
        
        # Información del proyecto
        info_data = [
            ['Usuario:', st.session_state.usuario_nombre],
            ['Fecha:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
            ['Nodos:', str(len(st.session_state.nodos))],
            ['Elementos:', str(len(st.session_state.elementos))],
            ['Grados de Libertad:', str(len(st.session_state.grados_libertad_info))],
            ['Factor de Escala Deformación:', f"{factor_escala_pdf}x"]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Generar y agregar gráficos
        try:
            # Gráfico estructura original
            fig_original = visualizar_estructura(mostrar_deformada=False)
            if fig_original:
                img_buffer_original = io.BytesIO()
                fig_original.savefig(img_buffer_original, format='png', dpi=150, bbox_inches='tight')
                img_buffer_original.seek(0)
                
                story.append(Paragraph("Estructura Original", styles['Heading2']))
                img_original = Image(img_buffer_original, width=6*inch, height=4*inch)
                story.append(img_original)
                story.append(Spacer(1, 20))
                plt.close(fig_original)
            
            # Gráfico estructura deformada
            fig_deformada = visualizar_estructura(mostrar_deformada=True, factor_escala=factor_escala_pdf)
            if fig_deformada:
                img_buffer_deformada = io.BytesIO()
                fig_deformada.savefig(img_buffer_deformada, format='png', dpi=150, bbox_inches='tight')
                img_buffer_deformada.seek(0)
                
                story.append(Paragraph(f"Estructura Deformada (Factor: {factor_escala_pdf}x)", styles['Heading2']))
                img_deformada = Image(img_buffer_deformada, width=6*inch, height=4*inch)
                story.append(img_deformada)
                story.append(Spacer(1, 20))
                plt.close(fig_deformada)
                
        except Exception as e:
            story.append(Paragraph("Error al generar gráficos", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Matriz K Global
        story.append(Paragraph("Matriz de Rigidez Global", styles['Heading2']))
        resultado = st.session_state.resultados
        K_data = [[''] + [f'GL{i+1}' for i in range(len(resultado['K_global']))]]
        
        for i, fila in enumerate(resultado['K_global']):
            fila_formateada = [f'GL{i+1}'] + [formatear_unidades(val, "presion") for val in fila]
            K_data.append(fila_formateada)
        
        K_table = Table(K_data)
        K_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(K_table)
        story.append(Spacer(1, 20))
        
        # Resultados de fuerzas y desplazamientos
        story.append(Paragraph("Resultados", styles['Heading2']))
        
        resultados_data = [['Grado de Libertad', 'Fuerza', 'Desplazamiento', 'Tipo']]
        
        for i, (info, fuerza, desplazamiento) in enumerate(zip(
            st.session_state.grados_libertad_info, 
            resultado['fuerzas'], 
            resultado['desplazamientos']
        )):
            nombre = st.session_state.nombres_fuerzas.get(i+1, f"F{i+1}")
            tipo_f = "Dato" if info['fuerza_conocida'] else "Calculado"
            tipo_u = "Dato" if info['desplazamiento_conocido'] else "Calculado"
            
            resultados_data.append([
                f"GL{i+1}",
                formatear_unidades(fuerza, "fuerza"),
                formatear_unidades(desplazamiento, "desplazamiento"),
                f"F:{tipo_f}, u:{tipo_u}"
            ])
        
        resultados_table = Table(resultados_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2*inch])
        resultados_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(resultados_table)
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error al generar PDF: {str(e)}")
        return None

def eliminar_nodo(nodo_id):
    """Eliminar un nodo y reorganizar"""
    st.session_state.nodos = [n for n in st.session_state.nodos if n['id'] != nodo_id]
    # Reorganizar IDs
    for i, nodo in enumerate(st.session_state.nodos):
        nodo['id'] = i + 1
        nodo['grados_libertad_globales'] = calcular_grados_libertad_globales(i + 1)
    
    # Limpiar elementos que usen este nodo
    st.session_state.elementos = [e for e in st.session_state.elementos 
                                 if e['nodo_inicio'] != nodo_id and e['nodo_fin'] != nodo_id]
    st.rerun()

def eliminar_elemento(elemento_id):
    """Eliminar un elemento"""
    st.session_state.elementos = [e for e in st.session_state.elementos if e['id'] != elemento_id]
    if elemento_id in st.session_state.matrices_elementos:
        del st.session_state.matrices_elementos[elemento_id]
    
    # Reorganizar IDs
    for i, elemento in enumerate(st.session_state.elementos):
        elemento['id'] = i + 1
    st.rerun()

def eliminar_nodo_interactivo(nodo_id):
    """Eliminar un nodo del modo interactivo"""
    st.session_state.nodos_interactivos = [n for n in st.session_state.nodos_interactivos if n['id'] != nodo_id]
    # Reorganizar IDs
    for i, nodo in enumerate(st.session_state.nodos_interactivos):
        nodo['id'] = i + 1
        nodo['grados_libertad_globales'] = calcular_grados_libertad_globales(i + 1)
    
    # Limpiar elementos que usen este nodo
    st.session_state.elementos_interactivos = [e for e in st.session_state.elementos_interactivos 
                                             if e['nodo_inicio'] != nodo_id and e['nodo_fin'] != nodo_id]
    st.rerun()

def eliminar_elemento_interactivo(elemento_id):
    """Eliminar un elemento del modo interactivo"""
    st.session_state.elementos_interactivos = [e for e in st.session_state.elementos_interactivos if e['id'] != elemento_id]
    
    # Reorganizar IDs
    for i, elemento in enumerate(st.session_state.elementos_interactivos):
        elemento['id'] = i + 1
    st.rerun()

def agregar_nodo_interactivo(x, y, tipo='libre'):
    """Agregar un nodo en el modo interactivo"""
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
    """Agregar un elemento en el modo interactivo"""
    if nodo_inicio_id == nodo_fin_id:
        return None
    
    # Verificar si ya existe un elemento con estos nodos
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
        'material': None,  # Se definirá después
        'tipo_seccion': None,  # Se definirá después
        'parametros_seccion': {}  # Se definirá después
    }
    
    st.session_state.elementos_interactivos.append(nuevo_elemento)
    return elemento_id

def transferir_datos_interactivos():
    """Transferir datos del modo interactivo al modo normal"""
    st.session_state.nodos = st.session_state.nodos_interactivos.copy()
    st.session_state.num_nodos = len(st.session_state.nodos)
    st.session_state.num_fijos = sum(1 for n in st.session_state.nodos if n['tipo'] == 'fijo')
    st.session_state.num_libres = st.session_state.num_nodos - st.session_state.num_fijos
    
    # Los elementos se transferirán después de configurar materiales y secciones
    st.session_state.elementos = []
    st.session_state.matrices_elementos = {}
    st.session_state.num_elementos = len(st.session_state.elementos_interactivos)
    
    # Avanzar al paso de configuración de elementos (paso 6)
    st.session_state.step = 6
    st.rerun()

# Auto-calcular cuando hay cambios
def auto_calcular():
    """Calcular automáticamente cuando hay datos suficientes"""
    if (st.session_state.elementos and 
        st.session_state.grados_libertad_info and 
        st.session_state.auto_calcular):
        resultado = resolver_sistema()
        if resultado and resultado['exito']:
            st.session_state.resultados = resultado

# Título principal
st.title("🏗️ Análisis Estructural Avanzado - Método de Matrices")

# Barra lateral con progreso
with st.sidebar:
    st.header("Progreso del Análisis")
    
    if st.session_state.step == 0:
        st.markdown("**➡️ Selección de Modo**")
    else:
        st.markdown("✅ Selección de Modo")
    
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
        if st.session_state.step == 0:
            st.markdown(f"⏳ {paso}")
        elif i == st.session_state.step:
            st.markdown(f"**➡️ {paso}**")
        elif i < st.session_state.step:
            st.markdown(f"✅ {paso}")
        else:
            st.markdown(f"⏳ {paso}")
    
    st.divider()
    
    if st.button("🔄 Reiniciar", type="secondary"):
        reset_app()
    
    if st.session_state.step > 1:
        if st.button("⬅️ Paso Anterior"):
            prev_step()
    
    # Información adicional
    st.divider()
    st.subheader("ℹ️ Información")
    
    if st.session_state.usuario_nombre:
        st.write(f"**Usuario:** {st.session_state.usuario_nombre}")
    
    if st.session_state.modo:
        st.write(f"**Modo:** {st.session_state.modo.capitalize()}")
    
    if st.session_state.step >= 4 and st.session_state.nodos:
        st.write(f"**Nodos:** {len(st.session_state.nodos)}")
    
    if st.session_state.step >= 6 and st.session_state.elementos:
        st.write(f"**Elementos:** {len(st.session_state.elementos)}")
    
    if st.session_state.step >= 7 and st.session_state.grados_libertad_info:
        st.write(f"**Grados de Libertad:** {len(st.session_state.grados_libertad_info)}")
    
    st.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
    st.write(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")

# Contenido principal según el paso
if st.session_state.step == 0:
    st.header("Selección de Modo de Análisis")
    st.write("Elija cómo desea definir su estructura para el análisis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modo Manual")
        st.info("""
        En el **modo manual**, usted ingresará:
        - Coordenadas exactas de cada nodo
        - Conexiones entre nodos para formar barras
        - Propiedades de materiales y secciones
        
        **Ideal para:**
        - Estructuras con geometría precisa
        - Cuando conoce las coordenadas exactas
        - Análisis detallado con valores específicos
        """)
        if st.button("MANUAL", type="primary", use_container_width=True):
            set_modo("manual")
    
    with col2:
        st.subheader("Modo Interactivo")
        st.info("""
        En el **modo interactivo**, usted podrá:
        - Colocar nodos haciendo clic en un gráfico
        - Conectar nodos visualmente para crear barras
        - Definir propiedades mediante formularios simples
        
        **Ideal para:**
        - Diseño rápido de estructuras
        - Exploración de diferentes configuraciones
        - Usuarios que prefieren interacción visual
        """)
        if st.button("INTERACTIVO", type="primary", use_container_width=True):
            set_modo("interactivo")

elif st.session_state.step == 1:
    st.header("Paso 1: Información del Usuario")
    st.write("Bienvenido al sistema de análisis estructural avanzado. Por favor, ingrese su información.")
    
    usuario_nombre = st.text_input("👤 Nombre completo:", 
                                  value=st.session_state.usuario_nombre,
                                  placeholder="Ej: Juan Pérez")
    
    if usuario_nombre:
        st.session_state.usuario_nombre = usuario_nombre
        
        st.markdown("""
        ## 🎯 Características del Sistema Avanzado:
        - ✅ Análisis de **barras** con matrices 4x4
        - ✅ **Base de datos de materiales** (Aluminio, Titanio, Fibra de Carbono, etc.)
        - ✅ **Formateo inteligente de unidades** (kPa, MPa, GPa)
        - ✅ Configuración automática de grados de libertad
        - ✅ **Exportación PDF** de resultados con gráficos
        - ✅ **Tablas copiables** para Word/Excel
        - ✅ Visualización gráfica optimizada con estructura deformada
        - ✅ **Cálculo automático** sin necesidad de botones
        - ✅ **Edición y eliminación** de nodos y elementos
        - ✅ **Secciones personalizables** (circular, rectangular)
        
        ### 🧪 Materiales Aeroespaciales Incluidos:
        - Aluminio 6061-T6, 7075-T6, 2024-T3
        - Titanio Ti-6Al-4V
        - Acero 4130
        - Fibra de Carbono T300
        - Magnesio AZ31B
        - + Opción de agregar materiales personalizados
        """)
        
        if st.button("Continuar", type="primary"):
            if st.session_state.modo == "manual":
                next_step()
            else:
                # Para modo interactivo, saltar al editor interactivo
                st.session_state.step = 2
                st.rerun()

elif st.session_state.modo == "manual" and st.session_state.step == 2:
    st.header("Paso 2: Número de Nodos")
    st.write("Ingrese el número total de nodos en el sistema")
    
    num_nodos = st.number_input("Número de Nodos", min_value=1, max_value=20, 
                               value=st.session_state.num_nodos)
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_nodos = num_nodos
        next_step()

elif st.session_state.modo == "manual" and st.session_state.step == 3:
    st.header("Paso 3: Clasificación de Nodos")
    st.write("Defina cuántos nodos son fijos y cuántos son libres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_fijos = st.number_input("Nodos Fijos", min_value=0, max_value=st.session_state.num_nodos, 
                                   value=st.session_state.num_fijos)
    
    with col2:
        num_libres = st.session_state.num_nodos - num_fijos
        st.metric("Nodos Libres", num_libres)
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_fijos = num_fijos
        st.session_state.num_libres = num_libres
        next_step()

elif st.session_state.modo == "manual" and st.session_state.step == 4:
    st.header("Paso 4: Coordenadas de Nodos")
    st.write("Ingrese las coordenadas para cada nodo")
    
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
        st.subheader("Coordenadas de Todos los Nodos")
        
        nodos_temp = []
        
        for i in range(st.session_state.num_nodos):
            nodo_id = i + 1
            tipo_actual = tipos_nodos[i]
            
            # Buscar nodo existente o usar valores por defecto
            nodo_existente = next((n for n in st.session_state.nodos if n['id'] == nodo_id), None)
            x_default = nodo_existente['x'] if nodo_existente else 0.0
            y_default = nodo_existente['y'] if nodo_existente else 0.0
            
            st.write(f"**Nodo {nodo_id} ({tipo_actual.title()})**")
            
            col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
            
            with col1:
                x = st.number_input(f"X{nodo_id}", value=x_default, format="%.2f", key=f"x_{nodo_id}")
            
            with col2:
                y = st.number_input(f"Y{nodo_id}", value=y_default, format="%.2f", key=f"y_{nodo_id}")
            
            with col3:
                gl_globales = calcular_grados_libertad_globales(nodo_id)
                st.write(f"GL Globales: {gl_globales}")
                st.write(f"GL{gl_globales[0]} → X, GL{gl_globales[1]} → Y")
            
            with col4:
                if len(st.session_state.nodos) >= nodo_id:
                    if st.form_submit_button(f"🗑️ Eliminar", key=f"eliminar_nodo_{nodo_id}"):
                        eliminar_nodo(nodo_id)
            
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
        st.subheader("📋 Nodos Configurados")
        df_nodos = pd.DataFrame(st.session_state.nodos)
        st.dataframe(df_nodos, use_container_width=True)
        
        if len(st.session_state.nodos) == st.session_state.num_nodos:
            if st.button("Continuar", type="primary"):
                next_step()

elif st.session_state.modo == "interactivo" and st.session_state.step == 2:
    st.header("Editor Interactivo de Estructura")
    st.write("Utilice el gráfico para crear su estructura. Haga clic para añadir nodos y conectarlos para formar barras.")
    
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
    
    # Crear gráfico interactivo
    fig = crear_grafico_interactivo()
    
    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)

    # Controles para añadir nodos manualmente
    st.subheader("Añadir Nodos Manualmente")
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
        st.subheader("Crear Barras")
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
        st.subheader("Nodos")
        if st.session_state.nodos_interactivos:
            for nodo in st.session_state.nodos_interactivos:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"Nodo {nodo['id']} ({nodo['tipo']}): ({nodo['x']:.2f}, {nodo['y']:.2f})")
                with col_b:
                    if st.button(f"🗑️", key=f"del_nodo_{nodo['id']}"):
                        eliminar_nodo_interactivo(nodo['id'])
    
    with col2:
        st.subheader("Barras")
        if st.session_state.elementos_interactivos:
            for elem in st.session_state.elementos_interactivos:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"Barra {elem['id']}: Nodo {elem['nodo_inicio']} → Nodo {elem['nodo_fin']}")
                with col_b:
                    if st.button(f"🗑️", key=f"del_elem_{elem['id']}"):
                        eliminar_elemento_interactivo(elem['id'])
    
    # Mostrar estado de selección
    if st.session_state.nodo_seleccionado_interactivo:
        st.info(f"🎯 Nodo {st.session_state.nodo_seleccionado_interactivo} seleccionado. Haga clic en otro nodo para crear una barra.")

elif st.session_state.step == 5:
    st.header("Paso 5: Número de Elementos")
    st.write("Ingrese el número total de elementos (barras) en el sistema")
    
    num_elementos = st.number_input("Número de Elementos", min_value=1, max_value=50, 
                                   value=st.session_state.num_elementos)
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_elementos = num_elementos
        next_step()

elif st.session_state.step == 6:
    st.header("Paso 6: Definición de Elementos")
    st.write("Configure cada elemento (barra) del sistema")
    
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
    
    # Mostrar materiales disponibles
    with st.expander("📚 Base de Datos de Materiales"):
        for nombre, props in MATERIALES_AEROESPACIALES.items():
            st.write(f"**{nombre}**: E = {formatear_unidades(props['modulo_young'], 'presion')}, "
                    f"ρ = {props['densidad']} kg/m³")
            st.caption(props['descripcion'])
    
    # Agregar material personalizado
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
                    st.session_state.materiales_personalizados[nombre_material] = {
                        'modulo_young': modulo_young,
                        'densidad': densidad,
                        'descripcion': descripcion
                    }
                    st.success(f"Material '{nombre_material}' agregado exitosamente")
    
    # Configurar elementos
    if st.session_state.modo == "interactivo":
        # Para modo interactivo, use elementos_interactivos como base
        elementos_base = st.session_state.elementos_interactivos
    else:
        # Para modo manual, use el número especificado
        elementos_base = [{'id': i+1} for i in range(st.session_state.num_elementos)]
    
    for i, elem_base in enumerate(elementos_base):
        elemento_id = elem_base['id']
        
        with st.expander(f"🔧 Elemento {elemento_id} (Barra)", expanded=True):
            # Buscar elemento existente
            elemento_existente = next((e for e in st.session_state.elementos if e['id'] == elemento_id), None)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.modo == "interactivo":
                    # En modo interactivo, los nodos ya están definidos
                    nodo_inicio = elem_base['nodo_inicio']
                    nodo_fin = elem_base['nodo_fin']
                    st.write(f"**Nodo Inicio:** {nodo_inicio}")
                    st.write(f"**Nodo Fin:** {nodo_fin}")
                else:
                    # En modo manual, permitir selección
                    nodos_disponibles = [n['id'] for n in st.session_state.nodos]
                    
                    nodo_inicio_default = elemento_existente['nodo_inicio'] if elemento_existente else nodos_disponibles[0]
                    nodo_fin_default = elemento_existente['nodo_fin'] if elemento_existente else nodos_disponibles[-1]
                    
                    nodo_inicio = st.selectbox(f"Nodo Inicio", nodos_disponibles, 
                                             index=nodos_disponibles.index(nodo_inicio_default) if nodo_inicio_default in nodos_disponibles else 0,
                                             key=f"inicio_{elemento_id}")
                    
                    nodo_fin = st.selectbox(f"Nodo Fin", nodos_disponibles,
                                          index=nodos_disponibles.index(nodo_fin_default) if nodo_fin_default in nodos_disponibles else -1,
                                          key=f"fin_{elemento_id}")
            
            with col2:
                # Selección de material
                todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
                nombres_materiales = list(todos_materiales.keys())
                
                material_default = elemento_existente['material'] if elemento_existente else nombres_materiales[0]
                material_idx = nombres_materiales.index(material_default) if material_default in nombres_materiales else 0
                
                material_seleccionado = st.selectbox(f"Material", nombres_materiales, 
                                                   index=material_idx, key=f"material_{elemento_id}")
                
                # Mostrar propiedades del material
                props_material = todos_materiales[material_seleccionado]
                st.write(f"E = {formatear_unidades(props_material['modulo_young'], 'presion')}")
                st.write(f"ρ = {props_material['densidad']} kg/m³")
            
            # Configuración de sección
            st.subheader("Tipo de Sección")
            
            tipo_seccion_default = elemento_existente.get('tipo_seccion', 'circular_solida') if elemento_existente else 'circular_solida'
            tipo_seccion = st.radio(
                "Seleccione el tipo de sección:",
                ["circular_solida", "circular_hueca", "rectangular"],
                format_func=lambda x: {
                    "circular_solida": "Circular Sólida",
                    "circular_hueca": "Circular Hueca (Cilindro)",
                    "rectangular": "Rectangular"
                }[x],
                index=["circular_solida", "circular_hueca", "rectangular"].index(tipo_seccion_default),
                key=f"tipo_seccion_{elemento_id}"
            )
            
            # Parámetros de sección según el tipo
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
            
            # Calcular área final
            area_final = calcular_area_seccion(tipo_seccion, parametros_seccion)
            
            # Botón para guardar elemento
            if st.button(f"💾 Guardar Elemento {elemento_id}", key=f"guardar_{elemento_id}"):
                # Buscar nodos
                nodo_inicio_obj = next((n for n in st.session_state.nodos if n['id'] == nodo_inicio), None)
                nodo_fin_obj = next((n for n in st.session_state.nodos if n['id'] == nodo_fin), None)
                
                if nodo_inicio_obj and nodo_fin_obj:
                    # Calcular propiedades geométricas
                    longitud = calcular_longitud_elemento(nodo_inicio_obj, nodo_fin_obj)
                    beta = calcular_angulo_beta(nodo_inicio_obj, nodo_fin_obj)
                    
                    # Grados de libertad globales
                    gl_globales = nodo_inicio_obj['grados_libertad_globales'] + nodo_fin_obj['grados_libertad_globales']
                    
                    # Crear elemento
                    nuevo_elemento = {
                        'id': elemento_id,
                        'nodo_inicio': nodo_inicio,
                        'nodo_fin': nodo_fin,
                        'tipo': 'Barra',
                        'material': material_seleccionado,
                        'tipo_seccion': tipo_seccion,
                        'parametros_seccion': parametros_seccion,
                        'area': area_final,
                        'longitud': longitud,
                        'beta': beta,
                        'grados_libertad_global': gl_globales
                    }
                    
                    # Actualizar o agregar elemento
                    elemento_idx = next((i for i, e in enumerate(st.session_state.elementos) if e['id'] == elemento_id), None)
                    if elemento_idx is not None:
                        st.session_state.elementos[elemento_idx] = nuevo_elemento
                    else:
                        st.session_state.elementos.append(nuevo_elemento)
                    
                    # Generar matrices
                    E = props_material['modulo_young']
                    A = area_final
                    L = longitud
                    
                    matriz_simbolica = [
                        [f"({E:.2e}*{A:.6f}/{L:.3f})*cos²β", f"({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"-({E:.2e}*{A:.6f}/{L:.3f})*cos²β", f"-({E:.2e}*{A:.6f}/{L:.3f})*cos*sin"],
                        [f"({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"({E:.2e}*{A:.6f}/{L:.3f})*sin²β", f"-({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"-({E:.2e}*{A:.6f}/{L:.3f})*sin²β"],
                        [f"-({E:.2e}*{A:.6f}/{L:.3f})*cos²β", f"-({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"({E:.2e}*{A:.6f}/{L:.3f})*cos²β", f"({E:.2e}*{A:.6f}/{L:.3f})*cos*sin"],
                        [f"-({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"-({E:.2e}*{A:.6f}/{L:.3f})*sin²β", f"({E:.2e}*{A:.6f}/{L:.3f})*cos*sin", f"({E:.2e}*{A:.6f}/{L:.3f})*sin²β"]
                    ]
                    
                    matriz_numerica = generar_matriz_rigidez_barra(E, A, L, beta)
                    
                    st.session_state.matrices_elementos[elemento_id] = {
                        'simbolica': matriz_simbolica,
                        'numerica': matriz_numerica.tolist()
                    }
                    
                    st.success(f"✅ Elemento {elemento_id} guardado correctamente")
                    st.rerun()
    
    # Mostrar elementos guardados
    if st.session_state.elementos:
        st.subheader("📋 Elementos Configurados")
        
        elementos_df = []
        for elem in st.session_state.elementos:
            elementos_df.append({
                'ID': elem['id'],
                'Tipo': elem['tipo'],
                'Nodo Inicio': elem['nodo_inicio'],
                'Nodo Fin': elem['nodo_fin'],
                'Material': elem['material'],
                'Sección': elem['tipo_seccion'],
                'Área (m²)': f"{elem['area']:.6f}",
                'Longitud (m)': f"{elem['longitud']:.3f}"
            })
        
        df = pd.DataFrame(elementos_df)
        st.dataframe(df, use_container_width=True)
        
        # Botón para continuar
        if len(st.session_state.elementos) >= st.session_state.num_elementos:
            if st.button("Continuar", type="primary"):
                # Generar información de grados de libertad
                max_gl = max([max(nodo['grados_libertad_globales']) for nodo in st.session_state.nodos])
                
                st.session_state.grados_libertad_info = []
                for i in range(max_gl):
                    gl_num = i + 1
                    
                    # Encontrar el nodo al que pertenece este GL
                    nodo_propietario = None
                    direccion = None
                    for nodo in st.session_state.nodos:
                        if gl_num in nodo['grados_libertad_globales']:
                            nodo_propietario = nodo
                            direccion = 'X' if nodo['grados_libertad_globales'].index(gl_num) == 0 else 'Y'
                            break
                    
                    # Determinar si es conocido (nodo fijo)
                    es_fijo = nodo_propietario['tipo'] == 'fijo' if nodo_propietario else False
                    
                    info_gl = {
                        'numero': gl_num,
                        'nodo': nodo_propietario['id'] if nodo_propietario else None,
                        'direccion': direccion,
                        'desplazamiento_conocido': es_fijo,
                        'valor_desplazamiento': 0.0 if es_fijo else None,
                        'fuerza_conocida': not es_fijo,
                        'valor_fuerza': 0.0 if not es_fijo else None
                    }
                    
                    st.session_state.grados_libertad_info.append(info_gl)
                
                next_step()

elif st.session_state.step == 7:
    st.header("Paso 7: Configuración de Incógnitas y Datos")
    st.write("Defina qué fuerzas y desplazamientos son conocidos (datos) o desconocidos (incógnitas)")

    st.info("💡 **Cálculo Automático:** Los resultados se calculan automáticamente cuando complete esta configuración.")

    st.subheader("⚙️ Configuración de Grados de Libertad")

    # Usar columnas para organizar mejor la información
    for i, info in enumerate(st.session_state.grados_libertad_info):
        with st.container():
            st.write(f"### Grado de Libertad {info['numero']}")
            
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
            
            with col1:
                # Nombre de la fuerza
                nombre_fuerza = st.text_input(
                    "Nombre de la Fuerza:",
                    value=st.session_state.nombres_fuerzas.get(info['numero'], f"F{info['numero']}"),
                    key=f"nombre_{i}"
                )
                # Actualizar inmediatamente
                st.session_state.nombres_fuerzas[info['numero']] = nombre_fuerza
            
            with col2:
                # Checkbox para fuerza conocida
                fuerza_conocida = st.checkbox(
                    "Fuerza Conocida", 
                    value=info['fuerza_conocida'],
                    key=f"fuerza_conocida_{i}"
                )
                # Actualizar inmediatamente
                st.session_state.grados_libertad_info[i]['fuerza_conocida'] = fuerza_conocida
            
            with col3:
                # Campo de valor de fuerza - HABILITADO cuando se marca el checkbox
                if fuerza_conocida:
                    valor_fuerza = st.number_input(
                        "Valor de la Fuerza (N):",
                        value=info['valor_fuerza'] or 0.0,
                        key=f"valor_fuerza_{i}",
                        format="%.6f"
                    )
                    # Actualizar inmediatamente
                    st.session_state.grados_libertad_info[i]['valor_fuerza'] = valor_fuerza
                else:
                    st.text_input(
                        "Valor de la Fuerza:",
                        value="Incógnita",
                        disabled=True,
                        key=f"valor_fuerza_disabled_{i}"
                    )
            
            with col4:
                # Checkbox para desplazamiento conocido
                desplazamiento_conocido = st.checkbox(
                    "Desplazamiento Conocido", 
                    value=info['desplazamiento_conocido'],
                    key=f"desplazamiento_conocido_{i}"
                )
                # Actualizar inmediatamente
                st.session_state.grados_libertad_info[i]['desplazamiento_conocido'] = desplazamiento_conocido
            
            with col5:
                # Campo de valor de desplazamiento - HABILITADO cuando se marca el checkbox
                if desplazamiento_conocido:
                    valor_desplazamiento = st.number_input(
                        "Valor del Desplazamiento (m):",
                        value=info['valor_desplazamiento'] or 0.0,
                        key=f"valor_desplazamiento_{i}",
                        format="%.6f"
                    )
                    # Actualizar inmediatamente
                    st.session_state.grados_libertad_info[i]['valor_desplazamiento'] = valor_desplazamiento
                else:
                    st.text_input(
                        "Valor del Desplazamiento:",
                        value="Incógnita",
                        disabled=True,
                        key=f"valor_desplazamiento_disabled_{i}"
                    )
            
            st.divider()

    # Auto-calcular cuando hay cambios
    auto_calcular()

    # Mostrar configuración actual
    st.subheader("📋 Resumen de Configuración")
    config_data = []
    for info in st.session_state.grados_libertad_info:
        fuerza_str = formatear_unidades(info['valor_fuerza'], "fuerza") if info['fuerza_conocida'] else "Incógnita"
        desplazamiento_str = formatear_unidades(info['valor_desplazamiento'], "desplazamiento") if info['desplazamiento_conocido'] else "Incógnita"
        
        config_data.append({
            'GL': info['numero'],
            'Nombre Fuerza': st.session_state.nombres_fuerzas[info['numero']],
            'Fuerza': fuerza_str,
            'Desplazamiento': desplazamiento_str
        })

    df_config = pd.DataFrame(config_data)
    st.dataframe(df_config, use_container_width=True)

    if st.button("Continuar", type="primary"):
        next_step()

elif st.session_state.step == 8:
    st.header("Paso 8: Resultados del Análisis")
    
    # Ejecutar cálculo automático si está habilitado
    auto_calcular()
    
    if st.session_state.resultados and st.session_state.resultados['exito']:
        resultado = st.session_state.resultados
        
        # Información general
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Determinante de K", f"{resultado['determinante']:.2e}")
        
        with col2:
            st.metric("Grados de Libertad", len(st.session_state.grados_libertad_info))
        
        with col3:
            st.metric("Elementos", len(st.session_state.elementos))
        
        # Matriz K Global
        st.subheader("🔧 Matriz de Rigidez Global")
        mostrar_matriz_formateada(resultado['K_global'], "Matriz K Global", es_simbolica=False)
        
        # Resultados de fuerzas y desplazamientos
        st.subheader("📊 Resultados de Fuerzas y Desplazamientos")
        
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
                'Dir': info['direccion'],
                'Fuerza': formatear_unidades(fuerza, "fuerza"),
                'Tipo F': tipo_f,
                'Desplazamiento': formatear_unidades(desplazamiento, "desplazamiento"),
                'Tipo u': tipo_u,
                'Nombre': nombre
            })
        
        df_resultados = pd.DataFrame(resultados_data)
        st.dataframe(df_resultados, use_container_width=True)
        
        # Visualización gráfica
        st.subheader("📈 Visualización de la Estructura")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            mostrar_deformada = st.checkbox("Mostrar Estructura Deformada", value=True)
            
            if mostrar_deformada:
                factor_escala = st.slider("Factor de Escala Deformación", 
                        min_value=1, max_value=100, value=10, step=1)
            else:
                factor_escala = 1
        
        with col2:
            fig = visualizar_estructura(mostrar_deformada=mostrar_deformada, factor_escala=factor_escala)
            if fig:
                st.pyplot(fig)
        
        # Exportación
        st.subheader("📄 Exportación de Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            factor_escala_pdf = st.number_input("Factor Escala PDF", min_value=1, max_value=1000, value=10)
        
        with col2:
            if st.button("📄 Generar PDF", type="primary"):
                pdf_data = generar_pdf_resultados(factor_escala_pdf)
                if pdf_data:
                    st.download_button(
                        label="⬇️ Descargar PDF",
                        data=pdf_data,
                        file_name=f"analisis_estructural_{st.session_state.usuario_nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
        
        with col3:
            # Exportar datos como CSV
            csv_data = df_resultados.to_csv(index=False)
            st.download_button(
                label="📊 Exportar CSV",
                data=csv_data,
                file_name=f"resultados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Análisis adicional
        st.subheader("🔍 Análisis Adicional")
        
        with st.expander("Análisis de Tensiones"):
            st.write("**Tensiones en cada elemento:**")
            
            for elemento in st.session_state.elementos:
                nodo_inicio = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_inicio']), None)
                nodo_fin = next((n for n in st.session_state.nodos if n['id'] == elemento['nodo_fin']), None)
                
                if nodo_inicio and nodo_fin:
                    # Obtener desplazamientos de los nodos
                    gl_inicio = nodo_inicio['grados_libertad_globales']
                    gl_fin = nodo_fin['grados_libertad_globales']
                    
                    u_inicio = [resultado['desplazamientos'][gl-1] for gl in gl_inicio]
                    u_fin = [resultado['desplazamientos'][gl-1] for gl in gl_fin]
                    
                    # Calcular deformación axial
                    dx = nodo_fin['x'] - nodo_inicio['x']
                    dy = nodo_fin['y'] - nodo_inicio['y']
                    L = elemento['longitud']
                    
                    # Deformación en dirección del elemento
                    deformacion_axial = ((u_fin[0] - u_inicio[0]) * dx + (u_fin[1] - u_inicio[1]) * dy) / L**2
                    
                    # Tensión
                    material = elemento['material']
                    todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
                    E = todos_materiales[material]['modulo_young']
                    
                    tension = E * deformacion_axial
                    
                    st.write(f"**Elemento {elemento['id']}:** {formatear_unidades(tension, 'presion')}")
        
        # Enlace de evaluación
        st.subheader("📝 Evaluación del Sistema")
        st.info("""
        **¡Su opinión es importante!** 
        
        Ayúdenos a mejorar este sistema de análisis estructural completando nuestra breve encuesta:
        
        [**📋 Evaluar Sistema de Análisis Estructural**](https://forms.gle/31KgSu263hf8dH5UA)
        
        Su retroalimentación nos ayuda a desarrollar mejores herramientas para la comunidad de ingeniería.
        """)
        
    else:
        st.error("❌ No se pudo resolver el sistema. Verifique la configuración.")
        
        if st.button("🔄 Intentar Resolver", type="primary"):
            resultado = resolver_sistema()
            if resultado and resultado['exito']:
                st.session_state.resultados = resultado
                st.rerun()
            else:
                st.error("El sistema no tiene solución única. Verifique las condiciones de frontera.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    🏗️ Sistema de Análisis Estructural Avanzado | Método de Matrices<br>
    Desarrollado para análisis de estructuras aeroespaciales y civiles<br>
    © 2024 - Versión 3.0 con Modo Interactivo y Secciones Personalizables
</div>
""", unsafe_allow_html=True)
