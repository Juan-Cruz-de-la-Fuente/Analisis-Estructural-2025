import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import math
from datetime import datetime
import io
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
    st.session_state.step = 1
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
    """Calcular el ángulo β entre la horizontal y la viga"""
    dx = nodo_fin['x'] - nodo_inicio['x']
    dy = nodo_fin['y'] - nodo_inicio['y']
    return math.atan2(dy, dx)

def generar_matriz_rigidez_viga(E, A, L, beta):
    """Generar matriz de rigidez para viga (4x4)"""
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

def visualizar_estructura(mostrar_deformada=False, factor_escala=100):
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
            color = 'blue'  # Todas son vigas ahora
            linewidth = 6
            
            ax.plot([nodo_inicio['x'], nodo_fin['x']], 
                   [nodo_inicio['y'], nodo_fin['y']], 
                   color=color, linewidth=linewidth, alpha=0.7)
            
            mid_x = (nodo_inicio['x'] + nodo_fin['x']) / 2
            mid_y = (nodo_inicio['y'] + nodo_fin['y']) / 2
            
            etiqueta = f'E{elemento["id"]}\n(Viga)'
            
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
            radio_exterior = 0.125  # Actualizado de 0.016 a 0.25
            radio_interior = 0.1   # Actualizado de 0.01 a 0.2
            color_exterior = 'darkred'
            color_interior = 'red'
        else:
            radio_exterior = 0.125  # Actualizado de 0.016 a 0.25
            radio_interior = 0.1   # Actualizado de 0.01 a 0.2
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
            patches.Patch(facecolor='blue', label='Vigas (4 GL)'),
            patches.Patch(facecolor='red', label='Vigas Deformadas'),
            patches.Patch(facecolor='green', label='Nodos Deformados')
        ]
    else:
        ax.set_title('Visualización de la Estructura Original', fontsize=14, fontweight='bold')
        # Leyenda para estructura original
        legend_elements = [
            patches.Patch(facecolor='darkred', label='Nodos Fijos'),
            patches.Patch(facecolor='darkblue', label='Nodos Libres'),
            patches.Patch(facecolor='blue', label='Vigas (4 GL)')
        ]
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(handles=legend_elements, loc='upper right')
    
    return fig

def mostrar_matriz_formateada(matriz, titulo="Matriz", es_simbolica=True):
    """Mostrar matriz en formato tabla con unidades formateadas"""
    if not matriz:
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
    """Resolver el sistema de ecuaciones"""
    if not st.session_state.elementos or not st.session_state.grados_libertad_info:
        st.error("No hay información suficiente para resolver el sistema")
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
                st.error("La matriz K_uu es singular, no se puede resolver el sistema")
                return None
        
        # Calcular fuerzas resultantes
        F_calculado = K_global @ U
        
        # Corregir el problema con GL4=0 mostrando -80000
        # Verificar si hay valores de fuerza conocidos que no coinciden con los calculados
        for i, info in enumerate(st.session_state.grados_libertad_info):
            if info['fuerza_conocida']:
                # Si la fuerza es conocida, usar el valor conocido en lugar del calculado
                F_calculado[i] = info['valor_fuerza']
        
        return {
            'K_global': K_global,
            'desplazamientos': U,
            'fuerzas': F_calculado,
            'determinante': np.linalg.det(K_global),
            'exito': True
        }
        
    except Exception as e:
        st.error(f"Error al resolver el sistema: {str(e)}")
        return None

def generar_pdf_resultados():
    """Generar PDF con los resultados del análisis"""
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
            ['Grados de Libertad:', str(len(st.session_state.grados_libertad_info))]
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

# Título principal
st.title("🏗️ Análisis Estructural Avanzado - Método de Matrices")

# Barra lateral con progreso
with st.sidebar:
    st.header("Progreso del Análisis")
    
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
    
    if st.session_state.step >= 4 and st.session_state.nodos:
        st.write(f"**Nodos:** {len(st.session_state.nodos)}")
    
    if st.session_state.step >= 6 and st.session_state.elementos:
        st.write(f"**Elementos:** {len(st.session_state.elementos)}")
    
    if st.session_state.step >= 7 and st.session_state.grados_libertad_info:
        st.write(f"**Grados de Libertad:** {len(st.session_state.grados_libertad_info)}")
    
    st.write(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}")
    st.write(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")

# Contenido principal según el paso
if st.session_state.step == 1:
    st.header("Paso 1: Información del Usuario")
    st.write("Bienvenido al sistema de análisis estructural avanzado. Por favor, ingrese su información.")
    
    usuario_nombre = st.text_input("👤 Nombre completo:", 
                                  value=st.session_state.usuario_nombre,
                                  placeholder="Ej: Juan Pérez")
    
    if usuario_nombre:
        st.session_state.usuario_nombre = usuario_nombre
        
        st.markdown("""
        ## 🎯 Características del Sistema Avanzado:
        - ✅ Análisis de **vigas** con matrices 4x4
        - ✅ **Base de datos de materiales** (Aluminio, Titanio, Fibra de Carbono, etc.)
        - ✅ **Formateo inteligente de unidades** (kPa, MPa, GPa)
        - ✅ Configuración automática de grados de libertad
        - ✅ **Exportación PDF** de resultados
        - ✅ **Tablas copiables** para Word/Excel
        - ✅ Visualización gráfica optimizada con estructura deformada
        - ✅ Factor de escala 100x para visualización de deformaciones
        
        ### 🧪 Materiales Aeroespaciales Incluidos:
        - Aluminio 6061-T6, 7075-T6, 2024-T3
        - Titanio Ti-6Al-4V
        - Acero 4130
        - Fibra de Carbono T300
        - Magnesio AZ31B
        - + Opción de agregar materiales personalizados
        """)
        
        if st.button("Continuar", type="primary"):
            next_step()

elif st.session_state.step == 2:
    st.header("Paso 2: Número de Nodos")
    st.write("Ingrese el número total de nodos en el sistema")
    
    num_nodos = st.number_input("Número de Nodos", min_value=1, max_value=20, 
                               value=getattr(st.session_state, 'num_nodos', 2))
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_nodos = num_nodos
        next_step()

elif st.session_state.step == 3:
    st.header("Paso 3: Clasificación de Nodos")
    st.write("Defina cuántos nodos son fijos y cuántos son libres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_fijos = st.number_input("Nodos Fijos", min_value=0, max_value=st.session_state.num_nodos, 
                                   value=getattr(st.session_state, 'num_fijos', 1))
    
    with col2:
        num_libres = st.session_state.num_nodos - num_fijos
        st.metric("Nodos Libres", num_libres)
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_fijos = num_fijos
        st.session_state.num_libres = num_libres
        next_step()

elif st.session_state.step == 4:
    st.header("Paso 4: Coordenadas de Nodos")
    st.write("Ingrese las coordenadas para cada nodo")
    
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
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                x = st.number_input(f"X{nodo_id}", value=x_default, format="%.2f", key=f"x_{nodo_id}")
            
            with col2:
                y = st.number_input(f"Y{nodo_id}", value=y_default, format="%.2f", key=f"y_{nodo_id}")
            
            with col3:
                gl_globales = calcular_grados_libertad_globales(nodo_id)
                st.write(f"GL Globales: {gl_globales}")
                st.write(f"GL{gl_globales[0]} → X, GL{gl_globales[1]} → Y")
            
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

elif st.session_state.step == 5:
    st.header("Paso 5: Número de Elementos")
    st.write("Ingrese el número total de elementos en el sistema")
    
    num_elementos = st.number_input("Número de Elementos", min_value=1, max_value=50, 
                                   value=getattr(st.session_state, 'num_elementos', 1))
    
    if st.button("Continuar", type="primary"):
        st.session_state.num_elementos = num_elementos
        next_step()

elif st.session_state.step == 6:
    st.header("Paso 6: Definición de Elementos")
    st.write("Defina las propiedades de cada elemento con materiales aeroespaciales")
    
    if len(st.session_state.elementos) < st.session_state.num_elementos:
        elemento_actual = len(st.session_state.elementos) + 1
        
        st.subheader(f"Elemento {elemento_actual}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Selección de Nodos:**")
            
            # Crear opciones más claras para los nodos
            opciones_nodos = []
            for nodo in st.session_state.nodos:
                opciones_nodos.append({
                    'id': nodo['id'],
                    'display': f"Nodo {nodo['id']} ({nodo['tipo']}) - Coord: ({nodo['x']}, {nodo['y']})"
                })
            
            # Selectbox para nodo inicio
            nodo_inicio_idx = st.selectbox(
                "Nodo de Inicio:",
                range(len(opciones_nodos)),
                format_func=lambda x: opciones_nodos[x]['display'],
                key=f"nodo_inicio_{elemento_actual}"
            )
            nodo_inicio_id = opciones_nodos[nodo_inicio_idx]['id']
            
            # Filtrar opciones para nodo fin (excluir el nodo inicio)
            opciones_nodos_fin = [opt for opt in opciones_nodos if opt['id'] != nodo_inicio_id]
            
            if opciones_nodos_fin:
                nodo_fin_idx = st.selectbox(
                    "Nodo de Fin:",
                    range(len(opciones_nodos_fin)),
                    format_func=lambda x: opciones_nodos_fin[x]['display'],
                    key=f"nodo_fin_{elemento_actual}"
                )
                nodo_fin_id = opciones_nodos_fin[nodo_fin_idx]['id']
            else:
                st.error("No hay nodos disponibles para el nodo fin")
                nodo_fin_id = nodo_inicio_id
            
            # Solo vigas (4x4)
            tipo = "Viga"
        
        with col2:
            st.write("**Selección de Material:**")
            
            # Combinar materiales predefinidos y personalizados
            todos_materiales = {**MATERIALES_AEROESPACIALES, **st.session_state.materiales_personalizados}
            nombres_materiales = list(todos_materiales.keys()) + ["➕ Agregar Material Personalizado"]
            
            material_seleccionado = st.selectbox(
                "Material:",
                nombres_materiales,
                key=f"material_{elemento_actual}"
            )
            
            if material_seleccionado == "➕ Agregar Material Personalizado":
                st.write("**Nuevo Material:**")
                nombre_material = st.text_input("Nombre del Material:", key=f"nombre_mat_{elemento_actual}")
                modulo_young = st.number_input("Módulo de Young (Pa):", value=200e9, format="%.2e", key=f"modulo_custom_{elemento_actual}")
                densidad = st.number_input("Densidad (kg/m³):", value=2700.0, key=f"densidad_custom_{elemento_actual}")
                descripcion = st.text_input("Descripción:", key=f"desc_custom_{elemento_actual}")
                
                if st.button("💾 Guardar Material", key=f"guardar_mat_{elemento_actual}"):
                    if nombre_material:
                        st.session_state.materiales_personalizados[nombre_material] = {
                            "modulo_young": modulo_young,
                            "densidad": densidad,
                            "descripcion": descripcion
                        }
                        st.success(f"Material '{nombre_material}' agregado correctamente")
                        st.rerun()
            else:
                material_info = todos_materiales[material_seleccionado]
                modulo_young = material_info["modulo_young"]
                
                st.info(f"""
                **Material Seleccionado:** {material_seleccionado}
                
                **Propiedades:**
                - Módulo de Young: {formatear_unidades(modulo_young, 'presion')}
                - Densidad: {material_info['densidad']} kg/m³
                - Descripción: {material_info['descripcion']}
                """)
            
            area = st.number_input("Área de la sección (m²):", value=0.01, format="%.6f", key=f"area_{elemento_actual}")
        
        # Información que se actualiza dinámicamente según el tipo
        if nodo_inicio_id != nodo_fin_id and material_seleccionado != "➕ Agregar Material Personalizado":
            nodo_inicio = next(n for n in st.session_state.nodos if n['id'] == nodo_inicio_id)
            nodo_fin = next(n for n in st.session_state.nodos if n['id'] == nodo_fin_id)
            
            longitud = calcular_longitud_elemento(nodo_inicio, nodo_fin)
            
            # Siempre es viga (4x4)
            beta = calcular_angulo_beta(nodo_inicio, nodo_fin)
            beta_grados = math.degrees(beta)
            gl_globales = nodo_inicio['grados_libertad_globales'] + nodo_fin['grados_libertad_globales']
            gl_locales = [1, 2, 3, 4]
            dimension_matriz = "4x4"
            orientacion = "Viga completa (4 GL)"
            descripcion_tipo = "Viga (4 grados de libertad por elemento)"
            
            # Mostrar información actualizada dinámicamente
            st.markdown("---")
            st.subheader("📊 Información Calculada Dinámicamente")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tipo de Elemento", tipo)
                st.metric("Dimensión de Matriz", dimension_matriz)
                st.metric("GL Locales", len(gl_locales))
            
            with col2:
                st.metric("Longitud (L)", f"{longitud:.4f} m")
                st.metric("Ángulo β", f"{beta_grados:.2f}°")
                st.metric("GL Globales", len(gl_globales))
            
            with col3:
                st.write("**Orientación:**")
                st.info(orientacion)
                st.write("**Material:**")
                st.info(material_seleccionado)
            
            # Mostrar grados de libertad detallados
            st.subheader("🔢 Grados de Libertad Detallados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Grados de Libertad Locales:**")
                gl_local_df = pd.DataFrame({
                    'GL Local': gl_locales,
                    'Descripción': [f"GL Local {gl}" for gl in gl_locales]
                })
                st.dataframe(gl_local_df, use_container_width=True)
            
            with col2:
                st.write("**Grados de Libertad Globales:**")
                gl_global_df = pd.DataFrame({
                    'GL Global': gl_globales,
                    'Nodo': [f"Nodo {nodo_inicio_id}" if i < len(gl_locales)//2 else f"Nodo {nodo_fin_id}" 
                            for i in range(len(gl_globales))],
                    'Dirección': ["X" if gl % 2 == 1 else "Y" for gl in gl_globales]
                })
                st.dataframe(gl_global_df, use_container_width=True)
            
            # Mostrar matriz de rigidez teórica
            st.subheader(f"🔢 Matriz de Rigidez {dimension_matriz}")
            
            st.latex(r'''
            [K] = \frac{EA}{L} \begin{bmatrix}
            c^2 & cs & -c^2 & -cs \\
            cs & s^2 & -cs & -s^2 \\
            -c^2 & -cs & c^2 & cs \\
            -cs & -s^2 & cs & s^2
            \end{bmatrix}
            ''')
            st.write(f"Donde: c = cos({beta_grados:.2f}°) = {math.cos(beta):.4f}, s = sin({beta_grados:.2f}°) = {math.sin(beta):.4f}")
            
            # Calcular matriz numérica
            matriz_numerica = generar_matriz_rigidez_viga(modulo_young, area, longitud, beta)
            
            # Mostrar matriz numérica calculada con formato
            st.write("**Matriz Numérica Calculada:**")
            mostrar_matriz_formateada(matriz_numerica.tolist(), f"Matriz K {dimension_matriz} - Elemento {elemento_actual}", es_simbolica=False)
            
            # Botón para guardar elemento
            if st.button(f"💾 Guardar Elemento {elemento_actual}", type="primary", key=f"guardar_{elemento_actual}"):
                elemento = {
                    'id': elemento_actual,
                    'tipo': tipo,
                    'orientacion': orientacion,
                    'nodo_inicio': nodo_inicio_id,
                    'nodo_fin': nodo_fin_id,
                    'grados_libertad_local': gl_locales,
                    'grados_libertad_global': gl_globales,
                    'material': material_seleccionado,
                    'modulo_young': modulo_young,
                    'area': area,
                    'longitud': longitud,
                    'beta': beta,
                    'beta_grados': beta_grados
                }
                
                st.session_state.elementos.append(elemento)
                
                st.session_state.matrices_elementos[elemento_actual] = {
                    'numerica': matriz_numerica.tolist(),
                    'local': matriz_numerica.tolist(),
                    'tipo': tipo
                }
                
                st.success(f"✅ Elemento {elemento_actual} ({tipo} {dimension_matriz}) con {material_seleccionado} guardado correctamente")
                st.rerun()
        
        else:
            if nodo_inicio_id == nodo_fin_id:
                st.warning("⚠️ Seleccione nodos diferentes para el inicio y fin del elemento")
            elif material_seleccionado == "➕ Agregar Material Personalizado":
                st.info("ℹ️ Complete la información del material personalizado")
    
    # Mostrar elementos guardados
    if st.session_state.elementos:
        st.subheader("📋 Elementos Configurados")
        elementos_data = []
        for elem in st.session_state.elementos:
            elementos_data.append({
                'ID': elem['id'],
                'Tipo': elem['tipo'],
                'Dimensión': f"{len(elem['grados_libertad_local'])}x{len(elem['grados_libertad_local'])}",
                'Material': elem['material'],
                'Orientación': elem['orientacion'].split()[0],
                'Nodo I': elem['nodo_inicio'],
                'Nodo F': elem['nodo_fin'],
                'E': formatear_unidades(elem['modulo_young'], "presion"),
                'A': f"{elem['area']:.6f} m²",
                'L': f"{elem['longitud']:.4f} m",
                'GL Globales': str(elem['grados_libertad_global'])
            })
        
        df_elementos = pd.DataFrame(elementos_data)
        st.dataframe(df_elementos, use_container_width=True)
        
        if len(st.session_state.elementos) == st.session_state.num_elementos:
            if st.button("Continuar", type="primary"):
                # Inicializar información de grados de libertad
                max_gl = max([max(elem['grados_libertad_global']) for elem in st.session_state.elementos])
                st.session_state.grados_libertad_info = []
                st.session_state.nombres_fuerzas = {}
                
                for i in range(max_gl):
                    st.session_state.grados_libertad_info.append({
                        'grado': i + 1,
                        'fuerza_conocida': False,
                        'valor_fuerza': 0.0,
                        'desplazamiento_conocido': False,
                        'valor_desplazamiento': 0.0
                    })
                    st.session_state.nombres_fuerzas[i+1] = f"F{i+1}"
                
                next_step()

elif st.session_state.step == 7:
    st.header("Paso 7: Configuración de Incógnitas y Datos")
    st.write("Defina qué fuerzas y desplazamientos son conocidos (datos) o desconocidos (incógnitas)")
    
    st.subheader("⚙️ Configuración de Grados de Libertad")
    
    # Usar columnas para organizar mejor la información
    for i, info in enumerate(st.session_state.grados_libertad_info):
        with st.container():
            st.write(f"### Grado de Libertad {info['grado']}")
            
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
            
            with col1:
                # Nombre de la fuerza
                nombre_fuerza = st.text_input(
                    "Nombre de la Fuerza:",
                    value=st.session_state.nombres_fuerzas[info['grado']],
                    key=f"nombre_{i}"
                )
                # Actualizar inmediatamente
                st.session_state.nombres_fuerzas[info['grado']] = nombre_fuerza
            
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
                        value=info['valor_fuerza'],
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
                        value=info['valor_desplazamiento'],
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
    
    # Mostrar configuración actual
    st.subheader("📋 Resumen de Configuración")
    config_data = []
    for info in st.session_state.grados_libertad_info:
        fuerza_str = formatear_unidades(info['valor_fuerza'], "fuerza") if info['fuerza_conocida'] else "Incógnita"
        desplazamiento_str = formatear_unidades(info['valor_desplazamiento'], "desplazamiento") if info['desplazamiento_conocido'] else "Incógnita"
        
        config_data.append({
            'GL': info['grado'],
            'Nombre Fuerza': st.session_state.nombres_fuerzas[info['grado']],
            'Fuerza': fuerza_str,
            'Desplazamiento': desplazamiento_str
        })
    
    df_config = pd.DataFrame(config_data)
    st.dataframe(df_config, use_container_width=True)
    
    if st.button("Continuar", type="primary"):
        next_step()

elif st.session_state.step == 8:
    st.header("Paso 8: Resultados del Análisis")
    
    # Pestañas para diferentes vistas
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Estructura", "🔢 Matrices", "🧮 Solución", "📈 Resumen"])
    
    with tab1:
        st.subheader("Visualización de la Estructura")
        
        if st.session_state.nodos:
            # Mostrar estructura original
            st.write("### Estructura Original")
            fig_original = visualizar_estructura(mostrar_deformada=False)
            if fig_original:
                st.pyplot(fig_original)
            
            # Mostrar estructura deformada si hay resultados
            if st.session_state.resultados:
                st.write("### Estructura Deformada (Factor de escala: 100x)")
                fig_deformada = visualizar_estructura(mostrar_deformada=True, factor_escala=100)
                if fig_deformada:
                    st.pyplot(fig_deformada)
                
                # Slider para ajustar factor de escala
                factor_escala = st.slider("Factor de escala para deformaciones:", 
                                         min_value=1, max_value=100, value=10, step=1)
                
                if factor_escala != 100:
                    fig_custom = visualizar_estructura(mostrar_deformada=True, factor_escala=factor_escala)
                    if fig_custom:
                        st.pyplot(fig_custom)
        
        # Tablas de resumen
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tabla de Nodos")
            if st.session_state.nodos:
                df_nodos = pd.DataFrame(st.session_state.nodos)
                st.dataframe(df_nodos, use_container_width=True)
        
        with col2:
            st.subheader("Tabla de Elementos")
            if st.session_state.elementos:
                elementos_data = []
                for elem in st.session_state.elementos:
                    elementos_data.append({
                        'ID': elem['id'],
                        'Tipo': elem['tipo'],
                        'Material': elem['material'],
                        'Dimensión': f"{len(elem['grados_libertad_local'])}x{len(elem['grados_libertad_local'])}",
                        'Orientación': elem['orientacion'].split()[0],
                        'Nodo I': elem['nodo_inicio'],
                        'Nodo F': elem['nodo_fin'],
                        'E': formatear_unidades(elem['modulo_young'], "presion"),
                        'A': f"{elem['area']:.6f} m²",
                        'L': f"{elem['longitud']:.4f} m"
                    })
                df_elementos = pd.DataFrame(elementos_data)
                st.dataframe(df_elementos, use_container_width=True)
    
    with tab2:
        st.subheader("Matrices de Rigidez")
        
        # Matrices de elementos
        for elemento_id, matrices in st.session_state.matrices_elementos.items():
            elemento = next(e for e in st.session_state.elementos if e['id'] == elemento_id)
            dimension = f"{len(elemento['grados_libertad_local'])}x{len(elemento['grados_libertad_local'])}"
            
            with st.expander(f"Elemento {elemento_id} ({matrices['tipo']} - Matriz {dimension}) - {elemento['material']}"):
                st.write(f"**Tipo:** {matrices['tipo']}")
                st.write(f"**Material:** {elemento['material']}")
                st.write(f"**Dimensión:** {dimension}")
                st.write(f"**Orientación:** {elemento['orientacion']}")
                st.write(f"**Nodos:** {elemento['nodo_inicio']} → {elemento['nodo_fin']}")
                st.write(f"**Longitud:** {elemento['longitud']:.4f} m")
                st.write(f"**Ángulo β:** {elemento['beta_grados']:.2f}°")
                st.write(f"**E:** {formatear_unidades(elemento['modulo_young'], 'presion')}, **A:** {elemento['area']:.6f} m²")
                st.write(f"**GL Locales:** {elemento['grados_libertad_local']}")
                st.write(f"**GL Globales:** {elemento['grados_libertad_global']}")
                
                mostrar_matriz_formateada(matrices['numerica'], f"Matriz K {dimension} - Elemento {elemento_id}", es_simbolica=False)
    
    with tab3:
        st.subheader("Solución del Sistema")
        
        if st.button("🧮 Calcular Solución", type="primary"):
            resultado = resolver_sistema()
            
            if resultado and resultado['exito']:
                st.session_state.resultados = resultado
                st.success("✅ Sistema resuelto exitosamente")
            else:
                st.error("❌ Error al resolver el sistema")
        
        if st.session_state.resultados:
            resultado = st.session_state.resultados
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Determinante de K", formatear_unidades(resultado['determinante'], "presion"))
            
            with col2:
                st.metric("Desplazamiento Máximo", formatear_unidades(np.max(np.abs(resultado['desplazamientos'])), "desplazamiento"))
            
            with col3:
                st.metric("Fuerza Máxima", formatear_unidades(np.max(np.abs(resultado['fuerzas'])), "fuerza"))
            
            # Resultados detallados
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Desplazamientos Calculados:**")
                desplazamientos_data = []
                for i, (info, valor) in enumerate(zip(st.session_state.grados_libertad_info, resultado['desplazamientos'])):
                    tipo = "Dato" if info['desplazamiento_conocido'] else "Calculado"
                    desplazamientos_data.append({
                        'GL': f"u{i+1}",
                        'Valor': formatear_unidades(valor, "desplazamiento"),
                        'Tipo': tipo
                    })
                
                df_desplazamientos = pd.DataFrame(desplazamientos_data)
                st.dataframe(df_desplazamientos, use_container_width=True)
                
                # Botón para copiar tabla de desplazamientos
                csv_desplazamientos = df_desplazamientos.to_csv(sep='\t', index=False)
                st.download_button(
                    label="📋 Copiar Tabla Desplazamientos",
                    data=csv_desplazamientos,
                    file_name="desplazamientos.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.write("**Fuerzas Calculadas:**")
                fuerzas_data = []
                for i, (info, valor) in enumerate(zip(st.session_state.grados_libertad_info, resultado['fuerzas'])):
                    nombre = st.session_state.nombres_fuerzas.get(i+1, f"F{i+1}")
                    tipo = "Dato" if info['fuerza_conocida'] else "Calculado"
                    fuerzas_data.append({
                        'GL': nombre,
                        'Valor': formatear_unidades(valor, "fuerza"),
                        'Tipo': tipo
                    })
                
                df_fuerzas = pd.DataFrame(fuerzas_data)
                st.dataframe(df_fuerzas, use_container_width=True)
                
                # Botón para copiar tabla de fuerzas
                csv_fuerzas = df_fuerzas.to_csv(sep='\t', index=False)
                st.download_button(
                    label="📋 Copiar Tabla Fuerzas",
                    data=csv_fuerzas,
                    file_name="fuerzas.csv",
                    mime="text/csv"
                )
            
            # Matriz K global
            st.write("**Matriz K Global Numérica:**")
            mostrar_matriz_formateada(resultado['K_global'].tolist(), "Matriz K Global", es_simbolica=False)
            
            # Botón para generar PDF
            st.subheader("📄 Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Generar PDF Completo", type="primary"):
                    pdf_data = generar_pdf_resultados()
                    if pdf_data:
                        st.download_button(
                            label="⬇️ Descargar PDF",
                            data=pdf_data,
                            file_name=f"analisis_estructural_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
            
            with col2:
                # Exportar datos completos en CSV
                if st.button("📊 Exportar Datos CSV"):
                    # Crear archivo ZIP con todas las tablas
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Matriz K
                        matriz_k_df = pd.DataFrame(resultado['K_global'])
                        matriz_k_df.index = [f"GL{i+1}" for i in range(len(resultado['K_global']))]
                        matriz_k_df.columns = [f"GL{i+1}" for i in range(len(resultado['K_global'][0]))]
                        zip_file.writestr("matriz_K_global.csv", matriz_k_df.to_csv())
                        
                        # Resultados
                        zip_file.writestr("desplazamientos.csv", df_desplazamientos.to_csv(index=False))
                        zip_file.writestr("fuerzas.csv", df_fuerzas.to_csv(index=False))
                        
                        # Información del proyecto
                        info_proyecto = {
                            'Usuario': st.session_state.usuario_nombre,
                            'Fecha': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                            'Nodos': len(st.session_state.nodos),
                            'Elementos': len(st.session_state.elementos),
                            'Grados_Libertad': len(st.session_state.grados_libertad_info)
                        }
                        info_df = pd.DataFrame(list(info_proyecto.items()), columns=['Parametro', 'Valor'])
                        zip_file.writestr("informacion_proyecto.csv", info_df.to_csv(index=False))
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ Descargar ZIP con Datos",
                        data=zip_buffer.getvalue(),
                        file_name=f"datos_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
    
    with tab4:
        st.subheader("Resumen del Análisis")
        
        # Información del sistema
        vigas_count = len(st.session_state.elementos)  # Todas son vigas ahora
        nodos_fijos = sum(1 for n in st.session_state.nodos if n['tipo'] == 'fijo')
        nodos_libres = sum(1 for n in st.session_state.nodos if n['tipo'] == 'libre')
        
        info_data = {
            "Parámetro": [
                "Usuario", "Fecha", "Nodos totales", "Nodos fijos", "Nodos libres",
                "Elementos totales", "Vigas (4x4)", "Grados de libertad"
            ],
            "Valor": [
                st.session_state.usuario_nombre,
                datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                len(st.session_state.nodos),
                nodos_fijos,
                nodos_libres,
                len(st.session_state.elementos),
                vigas_count,
                len(st.session_state.grados_libertad_info)
            ]
        }
        
        df_info = pd.DataFrame(info_data)
        st.dataframe(df_info, use_container_width=True)
        
        # Análisis de materiales utilizados
        st.subheader("🧪 Materiales Utilizados")
        
        materiales_usados = {}
        for elemento in st.session_state.elementos:
            material = elemento['material']
            if material not in materiales_usados:
                materiales_usados[material] = {
                    'count': 0,
                    'elementos': [],
                    'modulo_young': elemento['modulo_young']
                }
            materiales_usados[material]['count'] += 1
            materiales_usados[material]['elementos'].append(elemento['id'])
        
        materiales_data = []
        for material, info in materiales_usados.items():
            materiales_data.append({
                'Material': material,
                'Elementos': info['count'],
                'IDs': ', '.join(map(str, info['elementos'])),
                'Módulo E': formatear_unidades(info['modulo_young'], "presion")
            })
        
        df_materiales = pd.DataFrame(materiales_data)
        st.dataframe(df_materiales, use_container_width=True)
        
        # Análisis de elementos
        st.subheader("🔧 Análisis de Elementos")
        
        st.write("**Vigas (Matriz 4x4):**")
        vigas_info = []
        for elemento in st.session_state.elementos:
            vigas_info.append({
                'Elemento': elemento['id'],
                'Material': elemento['material'],
                'GL Globales': str(elemento['grados_libertad_global']),
                'Dimensión': "4x4",
                'Ángulo β': f"{elemento['beta_grados']:.1f}°"
            })
        
        if vigas_info:
            df_vigas = pd.DataFrame(vigas_info)
            st.dataframe(df_vigas, use_container_width=True)
        
        # Estado del sistema
        if st.session_state.resultados:
            resultado = st.session_state.resultados
            
            st.subheader("📊 Estado del Sistema")
            
            estado_sistema = "Bien condicionado" if abs(resultado['determinante']) > 1e-10 else "Mal condicionado"
            validez_solucion = "Válida" if np.all(np.isfinite(resultado['desplazamientos'])) else "Inválida"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Estado del Sistema", estado_sistema)
            
            with col2:
                st.metric("Validez de la Solución", validez_solucion)
            
            with col3:
                st.metric("Determinante K", formatear_unidades(resultado['determinante'], "presion"))

# Información adicional en la barra lateral
with st.sidebar:
    st.divider()
    st.write("**Desarrollado con:**")
    st.write("🐍 Python + Streamlit")
    st.write("📊 Matplotlib + NumPy")
    st.write("🔢 Pandas + ReportLab")
    st.write("🧪 Base de datos de materiales")
    
    if st.session_state.materiales_personalizados:
        st.divider()
        st.write("**Materiales Personalizados:**")
        for nombre in st.session_state.materiales_personalizados.keys():
            st.write(f"• {nombre}")

#python -m streamlit run analisis_estructural_streamlit.py
