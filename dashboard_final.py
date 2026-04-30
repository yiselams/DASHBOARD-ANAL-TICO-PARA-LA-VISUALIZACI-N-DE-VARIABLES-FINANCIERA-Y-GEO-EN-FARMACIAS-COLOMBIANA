import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Configuración
st.set_page_config(page_title="Dashboard Farmacias - Análisis Financiero", layout="wide")

# ==================== FUNCIONES DE CARGA Y LIMPIEZA ====================
@st.cache_data
def load_and_clean_data():
    """Carga y combina ambas bases de datos: BD_TG_YISELA (infraestructura) + CONSOLIDADO (financiero)"""
    
    # 1. CARGAR BD_TG_YISELA (Infraestructura, cánones, ubicación)
    bd_path = None
    if os.path.exists("BASES DE DATOS/BD_TG_YISELA.csv"):
        bd_path = "BASES DE DATOS/BD_TG_YISELA.csv"
    elif os.path.exists("BD_TG_YISELA.csv"):
        bd_path = "BD_TG_YISELA.csv"
    
    if not bd_path:
        st.error("❌ No se encontró BD_TG_YISELA.csv")
        st.stop()
    
    df_infra = pd.read_csv(bd_path, sep=';', encoding='utf-8-sig')
    
    # 2. CARGAR CONSOLIDADO (Datos financieros mensuales)
    consolidado_path = None
    if os.path.exists("BASES DE DATOS/CONSOLIDADO_FINAL_YISELA_MAYORGA.xlsx"):
        consolidado_path = "BASES DE DATOS/CONSOLIDADO_FINAL_YISELA_MAYORGA.xlsx"
    elif os.path.exists("CONSOLIDADO_FINAL_YISELA_MAYORGA.xlsx"):
        consolidado_path = "CONSOLIDADO_FINAL_YISELA_MAYORGA.xlsx"
    
    if not consolidado_path:
        st.error("❌ No se encontró CONSOLIDADO_FINAL_YISELA_MAYORGA.xlsx")
        st.stop()
    
    df_financiero = pd.read_excel(consolidado_path)
    
    # 3. COMBINAR ambas bases por CAF (left join para mantener toda la info de financiero)
    df = pd.merge(df_financiero, df_infra, on='CAF', how='left')
    
    # LIMPIEZA AUTOMÁTICA (intrínseca)
    df = df.copy()
    
    # Convertir columnas numéricas clave
    numeric_cols = ['Ingresos Final', 'Egresos Final', 'Formulación', 'CANON', 
                   'CANON X FO', 'CANON X M2', 'CAPACIDAD', 'HOLGURA', 'FO DIA ACTUAL']
    for col in numeric_cols:
        if col in df.columns:
            # Siempre intentar limpiar si parece ser un string
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Limpiar formato: $ 29.233.514 -> 29233514
                df[col] = (df[col].astype(str)
                          .str.replace('$', '', regex=False)
                          .str.replace('.', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.replace(' ', '', regex=False)
                          .str.replace('-', '', regex=False)
                          .str.strip())
                # Reemplazar strings vacíos con '0'
                df[col] = df[col].replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rellenar NaN con 0 (solo en columnas financieras)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Remover outliers extremos con percentiles 1-99 (SOLO para ingresos y egresos)
    outlier_cols = ['Ingresos Final', 'Egresos Final', 'Formulación']
    for col in outlier_cols:
        if col in df.columns:
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(p1, p99)
    
    # Agregar coordenadas geográficas
    coordenadas = {
        'BOGOTA': {'lat': 4.7110, 'lon': -74.0721},
        'MEDELLIN': {'lat': 6.2442, 'lon': -75.5812},
        'CALI': {'lat': 3.4516, 'lon': -76.5320},
        'BARRANQUILLA': {'lat': 10.9685, 'lon': -74.7813},
        'CARTAGENA': {'lat': 10.3910, 'lon': -75.4794},
        'CUCUTA': {'lat': 7.8939, 'lon': -72.5078},
        'SOLEDAD': {'lat': 10.9185, 'lon': -74.7644},
        'BUCARAMANGA': {'lat': 7.1254, 'lon': -73.1198},
        'SOACHA': {'lat': 4.5793, 'lon': -74.2169},
        'IBAGUE': {'lat': 4.4389, 'lon': -75.2322},
        'PEREIRA': {'lat': 4.8087, 'lon': -75.6906},
        'SANTA MARTA': {'lat': 11.2408, 'lon': -74.1990},
        'VALLEDUPAR': {'lat': 10.4631, 'lon': -73.2532},
        'MANIZALES': {'lat': 5.0689, 'lon': -75.5174},
        'PASTO': {'lat': 1.2136, 'lon': -77.2811},
        'NEIVA': {'lat': 2.9273, 'lon': -75.2819},
        'PALMIRA': {'lat': 3.5394, 'lon': -76.3036},
        'VILLAVICENCIO': {'lat': 4.1420, 'lon': -73.6266},
        'MONTERIA': {'lat': 8.7479, 'lon': -75.8814},
        'SINCELEJO': {'lat': 9.3047, 'lon': -75.3978},
        'POPAYAN': {'lat': 2.4448, 'lon': -76.6147},
        'TUNJA': {'lat': 5.5353, 'lon': -73.3678},
        'FLORENCIA': {'lat': 1.6144, 'lon': -75.6062},
        'ARMENIA': {'lat': 4.5339, 'lon': -75.6811},
        'RIOHACHA': {'lat': 11.5444, 'lon': -72.9072},
        'GIRON': {'lat': 7.0701, 'lon': -73.1692},
        'BARRANCABERMEJA': {'lat': 7.0653, 'lon': -73.8547},
        'TULUA': {'lat': 4.0847, 'lon': -76.1953},
        'BELLO': {'lat': 6.3373, 'lon': -75.5547},
        'ITAGUI': {'lat': 6.1845, 'lon': -75.5993},
        'ENVIGADO': {'lat': 6.1701, 'lon': -75.5783},
        'MAGANGUE': {'lat': 9.2415, 'lon': -74.7546},
        'PUEBLO NUEVO': {'lat': 8.9833, 'lon': -75.3000},
        'TIQUISIO NUEVO': {'lat': 8.5500, 'lon': -74.2667},
    }
    
    # Agregar coordenadas al dataframe
    if 'CIUDAD' in df.columns:
        import unicodedata
        def normalizar_ciudad(ciudad):
            if pd.isna(ciudad):
                return None
            # Remover tildes, puntos, y convertir a mayúsculas
            ciudad = str(ciudad).upper().strip()
            ciudad = ''.join(c for c in unicodedata.normalize('NFD', ciudad) 
                           if unicodedata.category(c) != 'Mn')
            return ciudad
        
        df['CIUDAD_NORM'] = df['CIUDAD'].apply(normalizar_ciudad)
        df['lat'] = df['CIUDAD_NORM'].map(lambda x: coordenadas.get(x, {}).get('lat', None) if x else None)
        df['lon'] = df['CIUDAD_NORM'].map(lambda x: coordenadas.get(x, {}).get('lon', None) if x else None)
    
    return df

# Cargar y limpiar datos
df = load_and_clean_data()

# ==================== SIDEBAR CON FILTROS ====================
st.sidebar.header("🔍 Filtros de Datos")
st.sidebar.markdown("---")

# Filtro por Ciudad
if 'CIUDAD' in df.columns:
    ciudades = sorted(df['CIUDAD'].dropna().unique().tolist())
    ciudad_opciones = ['Todas'] + ciudades
    ciudad_seleccionada = st.sidebar.selectbox("📍 Ciudad", ciudad_opciones, index=0)
    ciudades_seleccionadas = ciudades if ciudad_seleccionada == 'Todas' else [ciudad_seleccionada]
else:
    ciudades_seleccionadas = None

# Filtro por Departamento
if 'DEPARTAMENTO' in df.columns:
    departamentos = sorted(df['DEPARTAMENTO'].dropna().unique().tolist())
    dep_opciones = ['Todos'] + departamentos
    departamento_seleccionado = st.sidebar.selectbox("🗺️ Departamento", dep_opciones, index=0)
    departamentos_seleccionados = departamentos if departamento_seleccionado == 'Todos' else [departamento_seleccionado]
else:
    departamentos_seleccionados = None

# Filtro por Consorcio
if 'CONSORCIO' in df.columns:
    consorcios = sorted(df['CONSORCIO'].dropna().unique().tolist())
    consorcio_opciones = ['Todos'] + consorcios
    consorcio_seleccionado = st.sidebar.selectbox("🏢 Consorcio", consorcio_opciones, index=0)
    consorcios_seleccionados = consorcios if consorcio_seleccionado == 'Todos' else [consorcio_seleccionado]
else:
    consorcios_seleccionados = None

# Aplicar filtros
df_filtrado = df.copy()

if ciudades_seleccionadas is not None and 'CIUDAD' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['CIUDAD'].isin(ciudades_seleccionadas)]

if departamentos_seleccionados is not None and 'DEPARTAMENTO' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(departamentos_seleccionados)]

if consorcios_seleccionados is not None and 'CONSORCIO' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['CONSORCIO'].isin(consorcios_seleccionados)]

# Mostrar información del filtrado
st.sidebar.markdown("---")
st.sidebar.info(f"**Registros filtrados:** {len(df_filtrado):,} de {len(df):,}")

# Usar df_filtrado para todos los cálculos
df = df_filtrado.copy()

# ==================== CÁLCULO DE INDICADORES ====================
def calcular_indicadores(df):
    """Calcula indicadores de riesgo y salud financiera"""
    df = df.copy()
    
    if 'Ingresos Final' in df.columns and 'Egresos Final' in df.columns:
        df['Ratio_Egresos_Ingresos'] = (df['Egresos Final'] / (df['Ingresos Final'] + 1)).replace([np.inf, -np.inf], 0)
        df['Balance'] = df['Ingresos Final'] - df['Egresos Final']
        df['Margin_Ganancia'] = ((df['Ingresos Final'] - df['Egresos Final']) / (df['Ingresos Final'] + 1) * 100).fillna(0)
    
    if 'Formulación' in df.columns:
        df['Flag_Baja_Formulacion'] = df['Formulación'] < 500
    
    if 'FO DIA ACTUAL' in df.columns and 'CAPACIDAD' in df.columns:
        df['% Utilizacion'] = (df['FO DIA ACTUAL'] / (df['CAPACIDAD'] + 1) * 100).fillna(0)
    
    return df

# ==================== MODELO DE PREDICCIÓN ====================
@st.cache_data
def entrenar_modelo_riesgo(df_input):
    """Entrena modelo ML para predecir riesgo de cierre de farmacias"""
    df_modelo = df_input.copy()
    
    # Crear target: RIESGO = Margen < 0 O (Margen < 2% Y Formulación muy baja)
    umbrales_bajo = df_modelo['Formulación'].quantile(0.25)
    df_modelo['Riesgo'] = ((df_modelo['Margin_Ganancia'] < 0) | 
                           ((df_modelo['Margin_Ganancia'] < 2) & (df_modelo['Formulación'] < umbrales_bajo))).astype(int)
    
    # Variables predictoras
    features = ['Ratio_Egresos_Ingresos', 'Margin_Ganancia', 'Formulación', 
                'Balance', 'Flag_Baja_Formulacion']
    
    # Preparar datos
    X = df_modelo[features].fillna(0).values
    y = df_modelo['Riesgo'].values
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Predicciones y probabilidades
    df_modelo['Prediccion_Riesgo'] = model.predict(X)
    df_modelo['Prob_Riesgo'] = model.predict_proba(X)[:, 1]
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return df_modelo, model, feature_importance

df = calcular_indicadores(df)
df, modelo_riesgo, feature_imp = entrenar_modelo_riesgo(df)

# Crear versión agregada por CAF (promedio de todos los meses)
df_por_caf = df.groupby('CAF').agg({
    'Ingresos Final': 'mean',
    'Egresos Final': 'mean',
    'Formulación': 'mean',
    'Balance': 'mean',
    'Margin_Ganancia': 'mean',
    'Prob_Riesgo': 'mean',
    'Prediccion_Riesgo': lambda x: (x.mean() > 0.5).astype(int),  # Mayoría de meses en riesgo
    'Ratio_Egresos_Ingresos': 'mean',
    'Flag_Baja_Formulacion': 'mean'
}).reset_index()

# ==================== INTERFAZ ====================
st.title("💊 Dashboard Farmacias - Análisis Financiero y Operacional")
st.markdown("**Visión Integral de Desempeño Farmacéutico con Predicción de Riesgo**")

# KPIs PRINCIPALES
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    num_registros = df[df['Ingresos Final'] > 0]['CAF'].nunique()
    st.metric("📊 Farmacias Activas", num_registros)

with col2:
    ingresos_total = df['Ingresos Final'].sum() if 'Ingresos Final' in df.columns else 0
    st.metric("💰 Ingresos Totales", f"${ingresos_total/1e6:.1f}M")

with col3:
    egresos_total = df['Egresos Final'].sum() if 'Egresos Final' in df.columns else 0
    st.metric("📉 Egresos Totales", f"${egresos_total/1e6:.1f}M")

with col4:
    if 'Balance' in df.columns:
        balance = df['Balance'].sum()
        st.metric("⚖️ Balance Total", f"${balance/1e6:.1f}M")

with col5:
    if 'Formulación' in df.columns:
        formula_promedio = df['Formulación'].mean()
        st.metric("💊 Formulación Promedio", f"{formula_promedio:,.0f}")

# Segunda fila de KPIs - Métricas de Riesgo
st.markdown("")
col6, col7, col8, col9, col10 = st.columns(5)

with col6:
    if 'Margin_Ganancia' in df_por_caf.columns:
        caf_margen_negativo = len(df_por_caf[df_por_caf['Margin_Ganancia'] < 0])
        total_caf = len(df_por_caf)
        pct_negativo = (caf_margen_negativo / total_caf * 100) if total_caf > 0 else 0
        st.metric("🔴 CAFs Margen Negativo", caf_margen_negativo, 
                 delta=f"{pct_negativo:.1f}% del total", delta_color="inverse")

with col7:
    if 'Prob_Riesgo' in df_por_caf.columns:
        prob_riesgo_promedio = df_por_caf['Prob_Riesgo'].mean() * 100
        st.metric("⚠️ Prob. Riesgo Promedio", f"{prob_riesgo_promedio:.1f}%")

with col8:
    if 'Margin_Ganancia' in df_por_caf.columns:
        margen_promedio = df_por_caf['Margin_Ganancia'].mean()
        st.metric("📊 Margen Promedio", f"{margen_promedio:.1f}%",
                 delta="Positivo" if margen_promedio > 0 else "Negativo",
                 delta_color="normal" if margen_promedio > 0 else "inverse")

with col9:
    if 'Ratio_Egresos_Ingresos' in df_por_caf.columns:
        ratio_promedio = df_por_caf['Ratio_Egresos_Ingresos'].mean()
        st.metric("⚖️ Ratio E/I Promedio", f"{ratio_promedio:.2f}",
                 delta="Saludable" if ratio_promedio < 1 else "Crítico",
                 delta_color="normal" if ratio_promedio < 1 else "inverse")

with col10:
    if 'Prob_Riesgo' in df_por_caf.columns:
        caf_riesgo_critico = len(df_por_caf[df_por_caf['Prob_Riesgo'] > 0.7])
        st.metric("🚨 CAFs Riesgo Crítico", caf_riesgo_critico,
                 delta="Requieren intervención", delta_color="inverse")


# ==================== ANÁLISIS DE RIESGO PREDICTIVO ====================
st.markdown("---")
st.markdown("## 🚨 Análisis Predictivo - Riesgo de Cierre Farmacéutico")
st.caption("*Análisis a nivel de CAF (promedio de todos los meses disponibles)*")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    riesgo_alto = len(df_por_caf[df_por_caf['Prob_Riesgo'] > 0.7])
    st.metric("🔴 Riesgo ALTO (>70%)", riesgo_alto)

with pred_col2:
    riesgo_medio = len(df_por_caf[(df_por_caf['Prob_Riesgo'] >= 0.4) & (df_por_caf['Prob_Riesgo'] <= 0.7)])
    st.metric("🟡 Riesgo MEDIO (40-70%)", riesgo_medio)

with pred_col3:
    riesgo_bajo = len(df_por_caf[df_por_caf['Prob_Riesgo'] < 0.4])
    st.metric("🟢 Riesgo BAJO (<40%)", riesgo_bajo)

# Distribución de probabilidades
col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    st.markdown("### 📊 Distribución de Probabilidad de Riesgo")
    
    fig_prob = px.histogram(df_por_caf, x='Prob_Riesgo', nbins=30,
                           title="Distribución de Probabilidad de Cierre (por CAF)",
                           labels={'Prob_Riesgo': 'Probabilidad de Riesgo'},
                           color_discrete_sequence=['#3498db'])
    fig_prob.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                      annotation_text="Umbral Medio")
    fig_prob.add_vline(x=0.7, line_dash="dash", line_color="red", 
                      annotation_text="Umbral Alto")
    st.plotly_chart(fig_prob, use_container_width=True)

with col_pred2:
    st.markdown("### 🔑 Factores Más Influyentes en el Riesgo")
    
    fig_imp = px.bar(feature_imp, x='Importance', y='Feature',
                    title="Feature Importance del Modelo",
                    labels={'Importance': 'Importancia', 'Feature': 'Variable'},
                    color='Importance', color_continuous_scale='Reds')
    fig_imp.update_layout(height=350)
    st.plotly_chart(fig_imp, use_container_width=True)

# ==================== ANÁLISIS PRINCIPAL ====================
st.markdown("---")
st.markdown("## 📈 Análisis de Ingreso y Formulación")

col1, col2 = st.columns(2)

# TOP 15 CAF POR INGRESO
with col1:
    st.markdown("### 🏆 Top 15 CAF - Mayor Ingreso")
    
    if 'CAF' in df.columns and 'Ingresos Final' in df.columns:
        top_ingresos = df.groupby('CAF').agg({
            'Ingresos Final': 'sum',
            'Formulación': 'sum'
        }).nlargest(15, 'Ingresos Final').reset_index()
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=top_ingresos['CAF'],
            y=top_ingresos['Ingresos Final'],
            name='Ingresos',
            marker=dict(color='#2ecc71'),
            yaxis='y'
        ))
        fig1.add_trace(go.Scatter(
            x=top_ingresos['CAF'],
            y=top_ingresos['Formulación'],
            name='Formulación',
            mode='markers+lines',
            marker=dict(size=8, color='#3498db'),
            line=dict(width=2, color='#3498db'),
            yaxis='y2'
        ))
        
        fig1.update_layout(
            height=450,
            xaxis_title="Farmacia",
            yaxis=dict(title="Ingresos ($)", side='left'),
            yaxis2=dict(title="Formulación", overlaying='y', side='right'),
            hovermode='x unified',
            showlegend=True
        )
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)

# BOTTOM 15 CAF POR INGRESO
with col2:
    st.markdown("### 📉 Bottom 15 CAF - Menor Ingreso")
    
    if 'CAF' in df.columns and 'Ingresos Final' in df.columns:
        bottom_ingresos = df.groupby('CAF').agg({
            'Ingresos Final': 'sum',
            'Formulación': 'sum'
        }).nsmallest(15, 'Ingresos Final').reset_index()
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=bottom_ingresos['CAF'],
            y=bottom_ingresos['Ingresos Final'],
            name='Ingresos',
            marker=dict(color='#e74c3c'),
            yaxis='y'
        ))
        fig2.add_trace(go.Scatter(
            x=bottom_ingresos['CAF'],
            y=bottom_ingresos['Formulación'],
            name='Formulación',
            mode='markers+lines',
            marker=dict(size=8, color='#f39c12'),
            line=dict(width=2, color='#f39c12'),
            yaxis='y2'
        ))
        
        fig2.update_layout(
            height=450,
            xaxis_title="Farmacia",
            yaxis=dict(title="Ingresos ($)", side='left'),
            yaxis2=dict(title="Formulación", overlaying='y', side='right'),
            hovermode='x unified',
            showlegend=True
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

# ==================== ANÁLISIS DE FORMULACIÓN ====================
st.markdown("---")
st.markdown("## 💊 Análisis de Formulación")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🥇 Top 15 CAF - Mayor Formulación")
    
    if 'CAF' in df.columns and 'Formulación' in df.columns:
        top_formula = df.groupby('CAF').agg({
            'Formulación': 'sum',
            'Ingresos Final': 'sum'
        }).nlargest(15, 'Formulación').reset_index()
        
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=top_formula['CAF'],
            y=top_formula['Formulación'],
            name='Formulación',
            marker=dict(color='#9b59b6'),
            yaxis='y'
        ))
        fig3.add_trace(go.Scatter(
            x=top_formula['CAF'],
            y=top_formula['Ingresos Final'],
            name='Ingresos',
            mode='markers+lines',
            marker=dict(size=8, color='#2ecc71'),
            line=dict(width=2, color='#2ecc71'),
            yaxis='y2'
        ))
        
        fig3.update_layout(
            height=450,
            xaxis_title="Farmacia",
            yaxis=dict(title="Formulación", side='left'),
            yaxis2=dict(title="Ingresos ($)", overlaying='y', side='right'),
            hovermode='x unified',
            showlegend=True
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("### 📉 Top 15 CAF - Menor Formulación")
    
    if 'CAF' in df.columns and 'Formulación' in df.columns:
        bottom_formula = df.groupby('CAF').agg({
            'Formulación': 'sum',
            'Ingresos Final': 'sum'
        }).nsmallest(15, 'Formulación').reset_index()
        
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=bottom_formula['CAF'],
            y=bottom_formula['Formulación'],
            name='Formulación',
            marker=dict(color='#e67e22'),
            yaxis='y'
        ))
        fig4.add_trace(go.Scatter(
            x=bottom_formula['CAF'],
            y=bottom_formula['Ingresos Final'],
            name='Ingresos',
            mode='markers+lines',
            marker=dict(size=8, color='#2ecc71'),
            line=dict(width=2, color='#2ecc71'),
            yaxis='y2'
        ))
        
        fig4.update_layout(
            height=450,
            xaxis_title="Farmacia",
            yaxis=dict(title="Formulación", side='left'),
            yaxis2=dict(title="Ingresos ($)", overlaying='y', side='right'),
            hovermode='x unified',
            showlegend=True
        )
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)

# ==================== ANÁLISIS DE RIESGO FINANCIERO ====================
st.markdown("---")
st.markdown("## ⚠️ Análisis de Riesgo Financiero")

if 'Ratio_Egresos_Ingresos' in df.columns and 'Balance' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💥 Farmacias en Alerta - Egresos > Ingresos")
        
        df_alerta = df[df['Ratio_Egresos_Ingresos'] > 1.0].copy()
        
        if len(df_alerta) > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🚨 Cantidad", len(df_alerta))
            with col_b:
                pct = (len(df_alerta) / len(df[df['Ingresos Final'] > 0]) * 100)
                st.metric("% del Total", f"{pct:.1f}%")
            
            # Distribución
            fig5 = px.histogram(df_alerta, x='Ratio_Egresos_Ingresos', nbins=30,
                               title="Distribución del Ratio Egresos/Ingresos",
                               labels={'Ratio_Egresos_Ingresos': 'Ratio Egresos/Ingresos'})
            fig5.add_vline(x=1.0, line_dash="dash", line_color="red")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.success("✅ No hay farmacias en alerta")
    
    with col2:
        st.markdown("### 📊 Distribución de Márgenes de Ganancia")
        
        if 'Margin_Ganancia' in df.columns:
            df_margin = df[df['Ingresos Final'] > 0].copy()
            
            fig6 = px.box(df_margin, y='Margin_Ganancia',
                         title="Distribución de Margen de Ganancia (%)",
                         labels={'Margin_Ganancia': 'Margen %'})
            fig6.add_hline(y=0, line_dash="dash", line_color="red", 
                          annotation_text="Punto de Equilibrio")
            st.plotly_chart(fig6, use_container_width=True)

# ==================== CORRELACIONES CLAVE ====================
st.markdown("---")
st.markdown("## 🔗 Correlaciones Clave")

if 'Ingresos Final' in df.columns and 'Formulación' in df.columns and 'Egresos Final' in df.columns:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Ingresos vs Formulación")
        
        df_plot = df[df['Ingresos Final'] > 0].copy()
        
        fig7 = px.scatter(df_plot, x='Formulación', y='Ingresos Final',
                         title="Relación: Formulación → Ingresos",
                         labels={'Formulación': 'Formulación', 'Ingresos Final': 'Ingresos ($)'},
                         opacity=0.6)
        
        if len(df_plot) > 2:
            z = np.polyfit(df_plot['Formulación'], df_plot['Ingresos Final'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_plot['Formulación'].min(), df_plot['Formulación'].max(), 100)
            fig7.add_scatter(x=x_trend, y=p(x_trend), mode='lines', 
                            name='Tendencia', line=dict(color='red', width=2))
        
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        st.markdown("### Ingresos vs Egresos")
        
        df_plot = df[df['Ingresos Final'] > 0].copy()
        
        fig8 = px.scatter(df_plot, x='Egresos Final', y='Ingresos Final',
                         title="Relación: Egresos ↔ Ingresos",
                         labels={'Egresos Final': 'Egresos ($)', 'Ingresos Final': 'Ingresos ($)'},
                         opacity=0.6)
        
        max_val = max(df_plot['Egresos Final'].max(), df_plot['Ingresos Final'].max())
        fig8.add_scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                        name='Punto Equilibrio (y=x)', line=dict(color='red', width=2, dash='dash'))
        
        st.plotly_chart(fig8, use_container_width=True)
    
    with col3:
        st.markdown("### Balance vs Formulación")
        
        df_plot = df[df['Ingresos Final'] > 0].copy()
        
        fig9 = px.scatter(df_plot, x='Formulación', y='Balance',
                         title="Relación: Formulación → Balance",
                         labels={'Formulación': 'Formulación', 'Balance': 'Balance ($)'},
                         opacity=0.6)
        
        fig9.add_hline(y=0, line_dash="dash", line_color="red")
        
        st.plotly_chart(fig9, use_container_width=True)

# ==================== ANÁLISIS DE CAPACIDAD Y UTILIZACIÓN ====================
st.markdown("---")
st.markdown("## 📦 Análisis de Capacidad y Utilización")

if 'CAPACIDAD' in df.columns and 'FO DIA ACTUAL' in df.columns and '% Utilizacion' in df.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Distribución de % Utilización")
        
        df_capacidad = df[df['CAPACIDAD'] > 0].copy()
        
        if len(df_capacidad) > 0:
            fig_cap1 = px.histogram(df_capacidad, x='% Utilizacion', nbins=40,
                                   title="Distribución de % de Utilización de Capacidad",
                                   labels={'% Utilizacion': '% Utilización'},
                                   color_discrete_sequence=['#3498db'])
            fig_cap1.add_vline(x=80, line_dash="dash", line_color="orange", 
                              annotation_text="80% - Óptimo")
            fig_cap1.add_vline(x=100, line_dash="dash", line_color="red", 
                              annotation_text="100% - Máximo")
            st.plotly_chart(fig_cap1, use_container_width=True)
            
            # Métricas de utilización - AGRUPAR POR CAF PRIMERO
            df_cap_unico = df_capacidad.groupby('CAF')['% Utilizacion'].mean().reset_index()
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                subutilizadas = len(df_cap_unico[df_cap_unico['% Utilizacion'] < 50])
                st.metric("⚠️ Subutilizadas (<50%)", subutilizadas)
            with col_m2:
                optimas = len(df_cap_unico[(df_cap_unico['% Utilizacion'] >= 50) & (df_cap_unico['% Utilizacion'] <= 90)])
                st.metric("✅ Óptimas (50-90%)", optimas)
            with col_m3:
                saturadas = len(df_cap_unico[df_cap_unico['% Utilizacion'] > 90])
                st.metric("🔴 Saturadas (>90%)", saturadas)
    
    with col2:
        st.markdown("### 🔄 Capacidad vs Formulación Actual")
        
        df_cap_plot = df[(df['CAPACIDAD'] > 0) & (df['FO DIA ACTUAL'] > 0)].groupby('CAF').agg({
            'CAPACIDAD': 'mean',
            'FO DIA ACTUAL': 'mean',
            '% Utilizacion': 'mean'
        }).reset_index().nlargest(20, '% Utilizacion')
        
        fig_cap2 = go.Figure()
        fig_cap2.add_trace(go.Bar(
            x=df_cap_plot['CAF'],
            y=df_cap_plot['CAPACIDAD'],
            name='Capacidad Máxima',
            marker=dict(color='#95a5a6')
        ))
        fig_cap2.add_trace(go.Bar(
            x=df_cap_plot['CAF'],
            y=df_cap_plot['FO DIA ACTUAL'],
            name='Formulación Actual',
            marker=dict(color='#e74c3c')
        ))
        
        fig_cap2.update_layout(
            title="Top 20 CAF - Mayor % Utilización",
            xaxis_title="Farmacia",
            yaxis_title="Cantidad de Fórmulas",
            barmode='overlay',
            hovermode='x unified',
            height=400
        )
        fig_cap2.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cap2, use_container_width=True)

# ==================== ANÁLISIS DE CÁNONES ====================
st.markdown("---")
st.markdown("## 💰 Análisis de Cánones de Arrendamiento")

if 'CANON' in df.columns:
    # Verificar si hay datos válidos de canon
    df_canon_valido = df[(df['CANON'].notna()) & (df['CANON'] > 0)]
    
    if len(df_canon_valido) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💵 Top 15 CAF - Mayores Cánones")
            
            df_canon = df_canon_valido.groupby('CAF').agg({
                'CANON': 'mean',
                'Ingresos Final': 'mean',
                'Balance': 'mean'
            }).nlargest(15, 'CANON').reset_index()
            
            fig_canon1 = go.Figure()
            fig_canon1.add_trace(go.Bar(
                x=df_canon['CAF'],
                y=df_canon['CANON'],
                name='Canon Mensual',
                marker=dict(color='#e67e22'),
                yaxis='y'
            ))
            fig_canon1.add_trace(go.Scatter(
                x=df_canon['CAF'],
                y=df_canon['Balance'],
                name='Balance',
                mode='markers+lines',
                marker=dict(size=8, color='#2ecc71'),
                line=dict(width=2),
                yaxis='y2'
            ))
            
            fig_canon1.update_layout(
                height=450,
                xaxis_title="Farmacia",
                yaxis=dict(title="Canon ($)", side='left'),
                yaxis2=dict(title="Balance ($)", overlaying='y', side='right'),
                hovermode='x unified',
                showlegend=True
            )
            fig_canon1.update_xaxes(tickangle=45)
            st.plotly_chart(fig_canon1, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Canon vs Ingresos - Eficiencia")
            
            df_eficiencia = df_canon_valido[df_canon_valido['Ingresos Final'] > 0].copy()
            df_eficiencia['% Canon/Ingresos'] = (df_eficiencia['CANON'] / df_eficiencia['Ingresos Final'] * 100)
            
            if len(df_eficiencia) > 0:
                fig_canon2 = px.scatter(df_eficiencia, 
                                       x='CANON', 
                                       y='Ingresos Final',
                                       title="Relación: Canon → Ingresos",
                                       labels={'CANON': 'Canon Mensual ($)', 'Ingresos Final': 'Ingresos ($)'},
                                       opacity=0.6,
                                       color='% Canon/Ingresos',
                                       color_continuous_scale='RdYlGn_r',
                                       hover_data=['CAF'])
                
                fig_canon2.update_layout(height=450)
                st.plotly_chart(fig_canon2, use_container_width=True)
                
                # Estadísticas de canon
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    canon_promedio = df_eficiencia['CANON'].mean()
                    st.metric("💰 Canon Promedio", f"${canon_promedio/1e6:.2f}M" if canon_promedio >= 1e6 else f"${canon_promedio:,.0f}")
                with col_c2:
                    pct_canon_promedio = df_eficiencia['% Canon/Ingresos'].mean()
                    st.metric("📊 % Canon/Ingresos", f"{pct_canon_promedio:.1f}%")
            else:
                st.info("📊 No hay datos suficientes para mostrar la relación Canon/Ingresos")
    else:
        st.warning("⚠️ No hay datos de CANON disponibles en el dataset")

# ==================== MAPA GEOGRÁFICO ====================
st.markdown("---")
st.markdown("## 🗺️ Distribución Geográfica de Farmacias")

if 'lat' in df.columns and 'lon' in df.columns and 'CIUDAD' in df.columns:
    # Agrupar por ciudad y sumar cánones
    mapa_datos = df.groupby('CIUDAD').agg({
        'CANON': 'sum',
        'Ingresos Final': 'sum',
        'Formulación': 'sum',
        'lat': 'first',
        'lon': 'first',
        'CAF': 'count'
    }).reset_index()
    
    mapa_datos.columns = ['CIUDAD', 'CANON_TOTAL', 'INGRESOS_TOTAL', 'FORMULACION_TOTAL', 'lat', 'lon', 'NUM_CAF']
    
    # Filtrar ciudades sin coordenadas
    mapa_datos = mapa_datos.dropna(subset=['lat', 'lon'])
    
    # Asegurar que CANON_TOTAL sea positivo, usar valor absoluto si es necesario
    mapa_datos['CANON_TOTAL'] = mapa_datos['CANON_TOTAL'].abs()
    
    # Filtrar valores negativos o cero para el tamaño de las burbujas
    mapa_datos = mapa_datos[mapa_datos['CANON_TOTAL'] > 0]
    
    if len(mapa_datos) > 0:
        col_map1, col_map2 = st.columns([2, 1])
        
        with col_map1:
            st.markdown("### 📍 Mapa Interactivo de Cánones por Ciudad")
            
            fig_mapa = px.scatter_mapbox(
                mapa_datos,
                lat='lat',
                lon='lon',
                size='CANON_TOTAL',
                hover_name='CIUDAD',
                hover_data={
                    'CANON_TOTAL': ':,.0f',
                    'INGRESOS_TOTAL': ':,.0f',
                    'FORMULACION_TOTAL': ':,.0f',
                    'NUM_CAF': True,
                    'lat': False,
                    'lon': False
                },
                color='CANON_TOTAL',
                color_continuous_scale='Reds',
                size_max=50,
                zoom=5,
                height=500,
                labels={
                    'CANON_TOTAL': 'Canon Total ($)',
                    'INGRESOS_TOTAL': 'Ingresos Total ($)',
                    'FORMULACION_TOTAL': 'Formulación Total',
                    'NUM_CAF': 'Número de CAF'
                }
            )
            
            fig_mapa.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 0, "l": 0, "b": 0}
            )
            
            st.plotly_chart(fig_mapa, use_container_width=True)
        
        with col_map2:
            st.markdown("### 📊 Top 10 CAF por Canon Individual")
            
            # Mostrar CAF individuales ordenados por Canon
            if 'CANON' in df.columns and 'CAF' in df.columns:
                # Agrupar por CAF y tomar el promedio (debería ser el mismo valor para cada CAF)
                top_caf_canon = df[df['CANON'] > 0].groupby('CAF', as_index=False).agg({
                    'CANON': 'mean',  # Promedio del canon (todos los meses tienen el mismo)
                    'CIUDAD': 'first'
                }).nlargest(10, 'CANON')
                
                top_caf_canon['Canon'] = top_caf_canon['CANON'].apply(lambda x: f"${x/1e6:.2f}M")
                
                st.dataframe(top_caf_canon[['CAF', 'Canon', 'CIUDAD']], use_container_width=True, hide_index=True)
            else:
                st.info("Datos de Canon no disponibles")
            
            st.markdown("### 🏆 Top 5 Ciudades por Formulación")
            
            top_ciudades_form = mapa_datos.nlargest(5, 'FORMULACION_TOTAL')[['CIUDAD', 'FORMULACION_TOTAL', 'NUM_CAF']]
            top_ciudades_form['FORMULACION_TOTAL'] = top_ciudades_form['FORMULACION_TOTAL'].apply(lambda x: f"{x:,.0f}")
            top_ciudades_form.columns = ['Ciudad', 'Formulación', '# CAF']
            
            st.dataframe(top_ciudades_form, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No hay datos geográficos disponibles para el filtro seleccionado")
else:
    st.info("ℹ️ Datos de ubicación geográfica no disponibles en el dataset")

# ==================== FARMACIAS EN RIESGO ALTO ====================
st.markdown("---")
st.markdown("## 🚨 Farmacias en RIESGO ALTO - Requieren Intervención Inmediata")

df_riesgo_alto = df_por_caf[df_por_caf['Prob_Riesgo'] > 0.7].copy()

if len(df_riesgo_alto) > 0:
    df_riesgo_alto = df_riesgo_alto.sort_values('Prob_Riesgo', ascending=False)
    
    # Formatear para display
    df_display = df_riesgo_alto.copy()
    df_display['Prob_Riesgo'] = (df_display['Prob_Riesgo'] * 100).round(1).astype(str) + "%"
    df_display['Margin_Ganancia'] = df_display['Margin_Ganancia'].round(2)
    df_display['Ingresos Final'] = df_display['Ingresos Final'].apply(lambda x: f"${x/1e6:.2f}M")
    df_display['Egresos Final'] = df_display['Egresos Final'].apply(lambda x: f"${x/1e6:.2f}M")
    df_display['Balance'] = df_display['Balance'].apply(lambda x: f"${x/1e6:.2f}M")
    df_display['Formulación'] = df_display['Formulación'].round(0)
    
    tabla_riesgo = df_display[['CAF', 'Prob_Riesgo', 'Margin_Ganancia', 'Ingresos Final', 'Egresos Final', 'Balance', 'Formulación']]
    tabla_riesgo.columns = ['CAF', 'Prob. Cierre', 'Margen %', 'Ingresos', 'Egresos', 'Balance', 'Formulación']
    
    st.dataframe(tabla_riesgo.head(20), use_container_width=True, hide_index=True)
    
    st.warning(f"⚠️ **{len(df_riesgo_alto)} farmacias en riesgo crítico detectadas.** Recomendación: Revisar estructura de costos, renegociar cánones y estrategias de aumento de formulación.")
else:
    st.success("✅ No hay farmacias en riesgo alto por el momento")

# ==================== SEGMENTACIÓN ESTRATÉGICA ====================
st.markdown("---")
st.markdown("## 🎯 Segmentación Estratégica de Farmacias")

# Inicializar df_segment
df_segment = df_por_caf.copy()

if 'Margin_Ganancia' in df_por_caf.columns and 'Formulación' in df_por_caf.columns:
    # Crear segmentos basados en margen y formulación
    df_segment = df_por_caf.copy()
    
    # Umbrales
    margen_medio = df_segment['Margin_Ganancia'].median()
    formula_medio = df_segment['Formulación'].median()
    
    # Clasificación en 4 cuadrantes
    def clasificar_farmacia(row):
        if row['Margin_Ganancia'] >= margen_medio and row['Formulación'] >= formula_medio:
            return '🌟 Estrellas (Alto Margen + Alta Form.)'
        elif row['Margin_Ganancia'] >= margen_medio and row['Formulación'] < formula_medio:
            return '💎 Premium (Alto Margen + Baja Form.)'
        elif row['Margin_Ganancia'] < margen_medio and row['Formulación'] >= formula_medio:
            return '📈 Volumen (Bajo Margen + Alta Form.)'
        else:
            return '⚠️ Riesgo (Bajo Margen + Baja Form.)'
    
    df_segment['Segmento'] = df_segment.apply(clasificar_farmacia, axis=1)
    
    # Usar valor absoluto de Ingresos para el tamaño de las burbujas
    df_segment['Ingresos_Size'] = df_segment['Ingresos Final'].abs() + 1  # +1 para evitar ceros
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Matriz de Segmentación: Margen vs Formulación")
        
        fig_segment = px.scatter(df_segment,
                                x='Formulación',
                                y='Margin_Ganancia',
                                color='Segmento',
                                size='Ingresos_Size',
                                hover_data=['CAF', 'Balance', 'Prob_Riesgo'],
                                title="Segmentación Estratégica de Farmacias",
                                labels={'Formulación': 'Formulación Promedio',
                                       'Margin_Ganancia': 'Margen de Ganancia (%)'},
                                color_discrete_map={
                                    '🌟 Estrellas (Alto Margen + Alta Form.)': '#2ecc71',
                                    '💎 Premium (Alto Margen + Baja Form.)': '#3498db',
                                    '📈 Volumen (Bajo Margen + Alta Form.)': '#f39c12',
                                    '⚠️ Riesgo (Bajo Margen + Baja Form.)': '#e74c3c'
                                })
        
        fig_segment.add_hline(y=margen_medio, line_dash="dash", line_color="gray",
                             annotation_text="Margen Mediano")
        fig_segment.add_vline(x=formula_medio, line_dash="dash", line_color="gray",
                             annotation_text="Formulación Mediana")
        
        fig_segment.update_layout(height=500)
        st.plotly_chart(fig_segment, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Distribución por Segmento")
        
        segmento_counts = df_segment['Segmento'].value_counts().reset_index()
        segmento_counts.columns = ['Segmento', 'Cantidad']
        
        fig_pie = px.pie(segmento_counts,
                        values='Cantidad',
                        names='Segmento',
                        title="Distribución de CAFs por Segmento",
                        color='Segmento',
                        color_discrete_map={
                            '🌟 Estrellas (Alto Margen + Alta Form.)': '#2ecc71',
                            '💎 Premium (Alto Margen + Baja Form.)': '#3498db',
                            '📈 Volumen (Bajo Margen + Alta Form.)': '#f39c12',
                            '⚠️ Riesgo (Bajo Margen + Baja Form.)': '#e74c3c'
                        })
        
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Recomendaciones por segmento
    st.markdown("### 💡 Recomendaciones Estratégicas por Segmento")
    
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    
    with col_r1:
        count_estrellas = len(df_segment[df_segment['Segmento'] == '🌟 Estrellas (Alto Margen + Alta Form.)'])
        st.markdown(f"#### 🌟 Estrellas ({count_estrellas})")
        st.success("**✓ Mantener y potenciar**")
        st.markdown("""
        - Modelo exitoso
        - Replicar estrategia
        - Inversión en marketing
        - Mantener calidad
        """)
    
    with col_r2:
        count_premium = len(df_segment[df_segment['Segmento'] == '💎 Premium (Alto Margen + Baja Form.)'])
        st.markdown(f"#### 💎 Premium ({count_premium})")
        st.info("**⬆ Aumentar volumen**")
        st.markdown("""
        - Margen saludable
        - Aumentar formulación
        - Marketing local
        - Alianzas EPS
        """)
    
    with col_r3:
        count_volumen = len(df_segment[df_segment['Segmento'] == '📈 Volumen (Bajo Margen + Alta Form.)'])
        st.markdown(f"#### 📈 Volumen ({count_volumen})")
        st.warning("**💰 Optimizar costos**")
        st.markdown("""
        - Alto tráfico
        - Reducir costos fijos
        - Renegociar canon
        - Mejorar eficiencia
        """)
    
    with col_r4:
        count_riesgo = len(df_segment[df_segment['Segmento'] == '⚠️ Riesgo (Bajo Margen + Baja Form.)'])
        st.markdown(f"#### ⚠️ Riesgo ({count_riesgo})")
        st.error("**🚨 Intervención urgente**")
        st.markdown("""
        - Reestructuración
        - Evaluar viabilidad
        - Considerar cierre
        - Cambio de ubicación
        """)

# ==================== TABLA RESUMEN EJECUTIVA ====================
st.markdown("---")
st.markdown("## 📋 Resumen Ejecutivo - Top Y Bottom Farmacias")

if 'CAF' in df.columns:
    resumen = df.groupby('CAF').agg({
        'Ingresos Final': 'sum',
        'Egresos Final': 'sum',
        'Formulación': 'sum',
        'Balance': 'sum',
        'Ratio_Egresos_Ingresos': 'mean',
        'Margin_Ganancia': 'mean'
    }).round(2)
    
    # Clasificación por Margen de Ganancia (más crítico que ratio)
    resumen['Estado'] = resumen['Margin_Ganancia'].apply(
        lambda x: '🔴 ALERTA' if x < 0 else ('🟡 CUIDADO' if x < 2 else '🟢 OK')
    )
    
    # Ordenar por Margen % (rentabilidad) en lugar de Balance (volumen)
    resumen = resumen.sort_values('Margin_Ganancia', ascending=False)
    
    tab1, tab2 = st.tabs(["🏆 Top 10 - Mejor Desempeño (Margen %)", "📉 Bottom 10 - Menor Desempeño (Margen %)"])
    
    with tab1:
        top_10 = resumen.head(10).copy()
        top_10 = top_10[['Ingresos Final', 'Egresos Final', 'Balance', 'Formulación', 'Margin_Ganancia', 'Estado']]
        top_10.columns = ['Ingresos', 'Egresos', 'Balance', 'Formulación', 'Margen %', 'Estado']
        
        # Formatear columnas
        top_10['Ingresos'] = top_10['Ingresos'].apply(lambda x: f"${x:,.2f}")
        top_10['Egresos'] = top_10['Egresos'].apply(lambda x: f"${x:,.2f}")
        top_10['Balance'] = top_10['Balance'].apply(lambda x: f"${x:,.2f}")
        top_10['Margen %'] = top_10['Margen %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(top_10, use_container_width=True)
    
    with tab2:
        bottom_10 = resumen.tail(10).copy()
        bottom_10 = bottom_10[['Ingresos Final', 'Egresos Final', 'Balance', 'Formulación', 'Margin_Ganancia', 'Estado']]
        bottom_10.columns = ['Ingresos', 'Egresos', 'Balance', 'Formulación', 'Margen %', 'Estado']
        
        # Formatear columnas
        bottom_10['Ingresos'] = bottom_10['Ingresos'].apply(lambda x: f"${x:,.2f}")
        bottom_10['Egresos'] = bottom_10['Egresos'].apply(lambda x: f"${x:,.2f}")
        bottom_10['Balance'] = bottom_10['Balance'].apply(lambda x: f"${x:,.2f}")
        bottom_10['Margen %'] = bottom_10['Margen %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(bottom_10, use_container_width=True)

# ==================== INSIGHTS CLAVE Y CONCLUSIONES ====================
st.markdown("---")
st.markdown("## 💡 Insights Clave y Conclusiones")

col_i1, col_i2, col_i3 = st.columns(3)

with col_i1:
    st.markdown("### 🔍 Hallazgos Principales")
    
    if 'Margin_Ganancia' in df_por_caf.columns:
        pct_margen_negativo = (len(df_por_caf[df_por_caf['Margin_Ganancia'] < 0]) / len(df_por_caf) * 100)
        st.markdown(f"""
        - **{pct_margen_negativo:.1f}%** de las farmacias tienen margen negativo
        - **{len(df_por_caf[df_por_caf['Prob_Riesgo'] > 0.7])}** farmacias en riesgo crítico (>70%)
        - **{len(df_por_caf[df_por_caf['Prob_Riesgo'] < 0.4])}** farmacias con bajo riesgo (<40%)
        """)
    
    if 'Ingresos Final' in df.columns:
        st.markdown(f"""
        - Ingresos totales: **${df['Ingresos Final'].sum()/1e9:.2f}B**
        - Balance total: **${df['Balance'].sum()/1e9:.2f}B**
        """)

with col_i2:
    st.markdown("### 🎯 Factores Críticos de Riesgo")
    
    st.markdown("""
    **Según el modelo ML, los factores más influyentes son:**
    """)
    
    if len(feature_imp) > 0:
        for idx, row in feature_imp.head(3).iterrows():
            importancia_pct = row['Importance'] * 100
            st.markdown(f"- **{row['Feature']}**: {importancia_pct:.1f}% de importancia")
    
    st.markdown("""
    
    **Principales causas de riesgo:**
    - Egresos superiores a ingresos
    - Baja formulación (<500)
    - Balance negativo sostenido
    - Alta relación Egresos/Ingresos
    """)

with col_i3:
    st.markdown("### 🚀 Recomendaciones Prioritarias")
    
    st.markdown("""
    **1. Corto Plazo (0-3 meses):**
    - Intervenir farmacias en riesgo crítico
    - Auditar egresos de CAFs con margen negativo
    - Renegociar cánones altos
    
    **2. Mediano Plazo (3-6 meses):**
    - Optimizar rutas de distribución
    - Implementar KPIs semanales
    - Aumentar formulación en CAFs subutilizadas
    
    **3. Largo Plazo (6-12 meses):**
    - Cerrar o reubicar CAFs no viables
    - Expandir modelo de CAFs "Estrella"
    - Implementar sistema predictivo continuo
    """)

# Descargar datos
st.markdown("---")
st.markdown("### 📥 Exportar Datos")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    # CSV de farmacias en riesgo alto
    if len(df_riesgo_alto) > 0:
        csv_riesgo = df_riesgo_alto.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Descargar CAFs en Riesgo Alto (CSV)",
            data=csv_riesgo,
            file_name="farmacias_riesgo_alto.csv",
            mime="text/csv"
        )

with col_d2:
    # CSV de resumen por CAF
    if 'CAF' in df_por_caf.columns:
        csv_resumen = df_por_caf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📋 Descargar Resumen por CAF (CSV)",
            data=csv_resumen,
            file_name="resumen_por_caf.csv",
            mime="text/csv"
        )

with col_d3:
    # CSV de segmentación
    if 'Segmento' in df_segment.columns:
        csv_segment = df_segment.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🎯 Descargar Segmentación (CSV)",
            data=csv_segment,
            file_name="segmentacion_farmacias.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption("Dashboard automáticamente limpio y actualizado | Marzo 2026")
