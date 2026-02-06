import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
from shapely.geometry import Point
import re
import io

st.set_page_config(page_title="Sistema de Inteligencia Catemaco 2025", layout="wide", initial_sidebar_state="expanded")

# --- 1. MOTOR DE PROCESAMIENTO ROBUSTO ---

def clean_duplicates(df):
    """Elimina columnas duplicadas que causan conflicto."""
    return df.loc[:, ~df.columns.duplicated()].copy()

def safe_numeric(series):
    """Convierte columnas sucias (con *, N/D, texto) a números sumables."""
    # Convertir a string, quitar caracteres no numéricos excepto punto y negativo
    s_clean = series.astype(str).replace(r'[^\d\.-]', '', regex=True)
    # Convertir vacíos a 0
    s_clean = s_clean.replace('', '0')
    return pd.to_numeric(s_clean, errors='coerce').fillna(0)

def parse_coords(val):
    """Interpreta coordenadas en decimal o grados minutos segundos."""
    if pd.isna(val) or val == "": return None
    try:
        val_str = str(val).strip()
        # Caso decimal directo
        if re.match(r'^[-+]?\d*\.?\d+$', val_str):
            return float(val_str)
        # Caso DMS
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
        if len(nums) >= 3:
            d, m, s = float(nums[0]), float(nums[1]), float(nums[2])
            dec = d + m/60 + s/3600
            if any(c in val_str.upper() for c in ['W', 'O', 'S', '-']):
                dec = -dec
            return dec
    except:
        return None
    return None

def calculate_full_indicators(df):
    """
    Mapeo masivo con SUMA INTELIGENTE.
    Si una variable está en 'POBMAS' (Urbano) y 'HOMBRES' (Rural), suma ambas columnas.
    """
    
    def sum_variants(candidates):
        # Crea una serie de ceros del tamaño del dataframe
        total = pd.Series(0.0, index=df.index)
        for col in candidates:
            if col in df.columns:
                # Suma acumulativa reemplazando NaN con 0
                total = total.add(safe_numeric(df[col]), fill_value=0)
        return total

    # --- DEMOGRAFÍA ---
    df['Población Total'] = sum_variants(['POBTOT', 'P_TOTAL', 'POB_TOT'])
    df['Hombres'] = sum_variants(['POBMAS', 'HOMBRES', 'P_MAS'])
    df['Mujeres'] = sum_variants(['POBFEM', 'MUJERES', 'P_FEM'])
    
    # Grupos de Edad
    # Intentamos sumar totales directos (POB0_14 + NINOS_0_14)
    # Si da 0, sumamos los desagregados
    pob_0_14 = sum_variants(['POB0_14', 'NINOS_0_14'])
    if pob_0_14.sum() == 0:
        pob_0_14 = sum_variants(['P_0A2', 'P_0A2_F', 'P_0A2_M', 'P_3A5', 'P_6A11', 'P_12A14'])
    df['0-14 años'] = pob_0_14
    
    df['15-64 años'] = sum_variants(['POB15_64', 'P_15A64'])
    if df['15-64 años'].sum() == 0:
        df['15-64 años'] = sum_variants(['P_15A17', 'P_18A24', 'P_15A49_F']) * 2.1 # Estimación si falta dato directo
        
    df['65+ años'] = sum_variants(['POB65_MAS', 'P_60YMAS', 'ADULTOS_MAYORES'])
    
    # Migración
    df['Nacidos en la Entidad'] = sum_variants(['PNACENT'])
    df['Nacidos en Otra Entidad'] = sum_variants(['PNACOE'])
    
    # --- SOCIAL / CULTURA ---
    df['Pob. Indígena (3+ años)'] = sum_variants(['P3YM_HLI', 'POB_INDIGENA', 'PHOG_IND'])
    df['Pob. Afrodescendiente'] = sum_variants(['POB_AFRO', 'P_AFRO'])
    df['Habla Indígena'] = sum_variants(['P3HLI_HE', 'P3HLINHE'])
    
    # Religión
    df['Católica'] = sum_variants(['PCATOLICA'])
    df['Protestante/Evangélica'] = sum_variants(['PRO_CRIEVA'])
    df['Sin Religión'] = sum_variants(['PSIN_RELIG'])
    
    # --- SALUD / DISCAPACIDAD ---
    # Aquí unificamos nombres urbanos y rurales
    df['Discapacidad Total'] = sum_variants(['PCON_DISC', 'POB_DISCAPACITADA', 'P_DISCAP', 'DISCAPACIDAD_TOTAL'])
    df['Disc. Motriz'] = sum_variants(['PCDISC_MOT', 'PCDISC_MOT2'])
    df['Disc. Visual'] = sum_variants(['PCDISC_VIS'])
    df['Disc. Auditiva'] = sum_variants(['PCDISC_AUD'])
    df['Disc. Mental'] = sum_variants(['PCDISC_MEN', 'PCDISC_MEN2'])
    
    df['Con Derecho a Salud'] = sum_variants(['PDER_SS', 'CARENCIA_SALUD']) # Ajuste: suma lo disponible
    df['Afiliados IMSS'] = sum_variants(['PDER_IMSS'])
    df['Afiliados ISSSTE'] = sum_variants(['PDER_ISTE'])

    # --- EDUCACIÓN ---
    df['Analfabetas (15+)'] = sum_variants(['P15YM_AN', 'CARENCIA_EDU_AN'])
    df['Educ. Básica Incompleta'] = sum_variants(['P15YM_SE', 'CARENCIA_EDU_SE', 'P15YM_PI'])
    df['Asisten a Escuela (3-14)'] = sum_variants(['P3A14_ASISTE']) # Variable calculada inversa si existe
    df['Grado Promedio Escolaridad'] = sum_variants(['GRAPROES', 'GRA_PRO_ES']) # Promedio ponderado visual
    
    # --- ECONOMÍA ---
    df['Pob. Econ. Activa (PEA)'] = sum_variants(['PEA', 'P_ECON_ACT'])
    df['Pob. Ocupada'] = sum_variants(['POCUPADA', 'P_OCUPADA'])
    df['Pob. Desocupada'] = sum_variants(['PDESOCUP'])

    # --- VIVIENDA ---
    df['Viviendas Totales'] = sum_variants(['VIVTOT', 'TOTAL_VIVIENDAS'])
    df['Con Agua Entubada'] = sum_variants(['VPH_AGUADV'])
    df['Con Drenaje'] = sum_variants(['VPH_DRENAJ'])
    df['Con Electricidad'] = sum_variants(['VPH_C_ELEC'])
    df['Con Internet'] = sum_variants(['VPH_INTER'])
    df['Con Computadora'] = sum_variants(['VPH_PC'])
    df['Con Celular'] = sum_variants(['VPH_CEL'])
    df['Con Automóvil'] = sum_variants(['VPH_AUTOM'])
    df['Piso de Tierra'] = sum_variants(['VPH_PISOTI'])
    df['Con Refrigerador'] = sum_variants(['VPH_REFRI'])
    df['Con Lavadora'] = sum_variants(['VPH_LAVAD'])
    
    # Proyección 2025
    r = 0.00175
    df['Población 2025 (Est)'] = df['Población Total'] * ((1 + r)**5)
    
    return df

# --- FUNCIÓN DE TABLA DETALLADA (CORREGIDA PARA QUE NO BORRE FILAS) ---
def generar_tabla_localidades(df):
    """
    Agrupa Urbano en 1 fila.
    Deja Rural desglosado (241 filas).
    """
    # CORRECCIÓN: Usar las variables CALCULADAS (ej 'Pob. Econ. Activa (PEA)') en vez de las crudas
    cols_interes = [
        'Población Total', 'Población 2025 (Est)', 'Hombres', 'Mujeres', 
        '0-14 años', '65+ años', 'Pob. Indígena (3+ años)', 'Discapacidad Total',
        'Analfabetas (15+)', 'Pob. Econ. Activa (PEA)', 'Viviendas Totales', 'Con Agua Entubada', 
        'Con Internet', 'Con Automóvil'
    ]
    
    # Asegurar que existan
    cols = [c for c in cols_interes if c in df.columns]

    # Separar Urbano y Rural
    urb = df[df['AMBITO'] == 'Urbano'].copy()
    rur = df[df['AMBITO'] == 'Rural'].copy()

    # 1. Procesar Urbano (Agrupar en 1 fila)
    if not urb.empty:
        urb_row = urb[cols].sum().to_frame().T
        urb_row['NOM_LOC'] = 'Catemaco (Cabecera Municipal)'
        urb_row['AMBITO'] = 'Urbano'
        urb_row['Distancia_Laguna_KM'] = 0.0
    else:
        urb_row = pd.DataFrame()

    # 2. Procesar Rural (Mantener desglose)
    if not rur.empty:
        if 'NOM_LOC' not in rur.columns: rur['NOM_LOC'] = 'Localidad Rural'
        
        # CORRECCIÓN CRÍTICA: Agrupar por ID (LOC) para que no se fusionen localidades distintas
        # Si 'LOC' no está, usamos índice o un identificador único
        group_cols = ['NOM_LOC', 'AMBITO']
        if 'LOC' in rur.columns:
            group_cols.insert(0, 'LOC')
        elif 'CVE_LOC' in rur.columns:
            group_cols.insert(0, 'CVE_LOC')
            
        rur_grp = rur.groupby(group_cols)[cols].sum().reset_index()
        
        # Recuperar distancia (promedio o mínima por localidad)
        dist_df = rur.groupby('NOM_LOC')['Distancia_Laguna_KM'].min().reset_index()
        rur_grp = rur_grp.merge(dist_df, on='NOM_LOC', how='left')
    else:
        rur_grp = pd.DataFrame()

    # Unir
    tabla_final = pd.concat([urb_row, rur_grp], ignore_index=True)
    
    # Ordenar por población descendente
    if 'Población Total' in tabla_final.columns:
        tabla_final = tabla_final.sort_values('Población Total', ascending=False)
        
    return tabla_final


# --- 2. CARGA DE DATOS (CORREGIDA PARA USAR ARCHIVO GRANDE) ---
@st.cache_data
def load_data():
    try:
        # URBANO
        u = gpd.read_file('sits_urbano_fase2.geojson')
        u = clean_duplicates(u)
        u['AMBITO'] = 'Urbano'
        
        # RURAL
        # CORRECCIÓN: Intentamos cargar primero sits_capa_rural (241 items)
        # Si falla, intentamos sits_rural_fase2 (38 items)
        try:
            r = gpd.read_file('sits_capa_rural.geojson')
            # Verificación rápida: si tiene muy pocos datos, intentamos el otro
            if len(r) < 50:
                 r_alt = gpd.read_file('sits_rural_fase2.geojson')
                 if len(r_alt) > len(r):
                     r = r_alt
        except:
            r = gpd.read_file('sits_rural_fase2.geojson')
            
        r = clean_duplicates(r)
        r['AMBITO'] = 'Rural'
        
        # Reparar geometrías rurales para el mapa (sin borrar filas)
        if 'LATITUD' in r.columns:
            r['lat_f'] = r['LATITUD'].apply(parse_coords)
            r['lon_f'] = r['LONGITUD'].apply(parse_coords)
            
            # Solo asignamos punto donde hay coordenadas
            mask = r['lat_f'].notnull() & r['lon_f'].notnull()
            geoms = [Point(x, y) for x,y in zip(r.loc[mask, 'lon_f'], r.loc[mask, 'lat_f'])]
            r.loc[mask, 'geometry'] = gpd.GeoSeries(geoms, index=r[mask].index)
        
        # CONCATENAR
        df = pd.concat([u, r], ignore_index=True)
        
        # CALCULAR VARIABLES
        df = calculate_full_indicators(df)
        
        # DISTANCIA LAGUNA
        lake = Point(-95.1000, 18.4166)
        def get_d(g):
            if g is None or g.is_empty: return np.nan
            return g.centroid.distance(lake) * 111.1
        
        df['Distancia_Laguna_KM'] = df.geometry.apply(get_d)
        
        return df
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return pd.DataFrame()

# --- 3. INTERFAZ ---
df_all = load_data()

if not df_all.empty:
    # --- SIDEBAR ---
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Laguna_de_Catemaco.jpg/280px-Laguna_de_Catemaco.jpg", caption="Laguna de Catemaco")
    st.sidebar.header("🎛️ Filtros Avanzados")
    
    sel_ambito = st.sidebar.multiselect("Ámbito Geográfico", ["Urbano", "Rural"], default=["Urbano", "Rural"])
    
    use_dist = st.sidebar.toggle("Filtrar por Cercanía a Laguna", False)
    dist_max = 5.0
    if use_dist:
        dist_max = st.sidebar.slider("Radio (Km)", 0.5, 20.0, 5.0)
        
    no_duplicados = st.sidebar.checkbox("Excluir Cabecera (Evitar duplicidad)", True)

    # --- FILTRADO ---
    df_v = df_all.copy()
    
    # 1. Ámbito
    df_v = df_v[df_v['AMBITO'].isin(sel_ambito)]
    # 2. Distancia
    if use_dist:
        df_v = df_v[df_v['Distancia_Laguna_KM'] <= dist_max]
    # 3. Duplicados (Cabecera Rural)
    if no_duplicados:
        # Excluir si es Rural y su LOC es 1, 01, 0001
        df_v = df_v[~((df_v['AMBITO']=='Rural') & (df_v.get('LOC', '').astype(str).str.contains(r'^(0*1)$')))]

    # --- GENERAR MATRIZ DETALLADA (NUEVO) ---
    df_matriz = generar_tabla_localidades(df_v)

    # --- MAIN DASHBOARD ---
    st.title("📊 Sistema de Inteligencia Territorial Catemaco 2025")
    st.markdown(f"**Registros Analizados:** {len(df_v)} | **Filtros:** {', '.join(sel_ambito)} | **Radio:** {'Sin filtro' if not use_dist else f'{dist_max} km'}")

    # KPI PRINCIPALES
    k1, k2, k3, k4 = st.columns(4)
    pop_tot = df_v['Población Total'].sum()
    k1.metric("Población Total (2020)", f"{pop_tot:,.0f}")
    k2.metric("Proyección 2025", f"{df_v['Población 2025 (Est)'].sum():,.0f}")
    k3.metric("Pob. Indígena", f"{df_v['Pob. Indígena (3+ años)'].sum():,.0f}")
    k4.metric("Viviendas Habitadas", f"{df_v['Viviendas Totales'].sum():,.0f}")

    st.markdown("---")

    # --- GENERADOR DE TABLAS RESUMEN ---
    
    def create_summary(columns_list):
        # Agrupa por ámbito y suma
        summary = df_v.groupby('AMBITO')[columns_list].sum().T
        # Columna Total
        summary['TOTAL'] = summary.sum(axis=1)
        return summary

    # PESTAÑAS TEMÁTICAS (AQUÍ AGREGAMOS LA NUEVA PESTAÑA AL FINAL)
    tabs = st.tabs(["📋 Demografía", "🏥 Salud y Discapacidad", "💰 Economía y Educación", "🏠 Vivienda y Servicios", "🗺️ Mapa", "📍 Desglose por Localidad"])

    with tabs[0]: # DEMOGRAFIA
        st.subheader("Perfil Demográfico")
        cols_demo = ['Población Total', 'Hombres', 'Mujeres', '0-14 años', '15-64 años', '65+ años', 
                     'Nacidos en la Entidad', 'Nacidos en Otra Entidad', 
                     'Pob. Indígena (3+ años)', 'Pob. Afrodescendiente', 'Católica', 'Protestante/Evangélica']
        df_demo = create_summary(cols_demo)
        st.dataframe(df_demo.style.format("{:,.0f}"), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(names=['Hombres', 'Mujeres'], values=[df_v['Hombres'].sum(), df_v['Mujeres'].sum()], title="Sexo"), use_container_width=True)
        c2.plotly_chart(px.bar(df_demo.drop('Población Total').reset_index(), x='index', y='TOTAL', title="Indicadores Clave"), use_container_width=True)

    with tabs[1]: # SALUD
        st.subheader("Salud y Grupos Vulnerables")
        cols_salud = ['Con Derecho a Salud', 'Afiliados IMSS', 'Afiliados ISSSTE',
                      'Discapacidad Total', 'Disc. Motriz', 'Disc. Visual', 'Disc. Auditiva', 'Disc. Mental']
        st.dataframe(create_summary(cols_salud).style.format("{:,.0f}"), use_container_width=True)

    with tabs[2]: # ECONOMIA
        st.subheader("Economía y Educación")
        cols_eco = ['Pob. Econ. Activa (PEA)', 'Pob. Ocupada', 'Pob. Desocupada',
                    'Analfabetas (15+)', 'Educ. Básica Incompleta']
        st.dataframe(create_summary(cols_eco).style.format("{:,.0f}"), use_container_width=True)
        st.info(f"Grado Promedio de Escolaridad (Ponderado Global): {df_v['Grado Promedio Escolaridad'].mean():.2f} años")

    with tabs[3]: # VIVIENDA
        st.subheader("Características de la Vivienda")
        cols_viv = ['Viviendas Totales', 'Con Agua Entubada', 'Con Drenaje', 'Con Electricidad', 
                    'Con Internet', 'Con Automóvil', 'Piso de Tierra', 'Con Refrigerador', 'Con Lavadora']
        st.dataframe(create_summary(cols_viv).style.format("{:,.0f}"), use_container_width=True)

    with tabs[4]: # MAPA
        st.subheader("Distribución Geoespacial")
        # Mapa de Calor
        map_df = df_v[df_v['Distancia_Laguna_KM'].notnull()].copy()
        if not map_df.empty:
            map_df['lat'] = map_df.geometry.centroid.y
            map_df['lon'] = map_df.geometry.centroid.x
            st.map(map_df, size='Población Total', color='#FF4B4B')
            
            with st.expander("Ver lista de localidades en mapa"):
                st.dataframe(map_df[['NOM_LOC', 'AMBITO', 'Población Total', 'Distancia_Laguna_KM']].sort_values('Población Total', ascending=False))
        else:
            st.warning("No hay localidades con coordenadas válidas en esta selección.")

    with tabs[5]: # NUEVA PESTAÑA: DESGLOSE
        st.subheader("Matriz Detallada de Localidades")
        st.markdown("**Catemaco (Cabecera Municipal)** agrupa toda la zona urbana. Las demás filas son localidades rurales individuales.")
        st.dataframe(
            df_matriz.style.format({
                'Población Total': '{:,.0f}',
                'Población 2025 (Est)': '{:,.0f}',
                'Pob. Econ. Activa (PEA)': '{:,.0f}', # CORREGIDO: USA EL DATO CALCULADO
                'Distancia_Laguna_KM': '{:.2f} km'
            }), 
            height=600, 
            use_container_width=True
        )

    # --- DESCARGAS ---
    st.markdown("### 📥 Exportar Información")
    
    # Preparar Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_v.to_excel(writer, sheet_name='Base_Completa', index=False)
        create_summary(cols_demo).to_excel(writer, sheet_name='Resumen_Demografico')
        create_summary(cols_salud).to_excel(writer, sheet_name='Resumen_Salud')
        create_summary(cols_eco).to_excel(writer, sheet_name='Resumen_Economia')
        create_summary(cols_viv).to_excel(writer, sheet_name='Resumen_Vivienda')
        # NUEVA HOJA EN EXCEL
        df_matriz.to_excel(writer, sheet_name='Desglose_Por_Localidad', index=False)
        
    st.download_button(
        label="📥 Descargar Reporte Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="Reporte_Catemaco_2025.xlsx",
        mime="application/vnd.ms-excel"
    )
    
    st.download_button(
        label="📄 Descargar Datos CSV",
        data=df_v.to_csv(index=False).encode('utf-8'),
        file_name="datos_catemaco_full.csv",
        mime="text/csv"
    )

else:
    st.error("No se pudieron cargar los datos. Verifica que los archivos .geojson estén en la carpeta.")