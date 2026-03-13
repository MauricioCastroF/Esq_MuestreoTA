import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import base64
import json
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import simplekml
from datetime import datetime, timedelta
from fpdf import FPDF
import tempfile

# 1. ESTÉTICA "INGENIO PRO" - ANCLAJE ESTRUCTURAL
st.set_page_config(page_title="GIS Agro-MultiBatch Pro", page_icon="🛰️", layout="wide")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #0f172a 0%, #020617 100%); }
    .stMetric { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 15px; padding: 20px; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; padding: 10px 25px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #10b981; color: white; font-weight: bold; }
    .header-anchor { padding-top: 1rem; padding-bottom: 2rem; text-align: center; }
    div.stButton > button { background: linear-gradient(90deg, #10b981 0%, #059669 100%); border: none; border-radius: 10px; color: white; font-weight: 700; height: 3.5rem; }
    div.stDownloadButton > button { background: rgba(59, 130, 246, 0.1); border: 1px solid #3b82f6; border-radius: 10px; color: #3b82f6; height: 3.5rem; }
    </style>
    """, unsafe_allow_html=True)

# Encabezado Fijo
st.markdown("<div class='header-anchor'><h1 style='color: white; font-weight: 800; margin:0;'>🛰️ Estimación de Rendimiento</h1><p style='color: #10b981; font-size: 1.1rem; margin-top:5px;'>Esquema de Muestreo | Distribución por Cuantiles</p></div>", unsafe_allow_html=True)

# 2. AUTENTICACIÓN
def authenticate_gee():
    try:
        b64_str = os.environ.get("GEE_JSON_B64") or st.secrets.get("GEE_JSON_B64")
        if not b64_str: return False, "Falta Llave GEE"
        b64_str = b64_str.strip().strip('"').strip("'")
        missing_padding = len(b64_str) % 4
        if missing_padding: b64_str += '=' * (4 - missing_padding)
        json_key = json.loads(base64.b64decode(b64_str).decode('utf-8'))
        if 'private_key' in json_key: json_key['private_key'] = json_key['private_key'].replace('\\n', '\n')
        credentials = ee.ServiceAccountCredentials(json_key['client_email'], key_data=json_key['private_key'])
        ee.Initialize(credentials=credentials)
        ee.data._credentials = credentials 
        return True, "Conectado"
    except Exception as e: return False, str(e)

# 3. LÓGICA DE PROCESAMIENTO
def mask_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

def calculate_indices(image):
    ci_green = image.expression('(B8 / B3) - 1', {'B8': image.select('B8'), 'B3': image.select('B3')}).rename('CIgreen')
    return image.addBands(ci_green)

def main():
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status, st.session_state.auth_msg = authenticate_gee()

    with st.sidebar:
        st.markdown("<h2 style='color: #10b981;'>Control de Proyecto</h2>", unsafe_allow_html=True)
        if st.session_state.auth_status: st.success("🟢 Satélite En Línea")
        uploaded_file = st.file_uploader("📂 Cargar Lotes (KML/GeoJSON)", type=['geojson', 'kml'])
        n_clusters = st.select_slider("Segmentos (Cuantiles) por Lote", options=range(2, 11), value=5)
        date_range = st.date_input("Ventana Temporal", [datetime.now() - timedelta(days=90), datetime.now()])

    if uploaded_file and st.session_state.auth_status:
        try:
            gdf = gpd.read_file(uploaded_file)
            if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
            
            lote_names = [gdf.iloc[i].get('name') or gdf.iloc[i].get('Name') or f"Lote_{i+1}" for i in range(len(gdf))]
            selected_lote_name = st.selectbox("🎯 Seleccionar Lote para Visualizar", lote_names)
            selected_idx = lote_names.index(selected_lote_name)
            
            row = gdf.iloc[selected_idx]
            ee_geom = ee.Geometry(row.geometry.__geo_interface__)
            ee_geom_buffered = ee_geom.buffer(-50)
            
            with st.spinner(f"Analizando distribución por cuantiles para {selected_lote_name}..."):
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                             .filterBounds(ee_geom)
                             .filterDate(date_range[0].strftime('%Y-%m-%d'), date_range[1].strftime('%Y-%m-%d'))
                             .filter(ee.Filter.contains(rightValue=ee_geom, leftField='.geo'))
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                             .sort('CLOUDY_PIXEL_PERCENTAGE').limit(2))

                img_display = collection.map(mask_clouds).map(calculate_indices).select('CIgreen').median().clip(ee_geom)
                stats_info = img_display.reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 10).getInfo()
                v_min, v_max = stats_info.get('CIgreen_p2', 0), stats_info.get('CIgreen_p98', 4)

                img_sampling = img_display.clip(ee_geom_buffered)
                sample = img_sampling.sample(region=ee_geom_buffered, scale=10, numPixels=2500, geometries=True).getInfo()
                df_data = pd.DataFrame({
                    'CIgreen': [f['properties']['CIgreen'] for f in sample['features']],
                    'lon': [f['geometry']['coordinates'][0] for f in sample['features']],
                    'lat': [f['geometry']['coordinates'][1] for f in sample['features']]
                }).dropna()

                lote_avg = df_data['CIgreen'].mean()
                quant_levels = np.linspace(0, 1, n_clusters + 1)
                bin_edges = df_data['CIgreen'].quantile(quant_levels).values
                
                points = []
                for i in range(n_clusters):
                    low, high = bin_edges[i], bin_edges[i+1]
                    segment_df = df_data[(df_data['CIgreen'] >= low) & (df_data['CIgreen'] < high)]
                    if not segment_df.empty:
                        target_val = segment_df['CIgreen'].median()
                        idx_rep = (segment_df['CIgreen'] - target_val).abs().idxmin()
                        diff_pct = ((segment_df.loc[idx_rep, 'CIgreen'] - lote_avg) / lote_avg) * 100
                        points.append({
                            'Punto': i+1, 'Lat': segment_df.loc[idx_rep, 'lat'], 'Lon': segment_df.loc[idx_rep, 'lon'], 
                            'Valor': round(segment_df.loc[idx_rep, 'CIgreen'], 2), 'Relativo (%)': round(diff_pct, 1)
                        })
                df_res = pd.DataFrame(points)

            tab1, tab2 = st.tabs(["🗺️ Mapa del Lote", "📊 Análisis Estadístico"])

            with tab1:
                Map = geemap.Map()
                Map.add_basemap('HYBRID')
                Map.centerObject(ee_geom, 15)
                Map.addLayer(img_display, {'min': v_min, 'max': v_max, 'palette': ['#d73027','#ffffbf','#1a9850']}, 'Vigor Espectral')
                for _, p in df_res.iterrows():
                    txt = f"Ambiente: {int(p['Punto'])} | CIg: {p['Valor']}"
                    Map.add_marker([p['Lat'], p['Lon']], tooltip=txt, popup=txt)
                Map.to_streamlit(height=600)

            with tab2:
                # --- UI LIMPIA: SOLO HISTOGRAMA ---
                st.write("### Histograma de Vigor (Segmentación por Cuantiles)")
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='#020617')
                sns.histplot(df_data['CIgreen'], kde=True, color='#10b981', ax=ax)
                y_max = ax.get_ylim()[1]
                for _, p in df_res.iterrows():
                    ax.axvline(p['Valor'], color='white', linestyle='--', alpha=0.5)
                    ax.text(p['Valor'], y_max * 0.9, f"P{int(p['Punto'])}", color='white', ha='center', fontweight='bold')
                ax.set_facecolor('#020617')
                ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                st.pyplot(fig)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                    plot_path = tmpfile.name

                st.markdown("---")
                # EXPORTACIÓN (Se mantiene igual)
                kml = simplekml.Kml()
                for _, p in df_res.iterrows():
                    kml.newpoint(name=f"{selected_lote_name}_P{int(p['Punto'])}", coords=[(p['Lon'], p['Lat'])])
                
                col_btn1, col_btn2 = st.columns(2)
                col_btn1.download_button("📥 Descargar KML (GPS)", kml.kml(), file_name=f"puntos_{selected_lote_name}.kml")
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18); pdf.cell(0, 15, f"Reporte: {selected_lote_name}", 0, 1, 'C')
                pdf.set_font("Arial", '', 10); pdf.cell(0, 10, f"Promedio del Lote: {round(lote_avg, 2)}", 0, 1, 'R')
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "1. Coordenadas de Muestreo", 0, 1)
                pdf.set_font("Arial", '', 10)
                for _, p in df_res.iterrows():
                    pdf.cell(0, 8, f"- Punto {int(p['Punto'])}: Lat {p['Lat']}, Lon {p['Lon']} (Valor: {p['Valor']})", 0, 1)
                pdf.ln(10); pdf.image(plot_path, x=15, w=180)
                col_btn2.download_button("📄 Bajar Reporte PDF Científico", pdf.output(dest='S').encode('latin-1'), f"Reporte_{selected_lote_name}.pdf")

        except Exception as e:
            st.error(f"Error técnico: {e}")

if __name__ == "__main__":
    main()