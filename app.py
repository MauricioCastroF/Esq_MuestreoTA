import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.cluster import KMeans
from fpdf import FPDF
import simplekml
import os
import zipfile
import json
import base64
from datetime import datetime, timedelta
from google.oauth2 import service_account

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Muestreo Espectral UY", layout="wide")
st.title("🛰️ Sistema de Muestreo Espectral de Precisión")
st.markdown("""
Generación automatizada de puntos de muestreo basados en el índice **CIgreen** (Sentinel-2). 
Optimizado con limpieza de llaves PEM, filtrado IQR y clustering K-Means.
""")

# ==========================================
# 2. AUTENTICACIÓN ROBUSTA (Base64 + PEM Fix)
# ==========================================
def authenticate_gee():
    try:
        if "GEE_JSON_B64" in st.secrets:
            b64_str = st.secrets["GEE_JSON_B64"]
            decoded_bytes = base64.b64decode(b64_str)
            json_key = json.loads(decoded_bytes.decode('utf-8'))
            
            # Limpieza del PEM para evitar el error de bytes
            pk = json_key["private_key"]
            json_key["private_key"] = pk.replace("\\\\n", "\n").replace("\\n", "\n")

            credentials = service_account.Credentials.from_service_account_info(json_key)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            
            ee.Initialize(scoped_credentials)
            return True
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# ==========================================
# 3. FUNCIONES TÉCNICAS (Satelital)
# ==========================================
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).copyProperties(image, ['system:time_start'])

def add_cig_index(image):
    # CIgreen = (NIR / Green) - 1
    cig = image.expression('(NIR / GREEN) - 1', {
        'NIR': image.select('B8'),
        'GREEN': image.select('B3')
    }).rename('CIgreen')
    return image.addBands(cig).copyProperties(image, ['system:time_start'])

# ==========================================
# 4. INTERFAZ LATERAL
# ==========================================
st.sidebar.header("⚙️ Parámetros de Análisis")
uploaded_file = st.sidebar.file_uploader("Cargar Lotes (.geojson o .kml)", type=['geojson', 'kml'])
fecha_inicio = st.sidebar.date_input("Fecha Inicio", datetime.now() - timedelta(days=120))
fecha_fin = st.sidebar.date_input("Fecha Fin", datetime.now())
num_puntos = st.sidebar.slider("Puntos por lote", 3, 15, 5)

# ==========================================
# 5. LÓGICA DE PROCESAMIENTO
# ==========================================
if st.sidebar.button("🚀 Iniciar Muestreo"):
    if uploaded_file is not None and authenticate_gee():
        with st.spinner("Procesando imágenes Sentinel-2..."):
            # Guardar archivo temporal
            with open("temp_input", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Leer y proyectar a UTM 21S (Uruguay)
            gdf = gpd.read_file("temp_input")
            gdf_utm = gdf.to_crs(epsg=32721)
            
            puntos_finales = []
            archivos_a_comprimir = []
            palette_cig = ['#543005', '#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

            for index, row in gdf_utm.iterrows():
                lote_id = row.get('Name', f"Lote_{index}")
                
                # Buffer de seguridad (-15m) para evitar ruidos de bordes
                geom_inner = row.geometry.buffer(-15)
                if geom_inner.is_empty:
                    st.warning(f"⚠️ El lote {lote_id} es demasiado pequeño para el buffer. Saltando.")
                    continue

                # Convertir a WGS84 para Earth Engine
                geom_wgs = gpd.GeoSeries([geom_inner], crs="EPSG:32721").to_crs(epsg=4326).iloc[0]
                
                # Manejo de Polígonos y MultiPolígonos
                if isinstance(geom_wgs, Polygon):
                    ee_geom = ee.Geometry.Polygon(list(geom_wgs.exterior.coords))
                else:
                    coords = [list(part.exterior.coords) for part in geom_wgs.geoms]
                    ee_geom = ee.Geometry.MultiPolygon(coords)

                # Colección Sentinel-2
                s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(ee_geom)
                          .filterDate(str(fecha_inicio), str(fecha_fin))
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                          .map(mask_s2_clouds)
                          .map(add_cig_index))

                if s2_col.size().getInfo() == 0:
                    st.warning(f"☁️ Sin imágenes limpias para {lote_id}.")
                    continue

                # Mejor imagen (Quality Mosaic)
                best_img = s2_col.qualityMosaic('CIgreen').clip(ee_geom)
                
                # Visualización dinámica (p2 - p98)
                stats = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 20)
                cig_viz = best_img.select('CIgreen').visualize(
                    min=stats.get('CIgreen_p2'), 
                    max=stats.get('CIgreen_p98'), 
                    palette=palette_cig
                )
                
                # Descarga miniatura
                img_url = cig_viz.getThumbURL({'region': ee_geom.bounds().getInfo(), 'dimensions': 1024, 'format': 'png'})
                img_path = f"bg_{lote_id}.png"
                with open(img_path, 'wb') as f:
                    f.write(requests.get(img_url).content)

                # Muestreo IQR (Filtro de pureza espectral)
                perc = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([25, 75]), ee_geom, 20)
                mask_iqr = (best_img.select('CIgreen').gte(ee.Image.constant(perc.get('CIgreen_p25')))
                            .And(best_img.select('CIgreen').lte(ee.Image.constant(perc.get('CIgreen_p75')))))
                
                muestras = best_img.updateMask(mask_iqr).sample(region=ee_geom, scale=10, numPixels=800, geometries=True)
                pixel_data = muestras.aggregate_array('.geo').getInfo()
                
                if not pixel_data: continue

                # Clustering K-Means
                coords_np = np.array([p['coordinates'] for p in pixel_data])
                kmeans = KMeans(n_clusters=num_puntos, n_init=10, random_state=42).fit(coords_np)
                
                # Generación de Informe PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", 'B', 16)
                pdf.cell(0, 10, txt=f"REPORTE TÉCNICO DE MUESTREO - {lote_id}", ln=True, align='C')
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(plt.imread(img_path), extent=[row.geometry.bounds[0], row.geometry.bounds[2], row.geometry.bounds[1], row.geometry.bounds[3]])
                
                for i, center in enumerate(kmeans.cluster_centers_):
                    puntos_finales.append({'Lote': lote_id, 'Punto': i+1, 'geometry': Point(center[0], center[1])})
                    ax.scatter(center[0], center[1], color='red', s=60, edgecolors='white', zorder=5)
                    ax.text(center[0], center[1]+3, f"P{i+1}", color='white', weight='bold', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))

                ax.set_axis_off()
                plot_p = f"plot_{lote_id}.png"
                plt.savefig(plot_p, bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close()
                
                pdf.image(plot_p, x=15, y=35, w=180)
                pdf_name = f"Informe_{lote_id}.pdf"
                pdf.output(pdf_name)
                archivos_a_comprimir.append(pdf_name)
                
                os.remove(img_path); os.remove(plot_p)

            # Generación de ZIP
            if puntos_finales:
                # KML para Navegación
                kml = simplekml.Kml()
                pnt_gdf = gpd.GeoDataFrame(puntos_finales, crs="EPSG:4326")
                for l_id in pnt_gdf['Lote'].unique():
                    fol = kml.newfolder(name=l_id)
                    df_p = pnt_gdf[pnt_gdf['Lote'] == l_id]
                    for _, r in df_p.iterrows():
                        pnt = fol.newpoint(name=f"{l_id}-P{r['Punto']}", coords=[(r.geometry.x, r.geometry.y)])
                kml.save("Navegacion.kml")
                archivos_a_comprimir.append("Navegacion.kml")

                zip_name = "Resultados_Muestreo_UY.zip"
                with zipfile.ZipFile(zip_name, 'w') as zipf:
                    for f in archivos_a_comprimir:
                        zipf.write(f)
                        os.remove(f)
                
                with open(zip_name, "rb") as f:
                    st.success("✅ Análisis completado con éxito.")
                    st.download_button("📥 Descargar Resultados (.zip)", f, file_name=zip_name)
            
            if os.path.exists("temp_input"): os.remove("temp_input")
    else:
        st.info("👋 Sube tus lotes para iniciar el análisis pedométrico.")

# ==========================================
# 6. PIE DE PÁGINA
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("Herramienta GIS avanzada para el sector agropecuario.")