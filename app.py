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
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURACIÓN Y TÍTULOS
# ==========================================
st.set_page_config(page_title="Muestreo Pedometría UY", layout="wide")
st.title("🛰️ Sistema de Muestreo Espectral de Precisión")
st.markdown("""
Generación automatizada de puntos de muestreo basados en el índice **CIgreen** (Sentinel-2). 
Optimizado con filtrado IQR y clustering K-Means.
""")

# ==========================================
# 2. AUTENTICACIÓN REFORZADA
# ==========================================
def authenticate_gee():
    try:
        if "GEE_JSON" in st.secrets:
            json_text = st.secrets["GEE_JSON"]
            # Limpieza profunda de escapes dobles
            json_text = json_text.replace("\\\\n", "\\n")
            json_key = json.loads(json_text, strict=False)
            
            # Formatear la llave privada para el motor RSA
            if "private_key" in json_key:
                json_key["private_key"] = json_key["private_key"].replace("\\n", "\n")
            
            credentials = ee.ServiceAccountCredentials(
                json_key['client_email'], 
                key_data=json.dumps(json_key)
            )
            ee.Initialize(credentials)
            return True
        else:
            st.error("No se encontró el secreto GEE_JSON.")
            return False
    except Exception as e:
        st.error(f"Error de autenticación: {e}")
        return False

# ==========================================
# 3. FUNCIONES DE PROCESAMIENTO GEE
# ==========================================
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).copyProperties(image, ['system:time_start'])

def add_cig_index(image):
    # Índice CIgreen: (NIR / Green) - 1
    cig = image.expression('(NIR / GREEN) - 1', {
        'NIR': image.select('B8'),
        'GREEN': image.select('B3')
    }).rename('CIgreen')
    return image.addBands(cig).copyProperties(image, ['system:time_start'])

# ==========================================
# 4. INTERFAZ Y LÓGICA PRINCIPAL
# ==========================================
st.sidebar.header("⚙️ Parámetros")
uploaded_file = st.sidebar.file_uploader("Subir Lotes (.geojson, .kml)", type=['geojson', 'kml'])
fecha_inicio = st.sidebar.date_input("Fecha Inicio", datetime.now() - timedelta(days=120))
fecha_fin = st.sidebar.date_input("Fecha Fin", datetime.now())
num_puntos = st.sidebar.slider("Puntos de muestreo por lote", 3, 15, 5)

if st.sidebar.button("🚀 Iniciar Procesamiento"):
    if uploaded_file is not None and authenticate_gee():
        with st.spinner("Analizando firmas espectrales..."):
            # Guardar temporalmente para leer con GeoPandas
            with open("temp_input", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Cargar y asegurar proyección UTM (Uruguay - Zona 21S)
            gdf = gpd.read_file("temp_input")
            gdf_utm = gdf.to_crs(epsg=32721)
            
            puntos_finales = []
            archivos_a_comprimir = []
            palette_cig = ['#543005', '#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

            for index, row in gdf_utm.iterrows():
                lote_id = row.get('Name', f"Lote_{index}")
                
                # Buffer de -15m para evitar bordes (Critical para precisión pedométrica)
                geom_inner = row.geometry.buffer(-15)
                if geom_inner.is_empty:
                    st.warning(f"⚠️ El lote {lote_id} es muy pequeño para el buffer de -15m. Saltando.")
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
                    st.warning(f"☁️ Sin imágenes limpias para {lote_id} en este rango.")
                    continue

                # Mejor imagen (Mediana para reducir ruido)
                best_img = s2_col.qualityMosaic('CIgreen').clip(ee_geom)
                
                # Visualización con estiramiento dinámico (2-98%)
                stats = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 20)
                cig_viz = best_img.select('CIgreen').visualize(
                    min=stats.get('CIgreen_p2'), 
                    max=stats.get('CIgreen_p98'), 
                    palette=palette_cig
                )
                
                # Descarga de imagen para el PDF
                img_url = cig_viz.getThumbURL({'region': ee_geom.bounds().getInfo(), 'dimensions': 1024, 'format': 'png'})
                img_path = f"bg_{lote_id}.png"
                with open(img_path, 'wb') as f: f.write(requests.get(img_url).content)

                # Muestreo IQR (Filtro pedométrico para evitar extremos)
                perc = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([25, 75]), ee_geom, 20)
                mask_iqr = (best_img.select('CIgreen').gte(ee.Image.constant(perc.get('CIgreen_p25')))
                            .And(best_img.select('CIgreen').lte(ee.Image.constant(perc.get('CIgreen_p75')))))
                
                muestras = best_img.updateMask(mask_iqr).sample(region=ee_geom, scale=10, numPixels=800, geometries=True)
                pixel_data = muestras.aggregate_array('.geo').getInfo()
                
                if not pixel_data:
                    st.error(f"Error al extraer píxeles en {lote_id}")
                    continue

                # Clustering K-Means
                coords_np = np.array([p['coordinates'] for p in pixel_data])
                kmeans = KMeans(n_clusters=num_puntos, n_init=10, random_state=42).fit(coords_np)
                
                # --- GENERAR PDF ---
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", 'B', 16)
                pdf.cell(0, 10, txt=f"REPORTE DE MUESTREO - {lote_id}", ln=True, align='C')
                pdf.set_font("helvetica", size=10)
                pdf.cell(0, 10, txt=f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')

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
                
                # Limpiar archivos temporales de imagen
                os.remove(img_path); os.remove(plot_p)

            # --- GENERAR NAVEGACIÓN KML Y ZIP ---
            if puntos_finales:
                kml = simplekml.Kml()
                pnt_gdf = gpd.GeoDataFrame(puntos_finales, crs="EPSG:4326")
                for l_id in pnt_gdf['Lote'].unique():
                    fol = kml.newfolder(name=l_id)
                    df_l = pnt_gdf[pnt_gdf['Lote'] == l_id]
                    for _, r in df_l.iterrows():
                        pnt = fol.newpoint(name=f"{l_id}-P{r['Punto']}")
                        pnt.coords = [(r.geometry.x, r.geometry.y)]
                
                kml_path = "Puntos_Navegacion.kml"
                kml.save(kml_path)
                archivos_a_comprimir.append(kml_path)

                zip_name = "Resultados_Muestreo_GEE.zip"
                with zipfile.ZipFile(zip_name, 'w') as zipf:
                    for f in archivos_a_comprimir:
                        zipf.write(f)
                        os.remove(f) # Limpiar PDFs y KML locales tras comprimir
                
                with open(zip_name, "rb") as f:
                    st.success("✅ ¡Procesamiento Exitoso!")
                    st.download_button("📥 Descargar Resultados (.zip)", f, file_name=zip_name)
    else:
        st.info("👋 Esperando archivo de entrada y configuración de GEE.")

# ==========================================
# 5. PIE DE PÁGINA
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado para análisis pedométrico avanzado.")