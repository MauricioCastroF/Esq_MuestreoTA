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

# =================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =================================================================
st.set_page_config(page_title="Muestreo Espectral UY", layout="wide")
st.title("🛰️ Sistema de Muestreo Espectral de Precisión")
st.markdown("""
Generación automatizada de puntos de muestreo basados en el índice **CIgreen** (Sentinel-2).
Optimizado con decodificación Base64, filtrado IQR y clustering K-Means.
""")

# =================================================================
# 2. AUTENTICACIÓN BLINDADA (Base64)
# =================================================================
def authenticate_gee():
    try:
        if "GEE_JSON_B64" in st.secrets:
            # Recuperar y decodificar la cadena Base64
            b64_str = st.secrets["GEE_JSON_B64"]
            decoded_bytes = base64.b64decode(b64_str)
            json_key = json.loads(decoded_bytes.decode('utf-8'))
            
            # Formatear la llave privada para el motor RSA de Google
            if "private_key" in json_key:
                json_key["private_key"] = json_key["private_key"].replace("\\n", "\n")
            
            credentials = ee.ServiceAccountCredentials(
                json_key['client_email'], 
                key_data=json.dumps(json_key)
            )
            ee.Initialize(credentials)
            return True
        else:
            st.error("❌ No se encontró el secreto 'GEE_JSON_B64' en Streamlit Cloud.")
            st.info("Asegúrate de haber configurado el secreto en formato Base64.")
            return False
    except Exception as e:
        st.error(f"❌ Error crítico de autenticación: {e}")
        return False

# =================================================================
# 3. LÓGICA DE PROCESAMIENTO SATELITAL
# =================================================================
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

# =================================================================
# 4. INTERFAZ LATERAL (Configuración)
# =================================================================
st.sidebar.header("⚙️ Parámetros de Análisis")
uploaded_file = st.sidebar.file_uploader("Cargar Lotes (.geojson o .kml)", type=['geojson', 'kml'])
fecha_inicio = st.sidebar.date_input("Fecha Inicio", datetime.now() - timedelta(days=120))
fecha_fin = st.sidebar.date_input("Fecha Fin", datetime.now())
num_puntos = st.sidebar.slider("Puntos por lote", 3, 15, 5)

# =================================================================
# 5. EJECUCIÓN PRINCIPAL
# =================================================================
if st.sidebar.button("🚀 Ejecutar Muestreo"):
    if uploaded_file is not None and authenticate_gee():
        with st.spinner("Procesando imágenes satelitales Sentinel-2..."):
            # Guardar archivo temporalmente
            temp_file = "input_temp"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Cargar y proyectar a UTM 21S (Uruguay)
            gdf = gpd.read_file(temp_file)
            gdf_utm = gdf.to_crs(epsg=32721)
            
            puntos_finales = []
            archivos_zip = []
            palette_cig = ['#543005', '#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

            for idx, row in gdf_utm.iterrows():
                lote_id = row.get('Name', f"Lote_{idx}")
                
                # 5.1. Buffer de seguridad (-15m)
                geom_inner = row.geometry.buffer(-15)
                if geom_inner.is_empty:
                    st.warning(f"⚠️ El lote {lote_id} es demasiado pequeño para el buffer de -15m.")
                    continue

                # 5.2. Preparar geometría para GEE
                geom_wgs = gpd.GeoSeries([geom_inner], crs="EPSG:32721").to_crs(epsg=4326).iloc[0]
                if isinstance(geom_wgs, Polygon):
                    ee_geom = ee.Geometry.Polygon(list(geom_wgs.exterior.coords))
                else: # MultiPolygon
                    coords = [list(part.exterior.coords) for part in geom_wgs.geoms]
                    ee_geom = ee.Geometry.MultiPolygon(coords)

                # 5.3. Filtrar colección satelital
                s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                          .filterBounds(ee_geom)
                          .filterDate(str(fecha_inicio), str(fecha_fin))
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                          .map(mask_s2_clouds)
                          .map(add_cig_index))

                if s2_col.size().getInfo() == 0:
                    st.warning(f"☁️ Sin datos limpios para {lote_id} en el rango seleccionado.")
                    continue

                # 5.4. Seleccionar mejor imagen (Mosaico de calidad por CIgreen)
                best_img = s2_col.qualityMosaic('CIgreen').clip(ee_geom)
                
                # Visualización (Contraste 2-98%)
                stats = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 20)
                cig_viz = best_img.select('CIgreen').visualize(
                    min=stats.get('CIgreen_p2'), 
                    max=stats.get('CIgreen_p98'), 
                    palette=palette_cig
                )
                
                # Descargar miniatura para el PDF
                img_url = cig_viz.getThumbURL({'region': ee_geom.bounds().getInfo(), 'dimensions': 1024, 'format': 'png'})
                local_img = f"bg_{lote_id}.png"
                with open(local_img, 'wb') as f:
                    f.write(requests.get(img_url).content)

                # 5.5. Muestreo Inteligente (Filtrado por Rango Intercuartil - IQR)
                perc = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([25, 75]), ee_geom, 20)
                mask_iqr = (best_img.select('CIgreen').gte(ee.Image.constant(perc.get('CIgreen_p25')))
                            .And(best_img.select('CIgreen').lte(ee.Image.constant(perc.get('CIgreen_p75')))))
                
                samples = best_img.updateMask(mask_iqr).sample(region=ee_geom, scale=10, numPixels=1000, geometries=True)
                pixel_data = samples.aggregate_array('.geo').getInfo()
                
                if not pixel_data:
                    st.error(f"Error al extraer píxeles en {lote_id}")
                    continue

                # 5.6. Clustering K-Means para puntos de muestreo
                coords_np = np.array([p['coordinates'] for p in pixel_data])
                kmeans = KMeans(n_clusters=num_puntos, n_init=10, random_state=42).fit(coords_np)
                
                # 5.7. Generación de Reporte PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", 'B', 16)
                pdf.cell(0, 10, txt=f"REPORTE DE MUESTREO - {lote_id}", ln=True, align='C')
                
                # Generar el gráfico con Matplotlib
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(plt.imread(local_img), extent=[row.geometry.bounds[0], row.geometry.bounds[2], row.geometry.bounds[1], row.geometry.bounds[3]])
                
                for i, center in enumerate(kmeans.cluster_centers_):
                    puntos_finales.append({'Lote': lote_id, 'Punto': i+1, 'geometry': Point(center[0], center[1])})
                    ax.scatter(center[0], center[1], color='red', s=70, edgecolors='white', zorder=5)
                    ax.text(center[0], center[1]+4, f"P{i+1}", color='white', weight='bold', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

                ax.set_axis_off()
                local_plot = f"plot_{lote_id}.png"
                plt.savefig(local_plot, bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close()
                
                pdf.image(local_plot, x=15, y=35, w=180)
                pdf_file = f"Muestreo_{lote_id}.pdf"
                pdf.output(pdf_file)
                archivos_zip.append(pdf_file)
                
                # Limpiar temporales de imagen
                os.remove(local_img); os.remove(local_plot)

            # 6. GENERACIÓN DE ARCHIVOS DE SALIDA (KML y ZIP)
            if puntos_finales:
                # KML para Navegador GPS
                kml = simplekml.Kml()
                pnt_gdf = gpd.GeoDataFrame(puntos_finales, crs="EPSG:4326")
                for l_name in pnt_gdf['Lote'].unique():
                    folder = kml.newfolder(name=l_name)
                    df_lote = pnt_gdf[pnt_gdf['Lote'] == l_name]
                    for _, r in df_lote.iterrows():
                        pnt = folder.newpoint(name=f"{l_name}-P{r['Punto']}")
                        pnt.coords = [(r.geometry.x, r.geometry.y)]
                
                kml_output = "Puntos_Navegacion.kml"
                kml.save(kml_output)
                archivos_zip.append(kml_output)

                # Empaquetar todo en un ZIP
                zip_filename = "Resultados_Muestreo_UY.zip"
                with zipfile.ZipFile(zip_filename, 'w') as zipf:
                    for f in archivos_zip:
                        zipf.write(f)
                        os.remove(f) # Borrar del servidor tras comprimir
                
                with open(zip_filename, "rb") as f:
                    st.success("✅ Procesamiento completado. Los reportes están listos.")
                    st.download_button("📥 Descargar Resultados (.zip)", f, file_name=zip_filename)
            
            if os.path.exists(temp_file): os.remove(temp_file)

    else:
        if uploaded_file is None:
            st.info("👋 Por favor, carga un archivo GeoJSON o KML con tus lotes para comenzar.")

# =================================================================
# 6. PIE DE PÁGINA TÉCNICO
# =================================================================
st.sidebar.markdown("---")
st.sidebar.caption(f"Análisis basado en índice CIgreen optimizado.")