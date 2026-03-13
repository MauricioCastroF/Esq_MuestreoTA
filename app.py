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

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Muestreo Espectral UY", layout="wide")
st.title("🛰️ Sistema de Muestreo Espectral de Precisión")

# 2. AUTENTICACIÓN BLINDADA (Base64 con Auto-Fix de Padding)
def authenticate_gee():
    try:
        if "GEE_JSON_B64" in st.secrets:
            # 1. Recuperamos y limpiamos la cadena Base64
            b64_str = st.secrets["GEE_JSON_B64"].strip().strip('"').strip("'")
            
            # 2. Decodificamos a diccionario JSON
            decoded_bytes = base64.b64decode(b64_str)
            json_key = json.loads(decoded_bytes.decode('utf-8'))
            
            # 3. LIMPIEZA QUIRÚRGICA DE LA LLAVE (Solución al JWT Signature)
            pk = json_key["private_key"]
            
            # Primero: quitamos cualquier "ruido" de dobles escapes que Streamlit inyecta
            pk = pk.replace("\\\\n", "\n")
            # Segundo: convertimos los escapes de texto (\n) en saltos de línea reales
            pk = pk.replace("\\n", "\n")
            # Tercero: nos aseguramos de que empiece y termine limpio
            pk = pk.strip()
            
            json_key["private_key"] = pk

            # 4. Autenticación oficial
            credentials = service_account.Credentials.from_service_account_info(json_key)
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            
            ee.Initialize(scoped_credentials)
            return True
        else:
            st.error("❌ No se encontró el secreto 'GEE_JSON_B64'.")
            return False
    except Exception as e:
        # Mostramos el error detallado para saber si es un tema de permisos o de firma
        st.error(f"❌ Error de autenticación: {e}")
        return False

# 3. LÓGICA DE PROCESAMIENTO (Sentinel-2)
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).copyProperties(image, ['system:time_start'])

def add_cig_index(image):
    cig = image.expression('(NIR / GREEN) - 1', {
        'NIR': image.select('B8'),
        'GREEN': image.select('B3')
    }).rename('CIgreen')
    return image.addBands(cig).copyProperties(image, ['system:time_start'])

# 4. INTERFAZ Y EJECUCIÓN
st.sidebar.header("⚙️ Configuración")
uploaded_file = st.sidebar.file_uploader("Lotes (.geojson o .kml)", type=['geojson', 'kml'])
fecha_inicio = st.sidebar.date_input("Fecha Inicio", datetime.now() - timedelta(days=120))
fecha_fin = st.sidebar.date_input("Fecha Fin", datetime.now())
num_puntos = st.sidebar.slider("Puntos por lote", 3, 10, 5)

if st.sidebar.button("🚀 Iniciar Muestreo"):
    if uploaded_file is not None and authenticate_gee():
        with st.spinner("Procesando índices espectrales..."):
            with open("temp_input", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            gdf_utm = gpd.read_file("temp_input").to_crs(epsg=32721) # UTM 21S Uruguay
            puntos_finales = []
            archivos_zip = []
            palette_cig = ['#543005', '#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

            for idx, row in gdf_utm.iterrows():
                lote_id = row.get('Name', f"Lote_{idx}")
                geom_inner = row.geometry.buffer(-15) # Buffer de seguridad bordes
                if geom_inner.is_empty: continue

                geom_wgs = gpd.GeoSeries([geom_inner], crs="EPSG:32721").to_crs(epsg=4326).iloc[0]
                ee_geom = ee.Geometry.Polygon(list(geom_wgs.exterior.coords)) if isinstance(geom_wgs, Polygon) else ee.Geometry.MultiPolygon([list(p.exterior.coords) for p in geom_wgs.geoms])

                s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(ee_geom).filterDate(str(fecha_inicio), str(fecha_fin))
                      .map(mask_s2_clouds).map(add_cig_index))

                if s2.size().getInfo() == 0: continue

                best_img = s2.qualityMosaic('CIgreen').clip(ee_geom)
                stats = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 20)
                cig_viz = best_img.select('CIgreen').visualize(min=stats.get('CIgreen_p2'), max=stats.get('CIgreen_p98'), palette=palette_cig)
                
                img_url = cig_viz.getThumbURL({'region': ee_geom.bounds().getInfo(), 'dimensions': 1024, 'format': 'png'})
                local_img = f"bg_{lote_id}.png"
                with open(local_img, 'wb') as f: f.write(requests.get(img_url).content)

                # Muestreo Inteligente
                perc = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([25, 75]), ee_geom, 20)
                mask_iqr = best_img.select('CIgreen').gte(ee.Image.constant(perc.get('CIgreen_p25'))).And(best_img.select('CIgreen').lte(ee.Image.constant(perc.get('CIgreen_p75'))))
                muestras = best_img.updateMask(mask_iqr).sample(region=ee_geom, scale=10, numPixels=800, geometries=True)
                
                pixel_data = muestras.aggregate_array('.geo').getInfo()
                coords_np = np.array([p['coordinates'] for p in pixel_data])
                kmeans = KMeans(n_clusters=num_puntos, n_init=10).fit(coords_np)
                
                # --- PDF ---
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", 'B', 16)
                pdf.cell(0, 10, txt=f"Lote: {lote_id}", ln=True, align='C')
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(plt.imread(local_img), extent=[row.geometry.bounds[0], row.geometry.bounds[2], row.geometry.bounds[1], row.geometry.bounds[3]])
                for i, center in enumerate(kmeans.cluster_centers_):
                    puntos_finales.append({'Lote': lote_id, 'Punto': i+1, 'geometry': Point(center[0], center[1])})
                    ax.scatter(center[0], center[1], color='red', s=50, edgecolors='white')
                
                ax.set_axis_off()
                local_plot = f"plot_{lote_id}.png"
                plt.savefig(local_plot, bbox_inches='tight', dpi=150)
                plt.close()
                pdf.image(local_plot, x=15, y=35, w=180)
                pdf_name = f"Reporte_{lote_id}.pdf"
                pdf.output(pdf_name)
                archivos_zip.append(pdf_name)
                os.remove(local_img); os.remove(local_plot)

            if puntos_finales:
                kml = simplekml.Kml()
                for p in puntos_finales:
                    kml.newpoint(name=f"{p['Lote']}-P{p['Punto']}", coords=[(p['geometry'].x, p['geometry'].y)])
                kml.save("Navegacion.kml"); archivos_zip.append("Navegacion.kml")

                with zipfile.ZipFile("Muestreo.zip", 'w') as zipf:
                    for f in archivos_zip: zipf.write(f)
                
                with open("Muestreo.zip", "rb") as f:
                    st.success("✅ ¡Listo para descargar!")
                    st.download_button("📥 Resultados (.zip)", f, file_name="Muestreo_UY.zip")
    else:
        st.info("👋 Carga tus archivos para iniciar.")