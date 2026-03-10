import streamlit as st
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans
from fpdf import FPDF
import simplekml
import os
import zipfile
import json
from datetime import datetime, timedelta

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Muestreo de Precisión Uruguay", layout="wide")
st.title("🛰️ Sistema de Muestreo Espectral de Precisión")
st.markdown("Generación automatizada de puntos de muestreo basados en el índice CIgreen (Sentinel-2).")

# 2. AUTENTICACIÓN (Service Account)
def authenticate_gee():
    try:
        # Los secretos deben estar configurados en Streamlit Cloud
        if "GEE_JSON" in st.secrets:
            json_key = json.loads(st.secrets["GEE_JSON"])
            credentials = ee.ServiceAccountCredentials(json_key['client_email'], key_data=json.dumps(json_key))
            ee.Initialize(credentials)
        else:
            # Opción local para desarrollo
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Error de autenticación en GEE: {e}")
        return False

# 3. FUNCIONES TÉCNICAS (Lógica de tu Script)
def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).copyProperties(image, ['system:time_start'])

def cloudsT(image):
    mask = image.select('B2').lte(1500)
    return image.updateMask(mask).copyProperties(image, ['system:time_start'])

def addChlIndices(image):
    cig = image.expression('(NIR / GREEN) - 1', {
        'NIR': image.select('B8'),
        'GREEN': image.select('B3')
    }).rename('CIgreen')
    return image.addBands(cig).copyProperties(image, ['system:time_start'])

# 4. INTERFAZ LATERAL (Configuración)
st.sidebar.header("⚙️ Configuración del Análisis")
uploaded_file = st.sidebar.file_uploader("Cargar Lotes (.geojson o .kml)", type=['geojson', 'kml'])
fecha_inicio = st.sidebar.date_input("Fecha Inicio", datetime.now() - timedelta(days=180))
fecha_fin = st.sidebar.date_input("Fecha Fin", datetime.now() - timedelta(days=5))

if st.sidebar.button("🚀 Ejecutar Muestreo"):
    if uploaded_file is not None and authenticate_gee():
        with st.spinner("Procesando imágenes satelitales..."):
            # Procesamiento de archivos
            with open("temp_input", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            gdf_utm = gpd.read_file("temp_input").to_crs(epsg=32721)
            puntos_finales = []
            archivos_a_comprimir = []
            palette_cig = ['#543005', '#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

            for _, row in gdf_utm.iterrows():
                lote_id = row['Name']
                geom_buffer_utm = row.geometry.buffer(-15)
                if geom_buffer_utm.is_empty: continue

                geom_wgs = gpd.GeoSeries([geom_buffer_utm], crs="EPSG:32721").to_crs(epsg=4326).iloc[0]
                ee_geom = ee.Geometry.Polygon(list(geom_wgs.exterior.coords))
                
                s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(ee_geom)
                      .filterDate(str(fecha_inicio), str(fecha_fin))
                      .map(mask_s2_clouds).map(cloudsT).map(addChlIndices))
                
                if s2.size().getInfo() == 0: continue

                # Selección de mejor imagen y contraste dinámico
                ranked = s2.map(lambda img: img.set('mean_cig', img.select('CIgreen').reduceRegion(ee.Reducer.mean(), ee_geom, 20).get('CIgreen'))).sort('mean_cig', False)
                best_img = ee.Image(ranked.first()).clip(ee_geom)
                
                stats_viz = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([2, 98]), ee_geom, 20)
                cig_viz = best_img.select('CIgreen').visualize(min=stats_viz.get('CIgreen_p2'), max=stats_viz.get('CIgreen_p98'), palette=palette_cig)
                
                # Descarga miniatura
                url = cig_viz.getThumbURL({'region': ee_geom.bounds().getInfo(), 'dimensions': 1000, 'format': 'png'})
                img_path = f"bg_{lote_id}.png"
                with open(img_path, 'wb') as f: f.write(requests.get(url).content)

                # Muestreo IQR corregido (ee.Image.constant)
                perc = best_img.select('CIgreen').reduceRegion(ee.Reducer.percentile([25, 75]), ee_geom, 20)
                mask_iqr = best_img.select('CIgreen').gte(ee.Image.constant(perc.get('CIgreen_p25'))).And(best_img.select('CIgreen').lte(ee.Image.constant(perc.get('CIgreen_p75'))))
                muestras = best_img.updateMask(mask_iqr).sample(region=ee_geom, scale=10, numPixels=500, geometries=True)
                
                pixel_data = muestras.aggregate_array('.geo').getInfo()
                if not pixel_data:
                    pixel_data = best_img.select('CIgreen').sample(region=ee_geom, scale=10, numPixels=500, geometries=True).aggregate_array('.geo').getInfo()

                coords = np.array([p['coordinates'] for p in pixel_data])
                kmeans = KMeans(n_clusters=5, n_init=10).fit(coords)
                
                # Generar PDF Individual
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", 'B', 16)
                pdf.cell(0, 15, txt=f"Lote: {lote_id}", ln=True, align='C')
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(plt.imread(img_path), extent=[row.geometry.bounds[0], row.geometry.bounds[2], row.geometry.bounds[1], row.geometry.bounds[3]], origin='upper')
                gpd.GeoSeries([row.geometry]).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
                
                for i, center in enumerate(kmeans.cluster_centers_):
                    puntos_finales.append({'Lote': lote_id, 'Punto': i+1, 'geometry': Point(center[0], center[1])})
                    ax.scatter(center[0], center[1], color='red', s=50, edgecolors='white')
                    ax.text(center[0], center[1]+2, f"P{i+1}", color='yellow', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

                ax.set_axis_off()
                plot_p = f"plot_{lote_id}.png"
                plt.savefig(plot_p, bbox_inches='tight', dpi=150)
                plt.close()
                pdf.image(plot_p, x=10, y=30, w=190)
                pdf_name = f"Muestreo_{lote_id}.pdf"
                pdf.output(pdf_name)
                archivos_a_comprimir.append(pdf_name)
                os.remove(img_path); os.remove(plot_p)

            # Generar KML y ZIP
            if puntos_finales:
                kml = simplekml.Kml()
                puntos_wgs = gpd.GeoDataFrame(puntos_finales, crs="EPSG:4326")
                for l_name in puntos_wgs['Lote'].unique():
                    fol = kml.newfolder(name=l_name)
                    df_l = puntos_wgs[puntos_wgs['Lote'] == l_name].sort_values('Punto')
                    for _, r in df_l.iterrows():
                        pnt = fol.newpoint(name=f"P{r['Punto']}")
                        pnt.coords = [(r.geometry.x, r.geometry.y)]
                kml.save("Navegacion.kml")
                archivos_a_comprimir.append("Navegacion.kml")

                zip_name = "Resultados_Muestreo.zip"
                with zipfile.ZipFile(zip_name, 'w') as zipf:
                    for f in archivos_a_comprimir: zipf.write(f)
                
                with open(zip_name, "rb") as f:
                    st.success("✅ Procesamiento completado con éxito.")
                    st.download_button("📥 Descargar Archivos (.zip)", f, file_name=zip_name)
    else:
        st.warning("Por favor carga un archivo y asegúrate de que la configuración sea correcta.")