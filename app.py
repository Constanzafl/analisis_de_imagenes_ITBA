# -*- coding: utf-8 -*-
"""
Filtro Bilateral - Presentación Oral
TP2 - Punto 4 - ATIS ITBA
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuración
st.set_page_config(page_title="Filtro Bilateral", layout="wide")

st.title("🛰️ Filtro Bilateral - Análisis Comparativo")
st.markdown("Implementar el filtro bilateral y aplicarlo a imágenes y sus versiones contaminadas. Analizar los resultados y comparar con el filtro de Gauss y con el filtro de la mediana.")

# ============= FUNCIONES =============

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_salt_pepper(img, p=0.03):
    out = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    out[rnd < p] = 0
    out[rnd > 1 - p] = 255
    return out

def gaussian_rgb(img, sigma=2.0):
    kernel = int(2 * np.ceil(2 * sigma) + 1)
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)

def median_rgb(img, kernel=3):
    return cv2.medianBlur(img, kernel)

# ============= SIDEBAR =============

st.sidebar.markdown("## ⚙️ Control de Presentación")

# Selector de sección
seccion = st.sidebar.radio(
    "📍 Navegar:",
    ["🎯 1. Teoría", 
     "🔬 2. Ruido Gaussiano", 
     "⚡ 3. Ruido Sal & Pimienta",
     "📊 4. Comparación Final"]
)

st.sidebar.markdown("---")

# Cargar imagen
nombre_imagen = "small.jpg"  # 🔧 CAMBIA ESTO

try:
    image = Image.open(nombre_imagen)
    img_rgb = np.array(image)
    
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    
    st.sidebar.success(f"✅ {nombre_imagen}")
    
except:
    st.sidebar.error("❌ No se encontró la imagen")
    st.stop()

# Parámetros según la sección
if seccion == "🔬 2. Ruido Gaussiano":
    # Solo ruido Gaussiano
    st.sidebar.markdown("### 🎲 Ruido Gaussiano")
    sigma_noise = st.sidebar.slider("Intensidad (σ):", 10, 50, 30)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Parámetros Bilateral")
    d_bilateral = st.sidebar.slider("d (diámetro):", 5, 15, 9, 2)
    sigma_color = st.sidebar.slider("σr (color):", 30, 100, 75, 5)
    sigma_space = st.sidebar.slider("σs (espacio):", 30, 100, 75, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Otros Filtros")
    sigma_gauss = st.sidebar.slider("σ Gaussiano:", 1.0, 5.0, 2.0, 0.5)
    kernel_median = st.sidebar.slider("Kernel Mediana:", 3, 9, 3, 2)

elif seccion == "⚡ 3. Ruido Sal & Pimienta":
    # Solo ruido S&P
    st.sidebar.markdown("### 🎲 Ruido Sal & Pimienta")
    p_noise = st.sidebar.slider("Probabilidad:", 0.01, 0.10, 0.03, 0.01)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Parámetros Bilateral")
    d_bilateral = st.sidebar.slider("d (diámetro):", 5, 15, 9, 2)
    sigma_color = st.sidebar.slider("σr (color):", 30, 100, 75, 5)
    sigma_space = st.sidebar.slider("σs (espacio):", 30, 100, 75, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Otros Filtros")
    sigma_gauss = st.sidebar.slider("σ Gaussiano:", 1.0, 5.0, 2.0, 0.5)
    kernel_median = st.sidebar.slider("Kernel Mediana:", 3, 9, 3, 2)

elif seccion == "📊 4. Comparación Final":
    # Puede elegir entre ambos
    st.sidebar.markdown("### 🎲 Tipo de Ruido")
    tipo_ruido = st.sidebar.radio("", ["Gaussiano", "Sal & Pimienta"])
    
    if tipo_ruido == "Gaussiano":
        sigma_noise = st.sidebar.slider("Intensidad (σ):", 10, 50, 30)
    else:
        p_noise = st.sidebar.slider("Probabilidad:", 0.01, 0.10, 0.03, 0.01)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Parámetros Bilateral")
    d_bilateral = st.sidebar.slider("d (diámetro):", 5, 15, 9, 2)
    sigma_color = st.sidebar.slider("σr (color):", 30, 100, 75, 5)
    sigma_space = st.sidebar.slider("σs (espacio):", 30, 100, 75, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Otros Filtros")
    sigma_gauss = st.sidebar.slider("σ Gaussiano:", 1.0, 5.0, 2.0, 0.5)
    kernel_median = st.sidebar.slider("Kernel Mediana:", 3, 9, 3, 2)

# ============= CONTENIDO POR SECCIÓN =============

if seccion == "🎯 1. Teoría":
    st.markdown("---")
    st.markdown("## Fundamento del Filtro Bilateral")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Fórmula Matemática")
        st.latex(r'''
        I^{filt}(x) = \frac{1}{W_x} \sum_{x_i \in \Omega} I(x_i) \cdot 
        G_{\sigma_s}(\|x_i - x\|) \cdot G_{\sigma_r}(\|I(x_i) - I(x)\|)
        ''')
        
        st.markdown("""
        ### Componentes Clave:
        
        1. **G_σs (Espacial)**: Gaussiana basada en distancia geométrica
           - Píxeles cercanos → mayor peso
           - Similar al filtro Gaussiano clásico
        
        2. **G_σr (Rango/Color)**: Gaussiana basada en diferencia de intensidad
           - Píxeles similares → mayor peso
           - **Esto preserva los bordes**
        
        3. **Combinación**: Multiplica ambos pesos
           - Solo píxeles cercanos Y similares contribuyen significativamente
        """)
    
    with col2:
        st.markdown("### 🎛️ Parámetros")
        
        st.info("""
        **d (diámetro):**
        - Tamaño del vecindario
        - Mayor → más suavizado
        - Afecta tiempo de cómputo
        """)
        
        st.success("""
        **σs (espacial):**
        - Control de suavizado espacial
        - Mayor → se parece al Gaussiano
        - Rango típico: 30-100
        """)
        
        st.warning("""
        **σr (rango/color):**
        - Selectividad por intensidad
        - Mayor → menos selectivo
        - Menor → preserva bordes mejor
        - Rango típico: 30-100
        """)
    
    st.markdown("---")
    st.info("""
    **🔑 Ventaja Principal:** A diferencia del filtro Gaussiano que suaviza todo uniformemente,
    el bilateral adapta el suavizado según el contenido local, preservando estructuras importantes.
    """)

elif seccion == "🔬 2. Ruido Gaussiano":
    st.markdown("---")
    st.markdown("## Demo: Ruido Gaussiano")
    
    # Aplicar ruido Gaussiano
    noisy = add_gaussian_noise(img_rgb, sigma_noise)
    
    # Aplicar filtros
    bilateral = cv2.bilateralFilter(noisy, d_bilateral, sigma_color, sigma_space)
    gauss = gaussian_rgb(noisy, sigma_gauss)
    median = median_rgb(noisy, kernel_median)
    
    # Mostrar original vs ruidosa
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Imagen Original")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.markdown(f"### Con Ruido Gaussiano (σ={sigma_noise})")
        st.image(noisy, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## Comparación de Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Gaussiano**")
        st.image(gauss, caption=f"σ = {sigma_gauss}", use_container_width=True)
        st.caption("❌ Borra bordes")
        
    with col2:
        st.markdown("**Bilateral** ⭐")
        st.image(bilateral, caption=f"d={d_bilateral}, σr={sigma_color}, σs={sigma_space}", 
                 use_container_width=True)
        st.caption("✅ Reduce ruido + preserva bordes")
        
    with col3:
        st.markdown("**Mediana**")
        st.image(median, caption=f"Kernel {kernel_median}x{kernel_median}", 
                 use_container_width=True)
        st.caption("⚠️ Poco efecto en ruido Gaussiano")
    
    st.markdown("---")
    st.success("""
    **Conclusión:** Con ruido Gaussiano, el filtro bilateral es el más apropiado porque:
    - Reduce efectivamente el ruido (diferencias graduales)
    - Mantiene los bordes nítidos (gracias a G_σr)
    - El Gaussiano difumina todo uniformemente
    - La Mediana no tiene mucho efecto en este tipo de ruido
    """)

elif seccion == "⚡ 3. Ruido Sal & Pimienta":
    st.markdown("---")
    st.markdown("## Demo: Ruido Sal & Pimienta")
    
    # Aplicar ruido S&P
    noisy = add_salt_pepper(img_rgb, p_noise)
    
    # Aplicar filtros
    bilateral = cv2.bilateralFilter(noisy, d_bilateral, sigma_color, sigma_space)
    gauss = gaussian_rgb(noisy, sigma_gauss)
    median = median_rgb(noisy, kernel_median)
    
    # Mostrar original vs ruidosa
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Imagen Original")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.markdown(f"### Con Sal & Pimienta (p={p_noise})")
        st.image(noisy, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## Comparación de Filtros")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Gaussiano**")
        st.image(gauss, caption=f"σ = {sigma_gauss}", use_container_width=True)
        st.caption("❌ No mejora significativamente")
        
    with col2:
        st.markdown("**Bilateral**")
        st.image(bilateral, caption=f"d={d_bilateral}, σr={sigma_color}, σs={sigma_space}", 
                 use_container_width=True)
        st.caption("⚠️ Reduce pero no elimina")
        
    with col3:
        st.markdown("**Mediana** ⭐")
        st.image(median, caption=f"Kernel {kernel_median}x{kernel_median}", 
                 use_container_width=True)
        st.caption("✅ Más eficiente para S&P")
    
    st.markdown("---")
    st.warning("""
    **Explicación:** ¿Por qué la mediana es mejor aquí?
    
    - **Ruido S&P:** Valores extremos (0 o 255)
    - **Bilateral:** Los píxeles extremos reciben menor peso, pero aún sesgan el promedio ponderado
    - **Mediana:** Simplemente elige el valor central, descartando completamente los outliers
    - **Gaussiano:** Promedia todo, incluyendo los valores extremos
    
    **Conclusión:** La elección del filtro depende del tipo de ruido que queremos reducir.
    """)

else:  # Sección 4: Comparación Final
    st.markdown("---")
    st.markdown("## Comparación Final: Bilateral vs Otros Filtros")
    
    # # Aplicar el ruido elegido
    # if tipo_ruido == "Gaussiano":
    #     noisy = add_gaussian_noise(img_rgb, sigma_noise)
    # else:
    #     noisy = add_salt_pepper(img_rgb, p_noise)
    
    # # Aplicar filtros
    # bilateral = cv2.bilateralFilter(noisy, d_bilateral, sigma_color, sigma_space)
    # gauss = gaussian_rgb(noisy, sigma_gauss)
    # median = median_rgb(noisy, kernel_median)
    
    # # Mostrar comparación de imágenes
    # st.markdown(f"### Resultados con Ruido {tipo_ruido}")
    
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     st.markdown("**Original**")
    #     st.image(img_rgb, use_container_width=True)
        
    # with col2:
    #     st.markdown("**Con Ruido**")
    #     st.image(noisy, use_container_width=True)
        
    # with col3:
    #     st.markdown("**Bilateral**")
    #     st.image(bilateral, use_container_width=True)
        
    # with col4:
    #     st.markdown("**Gaussiano**")
    #     st.image(gauss, use_container_width=True)
    
    st.markdown("---")
    
    # Crear tabla comparativa
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Filtro Bilateral")
        st.success("""
        **Ventajas:**
        - ✅ Preserva bordes
        - ✅ Reduce ruido Gaussiano
        - ✅ Mantiene estructuras
        - ✅ Adaptativo al contenido
        
        **Desventajas:**
        - ❌ Computacionalmente costoso
        - ❌ Más parámetros a ajustar
        - ❌ No óptimo para ruido impulsivo
        
        **Uso ideal:**
        - Ruido Gaussiano
        - Imágenes con estructuras importantes
        - Preprocesamiento satelital
        """)
    
    with col2:
        st.markdown("### 📊 Filtro Gaussiano")
        st.info("""
        **Ventajas:**
        - ✅ Muy rápido
        - ✅ Simple (1 parámetro)
        - ✅ Matemáticamente bien definido
        
        **Desventajas:**
        - ❌ Difumina bordes
        - ❌ Pérdida de detalles
        - ❌ No selectivo
        
        **Uso ideal:**
        - Suavizado general
        - Cuando la velocidad es crítica
        - Preprocesamiento simple
        """)
    
    with col3:
        st.markdown("### 📈 Filtro Mediana")
        st.warning("""
        **Ventajas:**
        - ✅ Excelente para S&P
        - ✅ Preserva bordes
        - ✅ Robusto a outliers
        
        **Desventajas:**
        - ❌ Artefactos en escalones
        - ❌ No óptimo para Gaussiano
        - ❌ Puede eliminar detalles finos
        
        **Uso ideal:**
        - Ruido impulsivo (S&P)
        - Eliminación de outliers
        - Post-procesamiento
        """)
    
    st.markdown("---")
    st.markdown("## 🛰️ Aplicación en Imágenes Satelitales")
    
    st.info("""
    **¿Por qué el filtro bilateral es importante en teledetección?**
    
    1. **Ruido atmosférico:** Las imágenes satelitales sufren interferencia atmosférica
       que genera ruido tipo Gaussiano
    
    2. **Preservación de estructuras:** Es crucial mantener los límites de:
       - Terrenos agrícolas
       - Edificaciones urbanas
       - Caminos y vías
       - Cuerpos de agua
    
    3. **Análisis posterior:** Un buen preprocesamiento facilita:
       - Segmentación automática
       - Clasificación de cobertura
       - Detección de cambios
       - Análisis multitemporal
    
    **Trade-off:** Aunque es más lento, la calidad superior del resultado 
    justifica su uso en procesamiento satelital donde la precisión es prioritaria.
    """)

st.sidebar.markdown("---")
st.sidebar.info("*Alumnas Florio y Sansone - Análisis de Imágenes Satelitales - ITBA*")


