# Detector de Caries Dental

Aplicación con Streamlit que detecta caries en radiografías o imágenes dentales usando un modelo de Roboflow.

## Requisitos

- Python 3.10+
- Clave de API de [Roboflow](https://app.roboflow.com/settings/api)

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración

Copia el archivo de ejemplo y agrega tu API Key:

```bash
cp .env.example .env
```

Edita `.env` y reemplaza `tu_api_key_aqui` con tu clave de Roboflow.

## Uso

```bash
streamlit run app.py
```

Sube una imagen dental (JPG, PNG, BMP, TIFF) y la app mostrará las zonas con posible caries.

## Características

- Modelo alojado en Roboflow (dental-caries-i8vaj)
- Ajuste de umbral de confianza
- Selección de versión del modelo
- Descarga de la imagen anotada
- Manejo de errores y validación de API key
