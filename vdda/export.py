"""
===============================================================================
INSIGHTLAB — EXPORT MODULE (HTML Report Generation)
===============================================================================

Este módulo permite:
    • Construir reportes HTML estáticos de InsightLab
    • Combinar narrativas, visualizaciones, tablas, texto
    • Insertar estilos InsightLab automáticamente
    • Exportador ligero, silencioso, profesional

Salida final:
    → reporte.html (sin dependencias externas)

Comentarios en español.
Todo lo mostrado en inglés.

VERSION: 4.0 (Updated for unified core)
===============================================================================
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

from .core import CONFIG


# =============================================================================
# 1. PLANTILLA HTML BASE
# =============================================================================

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>

    <style>
        /* Estilos principales de InsightLab se insertan aquí */
        {style}

        body {{
            padding: 40px;
            margin: auto;
            max-width: 900px;
            background: #ffffff;
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            color: #1d085e;
            margin-bottom: 20px;
        }}

        h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            color: #1d085e;
            margin-top: 35px;
        }}

        img {{
            max-width: 100%;
            margin: 15px 0;
        }}

        /* Additional export-specific styles */
        .export-footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #13d6c1;
            text-align: center;
            color: #6c757d;
            font-size: 0.85rem;
        }}
    </style>
</head>

<body>
    <h1>{title}</h1>
    <p style="color:#6c757d; font-size:0.9rem;">
        Generated with InsightLab · {timestamp}
    </p>

    {content}

    <div class="export-footer">
        InsightLab — Professional Data Analysis & Storytelling
    </div>
</body>
</html>
"""


# =============================================================================
# 2. OBTENER CSS DE INSIGHTLAB
# =============================================================================

def _get_css() -> str:
    """
    Obtener CSS desde el sistema de estilos unificado.

    Usa CONFIG.Style.get_css_string() definido en core.py.
    """
    return CONFIG.Style.get_css_string()


# =============================================================================
# 3. EXPORTAR HTML SIMPLE
# =============================================================================

def export_html(path: str, sections: list, title: str = "InsightLab Report") -> str:
    """
    Exportar reporte HTML con narrativas y visualizaciones.

    Args:
        path: ruta del archivo HTML a crear
        sections: lista de strings HTML (cada uno es una sección)
        title: título del reporte

    Returns:
        str con la ruta del archivo creado
    """
    if not isinstance(sections, (list, tuple)):
        raise ValueError("sections must be a list of HTML strings")

    # Combinar todas las secciones
    content = "\n\n".join(str(s) for s in sections)

    # Obtener CSS del core unificado
    css = _get_css()

    # Generar HTML completo
    html = _HTML_TEMPLATE.format(
        title=title,
        style=css,
        content=content,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # Escribir archivo
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    return path


# =============================================================================
# 4. EXPORTAR NOTEBOOK A HTML (CON OPCIÓN DE EJECUCIÓN)
# =============================================================================

def export_notebook(
    input_path: str,
    output_path: str = "insightlab_report.html",
    title: str = "InsightLab Report",
    execute: bool = True,
    timeout: int = 600,
) -> str:
    """
    Exportar un notebook .ipynb a HTML estático profesional.

    Características:
        - Opcionalmente ejecuta el notebook en un kernel limpio (execute=True)
        - Sin celdas de código (solo outputs)
        - Estilos InsightLab aplicados

    Args:
        input_path: ruta del notebook .ipynb
        output_path: ruta del HTML a generar
        title: título del reporte (aplicado al <title> si existe)
        execute: si True, nbconvert ejecuta el notebook antes de exportar
        timeout: timeout de ejecución en segundos

    Returns:
        str con la ruta del archivo generado

    Raises:
        FileNotFoundError: si input_path no existe
        RuntimeError: si jupyter nbconvert falla
    """

    # 1) Validar que el input existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Notebook not found: {input_path}")

    # Directorio donde vive el notebook
    notebook_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    notebook_name = os.path.basename(input_path)

    # Nombre base (sin extensión) para nbconvert
    temp_base = "_insightlab_temp"
    temp_html = os.path.join(notebook_dir, temp_base + ".html")

    # 2) Comando nbconvert
    cmd = [
        "jupyter", "nbconvert",
        "--to", "html",
        "--TemplateExporter.exclude_input=True",
        "--output", temp_base,        # nbconvert producirá temp_base + ".html"
    ]

    if execute:
        cmd.extend([
            "--execute",
            f"--ExecutePreprocessor.timeout={timeout}",
        ])

    # Usamos el nombre del notebook relativo a notebook_dir
    cmd.append(notebook_name)

    # 3) Ejecutar conversión en el directorio del notebook
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=notebook_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Jupyter nbconvert failed:\n{result.stderr}")

    # 4) Leer HTML base generado por nbconvert
    if not os.path.exists(temp_html):
        raise RuntimeError(f"nbconvert did not create {temp_html}")

    with open(temp_html, "r", encoding="utf-8") as f:
        html_base = f.read()

    # 5) Inyectar estilos InsightLab
    css = _get_css()
    final_html = html_base.replace(
        "</head>",
        f"<style>{css}</style></head>",
    )

    # 6) Ajustar <title> si existe
    if "<title>" in final_html and "</title>" in final_html:
        start = final_html.index("<title>") + len("<title>")
        end = final_html.index("</title>")
        final_html = final_html[:start] + title + final_html[end:]

    # 7) Escribir archivo final (ruta que tú quieras)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    # 8) Eliminar temporal
    if os.path.exists(temp_html):
        os.remove(temp_html)

    return output_path



# =============================================================================
# 5. EXPORTAR CON IMÁGENES EMBEBIDAS
# =============================================================================

def export_html_with_images(
    path: str,
    sections: list,
    image_paths: list | None = None,
    title: str = "InsightLab Report",
) -> str:
    """
    Exportar HTML con imágenes convertidas a base64 (sin archivos externos).

    Args:
        path: ruta del HTML a crear
        sections: lista de secciones HTML
        image_paths: lista de rutas de imágenes a embedear
        title: título del reporte

    Returns:
        str con la ruta del archivo creado
    """
    import base64

    image_paths = image_paths or []

    # Si hay imágenes, convertir a base64 y agregarlas al final
    for img_path in image_paths:
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
                ext = os.path.splitext(img_path)[1][1:]  # png, jpg, etc.
                img_html = (
                    f'<img src="data:image/{ext};base64,{img_data}" '
                    f'style="max-width:100%;">'
                )
                sections.append(img_html)

    return export_html(path, sections, title)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "export_html",
    "export_notebook",
    "export_html_with_images",
]
