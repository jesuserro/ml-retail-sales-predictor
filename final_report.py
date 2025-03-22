# Auto-install dependencies if not found
import subprocess
import sys

required = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "xgboost", "jinja2", "weasyprint", "joblib"
]

def install_missing_packages():
    for pkg in required:
        try:
            __import__(pkg if pkg != "scikit-learn" else "sklearn")
        except ImportError:
            print(f"📦 Instalando paquete faltante: {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

install_missing_packages()

# final_report.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
from io import BytesIO
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from jinja2 import Template
from weasyprint import HTML

# ========================
# FUNCIONES AUXILIARES
# ========================

def plot_to_base64(fig, filename):
    os.makedirs("img", exist_ok=True)
    # Guardar imagen como archivo
    filepath = os.path.join("img", filename)
    fig.savefig(filepath, format="jpg", bbox_inches="tight")
    
    # Convertir a base64
    buf = BytesIO()
    fig.savefig(buf, format="jpg", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(metrics, images_b64, worst_cases_html, output_path="docs/report.html"):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Informe de Evaluación del Modelo</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 40px; }
            h1, h2 { color: #2c3e50; }
            img { max-width: 100%; margin: 20px 0; border: 1px solid #ccc; }
            .metric { font-size: 1.2em; margin-bottom: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: right; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>📊 Informe de Evaluación del Modelo</h1>
        <p><strong>Fecha:</strong> {{ date }}</p>

        <h2>✅ Métricas</h2>
        <div class="metric">🔹 R² Score: <strong>{{ r2 }}</strong></div>
        <div class="metric">🔹 RMSE: <strong>{{ rmse }}</strong></div>
        <div class="metric">🔹 MAE: <strong>{{ mae }}</strong></div>

        <h2>📈 Gráficas</h2>
        <h3>Predicciones vs Ventas Reales</h3>
        <img src="data:image/jpg;base64,{{ scatter }}"/>

        <h3>Distribución del Error</h3>
        <img src="data:image/jpg;base64,{{ error_hist }}"/>

        <h3>Correlación</h3>
        <img src="data:image/jpg;base64,{{ heatmap }}"/>

        {% if importance %}
        <h3>Importancia de Características</h3>
        <img src="data:image/jpg;base64,{{ importance }}"/>
        {% endif %}

        <h2>🔍 Casos con mayor error absoluto</h2>
        {{ worst_cases_html | safe }}

        <hr>
        <p>🧠 IronKaggle Report — Generado automáticamente</p>
    </body>
    </html>
    """
    template = Template(html_template)
    html = template.render(**metrics, **images_b64, worst_cases_html=worst_cases_html, date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Reporte HTML generado: {output_path}")

    # Generar también PDF
    pdf_path = output_path.replace(".html", ".pdf")
    HTML(string=html).write_pdf(pdf_path)
    print(f"📄 Reporte PDF generado: {pdf_path}")

# ========================
# CARGA DE DATOS
# ========================

# 1. Predicciones previas (con True_index)
pred_df = pd.read_csv("3_sales_predictions.csv")
real_df = pd.read_csv("4_sales_real_solutions.csv")

# 2. Unir por True_index
df = pd.merge(pred_df, real_df, on="True_index", how="inner")

# 3. Calcular métricas
df['error'] = df['Sales'] - df['Predicted_Sales']
df['abs_error'] = df['error'].abs()

r2 = r2_score(df['Sales'], df['Predicted_Sales'])
rmse = np.sqrt(mean_squared_error(df['Sales'], df['Predicted_Sales']))
mae = mean_absolute_error(df['Sales'], df['Predicted_Sales'])

# 4. Ejemplos con mayor error
worst_cases = df.sort_values(by='abs_error', ascending=False).head(10)
worst_cases_html = worst_cases[['True_index', 'Store_ID', 'Sales', 'Predicted_Sales', 'error', 'abs_error']] \
    .rename(columns={
        'Sales': 'Ventas Reales',
        'Predicted_Sales': 'Predicción',
        'error': 'Error',
        'abs_error': 'Error Absoluto'
    }).to_html(index=False, float_format="%.2f")

# ========================
# GRÁFICAS → BASE64
# ========================

# Gráfico 1: Scatter Real vs Predicho
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='Sales', y='Predicted_Sales', data=df, ax=ax1)
ax1.plot([df['Sales'].min(), df['Sales'].max()], [df['Sales'].min(), df['Sales'].max()], 'r--')
ax1.set(title="Predicciones vs Ventas Reales", xlabel="Ventas Reales", ylabel="Ventas Predichas")
scatter_b64 = plot_to_base64(fig1, "scatter_real_vs_pred.jpg")
plt.close(fig1)

# Gráfico 2: Histograma del error
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df['error'], kde=True, bins=30, color="orange", ax=ax2)
ax2.set(title="Distribución del Error de Predicción", xlabel="Error (Real - Predicho)")
error_b64 = plot_to_base64(fig2, "histogram_error.jpg")
plt.close(fig2)

# Gráfico 3: Heatmap de correlación
corr = df[['Sales', 'Predicted_Sales', 'error']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax3)
ax3.set_title("Correlación entre Predicción y Real")
heatmap_b64 = plot_to_base64(fig3, "heatmap_correlation.jpg")
plt.close(fig3)

# Gráfico 4: Importancia de variables (si existe el modelo)
importance_b64 = None
if os.path.exists("sales_model.pkl"):
    try:
        model = joblib.load("sales_model.pkl")
        importances = model.feature_importances_
        feature_names = model.get_booster().feature_names

        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature',
                    dodge=False, legend=False, palette='viridis', ax=ax4)
        ax4.set_title("Importancia de Características (ordenada)")
        importance_b64 = plot_to_base64(fig4, "feature_importance.jpg") 
        plt.close(fig4)
    except Exception as e:
        print("⚠️ Error cargando el modelo para importancia de variables:", e)

# ========================
# GENERAR INFORME
# ========================

generate_html_report(
    metrics={
        'r2': f"{r2:.4f}",
        'rmse': f"{rmse:.2f}",
        'mae': f"{mae:.2f}"
    },
    images_b64={
        'scatter': scatter_b64,
        'error_hist': error_b64,
        'heatmap': heatmap_b64,
        'importance': importance_b64
    },
    worst_cases_html=worst_cases_html,
    output_path="docs/report.html"
)

