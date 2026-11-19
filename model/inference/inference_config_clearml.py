"""Параметры inference для уже обученного Pipeline из ClearML.

Изменяйте MODEL_ID здесь или в секции `Inference` клонированного
ClearML-эксперимента. Не указывайте Task ID: нужен именно Model ID.
"""

PROJECT_NAME = "Ligand-Activity"
TASK_NAME = "INFERENCE: Pipeline from ClearML"

SELECTION = {
    "model_id": "PASTE_CLEARML_MODEL_ID_HERE",
    "save_predictions": True,
    "save_metrics": True,
}

# Локальные отчёты: внутри этой папки создаётся отдельный каталог по Task ID.
OUTPUT_DIR = "artifacts/inference"
