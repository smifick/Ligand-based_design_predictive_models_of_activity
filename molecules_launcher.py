"""Molecules: веб-панель для запуска экспериментов через ClearML Agent.

Установить: pip install streamlit clearml
Запустить из корня репозитория: streamlit run molecules_launcher.py

Переменные окружения (опционально):
    CLEARML_BASE_TASK_ID — ID ранее успешно созданного train.py Task
    CLEARML_QUEUE        — очередь агента (по умолчанию default)

Важно: менять исходный ML-пайплайн не требуется. Здесь создаётся клон
существующего ClearML Task, а не локальный процесс обучения.
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from clearml import Task

# Используем текущие реестры проекта без их изменения.
from model.clearML.registry_model_and_transformer import (
    COMMON_DESCRIPTOR_DEFAULTS,
    DESCRIPTOR_REGISTRY,
    MODEL_REGISTRY as MODEL_CONFIG_REGISTRY,
)
from model.models.create_model import MODEL_REGISTRY as MODEL_FACTORY_REGISTRY


def available_models() -> list[str]:
    """Выбираем только модели, которые поддерживают фабрика и resolver."""
    return sorted(set(MODEL_CONFIG_REGISTRY) & set(MODEL_FACTORY_REGISTRY))


def available_descriptors() -> list[str]:
    return sorted(DESCRIPTOR_REGISTRY)


def default_model_params(name: str) -> dict[str, Any]:
    return dict(MODEL_CONFIG_REGISTRY[name].get("default", {}))


def default_descriptor_params(name: str) -> dict[str, Any]:
    # Общие параметры FP тоже должны быть доступны для настройки.
    return {
        **COMMON_DESCRIPTOR_DEFAULTS,
        **DESCRIPTOR_REGISTRY[name].get("default", {}),
    }


def parse_overrides(raw: str, label: str) -> dict[str, Any]:
    """Только для необязательных дополнительных параметров без дефолта."""
    if not raw.strip():
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label}: некорректный JSON ({exc.msg}, строка {exc.lineno})"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label}: ожидается JSON-объект, например {{}}")
    return result


def parse_nullable_value(raw: str, label: str) -> Any:
    """Пустая строка → None; остальное — число, bool или строка."""
    value = raw.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Удобно вводить строковые значения без JSON-кавычек, например sqrt.
        if value.startswith(("[", "{", '"')):
            raise ValueError(f"{label}: некорректное JSON-значение")
        return value


def nullable_default_text(value: None) -> str:
    return ""  # None в UI означает пустое поле.


def render_value_field(scope: str, item: str, param: str, default: Any) -> Any:
    """Создаёт Streamlit-поле по типу параметра по умолчанию."""
    key = f"param:{scope}:{item}:{param}"
    label = param

    # В Python bool — подкласс int, поэтому проверяем раньше числа.
    if isinstance(default, bool):
        return st.checkbox(label, value=default, key=key)

    if isinstance(default, int):
        return int(st.number_input(label, value=default, step=1, key=key))

    if isinstance(default, float):
        return float(st.number_input(label, value=default, key=key, format="%.6g"))

    if default is None:
        raw = st.text_input(
            label,
            value=nullable_default_text(default),
            placeholder="None (пусто) или значение, например 12",
            help="Пусто → None. Можно ввести число, true/false или строку.",
            key=key,
        )
        # Валидация только по кнопке: некорректный ввод не должен ломать форму.
        return raw

    if isinstance(default, str):
        return st.text_input(label, value=default, key=key)

    if isinstance(default, (dict, list)):
        raw = st.text_area(
            label,
            value=json.dumps(default, ensure_ascii=False, indent=2),
            key=key,
            help="Для составных значений используется JSON.",
        )
        return raw

    # Редкий случай нестандартного типа в реестре.
    raw = st.text_input(label, value=str(default), key=key)
    if raw == str(default):
        return default
    return parse_nullable_value(raw, label)


def render_parameter_fields(
    scope: str, item: str, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Показывает поля для ВСЕХ defaults, возвращает ТОЛЬКО изменения."""
    values: dict[str, Any] = {}
    if not defaults:
        st.caption("Параметров по умолчанию в реестре пока нет.")
        return values

    first, second = st.columns(2)
    for index, (param, default) in enumerate(defaults.items()):
        with (first if index % 2 == 0 else second):
            values[param] = render_value_field(scope, item, param, default)

    return values


def collect_modified_values(
    values: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    """После отправки формы преобразует значения и отсеивает неизменённые."""
    modified: dict[str, Any] = {}
    for key, raw in values.items():
        default = defaults[key]
        if default is None:
            value = parse_nullable_value(raw, key)
        elif isinstance(default, (dict, list)):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{key}: некорректный JSON ({exc.msg})") from exc
            if not isinstance(value, type(default)):
                raise ValueError(f"{key}: ожидается {type(default).__name__}")
        else:
            value = raw
        if value != default:
            modified[key] = value
    return modified


def combine_overrides(
    edited: dict[str, Any], extras: dict[str, Any], defaults: dict[str, Any], label: str
) -> dict[str, Any]:
    collisions = set(extras) & set(defaults)
    if collisions:
        raise ValueError(
            f"{label}: эти ключи уже есть в полях формы: "
            + ", ".join(sorted(collisions))
        )
    return {**edited, **extras}


def reset_current_selection(model: str, descriptor: str) -> None:
    """Сбрасываем только значения выбранных компонентов, не весь интерфейс."""
    prefixes = (f"param:model:{model}:", f"param:descriptor:{descriptor}:")
    extra_keys = (f"extra:model:{model}", f"extra:descriptor:{descriptor}")
    for key in list(st.session_state):
        if key.startswith(prefixes) or key in extra_keys:
            del st.session_state[key]


def launch_experiment(
    *,
    base_task_id: str,
    queue_name: str,
    model: str,
    descriptor: str,
    seed: int,
    model_overrides: dict[str, Any],
    descriptor_overrides: dict[str, Any],
):
    """Клонирует шаблонный Task и отправляет его ClearML Agent."""
    if model not in available_models():
        raise ValueError(f"Модель {model!r} отсутствует в рабочих реестрах")
    if descriptor not in available_descriptors():
        raise ValueError(f"Дескриптор {descriptor!r} отсутствует в DESCRIPTOR_REGISTRY")
    if not base_task_id.strip():
        raise ValueError("Укажи ID шаблонного эксперимента ClearML")
    if not queue_name.strip():
        raise ValueError("Укажи очередь ClearML Agent")

    cloned = Task.clone(
        source_task=base_task_id.strip(),
        name=f"{model} + {descriptor} | seed={seed}",
    )
    cloned.set_parameter("Selection/model", model)
    cloned.set_parameter("Selection/descriptor", descriptor)
    cloned.set_parameter("Selection/seed", seed)
    cloned.set_configuration_object(
        name="Overrides",
        config_dict={
            "model": model_overrides,
            "descriptor": descriptor_overrides,
        },
    )
    Task.enqueue(cloned, queue_name=queue_name.strip())
    return cloned


def main() -> None:
    st.set_page_config(page_title="Molecules | Experiment Launcher", page_icon="🧬")
    st.title("🧬 Molecules — запуск эксперимента")
    st.caption(
        "Выбор модели и дескриптора автоматически меняет поля гиперпараметров. "
        "Обучение запускается через ClearML Agent."
    )

    models = available_models()
    descriptors = available_descriptors()
    if not models or not descriptors:
        st.error("Нет доступных моделей/дескрипторов. Проверь регистрации.")
        st.stop()

    unsupported_models = sorted(set(MODEL_FACTORY_REGISTRY) - set(MODEL_CONFIG_REGISTRY))
    if unsupported_models:
        st.info(
            "Модели без конфигураций в resolver пока недоступны: "
            + ", ".join(unsupported_models)
        )

    # ВНЕ st.form: при смене модели/дескриптора Streamlit сразу перерисует поля.
    left, right = st.columns(2)
    with left:
        model = st.selectbox("Модель", models, key="selected_model")
    with right:
        descriptor = st.selectbox("Дескриптор", descriptors, key="selected_descriptor")

    model_defaults = default_model_params(model)
    descriptor_defaults = default_descriptor_params(descriptor)

    if st.button("Сбросить параметры выбранных компонентов"):
        reset_current_selection(model, descriptor)
        st.rerun()

    with st.form("experiment_form"):
        st.subheader("Гиперпараметры")
        st.caption(
            "Поля создаются из default в реестрах. "
            "В ClearML передаются только параметры, отличающиеся от значений по умолчанию."
        )
        model_tab, descriptor_tab = st.tabs(
            [f"Модель: {model}", f"Дескриптор: {descriptor}"]
        )
        with model_tab:
            model_edited = render_parameter_fields("model", model, model_defaults)
        with descriptor_tab:
            descriptor_edited = render_parameter_fields(
                "descriptor", descriptor, descriptor_defaults
            )

        with st.expander("Дополнительные параметры (необязательно)"):
            st.caption(
                "Только для параметров, которых ещё нет в default. "
                "Обычные гиперпараметры редактируются выше без JSON."
            )
            left, right = st.columns(2)
            with left:
                model_extra = st.text_area(
                    "Дополнительные параметры модели (JSON)",
                    value="{}",
                    key=f"extra:model:{model}",
                )
            with right:
                descriptor_extra = st.text_area(
                    "Дополнительные параметры дескриптора (JSON)",
                    value="{}",
                    key=f"extra:descriptor:{descriptor}",
                )

        st.subheader("Настройки запуска")
        seed = st.number_input(
            "Seed", min_value=0, max_value=2_147_483_647, value=67, step=1
        )
        base_task_id = st.text_input(
            "ID шаблонного Task",
            value=os.getenv("CLEARML_BASE_TASK_ID", ""),
            help="ID успешно созданного эксперимента, исходный код которого выполняет train.py",
        )
        queue_name = st.text_input(
            "Очередь ClearML Agent", value=os.getenv("CLEARML_QUEUE", "default")
        )
        preview, submit = st.columns(2)
        with preview:
            show_preview = st.form_submit_button("Показать конфигурацию")
        with submit:
            enqueue = st.form_submit_button("Отправить в очередь", type="primary")

    if show_preview or enqueue:
        try:
            model_overrides = combine_overrides(
                collect_modified_values(model_edited, model_defaults),
                parse_overrides(model_extra, "Модель"),
                model_defaults,
                "Модель",
            )
            descriptor_overrides = combine_overrides(
                collect_modified_values(descriptor_edited, descriptor_defaults),
                parse_overrides(descriptor_extra, "Дескриптор"),
                descriptor_defaults,
                "Дескриптор",
            )
        except ValueError as exc:
            st.error(f"Ошибка конфигурации: {exc}")
            st.stop()

        if show_preview:
            st.subheader("Конфигурация для ClearML")
            st.json(
                {
                    "Selection": {
                        "model": model,
                        "descriptor": descriptor,
                        "seed": int(seed),
                    },
                    "Overrides": {
                        "model": model_overrides,
                        "descriptor": descriptor_overrides,
                    },
                }
            )
            st.caption("Это предпросмотр: задача в очередь не отправлена.")

        if enqueue:
            try:
                task = launch_experiment(
                    base_task_id=base_task_id,
                    queue_name=queue_name,
                    model=model,
                    descriptor=descriptor,
                    seed=int(seed),
                    model_overrides=model_overrides,
                    descriptor_overrides=descriptor_overrides,
                )
            except Exception as exc:
                st.error(f"Не удалось отправить эксперимент в очередь: {exc}")
            else:
                st.success(f"Эксперимент отправлен в очередь: {task.id}")
                st.link_button("Открыть в ClearML", task.get_output_log_web_page())


if __name__ == "__main__":
    main()
