import pathlib
import torch
import gc
from typing import Optional, Union, Any

from graphegfr.models import GraphEGFR


class GraphEGFRAdapter:
    def __init__(
            self,
            path_weights: str | pathlib.Path = None,
            device: str = 'cpu',
            **model_kwargs
    ):
        self.device = device

        self.model = GraphEGFR(**model_kwargs)

        if path_weights is not None:
            self.load_weights(path_weights)
        else:
            self.model.to(self.device)

    def load_weights(self, path: Union[str, pathlib.Path]):

        path = pathlib.Path(path)
        print(f"Загрузка весов из {path}...")

        loaded_obj = torch.load(path, map_location='cpu')

        # Достаем state_dict (на случай, если авторы сохранили кортеж) поправить как веса скачаю
        if isinstance(loaded_obj, tuple):
            state_dict = loaded_obj[0].state_dict()
        elif hasattr(loaded_obj, 'state_dict'):
            state_dict = loaded_obj.state_dict()
        else:
            state_dict = loaded_obj

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Загружаем веса в модель
        try:
            self.model.load_state_dict(state_dict)
            print("Веса успешно загружены!")
        except RuntimeError as e:
            print(f"Ошибка при загрузке весов. Возможно, архитектура не совпадает.\n{e}")
            raise

        # Переносим модель на нужное устройство
        self.model.to(self.device)

        # Очищаем память
        del loaded_obj, state_dict
        gc.collect()

    def predict(self, *args, **kwargs) -> Any:
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)