from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from typing import Any, Dict, List
import torch

class CheapDistillTask(Task[torch.Tensor]):
    def __init__(
        self,
        *,
        base_model_tensors: Dict[str, torch.Tensor],  # Тензоры маленькой модели
        big_model_tensors: Dict[str, torch.Tensor],   # Тензоры большой модели
        dtype: torch.dtype,
        parameters: ImmutableMap[str, Any],
    ):
        self.base_model_tensors = base_model_tensors
        self.big_model_tensors = big_model_tensors
        self.dtype = dtype
        self.parameters = parameters

    def arguments(self) -> Dict[str, Task]:
        return {
            "big_tensors": self.big_model_tensors,
            "small_tensors": self.base_model_tensors,
        }

    def priority(self) -> int:
        return 1  # Высокий приоритет для раннего выполнения

    def uses_accelerator(self) -> bool:
        return True  # Включаем GPU-ускорение

    def execute(
        self,
        big_tensors: Dict[str, torch.Tensor],
        small_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Реализация дешёвой дистилляции:
        - Переносим знания из больших слоёв в маленькие.
        - Используем dtype для оптимизации памяти.
        """
        distilled_tensors = {}

        for layer_name, small_tensor in small_tensors.items():
            if layer_name in big_tensors:
                big_tensor = big_tensors[layer_name].to(self.dtype)
                small_tensor = small_tensor.to(self.dtype)

                # Простое усреднение между большим и маленьким тензорами
                distilled_tensor = (big_tensor + small_tensor) / 2.0
                distilled_tensors[layer_name] = distilled_tensor
            else:
                # Если слой отсутствует в большой модели, оставляем маленький как есть
                distilled_tensors[layer_name] = small_tensor.to(self.dtype)

        return distilled_tensors


class CheapDistillMerge(MergeMethod):
    def name(self) -> str:
        return "cheap_distill"

    def pretty_name(self) -> str:
        return "Cheap Distillation"

    def reference_url(self) -> str:
        return "https://example.com/cheap-distill"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("dtype", str, required=False, default_value="float16"),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return []

    def make_task(
        self,
        *,
        output_weight: "WeightInfo",  # Предполагается, что WeightInfo определён в MergeKit
        tensors: Dict[str, torch.Tensor],  # Тензоры большой модели
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_tensors: Dict[str, torch.Tensor],  # Тензоры маленькой модели
        **kwargs,
    ) -> Task:
        dtype_str = parameters.get("dtype", "float16")
        dtype = getattr(torch, dtype_str)

        return CheapDistillTask(
            base_model_tensors=base_tensors,
            big_model_tensors=tensors,
            dtype=dtype,
            parameters=parameters,
        )
