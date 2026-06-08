from __future__ import annotations


def validate_device_type(device_type: str) -> str:
    normalized = str(device_type).strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise ValueError("device_type must be either 'cpu' or 'gpu'.")
    return normalized


def validate_gpu_device_id(gpu_device_id: int) -> int:
    if isinstance(gpu_device_id, bool) or int(gpu_device_id) < 0:
        raise ValueError("gpu_device_id must be a non-negative integer.")
    return int(gpu_device_id)


def lightgbm_device_params(device_type: str, gpu_device_id: int) -> dict[str, object]:
    validated_device = validate_device_type(device_type)
    params: dict[str, object] = {"device_type": validated_device}
    if validated_device == "gpu":
        params["gpu_device_id"] = validate_gpu_device_id(gpu_device_id)
    return params


def catboost_device_params(device_type: str, gpu_device_id: int) -> dict[str, object]:
    validated_device = validate_device_type(device_type)
    if validated_device == "gpu":
        return {"task_type": "GPU", "devices": str(validate_gpu_device_id(gpu_device_id))}
    return {"task_type": "CPU"}
