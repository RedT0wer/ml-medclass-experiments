"""Модуль моделей проекта."""

from .registry import (
    MODELS_REGISTRY,
    ClassifierProtocol,
    ModelRegistration,
    get_default_params,
    get_model,
    get_model_class,
    get_model_registration,
    list_models,
    register_model,
)

__all__ = [
    "MODELS_REGISTRY",
    "ClassifierProtocol",
    "ModelRegistration",
    "get_default_params",
    "get_model",
    "get_model_class",
    "get_model_registration",
    "list_models",
    "register_model",
]
