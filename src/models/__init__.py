"""Модуль моделей проекта."""

from .decision_tree import DecisionTreeModel
from .lightgbm import LightGBMModel
from .linear import RidgeClassifierModel, SGDClassifierModel
from .logistic_regression import LogisticRegressionModel
from .naive_bayes import ComplementNBModel
from .random_forest import RandomForestModel
from .registry import (
    MODELS_REGISTRY,
    ClassifierProtocol,
    get_default_params,
    get_model,
    get_model_class,
    get_model_registration,
    list_models,
    register_model,
)
from .svm import SVMModel

__all__ = [
    "DecisionTreeModel",
    "LightGBMModel",
    "RidgeClassifierModel",
    "SGDClassifierModel",
    "LogisticRegressionModel",
    "ComplementNBModel",
    "RandomForestModel",
    "SVMModel",
    "register_model",
    "get_model",
    "get_model_class",
    "get_default_params",
    "list_models",
    "get_model_registration",
    "MODELS_REGISTRY",
]
