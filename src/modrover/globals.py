import numpy as np
from numpy.typing import ArrayLike
from regmod.models import (
    TobitModel,
    create_binomial_model,
    create_gaussian_model,
    create_poisson_model,
)

model_type_dict = {
    "gaussian": create_gaussian_model,
    "poisson": create_poisson_model,
    "binomial": create_binomial_model,
    "tobit": TobitModel,
}


def get_r2(
    obs: ArrayLike, pred: ArrayLike, obs_mean: float | None = None
) -> float:
    ss_residual = np.sum((obs - pred) ** 2)
    if obs_mean is None:
        obs_mean = obs.mean()
    ss_total = np.sum((obs - obs_mean) ** 2)
    return float(min(max(0.0, 1 - ss_residual / ss_total), 1.0))


def get_rmse(obs: ArrayLike, pred: ArrayLike) -> float:
    metric = np.sqrt(np.mean((obs - pred) ** 2))
    metric = np.exp(-metric)
    return float(metric)


def get_mad(obs: ArrayLike, pred: ArrayLike) -> float:
    metric = np.median(np.abs(obs - pred))
    metric = np.exp(-metric)
    return float(metric)


def get_mape(obs: ArrayLike, pred: ArrayLike) -> float:
    metric = np.mean(np.abs((obs - pred) / obs))
    metric = np.exp(-metric)
    return float(metric)


metric_dict = {
    "r2": get_r2,
    "rmse": get_rmse,
    "mad": get_mad,
    "mape": get_mape,
}
