from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.prior import GaussianPrior
from regmod.variable import Variable
from scipy.stats import norm

LearnerID = tuple[int, ...]


class ModelStatus(Enum):
    SUCCESS = 0
    SINGULAR = 1
    CV_FAILED = 2
    SOLVER_FAILED = 3
    NOT_FITTED = -1


class Learner:
    """Individual learner class for one specific covariate configuration.

    Parameters
    ----------
    model_class
        Regmod model constructor
    obs
        Name corresponding to the observation column in the data frame
    main_param
        The main parameter we are exploring. This is aligned with the :class:`modrover.rover.Rover` class.
    param_specs
        Parameter settings for the regmod model
    weights
        Name corresponding to the weights column in the data frame
    get_score
        Function that evaluate the score of of the model

    """

    def __init__(
        self,
        model_class: type,
        obs: str,
        main_param: str,
        param_specs: dict[str, dict],
        weights: str = "weights",
        get_score: Callable | None = None,
        lam_ridge: float = 0.0,
    ) -> None:
        self.model_class = model_class
        self.obs = obs
        self.main_param = main_param
        self.weights = weights
        self.get_score = get_score
        self.lam_ridge = lam_ridge

        # convert str to Variable
        for param_spec in param_specs.values():
            priors = []
            if self.lam_ridge > 0.0:
                priors.append(
                    GaussianPrior(mean=0.0, sd=1.0 / np.sqrt(self.lam_ridge))
                )
            param_spec["variables"] = [
                Variable(name, priors=priors)
                for name in param_spec["variables"]
            ]
        self.param_specs = param_specs

        # initialize null model
        self.model = self._get_model()
        self.score = np.nan
        self.status = ModelStatus.NOT_FITTED

        # initialize cross validation model
        self._cv_models = defaultdict(self._get_model)
        self._cv_scores = defaultdict(lambda: None)
        self._cv_status = defaultdict(lambda: ModelStatus.NOT_FITTED)

    @property
    def coef(self) -> NDArray | None:
        """Coefficients in the regmod model."""
        return self.model.opt_coefs

    @coef.setter
    def coef(self, coef: NDArray):
        if len(coef) != self.model.size:
            raise ValueError("Provided coef size don't match")
        self.model.opt_coefs = coef

    @property
    def vcov(self) -> NDArray | None:
        """Variance-covarianace matrix for the coefficients in the regmod model."""
        return self.model.opt_vcov

    @vcov.setter
    def vcov(self, vcov: NDArray):
        if vcov.shape != (self.model.size, self.model.size):
            raise ValueError("Provided vcov shape don't match")
        self.model.opt_vcov = vcov

    def fit(
        self,
        data: DataFrame,
        holdouts: list[str] | None = None,
        **optimizer_options,
    ) -> None:
        """
        Fit a set of models on a series of holdout datasets.

        This method will fit a model over k folds of the dataset, where k is the
        length of the provided holdouts list. It is up to the user to decide the
        train-test splits for each holdout column.

        On each fold of the dataset, the trained model will predict out on the
        validation set and obtain a evaluate. The averaged evaluate across all
        folds becomes the model's overall score.

        Finally, a model is trained with all data in order to generate the final
        coefficients.

        Parameters
        ----------
        data
            A dataframe containing the training data
        holdouts
            Which column names to iterate over for cross validation. If it is
            `None`, insample performance score will be used to evaluate the
            model.
        **optimizer_options
            Extra options for the optimizer.

        """
        if self.status != ModelStatus.NOT_FITTED:
            return
        if holdouts:
            # If holdout cols are provided, loop through to calculate OOS score
            for holdout in holdouts:
                data_group = data.groupby(holdout)
                self._cv_status[holdout] = self._fit(
                    data_group.get_group(0),
                    self._cv_models[holdout],
                    **optimizer_options,
                )
                if self._cv_status[holdout] == ModelStatus.SUCCESS:
                    self._cv_scores[holdout] = self.evaluate(
                        data_group.get_group(1), self._cv_models[holdout]
                    )
                else:
                    self.status = ModelStatus.CV_FAILED
                    break
            if self.status != ModelStatus.CV_FAILED:
                self.score = np.mean(list(self._cv_scores.values()))
            # clear all cv models for storage efficiency
            self._cv_models.clear()

        # Fit final model with all data included
        if self.status != ModelStatus.CV_FAILED:
            self.status = self._fit(data, **optimizer_options)
            # If holdout cols not provided, use in-sample evaluate for the full data model
            if self.status == ModelStatus.SUCCESS and (not holdouts):
                self.score = self.evaluate(data)

    def predict(
        self,
        data: DataFrame,
        model: RegmodModel | None = None,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        """Generate prediction using regmod model. This function will return
        predictions for the :code:`main_param` with given data.

        Parameters
        ----------
        data
            A dataset to generate predictions from
        model
            A fitted RegmodModel. If it is ``None``, will use the overall model
            rather than the cross-validation model.
        return_ui
            If ``return_ui=True``, a matrix will be returned. The first row
            is the point prediction, second and thrid rows are the lower and
            upper bound of the prediction.
        alpha
            When ``return_ui=True``, function will return (1 - ``alpha``)
            uncertainty interval. By default, ``alpha=0.05``.

        """
        model = model or self.model
        model.data.attach_df(data)
        index = model.param_names.index(self.main_param)
        param = model.params[index]

        coef_index = model.indices[index]
        coef = model.opt_coefs[coef_index]

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = param.get_mat(model.data)
        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = model.opt_vcov[coef_index, coef_index]
            lin_param_sd = np.sqrt((mat.dot(vcov) * mat).sum(axis=1))
            lin_param_lower = norm.ppf(
                0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            lin_param_upper = norm.ppf(
                1 - 0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            pred = np.vstack(
                [
                    pred,
                    param.inv_link.fun(lin_param_lower),
                    param.inv_link.fun(lin_param_upper),
                ]
            )

        model.data.detach_df()
        return pred

    def evaluate(
        self, data: DataFrame, model: RegmodModel | None = None
    ) -> float:
        """Given a model and a test set, generate a performance score.

        Score is based on the provided evaluation metric, comparing the
        difference between observed and predicted values.

        Parameters
        ----------
        data
            The data set to generate predictions from
        model
            The fitted model to set predictions on. If ``None`` will use the
            overall model rather than the cross-validation model.

        """
        model = model or self.model
        if self.get_score is None:
            model.attach_df(data)
            pearson_r2 = model.get_pearson_residuals(model.opt_coefs) ** 2
            score = np.exp(-np.sqrt(np.mean(pearson_r2)))
            model = _detach_df(model)
        else:
            score = self.get_score(
                obs=data[self.obs].to_numpy(),
                pred=self.predict(data, model=model),
            )
        return score

    def _get_model(self) -> RegmodModel:
        # TODO: this shouldn't be necessary in regmod v1.0.0
        data = Data(
            col_obs=self.obs,
            col_weights=self.weights,
            subset_cols=False,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        model = self.model_class(
            data=data,
            param_specs=self.param_specs,
        )
        return model

    def _fit(
        self,
        data: DataFrame,
        model: RegmodModel | None = None,
        **optimizer_options,
    ) -> ModelStatus:
        model = model or self.model
        model.attach_df(data)
        mat = model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            status = ModelStatus.SINGULAR
        else:
            try:
                model.fit(**optimizer_options)
                status = ModelStatus.SUCCESS
            except Exception:
                status = ModelStatus.SOLVER_FAILED
        model = _detach_df(model)
        return status


def _detach_df(model: RegmodModel) -> RegmodModel:
    """Detach data and all the arrays from the regmod model."""
    model.data.detach_df()
    del model.mat
    del model.uvec
    del model.gvec
    del model.linear_uvec
    del model.linear_gvec
    del model.linear_umat
    del model.linear_gmat

    return model
