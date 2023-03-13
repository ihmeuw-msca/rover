from __future__ import annotations

from collections import defaultdict
from enum import Enum
from operator import attrgetter
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import Model as RegmodModel
from regmod.variable import Variable

from .globals import get_rmse

LearnerID = tuple[int, ...]


class ModelStatus(Enum):
    SUCCESS = 0
    SINGULAR = 1
    SOLVER_FAILED = 2
    NOT_FITTED = -1


class Learner:
    """Main model class for Rover explore.

    Parameters
    ----------
    model_class
        Regmod model constructor
    obs
        Name corresponding to the observation column in the data frame
    param_specs
        Parameter settings for the regmod model
    offset
        Name corresponding to the offset column in the data frame
    weights
        Name corresponding to the weights column in the data frame
    get_score
        Function that evaluate the score of of the model

    """

    def __init__(
        self,
        model_class: type,
        obs: str,
        param_specs: dict[str, dict],
        offset: str = "offset",
        weights: str = "weights",
        get_score: Callable = get_rmse,
    ) -> None:
        self.model_class = model_class
        self.obs = obs
        self.offset = offset
        self.weights = weights
        self.get_score = get_score

        # convert str to Variable
        for param_spec in param_specs.values():
            param_spec["variables"] = list(map(Variable, param_spec["variables"]))
        self.param_specs = param_specs

        # initialize null model
        self.model = self._get_model()
        self.score: Optional[float] = None
        self.status = ModelStatus.NOT_FITTED

        # initialize cross validation model
        self._cv_models = defaultdict(self._get_model)
        self._cv_scores = defaultdict(lambda: None)
        self._cv_status = defaultdict(lambda: ModelStatus.NOT_FITTED)

    @property
    def coef(self) -> Optional[NDArray]:
        return self.model.opt_coefs

    @coef.setter
    def coef(self, coef: NDArray):
        if len(coef) != self.model.size:
            raise ValueError("Provided coef size don't match")
        self.model.opt_coefs = coef

    @property
    def vcov(self) -> Optional[NDArray]:
        return self.model.opt_vcov

    @vcov.setter
    def vcov(self, vcov: NDArray):
        if vcov.shape != (self.model.size, self.model.size):
            raise ValueError("Provided vcov shape don't match")
        self.model.opt_vcov = vcov

    @property
    def df_coefs(self) -> Optional[DataFrame]:
        if not self.coef:
            return None
        # TODO: Update this datastructure to be flexible with multiple parameters.
        # Should reflect final all-data model, perhaps prefix with parameter name
        # Is this full structure necessary? Or just the means?
        data = DataFrame(
            {
                "cov_name": map(attrgetter("name"), self.model.params[0].variables),
                "mean": self.coef,
                "sd": np.sqrt(np.diag(self.vcov)),
            }
        )
        return data

    def _get_model(self) -> RegmodModel:
        # TODO: this shouldn't be necessary in regmod v1.0.0
        data = Data(
            col_obs=self.obs,
            col_offset=self.offset,
            col_weights=self.weights,
            subset_cols=False,
        )

        # Create regmod variables separately, by parameter
        # Initialize with fixed parameters
        model = self.model_class(
            data=data,
            param_specs=self.param_specs,
        )
        self.model = model
        return model

    def fit(
        self,
        data: DataFrame,
        holdouts: Optional[list[str]] = None,
        **optimizer_options,
    ):
        """
        Fit a set of models on a series of holdout datasets.

        This method will fit a model over k folds of the dataset, where k is the length
        of the provided holdouts list. It is up to the user to decide the train-test
        splits for each holdout column.

        On each fold of the dataset, the trained model will predict out on the validation set
        and obtain a evaluate. The averaged evaluate across all folds becomes the model's overall
        score.

        Finally, a model is trained with all data in order to generate the final coefficients.

        :param data: a dataframe containing the training data
        :param holdouts: which column names to iterate over for cross validation
        :return:
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
            score = np.mean(
                [
                    score
                    for holdout, score in self._cv_scores.items()
                    if self._cv_status[holdout] == ModelStatus.SUCCESS
                ]
            )

            # Learner score is average score across each k fold
            self.score = score

        # Fit final model with all data included
        self.status = self._fit(data, **optimizer_options)

        # If holdout cols not provided, use in sample evaluate for the full data model
        if not holdouts:
            self.score = self.evaluate(data)

    def _fit(
        self,
        data: DataFrame,
        model: Optional[RegmodModel] = None,
        **optimizer_options,
    ) -> ModelStatus:
        model = model or self.model
        model.attach_df(data)
        mat = model.mat[0]
        if np.linalg.matrix_rank(mat) < mat.shape[1]:
            return ModelStatus.SINGULAR

        try:
            model.fit(**optimizer_options)
        except:
            return ModelStatus.SOLVER_FAILED

        model.data.detach_df()
        return ModelStatus.SUCCESS

    def predict(self, data: DataFrame, model: Optional[RegmodModel] = None) -> NDArray:
        """
        Wraps regmod's predict method to avoid modifying input dataset.

        Can be removed if regmod models use a functionally pure predict function, otherwise
        we will raise SettingWithCopyWarnings repeatedly.

        :param model: a fitted RegmodModel
        :param test_set: a dataset to generate predictions from
        :param param_name: a string representing the parameter we are predicting out on
        :return: an array of predictions for the model parameter of interest
        """
        model = model or self.model
        df_pred = model.predict(data)
        col_pred = model.param_names[0]
        model.data.detach_df()
        return df_pred[col_pred].to_numpy()

    def evaluate(self, data: DataFrame, model: Optional[RegmodModel] = None) -> float:
        """
        Given a model and a test set, generate an aggregate evaluate.

        Score is based on the provided evaluation metric, comparing the difference between
        observed and predicted values.

        :param test_set: The holdout test set to generate predictions from
        :param model: The fitted model to set predictions on
        :return: a evaluate determined by the provided model evaluation metric
        """
        score = self.get_score(
            obs=data[self.obs].to_numpy(),
            pred=self.predict(data, model=model),
        )
        return score
