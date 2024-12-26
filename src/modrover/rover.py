from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from pandas import DataFrame

from .exceptions import InvalidConfigurationError, NotFittedError
from .globals import get_rmse, model_type_dict
from .learner import Learner, LearnerID, ModelStatus
from .strategies import get_strategy


class Rover:
    """Rover class explores model space and creates final super learner for
    prediction and inference.

    Parameters
    ----------
    model_type
        Type of the model. For example ``"gaussian"`` or ``"poisson"``
    obs
        The name of the column representing observations
    cov_fixed
        A list representing the covariates are present in every learner
    cov_exploring
        A list representing the covariates rover will explore over
    main_param
        The main parameter where the ``cov_fixed`` and ``cov_exploring`` are
        applied to. By default ``main_param=None``, and when the model only have
        one parameter, ``main_param`` will be automatically re-assigned to be
        that parameter. If we have multiple parameters in the model, user has to
        specify ``main_param``.
    param_specs
        Parameter settings including, link function, priors, etc
    weights
        Column name corresponding to the weights for each data point
    holdouts
        A list of column names containing 1's and 0's that represent folds in
        the rover cross-validation algorithm
    get_score
        A callable used to evaluate cross-validated score of sub-learners
        in rover

    """

    def __init__(
        self,
        model_type: str,
        obs: str,
        cov_fixed: list[str],
        cov_exploring: list[str],
        main_param: str | None = None,
        param_specs: dict[str, dict] | None = None,
        weights: str = "weights",
        holdouts: list[str] | None = None,
        get_score: Callable | None = get_rmse,
        lam_ridge: float = 0.0,
    ) -> None:
        self.model_type = self._as_model_type(model_type)
        self.obs = obs
        self.cov_fixed, self.cov_exploring = self._as_cov(
            cov_fixed, cov_exploring
        )
        self.main_param = self._as_main_param(main_param)
        self.param_specs = self._as_param_specs(param_specs)
        self.weights = weights
        self.holdouts = holdouts
        self.get_score = get_score
        self.lam_ridge = lam_ridge

        self.learners: dict[LearnerID, Learner] = {}

    @property
    def model_class(self) -> type:
        """Model class that ``model_type`` refers to."""
        return model_type_dict[self.model_type]

    @property
    def params(self) -> tuple[str, ...]:
        """A tuple of parameter names belong to the model class."""
        return self.model_class.param_names

    @property
    def variables(self) -> tuple[str, ...]:
        """A tuple of the variable names belong the model class with full list
        of covariates.
        """
        names = []
        for p in self.params:
            names.extend([f"{p}_{v}" for v in self.param_specs[p]["variables"]])
            if p == self.main_param:
                names.extend([f"{p}_{v}" for v in self.cov_exploring])
        return tuple(names)

    @property
    def num_vars(self) -> int:
        """Number of variables with full list of covariates."""
        return len(self.variables)

    @property
    def super_learner_id(self) -> tuple[int, ...]:
        """Learner id for the super learner."""
        return tuple(range(len(self.cov_exploring)))

    @property
    def super_learner(self) -> Learner:
        """Ensembled super learner."""
        if not hasattr(self, "_super_learner"):
            raise NotFittedError("Rover has not been ensembled yet")
        return self._super_learner

    @property
    def learner_info(self) -> DataFrame:
        """A data frame contains important information of fitted learners."""
        if not hasattr(self, "_learner_info"):
            raise NotFittedError("Rover has not been ensemble yet")
        return self._learner_info

    @property
    def summary(self) -> DataFrame:
        """A data frame contains the summary information of explored covariates
        across all fitted learners.
        """
        if not hasattr(self, "_summary"):
            raise NotFittedError("Rover has not been ensemble yet")
        return self._summary

    def fit(
        self,
        data: DataFrame,
        strategies: list[str],
        strategy_options: dict | None = None,
        top_pct_score: float = 0.1,
        top_pct_learner: float = 1.0,
        coef_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Fits the ensembled super learner.

        Explores over all covariate slices as defined by the input strategy, and
        fits the sublearners.

        The super learner coefficients are determined by the ensemble method
        parameters, and the super learner itself will be created - to be used in
        prediction and summarization.

        Parameters
        ----------
        data
            Training data to fit individual learners on.
        strategies
            The selection strategy to determine the model tree. Valid strategies
            include "forward", "backward" and "full".
        strategy_options
            A dictionary with key as the strategy name and value as the option
            with calling the strategy. By default, `strategy_options=None` where
            all default options will be used by the strategies.
        top_pct_score
            Only the learners with score that are greater or equal than
            ``best_score * (1 - top_score)`` can be selected. When
            ``top_score = 0`` only the best model will be selected.
        top_pct_learner
            Only the best ``top_pct_learner * num_learners`` will be selected.
        coef_bounds
            User pre-specified bounds for the coefficients. This is a dictionary
            with key as the covariate name and the value as the bounds. The
            learner will be marked valid or not if the coefficients are within
            the bounds. Invalid learners will not be used in ensemble process
            to create super learner. By default, ``coef_bounds=None``, where
            there is not validation based on the value of the coefficients.

        """
        self._explore(
            data=data, strategies=strategies, strategy_options=strategy_options
        )
        self._get_super_learner(
            top_pct_score=top_pct_score,
            top_pct_learner=top_pct_learner,
            coef_bounds=coef_bounds,
        )

    def predict(
        self, data: DataFrame, return_ui: bool = False, alpha: float = 0.05
    ) -> NDArray:
        """Predict with ensembled super learner.

        Parameters
        ----------
        data
            Testing data to predict
        return_ui
            If ``return_ui=True``, a matrix will be returned. The first row
            is the point prediction, second and thrid rows are the lower and
            upper bound of the prediction.
        alpha
            When ``return_ui=True``, function will return (1 - ``alpha``)
            uncertainty interval. By default, ``alpha=0.05``.

        """
        return self.super_learner.predict(
            data, return_ui=return_ui, alpha=alpha
        )

    def plot(self, bins: int | None = None) -> plt.Figure:
        """Plot the result of the exploration. Each panel of the figure
        corresponding to one covariate in the ``cov_exploring``. We plot the
        spread of the coefficients across all learners along with color
        represents their performance score.

        Parameters
        ----------
        bins
            When ``bins=None``, the coefficients will be spread along the y axis
            randomly to display the spread. When user pass in an integer, the
            x axis will be divided into bins and the y value will be assigned
            according to the ranking of score within the bin.

        """
        nrow = len(self.cov_exploring)
        fig, ax = plt.subplots(nrow, 1, figsize=(8, 2 * nrow), sharex=True)
        ax = [ax] if isinstance(ax, plt.Axes) else ax

        learner_info = self.learner_info[
            self.learner_info["status"] == ModelStatus.SUCCESS
        ].copy()
        if bins is not None:
            all_coef = learner_info[
                [f"{self.main_param}_{cov}" for cov in self.cov_exploring]
            ].to_numpy()
            cmin, cmax = all_coef.min(), all_coef.max()
            bins = np.linspace(cmin, cmax, bins + 1)

        summary = self.summary
        score_scaled = learner_info["score_scaled"].to_numpy()
        vmin, vmax = score_scaled.min(), score_scaled.max()
        highlight_index = {
            "final": learner_info["weight"] > 0,
            "invalid": ~learner_info["valid"],
        }
        highlight_config = {
            "single": {
                "marker": "^",
                "facecolor": "none",
                "edgecolor": "gray",
                "alpha": 0.5,
            },
            "final": {
                "marker": "o",
                "facecolor": "none",
                "edgecolor": "gray",
                "alpha": 0.5,
            },
            "invalid": {"marker": "x", "color": "gray", "alpha": 0.5},
        }
        for i, cov in enumerate(summary.sort_values("ranking")["cov"]):
            # plot the spread of the coef
            name = f"{self.main_param}_{cov}"
            coef = learner_info[name].to_numpy()
            coef_jitter = np.random.rand(coef.size)
            if bins is not None:
                learner_info["bin_id"] = np.digitize(learner_info[name], bins)
                coef_jitter = learner_info.groupby("bin_id")[
                    "score"
                ].rank() / len(learner_info)
            im = ax[i].scatter(
                coef,
                coef_jitter,
                alpha=0.2,
                c=score_scaled,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
            )
            # mark single, final and invalid models
            highlight_index["single"] = learner_info["learner_id"] == (
                self.cov_exploring.index(cov),
            )
            for key, index in highlight_index.items():
                ax[i].scatter(
                    coef[index], coef_jitter[index], **highlight_config[key]
                )
            # indicator of 0
            ax[i].axvline(0, linewidth=1, color="gray", linestyle="--")
            # colorbar
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.ax.set_yticks([vmin, vmax])
            # plot ensemble result
            index = summary["cov"] == cov
            super_coef = summary[index]["coef"].iloc[0]
            super_coef_lwr = summary[index]["coef_lwr"].iloc[0]
            super_coef_upr = summary[index]["coef_upr"].iloc[0]
            ax[i].axvline(super_coef, linewidth=1, color="#008080")
            ax[i].plot(
                [super_coef_lwr, super_coef_upr],
                [0.5, 0.5],
                linewidth=1,
                color="#008080",
            )
            # summary text
            text = [
                f"ranking = {summary[index]['ranking'].iloc[0]} / {len(summary)}",
                f"significant = {summary[index]['significant'].iloc[0]}",
                f"pct_present = {summary[index]['pct_present'].iloc[0]:.2%}",
                f"score_improvement = {summary[index]['score_improvement'].iloc[0]:.4}",
                f"coef = {super_coef:.2f} ({super_coef_lwr:.2f}, {super_coef_upr:.2f})",
            ]
            text = "\n".join(text)
            ax[i].text(
                1.15,
                1,
                text,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[i].transAxes,
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="none",
                    edgecolor="grey",
                    alpha=0.5,
                ),
            )
            # config
            ax[i].set_ylabel(cov)
            ax[i].xaxis.set_tick_params(labelbottom=True)
            ax[i].set_yticks([])
            if bins is not None:
                ax[i].set_yticks([0, 1])
        ax[0].set_title(
            f"models = {len(learner_info)}/{2 ** len(summary)}", loc="left"
        )

        return fig

    # validations ==============================================================
    def _as_model_type(self, model_type: str) -> str:
        if model_type not in model_type_dict:
            raise InvalidConfigurationError(
                f"{model_type=:} not known, "
                f"please select from {list(model_type_dict.keys())}"
            )
        return model_type

    def _as_cov(
        self, cov_fixed: list[str], cov_exploring: list[str]
    ) -> tuple[list[str], list[str]]:
        len_set = len(set(cov_fixed) | set(cov_exploring))
        len_sum = len(cov_fixed) + len(cov_exploring)
        if len_set != len_sum:
            raise InvalidConfigurationError(
                "Covariates in cov_fixed and cov_exploring cannot repeat"
            )
        return list(cov_fixed), list(cov_exploring)

    def _as_main_param(self, main_param: str | None) -> str:
        params = self.params
        if main_param is not None:
            if main_param not in params:
                raise InvalidConfigurationError(
                    f"{main_param=:} not know, " f"please select from {params}"
                )
        else:
            if len(params) > 1:
                raise InvalidConfigurationError(
                    "There are more than one model parameters, "
                    f"please select main_param from {params}"
                )
            main_param = params[0]
        return main_param

    def _as_param_specs(
        self, param_specs: dict[str, dict] | None
    ) -> dict[str, dict]:
        param_specs = param_specs or {}
        for param in self.params:
            if param != self.main_param:
                variables = param_specs.get(param, {}).get("variables", [])
                if len(variables) == 0:
                    example_param_specs = {param: {"variables": ["intercept"]}}
                    raise InvalidConfigurationError(
                        f"Please provide variables for {param}, "
                        f"for example, param_specs={example_param_specs}"
                    )
        param_specs.update({self.main_param: {"variables": self.cov_fixed}})
        return param_specs

    # construct learner ========================================================
    def _get_param_specs(self, learner_id: LearnerID) -> dict[str, dict]:
        param_specs = deepcopy(self.param_specs)
        param_specs[self.main_param]["variables"].extend(
            [self.cov_exploring[i] for i in learner_id]
        )
        return param_specs

    def _get_learner(
        self, learner_id: LearnerID, use_cache: bool = True
    ) -> Learner:
        if learner_id in self.learners and use_cache:
            return self.learners[learner_id]

        param_specs = self._get_param_specs(learner_id)
        return Learner(
            self.model_class,
            self.obs,
            self.main_param,
            param_specs,
            weights=self.weights,
            get_score=self.get_score,
            lam_ridge=self.lam_ridge,
        )

    # explore ==================================================================
    def _explore(
        self,
        data: DataFrame,
        strategies: list[str],
        strategy_options: dict | None = None,
    ):
        """Explore the entire tree of learners"""
        strategy_options = strategy_options or {}
        strategy_options = {
            strategy: strategy_options.get(strategy, {})
            for strategy in strategies
        }

        for strategy in strategies:
            options = strategy_options[strategy]
            strategy = get_strategy(strategy)(num_covs=len(self.cov_exploring))
            curr_ids = {strategy.base_learner_id}
            while curr_ids:
                for learner_id in curr_ids:
                    learner = self._get_learner(learner_id)
                    if learner.status == ModelStatus.NOT_FITTED:
                        learner.fit(data, self.holdouts)
                        self.learners[learner_id] = learner

                next_ids = strategy.get_next_layer(
                    curr_layer=curr_ids,
                    learners=self.learners,
                    **options,
                )
                curr_ids = next_ids

    # construct super learner ==================================================
    def _get_super_learner(
        self,
        top_pct_score: float,
        top_pct_learner: float,
        coef_bounds: dict[str, tuple[float, float]] | None,
    ) -> Learner:
        """Call at the end of fit, so model is configured at the end of fit."""
        df = self._get_learner_info(top_pct_score, top_pct_learner, coef_bounds)
        df = df[df["weight"] > 0.0]
        learner_ids, weights = df["learner_id"], df["weight"]
        coefs = df[list(self.variables)].to_numpy()
        super_coef = coefs.T.dot(weights)
        super_vcov = self._get_super_vcov(learner_ids, weights, super_coef)

        super_learner = self._get_learner(
            learner_id=self.super_learner_id, use_cache=False
        )
        super_learner.coef = super_coef
        super_learner.vcov = super_vcov
        self._super_learner = super_learner
        self._get_summary()
        return super_learner

    def _get_learner_info(
        self,
        top_pct_score: float = 0.1,
        top_pct_learner: float = 1.0,
        coef_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> DataFrame:
        df = DataFrame(
            columns=["learner_id", "status"] + list(self.variables) + ["score"]
        )
        for learner_id, learner in self.learners.items():
            row = [learner_id, learner.status]
            coef, score = np.repeat(np.nan, self.num_vars), np.nan
            if learner.status == ModelStatus.SUCCESS:
                coef = np.zeros(self.num_vars)
                coef_index = self._get_coef_index(learner_id)
                coef[coef_index] = learner.coef
                score = learner.score
            row.extend(list(coef) + [score])
            df.loc[len(df)] = row

        df["coef_valid"] = True
        if coef_bounds:
            coef_valid = []
            for cov, bounds in coef_bounds.items():
                if not any([cov.startswith(p + "_") for p in self.params]):
                    cov = "_".join([self.main_param, cov])
                coef_valid.append(
                    (df[cov] >= bounds[0]) & (df[cov] <= bounds[1])
                )
            df["coef_valid"] = np.vstack(coef_valid).all(axis=0)

        df["valid"] = (df["status"] == ModelStatus.SUCCESS) & df["coef_valid"]
        df["weight"] = 0.0
        df.loc[df["valid"], "weight"] = self._get_super_weights(
            df.loc[df["valid"], "learner_id"], top_pct_score, top_pct_learner
        )

        df["score_scaled"] = df["score"] / df["score"].dropna().max()
        self._learner_info = df
        return df

    def _get_super_coef(
        self, learner_ids: list[LearnerID], weights: NDArray
    ) -> NDArray:
        """Generates the weighted ensembled coefficients across all fitted
        learners.

        """
        super_coef = np.zeros(self.num_vars)
        for learner_id, weight in zip(learner_ids, weights):
            coef_index = self._get_coef_index(learner_id)
            super_coef[coef_index] += weight * self.learners[learner_id].coef
        return super_coef

    def _get_super_vcov(
        self,
        learner_ids: list[LearnerID],
        weights: NDArray,
        super_coef: NDArray,
    ) -> NDArray:
        super_vcov = np.zeros((self.num_vars, self.num_vars))
        for learner_id, weight in zip(learner_ids, weights):
            learner = self.learners[learner_id]
            coef_index = self._get_coef_index(learner_id)
            super_vcov[np.ix_(coef_index, coef_index)] += weight * (
                learner.vcov + np.outer(learner.coef, learner.coef)
            )
        super_vcov -= np.outer(super_coef, super_coef)
        return super_vcov

    def _get_coef_index(self, learner_id: LearnerID) -> list[int]:
        coef_index, pointer = [], 0
        for param in self.params:
            num_covs = len(self.param_specs[param]["variables"])
            coef_index.extend(list(range(pointer, pointer + num_covs)))
            pointer += num_covs
            if param == self.main_param:
                coef_index.extend([i + pointer for i in learner_id])
                pointer += len(self.cov_exploring)
        return coef_index

    def _get_super_weights(
        self,
        learner_ids: list[LearnerID],
        top_pct_score: float,
        top_pct_learner: float,
    ) -> NDArray:
        scores = np.array(
            [self.learners[learner_id].score for learner_id in learner_ids]
        )
        argsort = np.argsort(scores)[::-1]
        indices = scores >= scores[argsort[0]] * (1 - top_pct_score)
        num_learners = int(np.floor(len(scores) * top_pct_learner)) + 1
        indices[argsort[num_learners:]] = False

        scores[~indices] = 0.0
        weights = scores / scores.sum()
        return weights

    # diagnostics ==============================================================
    def _get_summary(self) -> DataFrame:
        # info
        variables = self.variables
        learner_info = self.learner_info[
            self.learner_info["status"] == ModelStatus.SUCCESS
        ]
        learner_scores = dict(
            zip(learner_info["learner_id"], learner_info["score"])
        )
        # ensemble info
        coef_index = [
            variables.index(f"{self.main_param}_{cov}")
            for cov in self.cov_exploring
        ]
        coef = self.super_learner.coef[coef_index]
        coef_sd = np.sqrt(np.diag(self.super_learner.vcov)[coef_index])
        # number of models the covariate is present
        pct_present = [
            (learner_info[f"{self.main_param}_{cov}"] != 0.0).sum()
            / len(learner_info)
            for cov in self.cov_exploring
        ]
        # score when only the selected covariate is present
        single_score = [
            learner_scores.get((i,), np.nan)
            for i in range(len(self.cov_exploring))
        ]
        # average score when selected covariate is present or not
        present_score = []
        not_present_score = []
        for cov in self.cov_exploring:
            present_index = learner_info[f"{self.main_param}_{cov}"] != 0.0
            ps, nps = 0.0, 0.0
            if any(present_index):
                ps = learner_info[present_index]["score"].mean()
            if not all(present_index):
                nps = learner_info[~present_index]["score"].mean()
            present_score.append(ps)
            not_present_score.append(nps)

        df = DataFrame(
            {
                "cov": self.cov_exploring,
                "coef": coef,
                "coef_sd": coef_sd,
                "pct_present": pct_present,
                "single_score": single_score,
                "present_score": present_score,
                "not_present_score": not_present_score,
            }
        )

        # derived quantities
        df["score_improvement"] = df["present_score"] / df["not_present_score"]
        df["ranking"] = (
            df["score_improvement"].rank(ascending=False).astype(int)
        )
        df["coef_lwr"] = coef - 1.96 * coef_sd
        df["coef_upr"] = coef + 1.96 * coef_sd
        df["significant"] = np.sign(df["coef_lwr"] * df["coef_upr"]) > 0
        self._summary = df
        return df
