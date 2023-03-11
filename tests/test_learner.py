import numpy as np
import pandas as pd
import pytest

from modrover.globals import model_type_dict
from modrover.learner import Learner


@pytest.fixture
def dataset():
    data = np.random.randn(25, 6)
    covariate_columns = ["var_a", "var_b", "var_c", "var_d", "var_e", "y"]
    dataframe = pd.DataFrame(data, columns=covariate_columns)
    # Fill in intercept and holdout columns
    dataframe["intercept"] = 1
    dataframe["holdout_1"] = np.random.randint(0, 2, 25)
    dataframe["holdout_2"] = np.random.randint(0, 2, 25)
    # Drop y covariate since it's the observed column
    return dataframe


@pytest.fixture
def model_specs(dataset):
    specs = dict(
        model_type=model_type_dict["gaussian"],
        obs="y",
        param_specs={"mu": {"variables": ["intercept", "var_a", "var_b", "var_c"]}},
        offset="offset",
    )
    return specs


def test_model_init(model_specs):
    # In param_specs, we'll select the intercept term plus 3/5 covariates
    model = Learner(**model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.coef is None
    assert model.performance is None
    model._initialize_model()


def test_model_fit(dataset, model_specs):
    model = Learner(**model_specs)

    # Fit the model, don't check for correctness
    model.fit(dataset, holdouts=["holdout_1", "holdout_2"])
    assert 0 <= model.performance <= 1
    assert model.coef is not None
    assert isinstance(model.coef, np.ndarray)
    assert len(model.coef == 4)  # Intercept + 3 covariate terms
    assert isinstance(model.vcov, np.ndarray)


def test_two_param_model_fit(dataset):
    # Sample two param model: a,b,c are mapped to mu, d,e to sigma
    model = Learner(
        model_type=model_type_dict["tobit"],
        obs="y",
        param_specs={
            "mu": {
                "variables": ["intercept", "var_a", "var_b", "var_c"],
            },
            "sigma": {
                "variables": ["intercept", "var_d", "var_e"],
            },
        },
    )

    # Should have 2 mu columns, 2 sigma columns, and the intercept
    model._initialize_model()
    model.fit(dataset, holdouts=["holdout_1", "holdout_2"])
    assert 0 <= model.performance <= 1
    assert model.coef is not None
    assert isinstance(model.coef, np.ndarray)
    # Result: intercept + 5 covariates + intercept = 7 coefficients
    assert len(model.coef) == 7
    assert isinstance(model.vcov, np.ndarray)


def test_initialize_model_with_coefs(model_specs):
    model = Learner(**model_specs)

    # Set some known coefficients, random number
    # Intercept + 3 covariates implies 4 coefficients
    expected_coefs = np.array([-0.5, -0.3, 0.3, 0.2])
    model.coef = expected_coefs
    assert np.isclose(model.coef, expected_coefs).all()

    with pytest.raises(ValueError):
        # Setting 5 coefficients on 4 variables should raise an error
        model.coef = np.append(expected_coefs, 0.4)
