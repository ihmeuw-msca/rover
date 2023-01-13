from modrover.learner import Learner
from modrover.learnerid import LearnerID

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dataset():
    data = np.random.randn(25, 6)
    columns = [
        'var_a',
        'var_b',
        'var_c',
        'var_d',
        'var_e',
        'y']
    dataframe = pd.DataFrame(data, columns=columns)
    # Fill in intercept and holdout columns
    dataframe['intercept'] = 1
    dataframe['holdout_1'] = np.random.randint(0, 2, 25)
    dataframe['holdout_2'] = np.random.randint(0, 2, 25)
    return dataframe


@pytest.fixture
def model_specs():
    specs = dict(
        model_type='gaussian',
        col_obs='y',
        col_fixed={'mu': ['intercept']},
        col_covs=['var_a', 'var_b', 'var_c', 'var_d', 'var_e'],
        model_param_name='mu'
    )
    return specs


def test_model_init(dataset, model_specs):
    # Arbitrary: select first 2 covariates out of 5
    model_id = LearnerID(cov_ids=(0, 1, 2))
    model = Learner(model_id=model_id, **model_specs)
    # Check that model is "new"
    assert not model.has_been_fit
    assert model.opt_coefs is None
    assert model.performance is None
    assert model.model_param_name == 'mu'

    # Only a and b are selected for this model
    regmod_model = model._initialize_model()
    assert set(regmod_model.data.col_covs) == \
           {'var_a', 'var_b', 'intercept'}

    # Should have 7 columns. y column, intercept, 2 covariates, weights, offset, trim
    assert regmod_model.data.df.shape == (0, 7)


def test_model_fit(dataset, model_specs):

    model_id = LearnerID(cov_ids=(0, 1, 2, 3))
    model = Learner(model_id=model_id, **model_specs)

    # Fit the model, don't check for correctness
    model.fit(dataset, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)


def test_two_param_model_fit(dataset):

    # Sample two param model: a,b,c are mapped to mu, d,e to sigma

    model_id = LearnerID(cov_ids=(0, 1, 2))

    model = Learner(
        model_id=model_id,
        model_type='tobit',
        col_obs='y',
        col_covs=['var_a', 'var_b', 'var_c'],
        col_fixed={
            'mu': ['intercept'],
            'sigma': ['intercept', 'var_d', 'var_e']
        },
        model_param_name='mu',
    )

    # Should have 2 mu columns, 2 sigma columns, and the intercept
    regmod_model = model._initialize_model()
    assert set(regmod_model.data.col_covs) == {'var_a', 'var_b', 'var_d', 'var_e', 'intercept'}

    model.fit(dataset, holdout_cols=['holdout_1', 'holdout_2'])
    assert 0 <= model.performance <= 1
    assert model.opt_coefs is not None
    assert isinstance(model.opt_coefs, np.ndarray)
    assert isinstance(model.vcov, np.ndarray)


def test_initialize_model_with_coefs(model_specs):

    model_id = LearnerID(cov_ids=(0, 1, 2))
    model = Learner(model_id=model_id, **model_specs)

    # Set some known coefficients, random number
    # 3 covariates implies 3 coefficients
    expected_coefs = np.array([-.5, -.3, .3])
    model.opt_coefs = expected_coefs
    assert np.isclose(model.opt_coefs, expected_coefs).all()

    with pytest.raises(ValueError):
        # Setting 4 coefficients on 3 variables should raise an error
        model.opt_coefs = np.append(expected_coefs, .4)
