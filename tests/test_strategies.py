from modrover.learner import Learner
from modrover.learnerid import LearnerID
from modrover.strategies import ForwardExplore, FullExplore, BackwardExplore
from modrover.strategies.base import RoverStrategy


class DummyModel(Learner):
    """Mock the Model class for testing. Only need a performance attribute."""

    def __init__(self, performance: float):
        self.performance = performance


class DummyStrategy(RoverStrategy):

    def __init__(self, num_covariates: int):
        self.num_covariates = num_covariates

    def generate_next_layer(self, *args, **kwargs):
        pass

    def get_upstream_learner_ids(self, *args, **kwargs):
        return set()


def test_basic_filtering():
    """Test that we can select the best learner IDs based on past performance."""
    num_covs = 5

    base_strategy = DummyStrategy(num_covariates=num_covs)
    first_layer = [LearnerID((0, i,)) for i in range(1, num_covs + 1)]

    # Test 1: select the n best prior_learners
    base_perf = 0
    delta = .2
    performances = {}
    for lid in first_layer:
        performances[lid] = DummyModel(base_perf)
        base_perf += delta

    best = base_strategy._filter_learner_ids(
        current_learner_ids=set(first_layer),
        prior_learners=performances,
    )
    assert best == {first_layer[-1]}

    best_two = base_strategy._filter_learner_ids(
        current_learner_ids=set(first_layer),
        prior_learners=performances,
        num_best=2
    )
    assert best_two == set(first_layer[-2:])


def test_parent_ratio():
    """Test that we can drop learner ids with better performing upstreams."""
    strategy = BackwardExplore(4)

    # Initialize a set of model ids and their children
    lid_1 = LearnerID((0, 1))
    lid_2 = LearnerID((0, 2))

    # Mock up some performances
    performances = {
        lid_1: DummyModel(5),
        lid_2: DummyModel(10)
    }

    upstreams = strategy.get_upstream_learner_ids(lid_1).\
        union(strategy.get_upstream_learner_ids(lid_2))
    for lid in upstreams:
        performances[lid] = DummyModel(7)

    # LID 1 should have worse performance than its upstreams, so should not be explored further
    # LID 2 has better performance than all parents, so we should be exploring further

    new_lids = strategy._filter_learner_ids(
        current_learner_ids={lid_1, lid_2},
        prior_learners=performances,
        num_best=2,
        threshold=1
    )
    assert new_lids == {lid_2}
