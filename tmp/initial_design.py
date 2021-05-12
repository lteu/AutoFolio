import logging
import typing

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
import numpy as np

from smac.intensification.intensification import Intensifier
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils import constants

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2019, AutoML"
__license__ = "3-clause BSD"


class InitialDesign:
    """Base class for initial design strategies that evaluates multiple configurations

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
    intensifier
    runhistory
    aggregate_func
    """

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 runhistory: RunHistory,
                 rng: np.random.RandomState,
                 intensifier: Intensifier,
                 aggregate_func: typing.Callable,
                 configs: typing.Optional[typing.List[Configuration]]=None,
                 n_configs_x_params: int=10,
                 max_config_fracs: float=0.25,
                 ):
        """Constructor

        Parameters
        ---------
        tae_runner: ExecuteTARun
            Target algorithm execution object.
        scenario: Scenario
            Scenario with all meta information (including configuration space).
        stats: Stats
            Statistics of experiments; needed in case initial design already
            exhausts the budget.
        traj_logger: TrajLogger
            Trajectory logging to add new incumbents found by the initial
            design.
        runhistory: RunHistory
            Runhistory with all target algorithm runs.
        rng: np.random.RandomState
            Random state
        intensifier: Intensifier
            Intensification object to issue a racing to decide the current
            incumbent.
        aggregate_func: typing:Callable
            Function to aggregate performance of a configuration across
            instances.
        configs: typing.Optional[typing.List[Configuration]]
            List of initial configurations.
        n_configs_x_params: int
            how many configurations will be used at most in the initial design (X*D)
        max_config_fracs: float
            use at most X*budget in the initial design. Not active if a time limit is given.
        """

        self.tae_runner = tae_runner
        self.stats = stats
        self.traj_logger = traj_logger
        self.scenario = scenario
        self.rng = rng
        self.configs = configs
        self.intensifier = intensifier
        self.runhistory = runhistory
        self.aggregate_func = aggregate_func

        self.logger = self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        n_params = len(self.scenario.cs.get_hyperparameters())
        self.init_budget = int(max(1, min(n_configs_x_params * n_params,
                          (max_config_fracs * scenario.ta_run_limit))))
        self.logger.info("Running initial design for %d configurations" %(self.init_budget))

    def select_configurations(self) -> typing.List[Configuration]:

        if self.configs is None:
            return self._select_configurations()
        else:
            return self.configs

    def _select_configurations(self) -> typing.List[Configuration]:
        # raise NotImplementedError
        config = self.scenario.cs.get_default_configuration()
        config.origin = 'Default'
        return [config]

    
