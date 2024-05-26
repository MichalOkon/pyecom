# Extends the BaseResource class to provide a storage resource

import numpy as np
from src.resources.base_resource import BaseResource

from typing import Union

class Storage(BaseResource):
    def __init__(self,
                 name: str,
                 value: Union[np.array, float],
                 cost_discharge: np.array,
                 cost_charge: np.array,
                 capacity_max: float,
                 capacity_min: float,
                 initial_charge: float,
                 discharge_efficiency: float,
                 discharge_max: float,
                 charge_efficiency: float,
                 charge_max: float,
                 capital_cost: float,
                 ):
        super().__init__(name, value)

        self.capacity_max = capacity_max
        self.capacity_min = capacity_min
        self.initial_charge = initial_charge
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.capital_cost = capital_cost

        self.discharge = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
        self.discharge_max = discharge_max
        self.cost_discharge = cost_discharge

        self.charge = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
        self.charge_max = charge_max
        self.cost_charge = cost_charge

        self.emin_relax = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
