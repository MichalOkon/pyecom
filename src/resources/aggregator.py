# Extends the BaseResource class to provide an aggregator resource

import numpy as np
from src.resources.base_resource import BaseResource

from typing import Union


class Aggregator():

    def __init__(self,
                 name: str,
                 imports: np.array,
                 exports: np.array,
                 import_cost: np.array,
                 export_cost: np.array,
                 import_max: np.array,
                 export_max: np.array):
        self.name = name

        self.imports = imports
        self.exports = exports

        self.import_max = import_max
        self.export_max = export_max

        self.import_cost = import_cost
        self.export_cost = export_cost

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return f'{self.name}'