import string
from typing import Union

import gymnasium as gym
import numpy as np
from copy import deepcopy
from src.resources.base_resource import BaseResource
from src.algorithms.rl.utils import separate_resources

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyCommunityMultiHouseholdsEnv_v0(MultiAgentEnv):

    metadata = {'name': 'EnergyCommunityMultiHouseholdsEnv_v0'}

    def __init__(self, households_resources: list[list[BaseResource]],
                 import_penalty,
                 export_penalty,
                 storage_action_reward,
                 storage_action_penalty,
                 balance_penalty,
                 max_timesteps):
        super().__init__()

        # Define the maximum number of timesteps to simulate
        self. max_timesteps = max_timesteps

        # Define the resources
        self.households_resources = deepcopy(households_resources)

        # Initialize households
        self.households = self._create_households(households_resources,
                                                  import_penalty,
                                                  export_penalty,
                                                  storage_action_reward,
                                                  storage_action_penalty,
                                                  balance_penalty)
        self.households_ids = set(self.households)

        # Initialize action space for each household
        self._create_action_space()

        # Initialize observation space for each household
        self._create_observation_space()

        # # Calculate total laods
        # self.total_loads =
        # # Calculate the sum of loads for each timestep
        # self.load_consumption = np.sum([load.value for load in self.loads], axis=0)

        # Initialize time counter
        self.current_timestep: int = 0

        # Initialize variables for production and consumption
        self.current_total_production: float = 0
        self.current_total_consumption: float = 0

        # # Initialize variables for currently available energy
        # self.current_total_available_energy: float = 0

        # Costs for each resource
        self.total_accumulated_generator_cost: float = 0.0
        self.total_accumulated_storage_cost: float = 0.0
        self.total_accumulated_import_cost: float = 0.0
        self.total_accumulated_export_cost: float = 0.0

        # Rewards
        self.storage_action_reward = storage_action_reward

        # Penalties
        self.storage_action_penalty = storage_action_penalty
        self.import_penalty = import_penalty
        self.export_penalty = export_penalty
        self.balance_penalty = balance_penalty

        # Set a penalty for each resource
        self.total_accumulated_generator_penalty: float = 0.0
        self.total_accumulated_storage_penalty: float = 0.0
        self.total_accumulated_import_penalty: float = 0.0
        self.total_accumulated_export_penalty: float = 0.0

        # Balance history
        self.balance_history = []
        self.reward_history = []
        self.final_logs = {}


        # Current real reward
        self.current_real_reward = {}

    def _create_households(self, households_resources: list[list[BaseResource]], import_penalty,
                           export_penalty,
                           storage_action_reward,
                           storage_action_penalty,
                           balance_penalty):
        households = {}
        for i, resources in enumerate(households_resources):
            households[i] = Household(i, resources, self, import_penalty,
                                      export_penalty,
                                      storage_action_reward,
                                      storage_action_penalty,
                                      balance_penalty)

        return households

    def _create_action_space(self):
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        for id in self.households:
            temp_action_space[id] = self.households[id].action_space

        self.action_space = gym.spaces.Dict(temp_action_space)

    def _create_observation_space(self):
        self._onservation_space_in_preferred_format = True
        temp_observation_space = {}
        for id in self.households:
            temp_observation_space[id] = self.households[id].observation_space

        self.observation_space = gym.spaces.Dict(temp_observation_space)

    # Reset environment
    def reset(self, *, seed=None, options=None):

        # Define the resources
        self.households_resources = deepcopy(self.households_resources)

        # Initialize households
        self.households = self._create_households(self.households_resources,
                                                  self.import_penalty,
                                                  self.export_penalty,
                                                  self.storage_action_reward,
                                                  self.storage_action_penalty,
                                                  self.balance_penalty)
        self.households_ids = set(self.households)

        # Set the initial pool of available energy to load consumption at the first timestep
        # self.current_total_available_energy = -self.load_consumption[0]

        # Reset environment variables
        self.current_timestep = 0
        self.current_total_production = 0
        self.current_total_consumption = 0

        # Reset penalties and costs
        self.total_accumulated_generator_cost = 0.0
        self.total_accumulated_storage_cost = 0.0
        self.total_accumulated_import_cost = 0.0
        self.total_accumulated_export_cost = 0.0

        self.total_accumulated_generator_penalty = 0.0
        self.total_accumulated_storage_penalty = 0.0
        self.total_accumulated_import_penalty = 0.0
        self.total_accumulated_export_penalty = 0.0

        # Clear the history
        self.balance_history = []
        self.reward_history = []
        self.final_logs = {}

        observations = self._get_households_observations()

        return observations, {}

    # Step function for environment transitions
    def step(self, action_dict: dict) -> tuple:
        """
        Step function for environment transitions
        Agents will act in the following order:
        1. Generators
        2. EVs
        3. Storages
        4. Aggregator

        :param action_dict: dict
        :return: tuple
        """

        # Check for completion of the episode
        if self.current_timestep >= self.max_timesteps:
            terminateds, truncateds = self._log_ending(True)
            observations, reward, info = {}, {}, {}

            return observations, reward, terminateds, truncateds, info

        # Check for existing actions
        exists_actions = len(action_dict) > 0

        # Observations
        observations = {}

        # Reward
        reward: dict = {}

        # Info dictionary
        info: dict = {}

        summed_available_energy = 0
        # Execute the actions
        if exists_actions:

            # Do the actions
            for action_id in action_dict.keys():
                # Execute the actions
                cost, penalty = self.households[action_id]. \
                    execute_action(action_dict[action_id])

                # Log the agent
                # self._log_households(deepcopy(self.households[action_id]))

                # Calculate the true reward
                self.current_real_reward[action_id] = self.households[action_id].calculate_reward(cost,
                                                                                                  penalty)
                summed_available_energy += self.households[action_id].current_available_energy
                info[action_id] = self.households[action_id].log_info()

            self.balance_history.append(summed_available_energy)
            self.reward_history.append(self.current_real_reward)
            # Terminations and truncations
            terminateds, truncateds = self._log_ending(False)
            #
            # # Check for end of episode
            # if self.current_timestep >= self.max_timesteps:
            #     terminateds, truncateds = self._log_ending(True)

            next_obs = self._get_households_observations()

            # Update the timestep
            self.current_timestep += 1

            return next_obs, self.current_real_reward, terminateds, truncateds, info

        # Terminations and truncations
        terminateds = {a: False for a in self.households}
        terminateds['__all__'] = False
        truncateds = {a: False for a in self.households}
        truncateds['__all__'] = False

        # Update the timestep
        self.current_timestep += 1

        return observations, reward, terminateds, truncateds, info

        # Get observations

    def _get_households_observations(self) -> dict:
        """
        Get the observations for the environment
        :return: dict
        """

        # Get the observation for the next resource
        observations = {}

        if self.current_timestep >= self.max_timesteps:
            return observations

        for household_id in self.households.keys():
            observations[household_id] = self.households[household_id].get_next_observations()

        return observations

    # Log ending of episode
    def _log_ending(self, is_over_flag: bool) -> tuple[dict, dict]:
        terminateds = {a: is_over_flag for a in self.households}
        terminateds['__all__'] = is_over_flag
        truncateds = {a: is_over_flag for a in self.households}
        truncateds['__all__'] = is_over_flag

        if is_over_flag:
            self.log_final_values()

        return terminateds, truncateds


    # Log final values
    def log_final_values(self):
        logs = {}

        # Aggregate values from different households
        total_imported_energy = [0.0] * self.max_timesteps
        total_exported_energy = [0.0] * self.max_timesteps
        for timestep in range(self.max_timesteps):
            timestep_imported_energy = 0.0
            timestep_exported_energy = 0.0
            for household_id in self.households:
                timestep_imported_energy += self.households[household_id].aggregator.imports[timestep]
                timestep_exported_energy += self.households[household_id].aggregator.exports[timestep]
            total_exported_energy[timestep] = timestep_exported_energy
            total_imported_energy[timestep] = timestep_exported_energy

        total_soc = [0.0] * self.max_timesteps
        total_produced_energy = [0.0] * self.max_timesteps
        for household_id in self.households:

            if self.households[household_id].generator is not None:
                for timestep in range(self.max_timesteps):
                    total_produced_energy[timestep] += self.households[household_id].generator.value[timestep]
            if self.households[household_id].storage is not None:
                for timestep in range(self.max_timesteps):
                    total_soc[timestep] += self.households[household_id].storage.value[timestep]

        logs['total_imported_energy'] = total_imported_energy
        logs['total_exported_energy'] = total_exported_energy
        logs['total_soc'] = total_soc
        logs['total_produced_energy'] = total_produced_energy

        # Log specific households
        logs['imported_energy'] = {}
        logs['exported_energy'] = {}
        logs['soc'] = {}
        logs['produced_energy'] = {}

        for household_id in self.households:
            logs['imported_energy'][household_id] = deepcopy(self.households[household_id].aggregator.imports)
            logs['exported_energy'][household_id] = deepcopy(self.households[household_id].aggregator.exports)
            if self.households[household_id].generator is not None:
                logs['produced_energy'][household_id] = deepcopy(self.households[household_id].generator.value)
            if self.households[household_id].storage is not None:
                logs['soc'][household_id] = deepcopy(self.households[household_id].storage.value)

        self.final_logs = logs

class Household:
    def __init__(self, id: int, resources, environment: EnergyCommunityMultiHouseholdsEnv_v0, import_penalty, export_penalty, storage_action_reward,
                 storage_action_penalty, balance_penalty):
        # Define the used environment (for features like timestep tracking)
        self.environment = environment

        # Define the resources
        self.resources = deepcopy(resources)

        # Define the household id
        self.id = id

        # Define current timestep
        self.current_production = 0

        # Split the incoming resources
        # temp_resources = separate_resources(self.resources)
        self.generator = self.resources['generator'] if 'generator' in self.resources.keys() else None
        self.load = self.resources['load'] if 'load' in self.resources.keys() else None
        self.storage =self.resources['storage'] if 'storage' in self.resources.keys() else None
        self.aggregator = self.resources['aggregator']  # required

        # Define max timestep
        self.max_timestep = self.aggregator.exports.shape[0]

        # TODO: This should go into environment?
        # # Calculate the sum of loads for each timestep
        # self.load_consumption = np.sum(self.load.value, axis=0)

        # Initialize variables for production and consumption
        self.current_production: float = 0
        self.current_consumption: float = 0

        # Initialize variables for currently available energy
        self.current_available_energy: float = 0

        # Observation space
        self._handle_observation_space()

        # Action space
        self._handle_action_space()

        # Costs for each resource
        self.accumulated_generator_cost: float = 0.0
        self.accumulated_storage_cost: float = 0.0
        self.accumulated_ev_cost: float = 0.0
        self.accumulated_import_cost: float = 0.0
        self.accumulated_export_cost: float = 0.0

        # Rewards for each resource
        self.storage_action_reward: float = storage_action_reward

        # Penalties
        self.storage_action_penalty = storage_action_penalty
        self.import_penalty = import_penalty
        self.export_penalty = export_penalty
        self.balance_penalty = balance_penalty

        # Set a penalty for each resource
        self.accumulated_generator_penalty: float = 0.0
        self.accumulated_storage_penalty: float = 0.0
        self.accumulated_ev_penalty: float = 0.0
        self.accumulated_ev_penalty_trip: float = 0.0
        self.accumulated_import_penalty: float = 0.0
        self.accumulated_export_penalty: float = 0.0

        # Balance history
        self.household_balance_history = []

        # History of the environment
        self.household_history = []
        self.household_history_dictionary = {}

        # Current real reward
        self.current_real_reward = {}

    # Handle Observation Space
    def _handle_observation_space(self) -> None:
        # Observation space
        self._obs_space_in_preferred_format = True
        temp_observation_space = {
            'current_available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
            'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'current_loads': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32)
        }

        if self.storage is not None:
            temp_observation_space['current_soc'] = gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)

        # Set the observation space
        self.observation_space = gym.spaces.Dict(temp_observation_space)

    # Handle Action Space
    def _handle_action_space(self) -> None:
        # Action space
        self._action_space_in_preferred_format = True
        temp_action_space = {}

        # Generator action space
        if self.generator is not None:
            temp_action_space.update(self.__create_generator_actions__())

        # Storage action space
        if self.storage is not None:
            temp_action_space.update(self.__create_storage_actions__())

        # Aggregator action space
        temp_action_space.update(self._create_aggregator_actions())

        # Set the action space
        self.action_space = gym.spaces.Dict(temp_action_space)

    # Create Generator Action Space
    def __create_generator_actions__(self) -> dict:
        """
        Create the action space for the generators
        Varies according to the resource's renewable variable
        Renewable generators will have the following actions:
        - production (float) -> Renewable generators can control their production
        Non-renewable generators will have the following actions:
        - active (bool)

        :return: dict
        """

        gen = self.generator
        if gen.is_renewable == 2.0 or gen.is_renewable is True:  # Hack for the Excel file
            generator_actions = {
                'production': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            }
        else:
            generator_actions = {
                'active': gym.spaces.Discrete(2)
            }

        return generator_actions

        # Execute generator actions

    def _execute_generator_actions(self, actions) -> tuple[float, float]:
        """
        Execute the actions for the generators
        :param gen: generator resource
        :param actions: actions to be executed
        :return: float
        """

        # Calculate the cost of the generator
        penalty: float = 0.0

        # Placeholder for produced energy
        produced_energy: float = 0.0

        # Check if actions has active or production
        if 'active' in actions.keys():
            produced_energy = (actions['active'] *
                               self.generator.upper_bound[self.environment.current_timestep])
        elif 'production' in actions.keys():
            produced_energy = (actions['production'][0] *
                               self.generator.upper_bound[self.environment.current_timestep])

        # Attribute the produced energy to the generator
        self.generator.value[self.environment.current_timestep] = produced_energy
        self.current_production += produced_energy
        self.current_available_energy += produced_energy

        # Update on the resource
        self.generator.value[self.environment.current_timestep] = produced_energy

        cost: float = self.generator.upper_bound[self.environment.current_timestep] - produced_energy

        return cost, penalty


    # Create Storage Action Space
    def __create_storage_actions__(self) -> dict:
        """
        Create the action space for the storages
        Will have the following actions:
        - ctl: control the storage (bool) -> 0/1/2 for none/charge/discharge
        - value: value to be charged or discharged (float)
        :return: dict
        """

        storage_actions = {
            'ctl': gym.spaces.Discrete(3),
            'value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        }

        return storage_actions

        # Execute storage actions

    def _execute_storage_actions(self, actions) -> tuple[float, float]:
        """
        Execute the actions for the storages
        :param storage: storage resource
        :param actions: actions to be executed
        :return: reward to be used as penalty
        """
        # TODO: verify if this function is correct

        # Calculate the cost of the storage
        cost: float = 0.0

        # Set up the penalty to be returned in case of illegal storage actions
        # Such as overcharging or discharging beyond the available energy.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        penalty: float = 0.0

        # Check if it is the first timestep
        if self.environment.current_timestep == 0:
            self.storage.value[self.environment.current_timestep] = self.storage.initial_charge
        else:
            self.storage.value[self.environment.current_timestep] = self.storage.value[self.environment.current_timestep - 1]

        # Idle state
        if actions['ctl'] == 0:
            self.storage.charge[self.environment.current_timestep] = 0.0
            self.storage.discharge[self.environment.current_timestep] = 0.0

            return cost, penalty

        # Charge state
        elif actions['ctl'] == 1:
            # Percent of the charge_max you're willing to use at a given moment?
            charge = actions['value'][0]

            # Set the charge as a percentage of the maximum charge allowed
            charge = charge * self.storage.charge_max/ self.storage.capacity_max

            if self.storage.value[self.environment.current_timestep] + charge > 1.0:
                # Calculate the deviation from the bounds
                deviation = self.storage.value[self.environment.current_timestep] + charge - 1.0
                charge = 1.0 - self.storage.value[self.environment.current_timestep]
                penalty = deviation

            # Get the cost of the energy
            # cost = charge * storage.cost_charge[self.environment.current_timestep]

            # Heavily penalize the storage action if it requires importing energy
            if self.current_available_energy - charge * self.storage.capacity_max < 0:
                penalty += self.storage_action_penalty
            elif self.current_available_energy - charge * self.storage.capacity_max > 0:
                penalty -= self.storage_action_reward

            # Remove energy from the pool
            self.current_available_energy -= charge * self.storage.capacity_max

            # Update as well on the resource
            self.storage.value[self.environment.current_timestep] += charge
            self.storage.charge[self.environment.current_timestep] = charge
            self.storage.discharge[self.environment.current_timestep] = 0.0

            return cost, penalty

        # Discharge state
        elif actions['ctl'] == 2:
            discharge = actions['value'][0]

            # Set discharge as a percentage of the maximum discharge allowed
            discharge = discharge * self.storage.discharge_max / self.storage.capacity_max

            if self.storage.value[self.environment.current_timestep] - discharge < 0.0:
                # Calculate the deviation from the bounds
                deviation = abs(self.storage.value[self.environment.current_timestep] - discharge)
                discharge = self.storage.value[self.environment.current_timestep]
                penalty = deviation

            # Get the cost of the energy
            # cost = discharge * storage.cost_discharge[self.environment.current_timestep]
            cost = self.storage.discharge_max - discharge * self.storage.capacity_max

            # Assign resource charge and discharge variables
            self.storage.value[self.environment.current_timestep] -= discharge
            self.storage.charge[self.environment.current_timestep] = 0.0
            self.storage.discharge[self.environment.current_timestep] = discharge

            # Update as well on the resource
            self.storage.value[self.environment.current_timestep] -= discharge
            self.storage.charge[self.environment.current_timestep] = 0.0
            self.storage.discharge[self.environment.current_timestep] = discharge

            # Add the energy to the pool
            self.current_available_energy += discharge * self.storage.capacity_max

            return cost, penalty

        return cost, penalty


    # Create Aggregator Action Space
    def _create_aggregator_actions(self) -> dict:
        """
        Create the action space for the aggregator
        Will have the following actions:
        - ctl: action to take 0/1/2 for none/import/export
        - value: value to be imported or exported (float)

        :return: dict
        """

        return {
            'ctl': gym.spaces.Discrete(3),
            'value': gym.spaces.Box(low=0, high=max(self.aggregator.import_max),
                                    shape=(1,), dtype=np.float32)
        }

        # Execute aggregator actions

    def _execute_aggregator_actions(self, actions) -> tuple[float, float]:
        """
        Execute the actions for the aggregator
        :param actions: actions to be executed
        :return: aggregator costs and penalty
        """

        # Set up the aggregator's costs
        cost: float = 0.0

        # Set up the reward to be returned in case of illegal aggregator actions
        # Such as importing or exporting beyond the allowed limits.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        penalty: float = 0.0

        # If we still have energy left, there is no need to import extra energy
        if self.current_available_energy > 0:

            # Force to export
            # 1 - Get the deviation from the bounds
            to_export = self.current_available_energy
            if to_export > self.aggregator.export_max[self.environment.current_timestep]:
                deviation = to_export - self.aggregator.export_max[self.environment.current_timestep]
                to_export = self.aggregator.export_max[self.environment.current_timestep]
                penalty = deviation

            # 2 - Set the exports for agent and resource
            self.aggregator.imports[self.environment.current_timestep] = 0.0
            self.aggregator.exports[self.environment.current_timestep] = to_export

            self.aggregator.imports[self.environment.current_timestep] = 0.0
            self.aggregator.exports[self.environment.current_timestep] = to_export

            # 3 - Update the available energy pool
            self.current_available_energy -= to_export

            # Update the cost of the export
            cost = to_export * self.aggregator.export_cost[self.environment.current_timestep]

            return -cost, penalty

        # Check if there is a defect of energy that needs to be imported
        if self.current_available_energy < 0:

            # If not, we are forced to import
            to_import = abs(self.current_available_energy)
            if to_import > self.aggregator.import_max[self.environment.current_timestep]:
                # Calculate the deviation from the bounds
                deviation = to_import - self.aggregator.import_max[self.environment.current_timestep]
                to_import = self.aggregator.import_max[self.environment.current_timestep]
                penalty = deviation

            # Set the imports for agent and resource
            self.aggregator.imports[self.environment.current_timestep] = to_import
            self.aggregator.exports[self.environment.current_timestep] = 0.0

            self.aggregator.imports[self.environment.current_timestep] = to_import
            self.aggregator.exports[self.environment.current_timestep] = 0.0

            # Update the available energy pool
            self.current_available_energy += to_import

            # Get the associated costs of importation
            cost = to_import * self.aggregator.import_cost[self.environment.current_timestep]

            return cost, penalty

        return cost, penalty

        # Get observations

    def get_next_observations(self) -> gym.spaces.Dict:
        """
        Get the observations for the environment
        :return: dict
        """

        # Get the observation for the next resource
        observations = {}

        if self.environment.current_timestep >= self.load.value.shape[0]:
            return observations

        observations: dict = {
            'current_available_energy': np.array([self.current_available_energy],
                                                 dtype=np.float32),
            'current_buy_price': np.array([self.aggregator.import_cost[self.environment.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.aggregator.export_cost[self.environment.current_timestep]],
                                           dtype=np.float32),
            'current_loads': np.array([self.load.value[self.environment.current_timestep]],
                                      dtype=np.float32)
        }
        if self.storage is not None:
            observations['current_soc'] = np.array(
                [self.storage.value[self.environment.current_timestep - 1] if self.environment.current_timestep > 0
                 else self.storage.initial_charge],
                dtype=np.float32)

        return observations

    def execute_action(self, actions) -> tuple[float, float]:
        # TODO: verify rewards and weights
        total_cost = 0
        total_penalty = 0
        if self.generator is not None:
            # Execute the actions for the generator
            generator_cost, generator_penalty = self._execute_generator_actions(actions)
            self.accumulated_generator_cost += generator_cost
            self.accumulated_generator_penalty += generator_penalty
            total_cost += generator_cost
            # TODO: should there be any penalty for generating power
            # total_penalty += generator_penalty

        if self.storage is not None:
            # Execute the actions for the storage
            storage_cost, storage_penalty = self._execute_storage_actions(actions)
            self.accumulated_storage_cost += storage_cost
            self.accumulated_storage_penalty += storage_penalty
            total_cost += storage_cost
            total_penalty += storage_penalty * self.storage_action_penalty

        # Execute the actions for the aggregator (currently actions don't mean anything)
        if self.current_available_energy < 0:
            import_cost, import_penalty = self._execute_aggregator_actions(actions)

            self.accumulated_import_cost += import_cost
            self.accumulated_import_penalty += import_penalty
            total_cost += import_cost
            total_penalty += import_penalty * self.import_penalty

        elif self.current_available_energy > 0:
            # TODO: Verify if the minus signs are correctly placed
            export_cost, export_penalty = self._execute_aggregator_actions(actions)
            self.accumulated_export_cost -= export_cost
            self.accumulated_export_penalty += export_penalty

            # Added self.export_penalty myself
            total_cost += export_cost
            total_penalty += export_penalty * self.export_penalty

        # Is it that bad if we stay with available energy
        total_penalty += self.balance_penalty * abs(self.current_available_energy)
        self.household_balance_history.append(self.current_available_energy)
        return total_cost, total_penalty

    def calculate_reward(self, cost: float, penalty: float) -> float:
        # Calculate the reward
        reward = - cost - penalty

        return reward

    # Build the info logs
    def log_info(self):

        info = {}

        # Generator
        if self.generator is not None:
            info.update( {
                'generator_cost': self.accumulated_generator_cost,
                'generator_penalty': self.accumulated_generator_penalty
            })

        # Storages
        if self.storage is not None:
            info.update({
                'storage_cost': self.accumulated_storage_cost,
                'storage_penalty': self.accumulated_storage_penalty
            })

        # Aggregator
        info.update({
            'aggregator_cost': self.accumulated_import_cost + self.accumulated_export_cost,
            'aggregator_penalty': self.accumulated_import_penalty + self.accumulated_export_penalty
        })

        return info


