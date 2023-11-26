import gymnasium as gym
import numpy as np
import random
from copy import copy
from collections import OrderedDict
from ...resources.base_resource import BaseResource
from ...resources.vehicle import Vehicle

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EVMultiAgent_v0(MultiAgentEnv):
    """
    A multi-agent environment for the EV charging problem.
    Penalties are applied for EVs not being charged at time of departure and illegal actions;
    Reward at each timestep is the negative sum of penalties and charging costs:
    reward = - (penalty + charging_cost)

    Penalties for illegal actions are applied as follows:
    penalty_action = coefficient * (action_delta)^2
    where action_delta is the difference between the current action and the allowed bounds

    Penalty for not being charged at time of departure is applied as a constant

    There are no restrictions for energy drawn from the grid, but the energy price can be defined

    The observation space is a dictionary with the following keys:
    - 'state': the current state of the EV
    - 'time_until_departure': time until departure, in timesteps
    - 'connected': whether the EV is connected to the charger
    - 'energy_price': the current energy price
    - 'time': the current timestep

    The action space is a dictionary with the following keys:
    - 'action_value': the amount of energy to charge/discharge the EV
    - 'action_indicator': whether the EV is idle, charging or discharging. 0: idle, 1: charging, 2: discharging
    """

    def __init__(self, resources: list[Vehicle],
                 penalty_action_coefficient: float = 1000,
                 penalty_not_charged: float = 1000,
                 energy_price: np.array = None,
                 ):
        super().__init__()

        # Define the Resources
        self.resources = resources
        self.energy_price = energy_price
        self.evs = resources.copy()

        # Define penalties
        self.penalty_action_coefficient = penalty_action_coefficient
        self.penalty_not_charged = penalty_not_charged

        # Initialize timestep counter
        self.current_timestep = 0

        # Create the agents
        self.possible_agents = [i.name for i in self.resources]
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Create observation space
        self._obs_space_in_preferred_format = True
        temp_observation_space = {}
        for agent in self.agents:
            ev_observation_space = gym.spaces.Dict({
                'state': gym.spaces.Box(low=self.agents[agent].min_charge,
                                        high=self.agents[agent].capacity_max,
                                        shape=(1,)),
                'time_until_departure': gym.spaces.Box(low=0,
                                                       high=self.agents[agent].schedule_connected.shape[0],
                                                       shape=(1,)),
                'connected': gym.spaces.Discrete(2),
                'energy_price': gym.spaces.Box(low=0,
                                               high=self.energy_price.max(),
                                               shape=(1,)),
                'time': gym.spaces.Box(low=0,
                                       high=self.agents[agent].schedule_connected.shape[0],
                                       shape=(1,)),
            })
            temp_observation_space[agent] = ev_observation_space
        self.observation_space = gym.spaces.Dict(temp_observation_space)

        # Create action space
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        for agent in self.agents:
            ev_action_space = gym.spaces.Dict({
                'action_value': gym.spaces.Box(low=0.0,
                                               high=self.agents[agent].capacity_max,
                                               shape=(1,)),
                'action_indicator': gym.spaces.Discrete(3),
            })
            temp_action_space[agent] = ev_action_space
        self.action_space = gym.spaces.Dict(temp_action_space)

    def __create_agents__(self):
        agents = {}
        for ev in self.resources:
            agents[str(ev.name)] = ev
        return agents

    # Implement the reset function
    def reset(self, *, seed=None, options=None):
        # Create the agents
        self.agents = self.__create_agents__()
        self.terminateds = set()
        self.truncateds = set()

        # Reset the timestep counter
        self.current_timestep = 0

        # Reset the EVs
        for ev in self.evs:
            ev.value = ev.initial_charge

        # Return the observations
        return self._get_observations(), {}

    def _get_observations(self) -> dict:
        """
        Returns the observations for all agents
        :return: dictionary with observations for all agents
        """

        if self.current_timestep >= self.energy_price.shape[0]:
            return self.observation_space.sample()

        observations = {}
        for ev in self.evs:
            observations[ev.name] = self._get_observation(ev)
        return observations

    def _get_observation(self, ev: Vehicle) -> dict:
        next_departure = np.where(ev.schedule_requirement_soc > self.current_timestep)[0]
        next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
            if len(next_departure) > 0 else ev.min_charge

        time_until_departure = abs(next_departure - self.current_timestep)

        observation = {'state': ev.value,
                       'time_until_departure': time_until_departure,
                       'required_soc': next_departure_soc,
                       'connected': ev.schedule_connected[self.current_timestep],
                       'energy_price': self.energy_price[self.current_timestep],
                       'time': self.current_timestep}
        return observation

    def step(self, action_dict: dict):

        # Check if end of episode
        if self.current_timestep >= self.energy_price.shape[0]:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
            terminations['__all__'] = True
            truncations['__all__'] = True

            observations = self._get_observations()
            reward = {a: 0 for a in self.agents}
            info = {a: {} for a in self.agents}

            return observations, reward, terminations, truncations, info

        # Check if there are actions
        exists_action = len(action_dict) > 0
        action_results = {}
        # Apply existing actions
        if exists_action:
            for ev in action_dict.keys():
                # Save the current state before taking action
                soc = self.agents[ev].value

                # Do action and get results
                updated_soc, charge, discharge, cost, \
                    penalty_action, penalty_charge = self._do_action(self.agents[ev], action_dict[ev])

                # Calculate reward
                reward = self._get_reward(cost, penalty_action, penalty_charge)

                # Save the results
                action_results[ev] = {'initial_soc': soc,
                                      'updated_soc': updated_soc,
                                      'charge': charge,
                                      'discharge': discharge,
                                      'cost': cost,
                                      'penalty_action': penalty_action,
                                      'penalty_charge': penalty_charge,
                                      'reward': reward}

        # Create reward dictionary
        rewards = {}
        for ev in self.agents.keys():
            rewards[ev] = action_results[ev]['reward'] if exists_action else 0

        # Update information dictionary
        info = {}
        for ev in self.agents.keys():
            info[ev] = action_results[ev] if exists_action else {}

        # Termination and truncation
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        terminations['__all__'] = False
        truncations['__all__'] = False

        # Check if end of episode
        if self.current_timestep >= self.energy_price.shape[0]:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
            terminations['__all__'] = True
            truncations['__all__'] = True
            self.agents = []

        # Update the timestep
        self.current_timestep += 1

        # Get the observations
        observations = self._get_observations()

        return observations, rewards, terminations, truncations, info

    def _do_action(self, ev, action) -> (float, float, float, float, float):
        """
        Returns the connected state of the EV, the charge and discharge values, as well as the discharge cost.
        :param action: dictionary with action_value and action_indicator
        :return: value, charge, discharge, cost, penalty_action, penalty_charge
        """

        # Initialize penalty and costs
        penalty_action = 0.0
        penalty_charge = 0.0
        cost = 0.0

        # Get the action
        action_value = action['action_value']
        action_indicator = action['action_indicator']

        # Get the current state
        state = ev.value
        connected = ev.schedule_connected[self.current_timestep]
        next_state = state
        charge = 0.0
        discharge = 0.0

        # Check if EV is arriving
        if ev.schedule_arrival_soc[self.current_timestep] > 0:
            ev.value = ev.schedule_arrival_soc[self.current_timestep]

        if ev.schedule_requirement_soc[self.current_timestep] > 0:
            if ev.value < ev.schedule_requirement_soc[self.current_timestep]:
                penalty_charge += self.penalty_not_charged

        # Get the next state
        if action_indicator == 0:
            # Idle
            next_state = state
        elif action_indicator == 1:
            # Charge
            charge = action_value * ev.charge_efficiency
        elif action_indicator == 2:
            # Discharge
            discharge = action_value / ev.discharge_efficiency
        else:
            raise ValueError('Action indicator should be 0, 1 or 2')

        # Check if the next state is within bounds
        if next_state - discharge < ev.min_charge:
            # If not, set the next state to the min charge
            next_state = ev.min_charge
            discharge = state - ev.min_charge

            # Difference between next state and min charge
            penalty_action = ev.min_charge - next_state

        if next_state + charge > ev.capacity_max:
            # If not, set the next state to the max charge
            next_state = ev.capacity_max
            charge = ev.capacity_max - state

            # Difference between next state and max charge
            penalty_action = next_state - ev.capacity_max

        # Get the cost
        if connected:
            if charge > 0:
                cost = self.energy_price[self.current_timestep] * action_value
            elif discharge > 0:
                cost = -self.energy_price[self.current_timestep] * action_value
        else:
            cost = 0

        # Update the EV
        ev.value = next_state + (charge - discharge) * connected

        return ev.value, charge, discharge, cost, penalty_action, penalty_charge

    def _get_reward(self, cost, penalty_action, penalty_charge):
        return -cost - penalty_charge - penalty_action ** 2
