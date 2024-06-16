import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from src.parsers import BaseParser
from src.resources import Aggregator, Load, Generator, Storage


class PecanParser(BaseParser):
    def __init__(self, timeseries_file_path: str, metadata_file_path: str, prices_file_path: str,
                 use_cache: bool = True):
        super().__init__(timeseries_file_path)
        self.timeseries_file_path = timeseries_file_path
        self.metadata_file_path = metadata_file_path
        self.prices_file_path = prices_file_path

        self.timeseries_data = None
        self.metadata = None
        self.price_data = None

        self.unique_household_ids = None

        # Prepare the variables
        self.household_resources = []

        self.use_cache = use_cache

        return

    def parse(self):
        # Check if the specs are cached
        if self.use_cache and self.is_cache_present():
            self.load_processed_data()
            return

        # Else parse the files
        self.read_files()
        self.process_tables()
        self.save_processed_data()
        # For testing
        self.print_tables()
        return

    def get_parsed_resources(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.timeseries_data['time'].iloc[0]
        else:
            start_time = pd.to_datetime(start_time).to_numpy().astype('datetime64[ns]')
        if end_time is None:
            end_time = self.timeseries_data['time'].iloc[-1]
        else:
            end_time = pd.to_datetime(end_time).to_numpy().astype('datetime64[ns]')

        relevant_price_data = self.price_data[
            (self.price_data['time'] >= start_time) & (self.price_data['time'] <= end_time)]
        relevant_timeseries_data = self.timeseries_data[
            (self.timeseries_data['time'] >= start_time) & (self.timeseries_data['time'] <= end_time)]

        resources = self.create_resources(price_data=relevant_price_data, timeseries_data=relevant_timeseries_data)
        return resources

    def read_files(self):
        # Read the files
        self.timeseries_data = pd.read_csv(self.timeseries_file_path)
        self.metadata = pd.read_csv(self.metadata_file_path)
        # Interpret ; as the delimiter and , as the decimal point
        # The prices are given as cents per kWh
        self.price_data = pd.read_csv(self.prices_file_path, delimiter=';', decimal=',')
        return

    def process_tables(self):
        self.unique_household_ids = self.timeseries_data['dataid'].unique()
        self.metadata = self.metadata[self.metadata['dataid'].isin(self.unique_household_ids)]
        # Take relevant columns from both files
        self.timeseries_data = self.timeseries_data[['dataid', 'local_15min', 'grid', 'solar']]
        self.metadata = self.metadata[['dataid', 'pv']]
        self.price_data = self.price_data[['DateTime', 'realtime_lbmp (avg) (nyiso)']]

        # Rename columns
        self.timeseries_data.rename(columns={'dataid': 'id', 'local_15min': 'time'}, inplace=True)
        self.metadata.rename(columns={'dataid': 'id'}, inplace=True)
        self.price_data.rename(columns={'DateTime': 'time', 'realtime_lbmp (avg) (nyiso)': 'price'}, inplace=True)

        self.timeseries_data['time'] = pd.to_datetime(self.timeseries_data['time'])

        # Resample the price data to 15 minutes
        self.price_data['time'] = pd.to_datetime(self.price_data['time'])
        self.price_data.set_index('time', inplace=True)
        self.price_data = self.price_data.resample('15T').mean().reset_index()

        print(self.timeseries_data.head())
        # Sort the timeseries data
        self.timeseries_data.sort_values(by=['id', 'time'], inplace=True)

        # Create a new column for the usage which is the sum of grid and solar
        self.timeseries_data['usage'] = self.timeseries_data['grid'] + self.timeseries_data['solar']

        self.timeseries_data['time'] = pd.to_datetime(self.timeseries_data['time']).dt.date
        self.timeseries_data.reset_index(drop=True, inplace=True)
        self.price_data['time'] = pd.to_datetime(self.price_data['time']).dt.date

        self.metadata.reset_index(drop=True, inplace=True)
        # Add predictions of prices and pv generation
        self.add_predictions()

        return

    def add_predictions(self):
        # Add predictions of prices and pv generation for each timestep as columns
        # Predictions are generated as the average of the same hour of the previous 7 days
        self.price_data['price_prediction'] = np.nan
        self.timeseries_data['pv_prediction'] = np.nan
        prediction_start = 4 * 24 * 7
        interval = 4 * 24
        for id in self.unique_household_ids:
            id_mask = self.timeseries_data['id'] == id
            household_row_count = np.sum(id_mask)
            # Continue if the household does not have pv
            if (self.metadata[self.metadata['id'] == id]['pv'] != 'yes').any() or pd.isna(
                    self.timeseries_data.loc[id_mask, 'solar'].iloc[0]):
                continue
            # Start iterating after 7 days
            for i in range(prediction_start, household_row_count):
                # print(f"Predicting solar for timestep {i} and household {id}")
                mean_past_pv = self.timeseries_data.loc[id_mask, 'solar'].iloc[i - prediction_start:i:interval].mean()
                actual_index = self.timeseries_data[id_mask].index[i]
                self.timeseries_data.at[actual_index, 'pv_prediction'] = mean_past_pv

        for i in range(prediction_start, len(self.price_data)):
            # Get dataframes with the same id in the previous 7 days
            # print("Predicting price for timestep", i)

            mean_past_prices = self.price_data.iloc[i - prediction_start:i:interval]['price'].mean()
            self.price_data.at[i, 'price_prediction'] = mean_past_prices

        return

    def create_resources(self, timeseries_data=None, metadata=None, price_data=None):
        # Use full tables if not provided
        if timeseries_data is None:
            timeseries_data = self.timeseries_data
        if metadata is None:
            metadata = self.metadata
        if price_data is None:
            price_data = self.price_data

        configuration = []
        # Take buy prices from the price data and sell prices as a minimum between the buy prices and a fixed value
        sell_price = 3.0
        sell_prices = np.minimum(sell_price, price_data['price'])
        aggregator_specs = {'buy_prices': price_data['price'].values, 'sell_prices': sell_prices,
                            'price_prediction': price_data['price_prediction']}
        for id in self.unique_household_ids:
            household_timeseries = timeseries_data[timeseries_data['id'] == id]

            print(f"Length of household {id} timeseries", len(household_timeseries))
            # Create pv specs
            has_pv = (metadata[metadata['id'] == id]['pv'] == 'yes').any()
            pv_specs = {'present': has_pv, 'values': household_timeseries['solar'].values,
                        'pv_prediction': household_timeseries['pv_prediction']}

            # Hard-coded batteries identical for each household
            battery_specs = self.create_battery_specs()

            # Create specs for load as the sum of grid and solar for each timestep
            usage = household_timeseries['usage'].values
            load_specs = {'values': usage}

            # Create a new resource
            accumulated_specs = {'id': id, 'generator': pv_specs, 'storage': battery_specs, 'load': load_specs,
                                 'aggregator': aggregator_specs}
            configuration.append(accumulated_specs)

        return self.specs_to_resources(configuration)

    def is_cache_present(self):
        return self.check_if_file_exists(
            'data/pecan_street/cache/processed_timeseries_data.csv') and self.check_if_file_exists(
            'data/pecan_street/cache/processed_price_data.csv') and self.check_if_file_exists(
            'data/pecan_street/cache/processed_metadata.csv')

    def save_processed_data(self):
        # Save processed tables to csv files
        self.timeseries_data.to_csv('data/pecan_street/cache/processed_timeseries_data.csv', index=False)
        self.metadata.to_csv('data/pecan_street/cache/processed_metadata.csv', index=False)
        self.price_data.to_csv('data/pecan_street/cache/processed_price_data.csv', index=False)

        return

    def load_processed_data(self):
        # Load processed tables from csv files
        self.timeseries_data = pd.read_csv('data/pecan_street/cache/processed_timeseries_data.csv')
        self.metadata = pd.read_csv('data/pecan_street/cache/processed_metadata.csv')
        self.price_data = pd.read_csv('data/pecan_street/cache/processed_price_data.csv')

        self.unique_household_ids = self.timeseries_data['id'].unique()
        return

    @staticmethod
    def check_if_file_exists(filename):
        # Check if the specs are cached
        try:
            with open(filename,
                      'r') as f:
                return True
        except FileNotFoundError:
            return False

    @staticmethod
    def specs_to_resources(specs):
        resources = []

        # Add the aggregator, prices are shared across households
        aggregator = Aggregator(name='aggregator_1',
                                imports=np.zeros(specs[0]['aggregator']['buy_prices'].shape),
                                exports=np.zeros(specs[0]['aggregator']['buy_prices'].shape),
                                import_max=1000,
                                export_max=1000,
                                import_cost=specs[0]['aggregator']['buy_prices'],
                                export_cost=specs[0]['aggregator']['sell_prices']
                                )

        max_timesteps = aggregator.imports.shape[0]
        # Generate the households
        # For simplicity, we assume there are as many households as there are loads
        for spec in specs:
            household_id = spec['id']
            household_resources = {}
            # Add the loads
            household_resources['load'] = Load(name='load_{:05d}'.format(household_id),
                                               value=spec['load']['values'],
                                               cost_cut=0,
                                               cost_reduce=0,
                                               cost_ens=0)

            # Add the generators
            if spec['generator']['present']:
                household_resources['generator'] = Generator(name='generator_{:05d}'.format(household_id),
                                                             value=np.zeros(spec['generator']['values'].shape),
                                                             lower_bound=np.array([-1] * max_timesteps),
                                                             upper_bound=spec['generator']['values'],
                                                             cost=np.zeros(spec['generator']['values'].shape),
                                                             cost_nde=np.zeros(spec['generator']['values'].shape),
                                                             is_renewable=True)

            # Add the storage
            household_resources['storage'] = (Storage(name='storage_{:05d}'.format(household_id),
                                                      value=np.zeros(spec['generator']['values'].shape),
                                                      cost_discharge=spec['storage']['discharge_cost'],
                                                      cost_charge=spec['storage']['charge_cost'],
                                                      capacity_max=spec['storage']['capacity_max'],
                                                      capacity_min=spec['storage']['capacity_min'],
                                                      initial_charge=spec['storage']['initial_charge'],
                                                      discharge_efficiency=spec['storage']['discharge_efficiency'],
                                                      charge_efficiency=spec['storage']['charge_efficiency'],
                                                      discharge_max=spec['storage']['p_discharge_max'],
                                                      charge_max=spec['storage']['p_charge_max'],
                                                      capital_cost=0))

            household_resources['aggregator'] = deepcopy(aggregator)
            resources.append(household_resources)
        return resources

    @staticmethod
    def create_battery_specs():
        # Create a battery using fixed values
        battery_specs = {'present': True, 'p_charge_max': 5, 'p_discharge_max': 5, 'capacity_max': 13.5,
                         'capacity_min': 0, 'initial_charge': 0, 'charge_efficiency': 0.8, 'discharge_efficiency': 0.8,
                         'charge_cost': 0, 'discharge_cost': 0}
        return battery_specs

    def print_tables(self):
        # Print first few rows of the tables
        print("Timeseries data:")
        print(self.timeseries_data.head())
        print("Metadata:")
        print(self.metadata.head())
        print("Price data:")
        print(self.price_data.head())

    def plot_tables(self):
        pass
