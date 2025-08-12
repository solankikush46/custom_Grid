# grid_env.py
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random 
import os
import csv
import pygame
import src.grid_gen as grid_gen
from src.constants import *
from src.utils import *
from src.reward_functions import compute_reward
from src.sensor import transmission_energy, reception_energy, compute_sensor_energy_loss, update_single_sensor_battery
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from src.path_planning import *

##==============================================================
## Logs custom metrics stored in `info` dict to TensorBoard
## and csv
##==============================================================
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # tracker
        self.episode_count = 0

        # CSV files and writers
        self.timestep_csv_file = None
        self.timestep_csv_writer = None
        self.timestep_fieldnames = []

        self.episode_csv_file = None
        self.episode_csv_writer = None
        self.episode_fieldnames = []

        self.subrewards_csv_file = None
        self.subrewards_csv_writer = None
        self.subrewards_fieldnames = []

        self.sensor_battery_csv_file = None
        self.sensor_battery_csv_writer = None
        self.sensor_battery_fieldnames = []

        # file to save non-line graph data
        self.txt = None
        
        self.verbose = verbose

    def _on_training_start(self) -> None:
        log_dir = self.logger.dir
        print(f"Logger directory resolved to: {log_dir}")

        self.timestep_csv_path = os.path.join(log_dir, f"timestep_metrics.csv")
        self.episode_csv_path = os.path.join(log_dir, f"episode_metrics.csv")
        self.subrewards_csv_path = os.path.join(log_dir, f"subrewards_metrics.csv")
        self.sensor_battery_csv_path = os.path.join(log_dir, "sensor_battery_levels.csv")
        
        self.timestep_csv_file = open(self.timestep_csv_path, "w", newline="")
        self.episode_csv_file = open(self.episode_csv_path, "w", newline="")
        self.subrewards_csv_file = open(self.subrewards_csv_path, "w", newline="")
        self.sensor_battery_csv_file = open(self.sensor_battery_csv_path, "w", newline="")

        self.txt_path = os.path.join(log_dir, "log.txt")
        self.txt_file = open(self.txt_path, "w")
        self.success_count = 0

        if self.verbose > 0:
            print(f"Timestep CSV logging to: {self.timestep_csv_path}")
            print(f"Episode CSV logging to: {self.episode_csv_path}")
            print(f"Subrewards CSV logging to: {self.subrewards_csv_path}")
            print(f"Sensor Battery CSV logging to: {self.sensor_battery_csv_path}")
            
    def _flatten_info(self, info):
        flat = {}
        subrewards = {}
        for k, v in info.items():
            if k == "subrewards" and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    subrewards[f"{sub_k}"] = sub_v
            elif not isinstance(v, (dict, list)):
                flat[k] = v
        return flat, subrewards

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue

            flat_info, subrewards_info = self._flatten_info(info)

            # Split metrics
            timestep_keys = [
                "current_reward", "current_battery",
                "distance_to_goal",
                "terminated", "truncated"
            ]
            episode_keys = [
                "cumulative_reward", "obstacle_hits",
                "visited_count", "average_battery_level",
                "episode_length", "revisit_count"
            ]

            timestep_data = {k: flat_info.get(k) for k in timestep_keys if k in flat_info}
            episode_data = {k: flat_info.get(k) for k in episode_keys if k in flat_info}

            # TensorBoard logging
            for k, v in timestep_data.items():
                self.logger.record(f"timestep/{k}", v)
            for k, v in episode_data.items():
                self.logger.record(f"episode/{k}", v)
            for k, v in subrewards_info.items():
                self.logger.record(f"subrewards/{k}", v)

            # CSV: timestep
            if not self.timestep_fieldnames:
                self.timestep_fieldnames = list(timestep_data.keys())
                self.timestep_csv_writer = csv.DictWriter(self.timestep_csv_file, fieldnames=self.timestep_fieldnames)
                self.timestep_csv_writer.writeheader()
            timestep_row = {k: timestep_data.get(k, "") for k in self.timestep_fieldnames}
            self.timestep_csv_writer.writerow(timestep_row)

            # CSV: subrewards
            if subrewards_info:
                # Optionally, attach step number if available
                if not self.subrewards_fieldnames:
                    self.subrewards_fieldnames = list(subrewards_info.keys())
                    self.subrewards_csv_writer = csv.DictWriter(self.subrewards_csv_file, fieldnames=self.subrewards_fieldnames)
                    self.subrewards_csv_writer.writeheader()
                subrewards_row = {k: subrewards_info.get(k, "") for k in self.subrewards_fieldnames}
                self.subrewards_csv_writer.writerow(subrewards_row)

            # CSV: episode
            if flat_info.get("terminated", False) or flat_info.get("truncated", False):
                # add episode count to csv data
                self.episode_count += 1
                episode_row = {"episode": self.episode_count}
                episode_row.update(episode_data)
                
                if not self.episode_fieldnames:
                    self.episode_fieldnames = ["episode"] + list(episode_data.keys())
                    self.episode_csv_writer = csv.DictWriter(self.episode_csv_file, fieldnames=self.episode_fieldnames)
                    self.episode_csv_writer.writeheader()
                    
                self.episode_csv_writer.writerow(episode_row)

                reached_exit = flat_info.get("terminated")
                self.success_count += int(reached_exit)

                # log to txt
                success_ratio = self.success_count / self.episode_count
                if self.txt_file:
                    self.txt_file.write(
                        f"Episode {self.episode_count}: Success Ratio = {self.success_count}/{self.episode_count} ({success_ratio:.1%})\n"
                    )
                    self.txt_file.flush()

            sensor_battery_snapshot = info.get("sensor_batteries")
            if sensor_battery_snapshot:
                if not self.sensor_battery_fieldnames:
                    # Use stringified sensor positions as column names
                    self.sensor_battery_fieldnames = list(map(str, sorted(sensor_battery_snapshot.keys())))
                    self.sensor_battery_csv_writer = csv.DictWriter(
                        self.sensor_battery_csv_file, fieldnames=self.sensor_battery_fieldnames
                    )
                    self.sensor_battery_csv_writer.writeheader()
                
                # Write battery values in sorted sensor position order
                row = {str(k): sensor_battery_snapshot[k] for k in sorted(sensor_battery_snapshot.keys())}
                self.sensor_battery_csv_writer.writerow(row)
            
        return True

    def _on_training_end(self) -> None:
        for f in [
                (self.timestep_csv_file, self.timestep_csv_path),
                (self.episode_csv_file, self.episode_csv_path),
                (self.subrewards_csv_file, self.subrewards_csv_path),
                (self.sensor_battery_csv_file, self.sensor_battery_csv_path)
        ]:
            if f[0]:
                f[0].close()
                if self.verbose > 0:
                    print(f"Closed CSV: {f[1]}")

        if self.txt_file:
            self.txt_file.close()
            if self.verbose > 0:
                print(f"Closed TXT: {self.txt_path}")

##==============================================================
## GridWorldEnv Class
##==============================================================
class GridWorldEnv(Env):
    def __init__(self,
                 reward_fn,
                 grid_file: str = None,
                 grid_height: int = None,
                 grid_width: int = None,
                 obstacle_percentage=None,
                 n_sensors=None, reset_kwargs={},
                 is_cnn=False, battery_truncation=False,
                 n_miners=12,
                 ):
        super(GridWorldEnv, self).__init__()

        ##=============== member variables ===============##
        # grid config
        self._init_grid_config(grid_file,
                               grid_height, grid_width,
                               obstacle_percentage,
                               n_sensors)
        self.max_steps = 2000
        self.reset_kwargs = reset_kwargs
        self.is_cnn = is_cnn
        self.battery_truncation = battery_truncation
        
        # pygame rendering
        self.pygame_initialized = False
        self.fixed_window_size = None
        self.cell_size = None
        self.screen = None
        self.font = None
        self.clock = None
        self.render_fps = 30

        # environment state variables
        self.grid = self.static_grid.copy()
        self.agent_pos = None
        self.visited = None
        self.current_battery_level = 0.0
        # self.battery_values_in_radar = None
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0
        self.last_action = -1
        self.miners = []
        self.n_miners = 12
        self.OBSTACLE_VALS = (OBSTACLE, SENSOR, BASE_STATION)
        self.n_miners = n_miners
        self.reward_fn = reward_fn

        # d*lite pathplanning assitance
        self.pathfinder = None
        self.cost_func_calls = 0
        
        # exclusively for graphing
        self.battery_levels_during_episode = []
            
        # observation/action spaces
        self._init_spaces()

        '''
        # radar around sensors
        self._init_radar_zone()
        '''
        ##================================================##

        self.reset()
