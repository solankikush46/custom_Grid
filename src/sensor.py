# sensor.py

import numpy as np
from .constants import (
    BATTERY_CAPACITY_JOULES, ALPHA_ELEC, ALPHA_SHORT, ALPHA_LONG,
    THRESHOLD_DIST, MAX_COMM_RANGE,
    K_BROADCAST, K_TO_MINER, K_TO_BASE
)

def transmission_energy(k_bits, distance):
    """Computes the transmission energy based on distance and message size."""
    if distance > MAX_COMM_RANGE:
        return 0.0
    if distance <= THRESHOLD_DIST:
        return k_bits * (ALPHA_ELEC + ALPHA_SHORT * distance ** 2)
    else:
        return k_bits * (ALPHA_ELEC + ALPHA_LONG * distance ** 4)

def reception_energy(k_bits):
    """Computes the energy to receive a message."""
    return k_bits * ALPHA_ELEC

def compute_sensor_energy_loss(sensor_pos, connected_miners_pos, base_stations_pos):
    """
    Calculates the total energy drained from a single sensor in one timestep.
    
    Args:
        sensor_pos (tuple): The (row, col) of the sensor.
        connected_miners_pos (list of tuples): A list of positions of miners connected to this sensor.
        base_stations_pos (list of tuples): A list of all base station positions.
    """
    total_energy_loss = 0.0

    # 1. Constant broadcast energy to general area
    # This represents a sensor's 'upkeep' cost per step.
    total_energy_loss += transmission_energy(K_BROADCAST, MAX_COMM_RANGE)

    # 2. Energy for communicating with each connected miner
    for miner_pos in connected_miners_pos:
        dist = np.linalg.norm(np.array(sensor_pos) - np.array(miner_pos))
        # Sensor both sends and receives data from a miner's device
        total_energy_loss += transmission_energy(K_TO_MINER, dist)
        total_energy_loss += reception_energy(K_TO_MINER)

    # 3. Energy for communicating with the nearest base station
    if base_stations_pos:
        distances_to_bs = [np.linalg.norm(np.array(sensor_pos) - np.array(bs_pos)) for bs_pos in base_stations_pos]
        nearest_dist = min(distances_to_bs)
        if nearest_dist <= MAX_COMM_RANGE:
            # Sensor sends its data and receives an acknowledgment
            total_energy_loss += transmission_energy(K_TO_BASE, nearest_dist)
            total_energy_loss += reception_energy(K_TO_BASE)
            
    return total_energy_loss
