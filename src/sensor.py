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

import numpy as np
from . import utils
from .constants import (
    BATTERY_CAPACITY_JOULES, ALPHA_ELEC, ALPHA_SHORT, ALPHA_LONG,
    THRESHOLD_DIST, MAX_COMM_RANGE,
    K_BROADCAST, K_TO_MINER, K_TO_BASE
)

# ===============================================
# === Core Energy Model (from your old logic) ===
# ===============================================

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
    This was previously named calculate_total_drain_on_sensor.
    """
    total_energy_loss = 0.0
    total_energy_loss += transmission_energy(K_BROADCAST, MAX_COMM_RANGE)

    for miner_pos in connected_miners_pos:
        dist = utils.euclidean_distance(sensor_pos, miner_pos)
        total_energy_loss += transmission_energy(K_TO_MINER, dist)
        total_energy_loss += reception_energy(K_TO_MINER)

    if base_stations_pos:
        distances_to_bs = utils.euclidean_distances(sensor_pos, base_stations_pos)
        if distances_to_bs.size > 0:
            nearest_dist = np.min(distances_to_bs)
            if nearest_dist <= MAX_COMM_RANGE:
                total_energy_loss += transmission_energy(K_TO_BASE, nearest_dist)
                total_energy_loss += reception_energy(K_TO_BASE)
            
    return total_energy_loss

# ====================================================================
# === Master Battery Update Function (replaces the old logic) ===
# ====================================================================
def update_all_sensor_batteries(sensor_positions, current_batteries, all_movers_pos, base_stations_pos):
    """
    Calculates and returns the new battery state for ALL sensors after one timestep.
    This single function encapsulates the entire battery depletion process.

    Args:
        sensor_positions (list): A list of all sensor (r, c) positions.
        current_batteries (dict): The dictionary of current battery levels.
        all_movers_pos (list): A list of all miner and agent (r, c) positions.
        base_stations_pos (list): A list of all base station (r, c) positions.

    Returns:
        dict: A new dictionary with the updated battery levels for all sensors.
    """
    # 1. Determine which movers are connected to which sensors
    connections = {s_pos: [] for s_pos in sensor_positions}
    if sensor_positions:
        for mover_pos in all_movers_pos:
            # Find the closest sensor for this mover
            closest_sensor = min(sensor_positions, key=lambda s_pos: utils.euclidean_distance(mover_pos, s_pos))
            connections[closest_sensor].append(mover_pos)
    
    # 2. Calculate the new battery level for each sensor
    new_batteries = current_batteries.copy()
    for sensor_pos, connected_miners in connections.items():
        # Calculate the total energy used by this sensor in this timestep
        energy_used = compute_sensor_energy_loss(
            sensor_pos,
            connected_miners,
            base_stations_pos
        )
        
        # Convert energy loss to percentage and update the battery level
        battery_loss_percent = (energy_used / BATTERY_CAPACITY_JOULES) * 100
        current_battery = new_batteries[sensor_pos]
        new_batteries[sensor_pos] = max(0.0, current_battery - battery_loss_percent)

    return new_batteries
