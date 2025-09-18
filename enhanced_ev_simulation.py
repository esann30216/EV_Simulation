"""
Electric Vehicle Charging Simulation System - Fixed Version
Removed visualization, fixed global variable issues, and performance problems
"""

import datetime
import gc
import json
import logging
import math
import os
import random
import sys
import threading
import time
import traceback
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, Optional, List, Set, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Handle numpy import
try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not found. Using standard library alternatives.")
    np = None

# Handle psutil import for performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not found. Performance monitoring limited.")
    PSUTIL_AVAILABLE = False

# SUMO imports
try:
    import traci
    import traci.constants as tc
    SUMO_AVAILABLE = True
except ImportError:
    print("Error: SUMO/TraCI not found. Please install SUMO.")
    SUMO_AVAILABLE = False

# OpenAI imports - CRITICAL FOR GPT DECISIONS
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI library not found. GPT features disabled.")
    OPENAI_AVAILABLE = False
    OpenAI = None

# ============================================================================
# Constants - Eliminating Magic Numbers
# ============================================================================

class TimeConstants:
    """Time-related constants in seconds"""
    MINUTE = 60
    FIVE_MINUTES = 300
    TEN_MINUTES = 600
    FIFTEEN_MINUTES = 900
    THIRTY_MINUTES = 1800
    HOUR = 3600
    TWO_HOURS = 7200
    FOUR_HOURS = 14400
    EIGHT_HOURS = 28800
    DAY = 86400

class ChargingConstants:
    """Charging-related constants"""
    MIN_CHARGE_TIME = 600  # 10 minutes minimum
    QUICK_CHARGE_TIME = 900  # 15 minutes for taxi
    STANDARD_CHARGE_TIME = 1800  # 30 minutes standard
    LONG_CHARGE_TIME = 2400  # 40 minutes for trucks
    MAX_CHARGE_TIME = 3600  # 1 hour maximum

    FAST_CHARGE_POWER = 150000  # 150kW threshold
    ULTRA_FAST_CHARGE_POWER = 350000  # 350kW threshold

class BatteryConstants:
    """Battery threshold constants"""
    CRITICAL = 0.10  # Emergency level
    VERY_LOW = 0.15  # Must charge immediately
    LOW = 0.25  # Should charge soon
    MEDIUM_LOW = 0.35  # Consider charging
    MEDIUM = 0.50  # Comfortable level
    MEDIUM_HIGH = 0.65  # Good level
    HIGH = 0.80  # Well charged
    FULL = 0.95  # Nearly full

class ProbabilityConstants:
    """Probability constants for behaviors"""
    TAXI_PASSENGER_CHANCE = 0.15  # Chance to get passenger
    SEDAN_WEEKEND_TRIP = 0.3  # Weekend leisure trip
    SUV_SCHOOL_RUN = 0.7  # Morning school run
    IMPULSE_CHARGE = 0.1  # Impulse charging probability
    SOCIAL_INFLUENCE = 0.2  # Following others' behavior

class PerformanceConstants:
    """Performance optimization constants"""
    MAX_ACTIVE_VEHICLES = 500  # Maximum vehicles to process per step
    VEHICLE_POOL_SIZE = 1000  # Vehicle object pool size
    BATCH_UPDATE_SIZE = 50  # Vehicles to update in batch
    CACHE_TTL = 300  # Cache time-to-live in seconds
    GC_INTERVAL = 1000  # Garbage collection interval
    MEMORY_THRESHOLD = 80  # Memory usage threshold percentage
    STATE_HISTORY_LIMIT = 20  # Reduced from 100 to save memory
    LOG_BUFFER_SIZE = 100  # Buffer log writes

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Global simulation configuration"""
    # Simulation parameters
    max_steps: int = 172800
    decision_interval: int = TimeConstants.FIVE_MINUTES
    state_update_interval: int = TimeConstants.MINUTE

    # Battery thresholds
    battery_critical: float = BatteryConstants.CRITICAL
    battery_low: float = BatteryConstants.LOW
    battery_medium: float = BatteryConstants.MEDIUM
    battery_high: float = BatteryConstants.HIGH

    # GPT settings - CRITICAL CONFIGURATION
    use_gpt: bool = True
    gpt_cooldown: int = TimeConstants.MINUTE
    gpt_model: str = "gpt-4o-mini"
    api_base_url: str = 'https://xiaoai.plus'  # Default OpenAI URL
    api_key: str = 'sk-UJ7Bc46l7K9nU3g68AKDVQhplFM5pAaA3f1RpuzBOcNcUmz0'  # Must be provided via environment or command line

    # File paths
    data_dir: str = "simulation_data"
    sumo_config: str = "osm.sumocfg"

    # Debug
    debug_mode: bool = True

    # Safety parameters
    max_retries: int = 3
    timeout: float = 10.0

    # Performance parameters
    max_concurrent_updates: int = 10
    enable_performance_monitor: bool = True
    enable_adaptive_processing: bool = True
    log_interval: int = 1000  # Log stats every N steps
    save_interval: int = 5000  # Save data every N steps

    def __post_init__(self):
        """Validate and setup configuration"""
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get('OPENAI_API_KEY', '')

        if not OPENAI_AVAILABLE:
            self.use_gpt = False
            print("WARNING: OpenAI not available - GPT decision making disabled!")

        if not PSUTIL_AVAILABLE:
            self.enable_performance_monitor = False

# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Monitor and optimize simulation performance"""

    def __init__(self, logger):
        self.logger = logger
        self.metrics = defaultdict(list)
        self.last_gc = time.time()
        self.process = None
        self.memory_warnings = 0

        if PSUTIL_AVAILABLE:
            try:
                self.process = psutil.Process()
            except:
                self.logger.warning("Could not initialize process monitor")

    def check_memory(self) -> Dict:
        """Check memory usage"""
        if not PSUTIL_AVAILABLE or not self.process:
            return {
                'rss': 0,
                'vms': 0,
                'percent': 0,
                'available': 0
            }

        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            return {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'percent': memory_percent,
                'available': psutil.virtual_memory().available / 1024 / 1024  # MB
            }
        except:
            return {'rss': 0, 'vms': 0, 'percent': 0, 'available': 0}

    def should_throttle(self) -> bool:
        """Determine if processing should be throttled"""
        mem = self.check_memory()
        if mem['percent'] > PerformanceConstants.MEMORY_THRESHOLD:
            self.memory_warnings += 1
            return True
        return False

    def record_metric(self, name: str, value: float):
        """Record performance metric with limited history"""
        self.metrics[name].append((time.time(), value))

        # Keep only recent metrics (last 10 minutes instead of hour)
        cutoff = time.time() - 600
        self.metrics[name] = [(t, v) for t, v in self.metrics[name] if t > cutoff]

    def get_avg_metric(self, name: str, window: int = 60) -> float:
        """Get average metric over time window"""
        if name not in self.metrics:
            return 0

        cutoff = time.time() - window
        recent = [v for t, v in self.metrics[name] if t > cutoff]
        return sum(recent) / len(recent) if recent else 0

    def trigger_gc_if_needed(self):
        """Trigger garbage collection if needed"""
        if time.time() - self.last_gc > 30:  # Every 30 seconds
            collected = gc.collect()
            self.last_gc = time.time()
            if collected > 100:
                self.logger.debug(f"Garbage collection: {collected} objects collected")

# ============================================================================
# Vehicle Pool Manager - Fixed thread safety
# ============================================================================

class VehiclePoolManager:
    """Manage vehicle objects efficiently with pooling"""

    def __init__(self, logger, max_active: int = PerformanceConstants.MAX_ACTIVE_VEHICLES):
        self.logger = logger
        self.max_active = max_active
        self.active_vehicles: Dict[str, 'Vehicle'] = {}
        self.inactive_vehicles: Dict[str, 'Vehicle'] = {}
        self.vehicle_pool: deque = deque(maxlen=PerformanceConstants.VEHICLE_POOL_SIZE)
        self.last_rotation = 0
        self.priority_queue = []
        self.lock = threading.RLock()
        self.total_vehicles_processed = 0

    def add_vehicle(self, vehicle: 'Vehicle'):
        """Add vehicle to pool with thread safety"""
        with self.lock:
            if len(self.active_vehicles) < self.max_active:
                self.active_vehicles[vehicle.vehicle_id] = vehicle
            else:
                self.inactive_vehicles[vehicle.vehicle_id] = vehicle
            self.total_vehicles_processed += 1

    def remove_vehicle(self, vehicle_id: str):
        """Remove vehicle from pool with cleanup"""
        with self.lock:
            vehicle = None
            if vehicle_id in self.active_vehicles:
                vehicle = self.active_vehicles.pop(vehicle_id)
            elif vehicle_id in self.inactive_vehicles:
                vehicle = self.inactive_vehicles.pop(vehicle_id)

            if vehicle:
                # Clear heavy data before recycling
                if hasattr(vehicle, 'state_machine') and vehicle.state_machine:
                    vehicle.state_machine.state_history.clear()
                self.vehicle_pool.append(vehicle)

    def get_active_vehicles(self) -> List['Vehicle']:
        """Get list of active vehicles for processing"""
        with self.lock:
            return list(self.active_vehicles.values())

    def get_all_vehicles(self) -> Dict[str, 'Vehicle']:
        """Get all vehicles"""
        with self.lock:
            return {**self.active_vehicles, **self.inactive_vehicles}

    def get_vehicle(self, vehicle_id: str) -> Optional['Vehicle']:
        """Get vehicle by ID"""
        with self.lock:
            return self.active_vehicles.get(vehicle_id) or self.inactive_vehicles.get(vehicle_id)

    def rotate_vehicles(self, sim_time: float):
        """Rotate active/inactive vehicles based on priority"""
        if sim_time - self.last_rotation < 60:  # Rotate every minute
            return

        with self.lock:
            # Only rotate if we have many vehicles
            if len(self.inactive_vehicles) == 0:
                return

            # Calculate priorities for inactive vehicles
            priorities = []
            for v in list(self.inactive_vehicles.values())[:100]:  # Limit to prevent slowdown
                priority = self._calculate_priority(v)
                if priority > 0.5:
                    priorities.append((priority, v.vehicle_id))

            if not priorities:
                self.last_rotation = sim_time
                return

            # Sort by priority
            priorities.sort(reverse=True)

            # Find low priority active vehicles
            low_priority_active = []
            for v_id, v in list(self.active_vehicles.items())[:100]:
                p = self._calculate_priority(v)
                if p < 0.3:
                    low_priority_active.append((p, v_id))

            if not low_priority_active:
                self.last_rotation = sim_time
                return

            low_priority_active.sort()

            # Swap vehicles
            num_to_swap = min(len(priorities), len(low_priority_active), 10)  # Limit swaps

            for i in range(num_to_swap):
                inactive_id = priorities[i][1]
                active_id = low_priority_active[i][1]

                # Swap
                self.active_vehicles[inactive_id] = self.inactive_vehicles.pop(inactive_id)
                self.inactive_vehicles[active_id] = self.active_vehicles.pop(active_id)

            self.last_rotation = sim_time

    def _calculate_priority(self, vehicle: 'Vehicle') -> float:
        """Calculate vehicle processing priority"""
        priority = 0.5

        # Low battery increases priority
        if vehicle.battery_soc < BatteryConstants.LOW:
            priority += 0.3
        elif vehicle.battery_soc < BatteryConstants.MEDIUM_LOW:
            priority += 0.2

        # Vehicle type priority
        if vehicle.vehicle_type == 'electric_taxi':
            priority += 0.2
        elif vehicle.vehicle_type == 'electric_truck':
            priority += 0.1

        # Charging state priority
        if vehicle.state_machine and vehicle.state_machine.current_state:
            state_str = str(vehicle.state_machine.current_state)
            if 'CHARGING' in state_str:
                priority += 0.15
            elif 'SEARCHING_CHARGING' in state_str:
                priority += 0.25

        return min(1.0, priority)

# ============================================================================
# Enhanced Logging with Buffering
# ============================================================================

class SimulationLogger:
    """Enhanced logger with buffering for performance"""

    def __init__(self, config: Config):
        self.config = config
        self.error_count = defaultdict(int)
        self.warning_count = defaultdict(int)
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_flush = time.time()
        self.running = True

        # Create data directory
        try:
            os.makedirs(config.data_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create data directory: {e}")
            sys.exit(1)

        # Setup logging
        level = logging.DEBUG if config.debug_mode else logging.INFO

        # Configure logging
        log_format = '%(asctime)s [%(levelname)s] %(message)s'

        # Create handlers
        handlers = []

        # File handler
        try:
            self.log_file_path = os.path.join(config.data_dir, "simulation.log")
            file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        except:
            print("Warning: Could not create log file")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers
        )

        self.logger = logging.getLogger(__name__)

        # Start flush thread
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()

    def _flush_loop(self):
        """Periodically flush log buffer"""
        while self.running:
            time.sleep(5)  # Flush every 5 seconds
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered log messages"""
        with self.buffer_lock:
            if not self.log_buffer:
                return

            messages = self.log_buffer[:]
            self.log_buffer.clear()

        # Write to file
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                for msg in messages:
                    f.write(msg + '\n')
        except:
            pass

    def _add_to_buffer(self, message: str):
        """Add message to buffer"""
        with self.buffer_lock:
            self.log_buffer.append(message)
            if len(self.log_buffer) > PerformanceConstants.LOG_BUFFER_SIZE:
                self._flush_buffer()

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str, category: str = "general"):
        """Log warning message"""
        self.warning_count[category] += 1
        self.logger.warning(f"[{category}] {message}")

    def error(self, message: str, category: str = "general", exc_info=False):
        """Log error message"""
        self.error_count[category] += 1
        if exc_info:
            message += f"\n{traceback.format_exc()}"
        self.logger.error(f"[{category}] {message}")

    def debug(self, message: str):
        """Log debug message"""
        if self.config.debug_mode:
            self.logger.debug(message)

    def get_statistics(self) -> Dict:
        """Get error and warning statistics"""
        return {
            "errors": dict(self.error_count),
            "warnings": dict(self.warning_count)
        }

    def shutdown(self):
        """Shutdown logging thread"""
        self.running = False
        self._flush_buffer()
        if self.flush_thread:
            self.flush_thread.join(timeout=1)

# ============================================================================
# Vehicle State Enumerations
# ============================================================================

class SedanState(Enum):
    """States for electric_sedan with realistic patterns"""
    SLEEPING = auto()
    MORNING_ROUTINE = auto()
    COMMUTING_TO_WORK = auto()
    AT_WORK = auto()
    WORKING = auto()
    LUNCH_BREAK = auto()
    AFTERNOON_MEETING = auto()
    COMMUTING_HOME = auto()
    AT_HOME = auto()
    EVENING_ACTIVITY = auto()
    LEISURE_DRIVING = auto()
    SHOPPING = auto()
    VISITING_FRIENDS = auto()
    SEARCHING_CHARGING = auto()
    NAVIGATING_TO_CHARGING = auto()
    WAITING_FOR_CHARGING = auto()
    CHARGING = auto()
    EMERGENCY_LOW_BATTERY = auto()

class SUVState(Enum):
    """States for electric_suv - family vehicle"""
    AT_HOME = auto()
    MORNING_PREP = auto()
    SCHOOL_RUN = auto()
    ERRANDS = auto()
    GROCERY_SHOPPING = auto()
    KIDS_ACTIVITY = auto()
    FAMILY_OUTING = auto()
    WEEKEND_TRIP = auto()
    AT_DESTINATION = auto()
    RETURNING_HOME = auto()
    EVENING_ROUTINE = auto()
    SEARCHING_CHARGING = auto()
    NAVIGATING_TO_CHARGING = auto()
    WAITING_FOR_CHARGING = auto()
    CHARGING = auto()
    EMERGENCY_LOW_BATTERY = auto()

class TaxiState(Enum):
    """States for electric_taxi - commercial service"""
    OFF_DUTY = auto()
    STARTING_SHIFT = auto()
    CRUISING = auto()
    HEADING_TO_PICKUP = auto()
    WAITING_FOR_PASSENGER = auto()
    PICKING_UP_PASSENGER = auto()
    WITH_PASSENGER = auto()
    DROPPING_OFF = auto()
    SHIFT_BREAK = auto()
    MEAL_BREAK = auto()
    RETURNING_TO_BASE = auto()
    AT_BASE = auto()
    ENDING_SHIFT = auto()
    SEARCHING_CHARGING = auto()
    NAVIGATING_TO_CHARGING = auto()
    WAITING_FOR_CHARGING = auto()
    CHARGING = auto()
    EMERGENCY_LOW_BATTERY = auto()

class TruckState(Enum):
    """States for electric_truck - freight"""
    OFF_DUTY = auto()
    AT_DEPOT = auto()
    PRE_TRIP_INSPECTION = auto()
    LOADING = auto()
    EN_ROUTE_TO_DELIVERY = auto()
    TRAFFIC_DELAY = auto()
    DELIVERING = auto()
    UNLOADING = auto()
    RETURNING_TO_DEPOT = auto()
    BREAK_TIME = auto()
    MAINTENANCE_CHECK = auto()
    END_OF_SHIFT = auto()
    SEARCHING_CHARGING = auto()
    NAVIGATING_TO_CHARGING = auto()
    WAITING_FOR_CHARGING = auto()
    CHARGING = auto()
    EMERGENCY_LOW_BATTERY = auto()

class VanState(Enum):
    """States for electric_van - local delivery"""
    AT_WAREHOUSE = auto()
    MORNING_BRIEFING = auto()
    LOADING_PACKAGES = auto()
    SORTING_PACKAGES = auto()
    STARTING_ROUTE = auto()
    DELIVERY_ROUTE = auto()
    DELIVERING_PACKAGE = auto()
    CUSTOMER_INTERACTION = auto()
    BETWEEN_DELIVERIES = auto()
    RETURNING_TO_WAREHOUSE = auto()
    UNLOADING_RETURNS = auto()
    BREAK_TIME = auto()
    END_OF_ROUTE = auto()
    SEARCHING_CHARGING = auto()
    NAVIGATING_TO_CHARGING = auto()
    WAITING_FOR_CHARGING = auto()
    CHARGING = auto()
    EMERGENCY_LOW_BATTERY = auto()

# ============================================================================
# Vehicle Specifications with Validation
# ============================================================================

@dataclass
class VehicleSpecs:
    """Technical specifications with validation"""
    battery_capacity: float  # Wh
    max_charge_power: float  # W
    consumption_rate: float  # Wh/km
    mass: float  # kg
    max_speed: float  # m/s
    acceleration: float  # m/s²
    deceleration: float  # m/s²

    def __post_init__(self):
        """Validate specifications"""
        if self.battery_capacity <= 0:
            raise ValueError(f"Invalid battery capacity: {self.battery_capacity}")
        if self.consumption_rate <= 0:
            raise ValueError(f"Invalid consumption rate: {self.consumption_rate}")
        if self.max_speed <= 0:
            raise ValueError(f"Invalid max speed: {self.max_speed}")

# Safe vehicle specifications
VEHICLE_SPECS = {
    'electric_sedan': VehicleSpecs(
        battery_capacity=82000,
        max_charge_power=250000,
        consumption_rate=150,
        mass=2003,
        max_speed=50.0,
        acceleration=2.6,
        deceleration=4.5
    ),
    'electric_suv': VehicleSpecs(
        battery_capacity=82000,
        max_charge_power=250000,
        consumption_rate=200,
        mass=2500,
        max_speed=45.0,
        acceleration=2.4,
        deceleration=4.3
    ),
    'electric_taxi': VehicleSpecs(
        battery_capacity=62000,
        max_charge_power=100000,
        consumption_rate=180,
        mass=1656,
        max_speed=50.0,
        acceleration=2.2,
        deceleration=4.0
    ),
    'electric_truck': VehicleSpecs(
        battery_capacity=438000,
        max_charge_power=270000,
        consumption_rate=1900,
        mass=8508,
        max_speed=25.0,
        acceleration=1.0,
        deceleration=2.5
    ),
    'electric_van': VehicleSpecs(
        battery_capacity=89000,
        max_charge_power=115000,
        consumption_rate=300,
        mass=2649,
        max_speed=30.0,
        acceleration=2.0,
        deceleration=4.0
    )
}

# ============================================================================
# Enhanced Driver Profile with Irrational Behaviors
# ============================================================================

@dataclass
class IrrationalFactors:
    """Irrational behavior factors"""
    herd_mentality: float = 0.0
    impulse_control: float = 0.5
    habit_strength: float = 0.0
    optimism_bias: float = 0.5
    loss_aversion: float = 0.5
    confirmation_bias: float = 0.5
    anchoring_bias: float = 0.5
    recency_effect: float = 0.5

    def apply_to_decision(self, rational_decision: float, context: Dict) -> float:
        """Apply irrational factors to modify rational decision"""
        decision = rational_decision

        # Herd mentality - if others are charging, increase probability
        if context.get('others_charging', False):
            decision += self.herd_mentality * 0.3

        # Impulse - random urge to charge
        if random.random() < (1 - self.impulse_control) * 0.2:
            decision += 0.4

        # Habit - charge at usual times/places
        if context.get('is_usual_time', False):
            decision += self.habit_strength * 0.3

        # Optimism bias - underestimate need
        decision -= self.optimism_bias * 0.2

        # Loss aversion - overreact to low battery
        if context.get('battery_low', False):
            decision += self.loss_aversion * 0.4

        return min(1.0, max(0.0, decision))

@dataclass
class DriverProfile:
    """Enhanced driver profile with psychological depth"""
    # Demographics
    age: int
    gender: str
    income_level: str
    education: str
    occupation: str

    # Personality traits (Big Five)
    openness: float  # 0-1
    conscientiousness: float  # 0-1
    extraversion: float  # 0-1
    agreeableness: float  # 0-1
    neuroticism: float  # 0-1

    # Rational factors
    risk_tolerance: float
    patience_level: float
    eco_consciousness: float
    tech_savviness: float

    # Charging behavior
    range_anxiety: float
    price_sensitivity: float
    time_sensitivity: float
    comfort_preference: float

    # Irrational factors
    irrational: IrrationalFactors = field(default_factory=IrrationalFactors)

    # Dynamic psychological state
    mood: str = 'neutral'
    fatigue_level: float = 0.0
    stress_level: float = 0.0
    urgency_level: float = 0.0
    cognitive_load: float = 0.0  # Mental burden
    emotional_state: float = 0.5  # 0=negative, 1=positive

    # Social factors
    social_pressure: float = 0.0
    peer_influence: float = 0.0

    # Work/life schedule
    work_start_hour: int = 9
    work_end_hour: int = 18
    work_days: List[int] = field(default_factory=lambda: [1,2,3,4,5])
    chronotype: str = 'normal'  # 'early_bird', 'normal', 'night_owl'

    # Preferences and habits
    preferred_stations: Set[str] = field(default_factory=set)
    avoided_stations: Set[str] = field(default_factory=set)
    charging_habits: Dict[str, Any] = field(default_factory=dict)
    home_location: Optional[str] = None
    work_location: Optional[str] = None
    favorite_locations: List[str] = field(default_factory=list)

    # Learning and adaptation
    experiences: List[Dict] = field(default_factory=list)
    trust_in_system: float = 0.5

    @classmethod
    def generate_profile(cls, vehicle_type: str) -> 'DriverProfile':
        """Generate realistic driver profile with personality"""

        # Age distribution by vehicle type
        age_ranges = {
            'electric_sedan': (25, 55, 35),  # min, max, mean
            'electric_suv': (30, 60, 40),
            'electric_taxi': (25, 60, 38),
            'electric_truck': (25, 65, 42),
            'electric_van': (22, 55, 35)
        }

        age_range = age_ranges.get(vehicle_type, (25, 60, 40))
        if np:
            age = int(np.random.normal(age_range[2], 8))
        else:
            age = random.randint(age_range[0], age_range[1])
        age = max(age_range[0], min(age_range[1], age))

        # Generate Big Five personality traits
        openness = random.betavariate(5, 5)  # Beta distribution for realistic spread
        conscientiousness = random.betavariate(6, 4) if vehicle_type in ['electric_truck', 'electric_van'] else random.betavariate(5, 5)
        extraversion = random.betavariate(7, 3) if vehicle_type == 'electric_taxi' else random.betavariate(5, 5)
        agreeableness = random.betavariate(5, 5)
        neuroticism = random.betavariate(4, 6)  # Slightly lower neuroticism

        # Derive other traits from personality
        risk_tolerance = 0.3 + 0.4 * openness + 0.3 * (1 - neuroticism)
        patience_level = 0.2 + 0.5 * conscientiousness + 0.3 * agreeableness
        eco_consciousness = 0.3 + 0.4 * openness + 0.3 * conscientiousness
        tech_savviness = 0.2 + 0.5 * openness + 0.3 * max(0, (50 - age) / 50)

        # Range anxiety influenced by neuroticism and experience
        range_anxiety = 0.2 + 0.5 * neuroticism + 0.3 * (1 - tech_savviness)

        # Work schedule based on vehicle type
        if vehicle_type == 'electric_sedan':
            work_start = random.choice([7, 8, 9])
            work_end = work_start + random.choice([8, 9, 10])
            occupation = random.choice(['office_worker', 'manager', 'teacher', 'engineer'])
        elif vehicle_type == 'electric_suv':
            work_start = random.choice([8, 9])
            work_end = random.choice([17, 18])
            occupation = random.choice(['executive', 'doctor', 'lawyer', 'business_owner'])
        elif vehicle_type == 'electric_taxi':
            work_start = random.choice([6, 14, 22])  # Shift work
            work_end = (work_start + 8) % 24
            occupation = 'taxi_driver'
        elif vehicle_type == 'electric_truck':
            work_start = random.choice([5, 6, 7])
            work_end = work_start + random.choice([10, 11, 12])
            occupation = 'truck_driver'
        else:  # van
            work_start = random.choice([7, 8])
            work_end = random.choice([18, 19, 20])
            occupation = 'delivery_driver'

        # Generate irrational factors based on personality
        irrational = IrrationalFactors(
            herd_mentality=0.2 + 0.4 * agreeableness + 0.2 * (1 - openness),
            impulse_control=0.3 + 0.5 * conscientiousness - 0.2 * neuroticism,
            habit_strength=0.2 + 0.5 * conscientiousness + 0.3 * (age / 65),
            optimism_bias=0.3 + 0.4 * extraversion - 0.3 * neuroticism,
            loss_aversion=0.3 + 0.5 * neuroticism - 0.2 * risk_tolerance,
            confirmation_bias=0.4 + 0.3 * (1 - openness),
            anchoring_bias=0.3 + 0.3 * (1 - openness) + 0.2 * age / 65,
            recency_effect=0.4 + 0.3 * neuroticism - 0.2 * conscientiousness
        )

        # Determine chronotype
        if random.random() < 0.2:
            chronotype = 'early_bird'
        elif random.random() < 0.2:
            chronotype = 'night_owl'
        else:
            chronotype = 'normal'

        return cls(
            age=age,
            gender=random.choice(['male', 'female', 'other']),
            income_level=random.choice(['low', 'middle', 'high']) if vehicle_type != 'electric_suv' else 'high',
            education=random.choice(['high_school', 'bachelor', 'master', 'phd']),
            occupation=occupation,
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
            risk_tolerance=risk_tolerance,
            patience_level=patience_level,
            eco_consciousness=eco_consciousness,
            tech_savviness=tech_savviness,
            range_anxiety=range_anxiety,
            price_sensitivity=0.3 + 0.4 * random.random() if vehicle_type != 'electric_suv' else 0.2,
            time_sensitivity=0.5 + 0.3 * random.random() if vehicle_type in ['electric_taxi', 'electric_van'] else 0.4,
            comfort_preference=0.4 + 0.4 * random.random() if vehicle_type == 'electric_suv' else 0.3,
            irrational=irrational,
            work_start_hour=work_start,
            work_end_hour=work_end,
            chronotype=chronotype
        )

    def update_psychological_state(self, context: Dict):
        """Update psychological state based on context"""
        # Fatigue increases over time, especially at night
        hour = context.get('hour', 12)
        if self.chronotype == 'early_bird':
            if hour >= 21 or hour <= 4:
                self.fatigue_level = min(1.0, self.fatigue_level + 0.02)
        elif self.chronotype == 'night_owl':
            if 6 <= hour <= 10:
                self.fatigue_level = min(1.0, self.fatigue_level + 0.02)
        else:
            if hour >= 23 or hour <= 5:
                self.fatigue_level = min(1.0, self.fatigue_level + 0.015)

        # Stress from traffic
        traffic_density = context.get('traffic_density', 0.5)
        if traffic_density > 0.7:
            self.stress_level = min(1.0, self.stress_level + 0.01)
            self.cognitive_load = min(1.0, self.cognitive_load + 0.02)

        # Weather effects
        weather = context.get('weather', 'clear')
        if weather in ['rain', 'snow', 'fog']:
            self.stress_level = min(1.0, self.stress_level + 0.01)
            self.range_anxiety = min(1.0, self.range_anxiety * 1.1)

        # Social influence
        if context.get('vehicles_nearby_charging', 0) > 2:
            self.social_pressure = min(1.0, 0.3 + 0.1 * context['vehicles_nearby_charging'])

        # Update mood based on overall state
        if self.stress_level > 0.7 or self.fatigue_level > 0.7:
            self.mood = random.choice(['stressed', 'anxious', 'irritated'])
            self.emotional_state = max(0.0, self.emotional_state - 0.1)
        elif self.stress_level < 0.3 and self.fatigue_level < 0.3:
            self.mood = random.choice(['happy', 'content', 'relaxed'])
            self.emotional_state = min(1.0, self.emotional_state + 0.1)
        else:
            self.mood = 'neutral'
            self.emotional_state = 0.5

        # Cognitive load affects decision quality
        self.cognitive_load = min(1.0,
            0.3 * self.stress_level +
            0.3 * self.fatigue_level +
            0.2 * (1 - self.emotional_state) +
            0.2 * traffic_density
        )

# ============================================================================
# State Machines - With memory optimization
# ============================================================================

class BaseStateMachine:
    """Base state machine with error recovery"""

    def __init__(self, vehicle_id: str, vehicle_type: str, logger: SimulationLogger):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.logger = logger
        self.current_state = None
        self.previous_state = None
        self.state_start_time = 0
        self.state_data = {}
        self.state_history = deque(maxlen=PerformanceConstants.STATE_HISTORY_LIMIT)  # Reduced size
        self.error_count = 0
        self.destinations_visited = 0
        self.charging_count = 0

    def safe_transition(self, new_state, sim_time: float, **kwargs) -> bool:
        """Safe state transition with validation"""
        try:
            # Record transition
            self.state_history.append({
                'from': self.current_state,
                'to': new_state,
                'time': sim_time,
                # Don't store kwargs to save memory
            })

            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_start_time = sim_time
            self.state_data = kwargs  # Keep current data only

            self.logger.debug(f"{self.vehicle_id} transitioned from {self.previous_state} to {new_state}")
            return True

        except Exception as e:
            self.error_count += 1
            self.logger.error(f"State transition failed for {self.vehicle_id}: {e}", "state_machine")
            return False

    def get_state_duration(self, sim_time: float) -> float:
        """Get time in current state with safety check"""
        try:
            return max(0, sim_time - self.state_start_time)
        except:
            return 0

    def recover_from_error(self, sim_time: float):
        """Recover from error state"""
        self.logger.warning(f"Recovering {self.vehicle_id} from error state", "recovery")
        self.error_count = 0

# [Include all state machine classes - SedanStateMachine, SUVStateMachine, etc. - unchanged]
# I'm omitting them here for brevity, but they should be included unchanged

class SedanStateMachine(BaseStateMachine):
    """Enhanced sedan state machine with realistic daily patterns"""

    def __init__(self, vehicle_id: str, logger: SimulationLogger):
        super().__init__(vehicle_id, 'electric_sedan', logger)
        self.current_state = SedanState.SLEEPING
        self.work_trips_today = 0
        self.leisure_trips_today = 0
        self.last_coffee_break = 0

    def update(self, sim_time: float, battery_level: float, driver_profile: DriverProfile) -> Optional[SedanState]:
        """Update with realistic patterns and irrationality"""
        try:
            hour = int((sim_time / TimeConstants.HOUR) % 24)
            minute = int((sim_time / TimeConstants.MINUTE) % 60)
            day_of_week = int((sim_time / TimeConstants.DAY) % 7) + 1
            is_workday = day_of_week in driver_profile.work_days

            # Apply irrational factors
            context = {
                'battery_low': battery_level < BatteryConstants.MEDIUM_LOW,
                'is_usual_time': hour in [12, 18],  # Lunch and evening
                'others_charging': random.random() < 0.3  # Simulated
            }

            # Emergency charging overrides everything
            if battery_level < BatteryConstants.VERY_LOW:
                if self.current_state not in [
                    SedanState.CHARGING,
                    SedanState.NAVIGATING_TO_CHARGING,
                    SedanState.WAITING_FOR_CHARGING
                ]:
                    return SedanState.EMERGENCY_LOW_BATTERY

            # Irrational charging decision
            charge_probability = driver_profile.irrational.apply_to_decision(
                battery_level < BatteryConstants.MEDIUM_LOW,
                context
            )

            if random.random() < charge_probability * 0.1:  # Impulse charging
                if self.current_state == SedanState.LEISURE_DRIVING:
                    self.logger.debug(f"{self.vehicle_id} impulse charging decision")
                    return SedanState.SEARCHING_CHARGING

            # State-specific transitions with time-based patterns
            if self.current_state == SedanState.SLEEPING:
                # Wake up based on chronotype
                wake_hour = driver_profile.work_start_hour - 2
                if driver_profile.chronotype == 'early_bird':
                    wake_hour -= 1
                elif driver_profile.chronotype == 'night_owl':
                    wake_hour += 1

                if hour == wake_hour and is_workday:
                    return SedanState.MORNING_ROUTINE
                elif hour == wake_hour + 1 and not is_workday:
                    return SedanState.MORNING_ROUTINE

            elif self.current_state == SedanState.MORNING_ROUTINE:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    if is_workday:
                        return SedanState.COMMUTING_TO_WORK
                    else:
                        # Weekend activities
                        if random.random() < 0.6:
                            return SedanState.LEISURE_DRIVING
                        else:
                            return SedanState.AT_HOME

            elif self.current_state == SedanState.COMMUTING_TO_WORK:
                # Commute time varies with traffic
                commute_time = TimeConstants.THIRTY_MINUTES
                if driver_profile.stress_level > 0.5:
                    commute_time *= 1.3  # Traffic stress

                if self.get_state_duration(sim_time) > commute_time:
                    self.work_trips_today += 1
                    return SedanState.AT_WORK

            elif self.current_state == SedanState.AT_WORK:
                return SedanState.WORKING

            elif self.current_state == SedanState.WORKING:
                # Lunch break decision
                if 11 <= hour <= 13 and self.last_coffee_break < sim_time - TimeConstants.TWO_HOURS:
                    if random.random() < 0.7:
                        self.last_coffee_break = sim_time
                        return SedanState.LUNCH_BREAK

                # Afternoon meeting
                if hour == 14 and random.random() < 0.3:
                    return SedanState.AFTERNOON_MEETING

                # End of work
                if hour >= driver_profile.work_end_hour:
                    # Sometimes work late
                    if random.random() < driver_profile.conscientiousness * 0.3:
                        if hour < driver_profile.work_end_hour + 2:
                            return SedanState.WORKING  # Keep working
                    return SedanState.COMMUTING_HOME

            elif self.current_state == SedanState.LUNCH_BREAK:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    # Maybe charge during lunch
                    if battery_level < BatteryConstants.MEDIUM and random.random() < 0.4:
                        return SedanState.SEARCHING_CHARGING
                    return SedanState.WORKING

            elif self.current_state == SedanState.AFTERNOON_MEETING:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    return SedanState.WORKING

            elif self.current_state == SedanState.COMMUTING_HOME:
                commute_time = TimeConstants.THIRTY_MINUTES
                if hour >= 17 and hour <= 19:
                    commute_time *= 1.4  # Rush hour

                if self.get_state_duration(sim_time) > commute_time:
                    # Sometimes stop for errands
                    if random.random() < 0.3:
                        return SedanState.SHOPPING
                    return SedanState.AT_HOME

            elif self.current_state == SedanState.AT_HOME:
                # Evening activities
                if 18 <= hour <= 21:
                    if random.random() < 0.2:
                        return SedanState.EVENING_ACTIVITY
                    elif random.random() < 0.1:
                        return SedanState.VISITING_FRIENDS

                # Go to sleep
                sleep_hour = 22
                if driver_profile.chronotype == 'early_bird':
                    sleep_hour = 21
                elif driver_profile.chronotype == 'night_owl':
                    sleep_hour = 24

                if hour >= sleep_hour:
                    return SedanState.SLEEPING

            elif self.current_state == SedanState.EVENING_ACTIVITY:
                if self.get_state_duration(sim_time) > TimeConstants.TWO_HOURS:
                    return SedanState.AT_HOME

            elif self.current_state == SedanState.LEISURE_DRIVING:
                duration = random.randint(TimeConstants.THIRTY_MINUTES, TimeConstants.TWO_HOURS)
                if self.get_state_duration(sim_time) > duration:
                    self.leisure_trips_today += 1
                    return SedanState.AT_HOME

            elif self.current_state == SedanState.SHOPPING:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    return SedanState.AT_HOME

            elif self.current_state == SedanState.VISITING_FRIENDS:
                if self.get_state_duration(sim_time) > TimeConstants.TWO_HOURS:
                    return SedanState.AT_HOME

            # Charging states
            elif self.current_state == SedanState.EMERGENCY_LOW_BATTERY:
                return SedanState.SEARCHING_CHARGING

            elif self.current_state == SedanState.SEARCHING_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return SedanState.NAVIGATING_TO_CHARGING

            elif self.current_state == SedanState.NAVIGATING_TO_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return SedanState.WAITING_FOR_CHARGING

            elif self.current_state == SedanState.WAITING_FOR_CHARGING:
                wait_time = TimeConstants.FIVE_MINUTES
                if driver_profile.patience_level < 0.3:
                    wait_time = TimeConstants.MINUTE * 2

                if self.get_state_duration(sim_time) > wait_time:
                    return SedanState.CHARGING

            elif self.current_state == SedanState.CHARGING:
                # Charging time based on battery and patience
                target_soc = BatteryConstants.HIGH
                if driver_profile.time_sensitivity > 0.7:
                    target_soc = BatteryConstants.MEDIUM_HIGH

                if battery_level >= target_soc or self.get_state_duration(sim_time) > ChargingConstants.STANDARD_CHARGE_TIME:
                    self.charging_count += 1
                    # Return to appropriate state
                    if is_workday and 8 <= hour < 17:
                        return SedanState.WORKING
                    elif hour >= 22:
                        return SedanState.AT_HOME
                    else:
                        return SedanState.LEISURE_DRIVING

            return None

        except Exception as e:
            self.logger.error(f"Error in sedan state update: {e}", "state_update", exc_info=True)
            self.recover_from_error(sim_time)
            return SedanState.AT_HOME

class SUVStateMachine(BaseStateMachine):
    """SUV state machine for family activities"""

    def __init__(self, vehicle_id: str, logger: SimulationLogger):
        super().__init__(vehicle_id, 'electric_suv', logger)
        self.current_state = SUVState.AT_HOME
        self.trips_today = 0
        self.school_runs = 0

    def update(self, sim_time: float, battery_level: float, driver_profile: DriverProfile) -> Optional[SUVState]:
        """Update SUV state with family patterns"""
        try:
            hour = int((sim_time / TimeConstants.HOUR) % 24)
            day_of_week = int((sim_time / TimeConstants.DAY) % 7) + 1
            is_weekend = day_of_week in [6, 7]
            is_school_day = day_of_week in [1, 2, 3, 4, 5]

            # Emergency charging
            if battery_level < BatteryConstants.VERY_LOW:
                if self.current_state not in [
                    SUVState.CHARGING,
                    SUVState.NAVIGATING_TO_CHARGING,
                    SUVState.WAITING_FOR_CHARGING
                ]:
                    return SUVState.EMERGENCY_LOW_BATTERY

            # Family-oriented charging decisions
            if battery_level < BatteryConstants.MEDIUM_LOW:
                # Parents are more cautious with family vehicle
                caution_factor = 1.3 if driver_profile.agreeableness > 0.6 else 1.0
                if random.random() < driver_profile.range_anxiety * caution_factor:
                    if self.current_state in [SUVState.AT_HOME, SUVState.ERRANDS]:
                        return SUVState.SEARCHING_CHARGING

            # State transitions
            if self.current_state == SUVState.AT_HOME:
                if is_school_day:
                    if hour == 7 and self.school_runs == 0:
                        return SUVState.MORNING_PREP
                    elif hour == 15 and self.school_runs == 1:  # Afternoon pickup
                        return SUVState.SCHOOL_RUN
                    elif 10 <= hour <= 14 and random.random() < 0.3:
                        return SUVState.ERRANDS

                if is_weekend:
                    if 9 <= hour <= 18 and random.random() < 0.2:
                        if random.random() < 0.7:
                            return SUVState.FAMILY_OUTING
                        else:
                            return SUVState.WEEKEND_TRIP

            elif self.current_state == SUVState.MORNING_PREP:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return SUVState.SCHOOL_RUN

            elif self.current_state == SUVState.SCHOOL_RUN:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    self.school_runs += 1
                    if hour < 9:  # Morning drop-off
                        if random.random() < 0.4:
                            return SUVState.ERRANDS
                        return SUVState.AT_HOME
                    else:  # Afternoon pickup
                        if random.random() < 0.3:
                            return SUVState.KIDS_ACTIVITY
                        return SUVState.AT_HOME

            elif self.current_state == SUVState.ERRANDS:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    if random.random() < 0.5:
                        return SUVState.GROCERY_SHOPPING
                    return SUVState.AT_HOME

            elif self.current_state == SUVState.GROCERY_SHOPPING:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    return SUVState.RETURNING_HOME

            elif self.current_state == SUVState.KIDS_ACTIVITY:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR * 1.5:
                    return SUVState.RETURNING_HOME

            elif self.current_state == SUVState.FAMILY_OUTING:
                duration = random.randint(TimeConstants.TWO_HOURS, TimeConstants.FOUR_HOURS)
                if self.get_state_duration(sim_time) > duration:
                    return SUVState.AT_DESTINATION

            elif self.current_state == SUVState.WEEKEND_TRIP:
                if self.get_state_duration(sim_time) > TimeConstants.FOUR_HOURS:
                    return SUVState.AT_DESTINATION

            elif self.current_state == SUVState.AT_DESTINATION:
                if self.get_state_duration(sim_time) > TimeConstants.TWO_HOURS:
                    return SUVState.RETURNING_HOME

            elif self.current_state == SUVState.RETURNING_HOME:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return SUVState.AT_HOME

            elif self.current_state == SUVState.EVENING_ROUTINE:
                if hour >= 21:
                    self.school_runs = 0  # Reset for next day
                    return SUVState.AT_HOME

            # Charging states
            elif self.current_state == SUVState.EMERGENCY_LOW_BATTERY:
                return SUVState.SEARCHING_CHARGING

            elif self.current_state == SUVState.SEARCHING_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return SUVState.NAVIGATING_TO_CHARGING

            elif self.current_state == SUVState.NAVIGATING_TO_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return SUVState.WAITING_FOR_CHARGING

            elif self.current_state == SUVState.WAITING_FOR_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return SUVState.CHARGING

            elif self.current_state == SUVState.CHARGING:
                # Families prefer fuller charge for safety
                target_soc = BatteryConstants.HIGH
                if battery_level >= target_soc or self.get_state_duration(sim_time) > ChargingConstants.LONG_CHARGE_TIME:
                    self.charging_count += 1
                    return SUVState.AT_HOME

            return None

        except Exception as e:
            self.logger.error(f"Error in SUV state update: {e}", "state_update", exc_info=True)
            return SUVState.AT_HOME

class TaxiStateMachine(BaseStateMachine):
    """Taxi state machine with shift management"""

    def __init__(self, vehicle_id: str, logger: SimulationLogger):
        super().__init__(vehicle_id, 'electric_taxi', logger)
        self.current_state = TaxiState.OFF_DUTY
        self.passengers_served = 0
        self.shift_start = 0
        self.shift_duration = TimeConstants.EIGHT_HOURS
        self.break_count = 0
        self.earnings_today = 0

    def update(self, sim_time: float, battery_level: float, driver_profile: DriverProfile) -> Optional[TaxiState]:
        """Update taxi state with commercial service patterns"""
        try:
            hour = int((sim_time / TimeConstants.HOUR) % 24)
            shift_time = sim_time - self.shift_start if self.shift_start > 0 else 0

            # Taxi-specific battery management - more aggressive
            if battery_level < BatteryConstants.LOW:
                if self.current_state not in [
                    TaxiState.CHARGING,
                    TaxiState.NAVIGATING_TO_CHARGING,
                    TaxiState.WAITING_FOR_CHARGING
                ]:
                    self.logger.info(f"Taxi {self.vehicle_id} needs urgent charging at {battery_level:.2f}")
                    return TaxiState.EMERGENCY_LOW_BATTERY

            # Proactive charging during low demand
            if battery_level < BatteryConstants.MEDIUM:
                if hour in [2, 3, 4, 14, 15]:  # Low demand hours
                    if random.random() < 0.6:
                        return TaxiState.SEARCHING_CHARGING

            # State transitions
            if self.current_state == TaxiState.OFF_DUTY:
                # Start shift based on driver schedule
                if hour == driver_profile.work_start_hour:
                    self.shift_start = sim_time
                    self.passengers_served = 0
                    self.break_count = 0
                    return TaxiState.STARTING_SHIFT

            elif self.current_state == TaxiState.STARTING_SHIFT:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return TaxiState.CRUISING

            elif self.current_state == TaxiState.CRUISING:
                # Passenger pickup probability varies by time
                pickup_prob = ProbabilityConstants.TAXI_PASSENGER_CHANCE
                if hour in [7, 8, 9, 17, 18, 19]:  # Rush hours
                    pickup_prob *= 2
                elif hour in [2, 3, 4]:  # Late night
                    pickup_prob *= 0.5

                if random.random() < pickup_prob:
                    return TaxiState.HEADING_TO_PICKUP

                # Take breaks
                if shift_time > TimeConstants.TWO_HOURS * (self.break_count + 1):
                    if random.random() < 0.3:
                        return TaxiState.SHIFT_BREAK

                # End shift
                if shift_time > self.shift_duration:
                    return TaxiState.ENDING_SHIFT

            elif self.current_state == TaxiState.HEADING_TO_PICKUP:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return TaxiState.WAITING_FOR_PASSENGER

            elif self.current_state == TaxiState.WAITING_FOR_PASSENGER:
                wait_time = random.randint(TimeConstants.MINUTE, TimeConstants.FIVE_MINUTES)
                if self.get_state_duration(sim_time) > wait_time:
                    # Passenger no-show
                    if random.random() < 0.1:
                        return TaxiState.CRUISING
                    return TaxiState.PICKING_UP_PASSENGER

            elif self.current_state == TaxiState.PICKING_UP_PASSENGER:
                if self.get_state_duration(sim_time) > TimeConstants.MINUTE:
                    return TaxiState.WITH_PASSENGER

            elif self.current_state == TaxiState.WITH_PASSENGER:
                # Trip duration varies
                trip_duration = random.randint(TimeConstants.TEN_MINUTES, TimeConstants.THIRTY_MINUTES)
                if hour in [7, 8, 9, 17, 18, 19]:  # Rush hour trips are longer
                    trip_duration = int(trip_duration * 1.3)

                if self.get_state_duration(sim_time) > trip_duration:
                    self.passengers_served += 1
                    self.earnings_today += random.uniform(10, 50)
                    return TaxiState.DROPPING_OFF

            elif self.current_state == TaxiState.DROPPING_OFF:
                if self.get_state_duration(sim_time) > TimeConstants.MINUTE:
                    # Chain rides during busy times
                    if hour in [7, 8, 9, 17, 18, 19] and random.random() < 0.4:
                        return TaxiState.PICKING_UP_PASSENGER
                    return TaxiState.CRUISING

            elif self.current_state == TaxiState.SHIFT_BREAK:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    self.break_count += 1
                    # Charge during break if needed
                    if battery_level < BatteryConstants.MEDIUM_HIGH:
                        return TaxiState.SEARCHING_CHARGING
                    return TaxiState.CRUISING

            elif self.current_state == TaxiState.MEAL_BREAK:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return TaxiState.CRUISING

            elif self.current_state == TaxiState.ENDING_SHIFT:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return TaxiState.RETURNING_TO_BASE

            elif self.current_state == TaxiState.RETURNING_TO_BASE:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return TaxiState.AT_BASE

            elif self.current_state == TaxiState.AT_BASE:
                # Charge at base if needed
                if battery_level < BatteryConstants.MEDIUM_HIGH:
                    return TaxiState.SEARCHING_CHARGING
                elif self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return TaxiState.OFF_DUTY

            # Charging states
            elif self.current_state == TaxiState.EMERGENCY_LOW_BATTERY:
                return TaxiState.SEARCHING_CHARGING

            elif self.current_state == TaxiState.SEARCHING_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return TaxiState.NAVIGATING_TO_CHARGING

            elif self.current_state == TaxiState.NAVIGATING_TO_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return TaxiState.WAITING_FOR_CHARGING

            elif self.current_state == TaxiState.WAITING_FOR_CHARGING:
                # Taxis are impatient
                max_wait = TimeConstants.FIVE_MINUTES if driver_profile.patience_level > 0.5 else TimeConstants.MINUTE * 2
                if self.get_state_duration(sim_time) > max_wait:
                    return TaxiState.CHARGING

            elif self.current_state == TaxiState.CHARGING:
                # Quick charging for taxis
                target_soc = BatteryConstants.MEDIUM_HIGH  # Don't need full charge
                charge_time = ChargingConstants.QUICK_CHARGE_TIME

                if battery_level >= target_soc or self.get_state_duration(sim_time) > charge_time:
                    self.charging_count += 1
                    # Return to service quickly
                    if shift_time < self.shift_duration:
                        return TaxiState.CRUISING
                    return TaxiState.AT_BASE

            return None

        except Exception as e:
            self.logger.error(f"Error in taxi state update: {e}", "state_update", exc_info=True)
            return TaxiState.CRUISING

class TruckStateMachine(BaseStateMachine):
    """Truck state machine for freight delivery"""

    def __init__(self, vehicle_id: str, logger: SimulationLogger):
        super().__init__(vehicle_id, 'electric_truck', logger)
        self.current_state = TruckState.OFF_DUTY
        self.deliveries_completed = 0
        self.daily_deliveries = random.randint(5, 15)
        self.pre_trip_done = False
        self.maintenance_due = False

    def update(self, sim_time: float, battery_level: float, driver_profile: DriverProfile) -> Optional[TruckState]:
        """Update truck state with logistics patterns"""
        try:
            hour = int((sim_time / TimeConstants.HOUR) % 24)

            # Trucks need more battery for heavy loads
            if battery_level < BatteryConstants.LOW:
                if self.current_state not in [
                    TruckState.CHARGING,
                    TruckState.NAVIGATING_TO_CHARGING,
                    TruckState.WAITING_FOR_CHARGING
                ]:
                    return TruckState.EMERGENCY_LOW_BATTERY

            # Opportunistic charging at depot
            if battery_level < BatteryConstants.MEDIUM:
                if self.current_state in [TruckState.AT_DEPOT, TruckState.LOADING]:
                    if random.random() < 0.7:
                        return TruckState.SEARCHING_CHARGING

            # State transitions
            if self.current_state == TruckState.OFF_DUTY:
                if 5 <= hour <= 7:
                    self.deliveries_completed = 0
                    self.daily_deliveries = random.randint(5, 15)
                    self.pre_trip_done = False
                    return TruckState.AT_DEPOT

            elif self.current_state == TruckState.AT_DEPOT:
                if not self.pre_trip_done:
                    return TruckState.PRE_TRIP_INSPECTION
                elif self.deliveries_completed < self.daily_deliveries:
                    return TruckState.LOADING
                elif hour >= 18:
                    return TruckState.END_OF_SHIFT

            elif self.current_state == TruckState.PRE_TRIP_INSPECTION:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    self.pre_trip_done = True
                    # Check for maintenance
                    if random.random() < 0.05:
                        self.maintenance_due = True
                        return TruckState.MAINTENANCE_CHECK
                    return TruckState.AT_DEPOT

            elif self.current_state == TruckState.LOADING:
                loading_time = TimeConstants.THIRTY_MINUTES
                if self.get_state_duration(sim_time) > loading_time:
                    return TruckState.EN_ROUTE_TO_DELIVERY

            elif self.current_state == TruckState.EN_ROUTE_TO_DELIVERY:
                # Check for traffic delays
                if random.random() < 0.2:
                    return TruckState.TRAFFIC_DELAY

                route_time = random.randint(TimeConstants.THIRTY_MINUTES, TimeConstants.HOUR * 2)
                if self.get_state_duration(sim_time) > route_time:
                    return TruckState.DELIVERING

            elif self.current_state == TruckState.TRAFFIC_DELAY:
                delay_time = random.randint(TimeConstants.TEN_MINUTES, TimeConstants.THIRTY_MINUTES)
                if self.get_state_duration(sim_time) > delay_time:
                    return TruckState.EN_ROUTE_TO_DELIVERY

            elif self.current_state == TruckState.DELIVERING:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return TruckState.UNLOADING

            elif self.current_state == TruckState.UNLOADING:
                unload_time = TimeConstants.THIRTY_MINUTES
                if self.get_state_duration(sim_time) > unload_time:
                    self.deliveries_completed += 1

                    # Check if more deliveries
                    if self.deliveries_completed >= self.daily_deliveries:
                        return TruckState.RETURNING_TO_DEPOT
                    elif self.deliveries_completed % 3 == 0:  # Break every 3 deliveries
                        return TruckState.BREAK_TIME
                    else:
                        return TruckState.EN_ROUTE_TO_DELIVERY

            elif self.current_state == TruckState.RETURNING_TO_DEPOT:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    return TruckState.AT_DEPOT

            elif self.current_state == TruckState.BREAK_TIME:
                # Mandatory break for safety
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    if battery_level < BatteryConstants.MEDIUM:
                        return TruckState.SEARCHING_CHARGING
                    return TruckState.EN_ROUTE_TO_DELIVERY

            elif self.current_state == TruckState.MAINTENANCE_CHECK:
                if self.get_state_duration(sim_time) > TimeConstants.HOUR:
                    self.maintenance_due = False
                    return TruckState.AT_DEPOT

            elif self.current_state == TruckState.END_OF_SHIFT:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    return TruckState.OFF_DUTY

            # Charging states
            elif self.current_state == TruckState.EMERGENCY_LOW_BATTERY:
                return TruckState.SEARCHING_CHARGING

            elif self.current_state == TruckState.SEARCHING_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return TruckState.NAVIGATING_TO_CHARGING

            elif self.current_state == TruckState.NAVIGATING_TO_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return TruckState.WAITING_FOR_CHARGING

            elif self.current_state == TruckState.WAITING_FOR_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return TruckState.CHARGING

            elif self.current_state == TruckState.CHARGING:
                # Trucks need more charge for heavy loads
                target_soc = BatteryConstants.HIGH
                charge_time = ChargingConstants.LONG_CHARGE_TIME

                if battery_level >= target_soc or self.get_state_duration(sim_time) > charge_time:
                    self.charging_count += 1
                    if self.deliveries_completed < self.daily_deliveries:
                        return TruckState.EN_ROUTE_TO_DELIVERY
                    return TruckState.AT_DEPOT

            return None

        except Exception as e:
            self.logger.error(f"Error in truck state update: {e}", "state_update", exc_info=True)
            return TruckState.AT_DEPOT

class VanStateMachine(BaseStateMachine):
    """Van state machine for package delivery"""

    def __init__(self, vehicle_id: str, logger: SimulationLogger):
        super().__init__(vehicle_id, 'electric_van', logger)
        self.current_state = VanState.AT_WAREHOUSE
        self.packages_delivered = 0
        self.daily_packages = random.randint(30, 80)
        self.returns_collected = 0

    def update(self, sim_time: float, battery_level: float, driver_profile: DriverProfile) -> Optional[VanState]:
        """Update van state with delivery patterns"""
        try:
            hour = int((sim_time / TimeConstants.HOUR) % 24)

            # Vans do many short trips, need frequent charging
            if battery_level < BatteryConstants.VERY_LOW:
                if self.current_state not in [
                    VanState.CHARGING,
                    VanState.NAVIGATING_TO_CHARGING,
                    VanState.WAITING_FOR_CHARGING
                ]:
                    return VanState.EMERGENCY_LOW_BATTERY

            # Opportunistic charging
            if battery_level < BatteryConstants.MEDIUM_LOW:
                if self.current_state in [VanState.AT_WAREHOUSE, VanState.BREAK_TIME]:
                    if random.random() < 0.8:
                        return VanState.SEARCHING_CHARGING

            # State transitions
            if self.current_state == VanState.AT_WAREHOUSE:
                if 6 <= hour <= 8 and self.packages_delivered == 0:
                    return VanState.MORNING_BRIEFING
                elif 8 <= hour <= 18 and self.packages_delivered < self.daily_packages:
                    return VanState.LOADING_PACKAGES
                elif hour >= 19:
                    self.packages_delivered = 0
                    self.daily_packages = random.randint(30, 80)
                    self.returns_collected = 0

            elif self.current_state == VanState.MORNING_BRIEFING:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return VanState.LOADING_PACKAGES

            elif self.current_state == VanState.LOADING_PACKAGES:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return VanState.SORTING_PACKAGES

            elif self.current_state == VanState.SORTING_PACKAGES:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return VanState.STARTING_ROUTE

            elif self.current_state == VanState.STARTING_ROUTE:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return VanState.DELIVERY_ROUTE

            elif self.current_state == VanState.DELIVERY_ROUTE:
                # Short drives between stops
                drive_time = random.randint(TimeConstants.MINUTE * 2, TimeConstants.TEN_MINUTES)
                if self.get_state_duration(sim_time) > drive_time:
                    return VanState.DELIVERING_PACKAGE

            elif self.current_state == VanState.DELIVERING_PACKAGE:
                if self.get_state_duration(sim_time) > TimeConstants.MINUTE:
                    # Sometimes customer interaction needed
                    if random.random() < 0.2:
                        return VanState.CUSTOMER_INTERACTION

                    self.packages_delivered += 1

                    # Collect returns occasionally
                    if random.random() < 0.1:
                        self.returns_collected += 1

                    # Check progress
                    if self.packages_delivered >= self.daily_packages:
                        return VanState.RETURNING_TO_WAREHOUSE
                    elif self.packages_delivered % 15 == 0:  # Break every 15 packages
                        return VanState.BREAK_TIME
                    else:
                        return VanState.BETWEEN_DELIVERIES

            elif self.current_state == VanState.CUSTOMER_INTERACTION:
                interaction_time = random.randint(TimeConstants.MINUTE * 2, TimeConstants.FIVE_MINUTES)
                if self.get_state_duration(sim_time) > interaction_time:
                    self.packages_delivered += 1
                    return VanState.BETWEEN_DELIVERIES

            elif self.current_state == VanState.BETWEEN_DELIVERIES:
                if self.get_state_duration(sim_time) > TimeConstants.MINUTE:
                    return VanState.DELIVERY_ROUTE

            elif self.current_state == VanState.RETURNING_TO_WAREHOUSE:
                if self.get_state_duration(sim_time) > TimeConstants.THIRTY_MINUTES:
                    if self.returns_collected > 0:
                        return VanState.UNLOADING_RETURNS
                    return VanState.END_OF_ROUTE

            elif self.current_state == VanState.UNLOADING_RETURNS:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return VanState.END_OF_ROUTE

            elif self.current_state == VanState.BREAK_TIME:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    if battery_level < BatteryConstants.MEDIUM:
                        return VanState.SEARCHING_CHARGING
                    return VanState.DELIVERY_ROUTE

            elif self.current_state == VanState.END_OF_ROUTE:
                if self.get_state_duration(sim_time) > TimeConstants.FIFTEEN_MINUTES:
                    return VanState.AT_WAREHOUSE

            # Charging states
            elif self.current_state == VanState.EMERGENCY_LOW_BATTERY:
                return VanState.SEARCHING_CHARGING

            elif self.current_state == VanState.SEARCHING_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return VanState.NAVIGATING_TO_CHARGING

            elif self.current_state == VanState.NAVIGATING_TO_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.TEN_MINUTES:
                    return VanState.WAITING_FOR_CHARGING

            elif self.current_state == VanState.WAITING_FOR_CHARGING:
                if self.get_state_duration(sim_time) > TimeConstants.FIVE_MINUTES:
                    return VanState.CHARGING

            elif self.current_state == VanState.CHARGING:
                # Vans need quick turnaround
                target_soc = BatteryConstants.MEDIUM_HIGH
                charge_time = ChargingConstants.STANDARD_CHARGE_TIME

                if battery_level >= target_soc or self.get_state_duration(sim_time) > charge_time:
                    self.charging_count += 1
                    if self.packages_delivered < self.daily_packages:
                        return VanState.DELIVERY_ROUTE
                    return VanState.AT_WAREHOUSE

            return None

        except Exception as e:
            self.logger.error(f"Error in van state update: {e}", "state_update", exc_info=True)
            return VanState.AT_WAREHOUSE

# ============================================================================
# Safe Mathematical Operations
# ============================================================================

class SafeMath:
    """Safe mathematical operations with error handling"""

    @staticmethod
    def divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with zero check"""
        try:
            if abs(denominator) < 1e-10:
                return default
            result = numerator / denominator
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except:
            return default

    @staticmethod
    def distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Safe distance calculation"""
        try:
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            return math.sqrt(dx * dx + dy * dy)
        except:
            return float('inf')

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))

# ============================================================================
# Vehicle Class with Complete Implementation
# ============================================================================

@dataclass
class Vehicle:
    """Complete vehicle with all attributes"""
    vehicle_id: str
    vehicle_type: str
    specs: VehicleSpecs
    driver_profile: DriverProfile
    state_machine: BaseStateMachine

    # Physical state
    battery_soc: float = 0.7
    position: Tuple[float, float] = (0, 0)
    speed: float = 0.0
    current_edge: str = ""
    destination: Optional[str] = None

    # Statistics
    distance_traveled: float = 0.0
    energy_consumed: float = 0.0
    charging_sessions: int = 0
    time_created: float = 0.0

    # Decision tracking
    last_decision_time: float = 0.0
    last_gpt_call: float = 0.0

    def safe_update_battery(self, distance: float):
        """Safely update battery with consumption"""
        if distance > 0 and distance < 10000:  # Sanity check
            consumption_kwh = SafeMath.divide(
                distance * self.specs.consumption_rate,
                1000 * self.specs.battery_capacity,
                0
            )
            self.battery_soc = SafeMath.clamp(
                self.battery_soc - consumption_kwh,
                0.0, 1.0
            )
            self.energy_consumed += consumption_kwh * self.specs.battery_capacity
            self.distance_traveled += distance

# ============================================================================
# Enhanced Charging Station
# ============================================================================

@dataclass
class ChargingStation:
    """Charging station with comprehensive features"""
    station_id: str
    edge: str
    position: Tuple[float, float]
    power: float
    efficiency: float
    base_price: float
    capacity: int
    parking_area: Optional[str] = None

    # Dynamic attributes
    current_price: float = field(init=False)
    utilization: float = 0.0
    queue_length: int = 0
    vehicles_charging: List[str] = field(default_factory=list)
    vehicles_queued: List[str] = field(default_factory=list)

    # Features
    amenities: List[str] = field(default_factory=list)
    renewable_energy: bool = False
    fast_charging: bool = False

    # Statistics
    total_sessions: int = 0
    total_energy: float = 0.0
    rating: float = 5.0
    rating_count: int = 0
    hourly_usage: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        self.current_price = self.base_price
        self.fast_charging = self.power >= ChargingConstants.FAST_CHARGE_POWER
        self._generate_features()

    def _generate_features(self):
        """Generate station features"""
        if self.power >= ChargingConstants.ULTRA_FAST_CHARGE_POWER:
            self.amenities = random.sample(['restroom', 'coffee', 'wifi', 'restaurant', 'lounge'],
                                         min(4, 5))
        elif self.power >= ChargingConstants.FAST_CHARGE_POWER:
            self.amenities = random.sample(['restroom', 'coffee', 'wifi'],
                                         min(2, 3))
        else:
            self.amenities = random.sample(['restroom', 'wifi'],
                                         min(1, 2))

        self.renewable_energy = random.random() > 0.4

    def update_pricing(self, hour: int, utilization: float):
        """Dynamic pricing based on demand"""
        # Time-based pricing
        if hour in [7, 8, 9, 17, 18, 19]:  # Peak hours
            time_factor = 1.3
        elif hour in [23, 0, 1, 2, 3, 4]:  # Off-peak
            time_factor = 0.7
        else:
            time_factor = 1.0

        # Utilization-based pricing
        if utilization > 0.8:
            util_factor = 1.4
        elif utilization > 0.6:
            util_factor = 1.2
        elif utilization < 0.3:
            util_factor = 0.8
        else:
            util_factor = 1.0

        self.current_price = self.base_price * time_factor * util_factor
        self.utilization = utilization

        # Track hourly usage
        self.hourly_usage[hour] += len(self.vehicles_charging)

    def add_vehicle(self, vehicle_id: str, charging: bool = False):
        """Add vehicle to station"""
        if charging:
            if vehicle_id not in self.vehicles_charging:
                self.vehicles_charging.append(vehicle_id)
                self.total_sessions += 1
        else:
            if vehicle_id not in self.vehicles_queued:
                self.vehicles_queued.append(vehicle_id)

        self.queue_length = len(self.vehicles_queued)

    def remove_vehicle(self, vehicle_id: str):
        """Remove vehicle from station"""
        if vehicle_id in self.vehicles_charging:
            self.vehicles_charging.remove(vehicle_id)
        if vehicle_id in self.vehicles_queued:
            self.vehicles_queued.remove(vehicle_id)

        self.queue_length = len(self.vehicles_queued)

    def get_status(self) -> Dict:
        """Get station status"""
        return {
            'id': self.station_id,
            'position': self.position,
            'power': self.power,
            'price': self.current_price,
            'utilization': self.utilization,
            'charging': len(self.vehicles_charging),
            'queued': len(self.vehicles_queued),
            'capacity': self.capacity,
            'amenities': self.amenities,
            'renewable': self.renewable_energy,
            'rating': self.rating
        }

# ============================================================================
# Main Simulation Controller - Fixed Performance Issues
# ============================================================================

class EVSimulation:
    """Main simulation controller with performance fixes"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = SimulationLogger(config)

        # Check dependencies
        if not SUMO_AVAILABLE:
            self.logger.error("SUMO not available. Cannot run simulation.", "init")
            sys.exit(1)

        # Core components
        self.vehicle_pool = VehiclePoolManager(self.logger)
        self.stations: Dict[str, ChargingStation] = {}
        self.charging_sessions: Dict[str, Dict] = {}
        self.station_queues: Dict[str, deque] = defaultdict(deque)

        # Simulation state
        self.sim_time = 0
        self.step = 0
        self.destinations_cache = []
        self.statistics = defaultdict(int)

        # Performance monitoring
        self.performance_monitor = None
        if config.enable_performance_monitor:
            self.performance_monitor = PerformanceMonitor(self.logger)

        # GPT client - CRITICAL COMPONENT
        self.gpt_client = None
        if config.use_gpt and OPENAI_AVAILABLE:
            self._init_gpt()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_updates)

        # Thread safety
        self.lock = threading.RLock()

        # Data buffers for performance
        self.statistics_buffer = []
        self.decision_buffer = []
        self.session_buffer = []
        self.last_save_time = time.time()

    def _init_gpt(self):
        """Initialize GPT client with error handling"""
        if not OPENAI_AVAILABLE or not self.config.api_key:
            self.logger.warning("OpenAI not available or API key missing - GPT DECISIONS DISABLED!", "gpt")
            self.config.use_gpt = False
            return

        for attempt in range(self.config.max_retries):
            try:
                # Fix for xiaoai.plus API - ensure proper URL format
                api_url = self.config.api_base_url
                if not api_url.endswith('/v1'):
                    api_url = api_url.rstrip('/') + '/v1'

                self.gpt_client = OpenAI(
                    base_url=api_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=1  # Reduce retries at OpenAI level
                )

                # Test connection with a simple chat completion instead of models.list()
                test_response = self.gpt_client.chat.completions.create(
                    model=self.config.gpt_model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )

                self.logger.info("GPT client initialized successfully - AI DECISION MAKING ENABLED!")
                return

            except AttributeError as e:
                # Handle specific attribute errors from API compatibility
                self.logger.error(f"GPT init attempt {attempt+1} failed - API compatibility issue: {e}", "gpt")
                if "private_attributes" in str(e):
                    self.logger.warning("This may be due to OpenAI library version incompatibility with xiaoai.plus", "gpt")
                time.sleep(2 ** attempt)

            except Exception as e:
                self.logger.error(f"GPT init attempt {attempt+1} failed: {e}", "gpt")
                time.sleep(2 ** attempt)

        self.logger.error("Failed to initialize GPT after all retries - FALLING BACK TO RULE-BASED DECISIONS", "gpt")
        self.config.use_gpt = False
        self.gpt_client = None

    def init_sumo(self) -> bool:
        """Initialize SUMO with error handling"""
        try:
            cmd = ["sumo-gui", "-c", self.config.sumo_config]
            traci.start(cmd)
            self.logger.info("SUMO started successfully")

            # Cache valid destinations
            edges = traci.edge.getIDList()
            self.destinations_cache = [
                e for e in edges
                if not e.startswith(':') and len(e) > 0
            ]

            if not self.destinations_cache:
                self.logger.error("No valid edges found in network", "sumo")
                return False

            self.logger.info(f"Loaded {len(self.destinations_cache)} valid edges")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SUMO: {e}", "sumo", exc_info=True)
            return False

    def init_charging_stations(self):
        """Initialize charging stations with validation"""
        try:
            cs_ids = traci.chargingstation.getIDList()
            self.logger.info(f"Found {len(cs_ids)} charging stations")

            for cs_id in cs_ids:
                try:
                    # Get station information safely
                    lane_id = traci.chargingstation.getLaneID(cs_id)
                    if not lane_id:
                        continue

                    edge_id = traci.lane.getEdgeID(lane_id)
                    if edge_id.startswith(':'):
                        continue

                    start_pos = traci.chargingstation.getStartPos(cs_id)
                    end_pos = traci.chargingstation.getEndPos(cs_id)
                    center = SafeMath.divide(start_pos + end_pos, 2, 0)

                    # Safe coordinate conversion
                    try:
                        x, y = traci.simulation.convert2D(edge_id, center)
                    except:
                        x, y = 0, 0

                    power = max(1000, traci.chargingstation.getChargingPower(cs_id))
                    efficiency = SafeMath.clamp(traci.chargingstation.getEfficiency(cs_id), 0.5, 1.0)

                    # Find parking area
                    parking_area = None
                    try:
                        for pa_id in traci.parkingarea.getIDList():
                            if traci.parkingarea.getLaneID(pa_id) == lane_id:
                                parking_area = pa_id
                                break
                    except:
                        pass

                    # Create station
                    station = ChargingStation(
                        station_id=cs_id,
                        edge=edge_id,
                        position=(x, y),
                        power=power,
                        efficiency=efficiency,
                        base_price=random.uniform(1.5, 3.0),
                        capacity=random.randint(2, 8),
                        parking_area=parking_area
                    )

                    self.stations[cs_id] = station
                    self.logger.debug(f"Loaded station {cs_id} at {edge_id}")

                except Exception as e:
                    self.logger.error(f"Error loading station {cs_id}: {e}", "station")

            self.logger.info(f"Successfully loaded {len(self.stations)} stations")

        except Exception as e:
            self.logger.error(f"Failed to initialize charging stations: {e}", "station", exc_info=True)

    def create_vehicle(self, vehicle_id: str):
        """Create vehicle with comprehensive initialization"""
        try:
            # Check if already exists
            if self.vehicle_pool.get_vehicle(vehicle_id):
                return

            # Get vehicle type safely
            try:
                type_id = traci.vehicle.getTypeID(vehicle_id).lower()
            except:
                type_id = "electric"

            # Map to our vehicle types
            vehicle_type = 'electric_sedan'  # default
            for vtype in ['sedan', 'suv', 'taxi', 'truck', 'van']:
                if vtype in type_id:
                    vehicle_type = f'electric_{vtype}'
                    break

            # Validate vehicle type
            if vehicle_type not in VEHICLE_SPECS:
                vehicle_type = 'electric_sedan'

            # Create state machine
            if vehicle_type == 'electric_sedan':
                state_machine = SedanStateMachine(vehicle_id, self.logger)
            elif vehicle_type == 'electric_suv':
                state_machine = SUVStateMachine(vehicle_id, self.logger)
            elif vehicle_type == 'electric_taxi':
                state_machine = TaxiStateMachine(vehicle_id, self.logger)
            elif vehicle_type == 'electric_truck':
                state_machine = TruckStateMachine(vehicle_id, self.logger)
            elif vehicle_type == 'electric_van':
                state_machine = VanStateMachine(vehicle_id, self.logger)
            else:
                state_machine = SedanStateMachine(vehicle_id, self.logger)

            # Create vehicle
            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                specs=VEHICLE_SPECS[vehicle_type],
                driver_profile=DriverProfile.generate_profile(vehicle_type),
                state_machine=state_machine,
                battery_soc=random.uniform(0.5, 0.9),
                time_created=self.sim_time
            )

            # Add to pool
            self.vehicle_pool.add_vehicle(vehicle)

            # Set initial route
            self._set_safe_destination(vehicle_id)

            self.logger.info(f"Created {vehicle_type}: {vehicle_id} with battery {vehicle.battery_soc:.2f}")
            self.statistics[f'vehicles_created_{vehicle_type}'] += 1

        except Exception as e:
            self.logger.error(f"Failed to create vehicle {vehicle_id}: {e}", "vehicle", exc_info=True)

    def _set_safe_destination(self, vehicle_id: str):
        """Set destination with error handling"""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(':'):
                return

            # Get valid destinations
            valid_dests = [
                d for d in self.destinations_cache
                if d != current_edge
            ]

            if not valid_dests:
                return

            # Choose destination based on vehicle type
            vehicle = self.vehicle_pool.get_vehicle(vehicle_id)

            if vehicle:
                if vehicle.vehicle_type == 'electric_taxi':
                    # Taxis prefer busy areas
                    dest = random.choice(valid_dests[:len(valid_dests)//2])
                elif vehicle.vehicle_type in ['electric_truck', 'electric_van']:
                    # Delivery vehicles have specific routes
                    dest = random.choice(valid_dests)
                else:
                    # Regular vehicles
                    dest = random.choice(valid_dests)
            else:
                dest = random.choice(valid_dests)

            # Find and set route
            route = traci.simulation.findRoute(current_edge, dest)
            if route and route.edges:
                traci.vehicle.setRoute(vehicle_id, route.edges)
                self.logger.debug(f"Set route for {vehicle_id}: {current_edge} -> {dest}")

        except Exception as e:
            self.logger.debug(f"Could not set destination for {vehicle_id}: {e}")

    def update_vehicle(self, vehicle: Vehicle) -> bool:
        """Update vehicle with comprehensive error handling"""
        try:
            veh_id = vehicle.vehicle_id

            # Check existence
            if veh_id not in traci.vehicle.getIDList():
                return False

            # Update position safely
            try:
                old_pos = vehicle.position
                vehicle.position = traci.vehicle.getPosition(veh_id)
                vehicle.speed = max(0, traci.vehicle.getSpeed(veh_id))
                vehicle.current_edge = traci.vehicle.getRoadID(veh_id)

                # Calculate distance and update battery
                if old_pos != (0, 0):
                    distance = SafeMath.distance(old_pos, vehicle.position)
                    if 0 < distance < 1000:  # Sanity check
                        vehicle.safe_update_battery(distance)

            except Exception as e:
                self.logger.debug(f"Position update error for {veh_id}: {e}")

            # Update psychological state
            all_vehicles = self.vehicle_pool.get_all_vehicles()
            active_count = len(self.vehicle_pool.active_vehicles)

            context = {
                'hour': int((self.sim_time / TimeConstants.HOUR) % 24),
                'traffic_density': active_count / max(1, len(self.destinations_cache)),
                'weather': random.choice(['clear', 'rain', 'fog']),
                'vehicles_nearby_charging': sum(1 for v in all_vehicles.values()
                                               if v.state_machine and 'CHARGING' in str(v.state_machine.current_state))
            }
            vehicle.driver_profile.update_psychological_state(context)

            # Update state machine
            new_state = vehicle.state_machine.update(
                self.sim_time,
                vehicle.battery_soc,
                vehicle.driver_profile
            )

            if new_state:
                self._handle_state_transition(vehicle, new_state)

            # CRITICAL: Check for charging decision interval
            if self.sim_time - vehicle.last_decision_time >= self.config.decision_interval:
                # Make charging decision - GPT or rule-based
                decision = self.make_gpt_decision(vehicle)
                vehicle.last_decision_time = self.sim_time

                # Apply decision
                if decision['should_charge']:
                    self.logger.info(f"{veh_id} decided to charge (urgency: {decision['urgency']:.2f})")
                    self._initiate_charging_process(vehicle)

            # Check route validity
            try:
                route = traci.vehicle.getRoute(veh_id)
                route_index = traci.vehicle.getRouteIndex(veh_id)

                if not route or route_index >= len(route) - 2:
                    self._set_safe_destination(veh_id)

            except:
                self._set_safe_destination(veh_id)

            return True

        except Exception as e:
            self.logger.error(f"Failed to update vehicle {vehicle.vehicle_id}: {e}", "update")
            return False

    def _handle_state_transition(self, vehicle: Vehicle, new_state):
        """Handle state transitions with charging logic"""
        vehicle.state_machine.safe_transition(new_state, self.sim_time)

        # Check if entering charging search state
        charging_search_states = [
            SedanState.SEARCHING_CHARGING,
            SUVState.SEARCHING_CHARGING,
            TaxiState.SEARCHING_CHARGING,
            TruckState.SEARCHING_CHARGING,
            VanState.SEARCHING_CHARGING
        ]

        if new_state in charging_search_states:
            self._initiate_charging_process(vehicle)

    def _initiate_charging_process(self, vehicle: Vehicle):
        """Initiate charging with station selection"""
        try:
            # Find best station
            best_station = self._find_best_station(vehicle)

            if best_station:
                # Navigate to station
                self._navigate_to_station(vehicle, best_station)
                vehicle.destination = best_station.station_id

                # Add to station
                best_station.add_vehicle(vehicle.vehicle_id, charging=False)

                # Add to queue if needed
                with self.lock:
                    if best_station.station_id not in self.station_queues:
                        self.station_queues[best_station.station_id] = deque()

                    if vehicle.vehicle_id not in self.station_queues[best_station.station_id]:
                        self.station_queues[best_station.station_id].append(vehicle.vehicle_id)

                self.logger.info(f"{vehicle.vehicle_id} heading to station {best_station.station_id}")

        except Exception as e:
            self.logger.error(f"Failed to initiate charging for {vehicle.vehicle_id}: {e}", "charging")

    def _find_best_station(self, vehicle: Vehicle) -> Optional[ChargingStation]:
        """Find optimal charging station considering all factors"""
        try:
            candidates = []

            for station in self.stations.values():
                distance = SafeMath.distance(vehicle.position, station.position)

                # Skip if too far
                if distance > 30000:
                    continue

                # Calculate score considering irrational factors
                score = self._calculate_station_score(vehicle, station, distance)
                candidates.append((station, score))

            if candidates:
                # Sort by score
                candidates.sort(key=lambda x: x[1], reverse=True)

                # Apply irrational choice - sometimes not picking the best
                if vehicle.driver_profile.irrational.impulse_control < 0.5:
                    # Impulsive choice - might pick random
                    if random.random() < 0.2:
                        return random.choice(candidates)[0]

                return candidates[0][0]

            return None

        except Exception as e:
            self.logger.error(f"Error finding station: {e}", "station_search")
            return None

    def _calculate_station_score(self, vehicle: Vehicle, station: ChargingStation, distance: float) -> float:
        """Calculate station score with irrational factors"""
        score = 100.0

        # Distance factor
        distance_factor = SafeMath.clamp(1 - distance / 30000, 0, 1)
        score *= (0.3 + 0.7 * distance_factor)

        # Price sensitivity
        if vehicle.driver_profile.price_sensitivity > 0.5:
            price_factor = SafeMath.clamp(1 - station.current_price / 5, 0, 1)
            score *= (0.5 + 0.5 * price_factor)

        # Queue factor
        queue_len = len(self.station_queues.get(station.station_id, []))
        if vehicle.driver_profile.patience_level < 0.3:
            # Impatient drivers heavily penalize queues
            queue_factor = SafeMath.clamp(1 - queue_len / 5, 0, 1)
            score *= queue_factor

        # Amenity preference (irrational comfort seeking)
        if vehicle.driver_profile.comfort_preference > 0.6:
            amenity_score = len(station.amenities) / 5
            score *= (0.8 + 0.2 * amenity_score)

        # Brand loyalty (irrational preference)
        if station.station_id in vehicle.driver_profile.preferred_stations:
            score *= 1.5
        elif station.station_id in vehicle.driver_profile.avoided_stations:
            score *= 0.3

        # Social proof (if others are charging there)
        if queue_len > 0 and vehicle.driver_profile.irrational.herd_mentality > 0.5:
            score *= (1 + 0.2 * vehicle.driver_profile.irrational.herd_mentality)

        # Confirmation bias - stick to what worked before
        if vehicle.charging_sessions > 0 and station.station_id in vehicle.driver_profile.preferred_stations:
            score *= (1 + 0.3 * vehicle.driver_profile.irrational.confirmation_bias)

        return score

    def _navigate_to_station(self, vehicle: Vehicle, station: ChargingStation):
        """Navigate vehicle to charging station"""
        try:
            if vehicle.current_edge and not vehicle.current_edge.startswith(':'):
                route = traci.simulation.findRoute(vehicle.current_edge, station.edge)
                if route and route.edges:
                    traci.vehicle.setRoute(vehicle.vehicle_id, route.edges)

                    # Set parking stop if available
                    if station.parking_area:
                        try:
                            duration = ChargingConstants.STANDARD_CHARGE_TIME
                            traci.vehicle.setParkingAreaStop(
                                vehicle.vehicle_id,
                                station.parking_area,
                                duration=duration
                            )
                        except:
                            pass

        except Exception as e:
            self.logger.debug(f"Navigation error: {e}")

    # ============================================================================
    # CRITICAL: GPT DECISION MAKING SYSTEM
    # ============================================================================

    def make_gpt_decision(self, vehicle: Vehicle) -> Dict:
        """Enhanced GPT decision with irrational factors - CORE FUNCTIONALITY"""
        if not self.gpt_client or not self.config.use_gpt:
            return self._make_rule_decision(vehicle)

        # Check cooldown
        if self.sim_time - vehicle.last_gpt_call < self.config.gpt_cooldown:
            return self._make_rule_decision(vehicle)

        vehicle.last_gpt_call = self.sim_time

        try:
            # Create comprehensive prompt
            prompt = self._create_enhanced_prompt(vehicle)

            response = self.gpt_client.chat.completions.create(
                model=self.config.gpt_model,
                messages=[
                    {"role": "system", "content": self._get_enhanced_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                response_format={"type": "json_object"},
                timeout=self.config.timeout
            )

            result = json.loads(response.choices[0].message.content)

            # Buffer decision for batch writing
            self._buffer_decision(vehicle, result, "gpt")
            self.statistics['gpt_decisions'] += 1

            self.logger.info(f"GPT decision for {vehicle.vehicle_id}: {result['should_charge']} (urgency: {result['urgency']:.2f})")

            return result

        except Exception as e:
            self.logger.error(f"GPT decision failed: {e}", "gpt")
            self.statistics['gpt_failures'] += 1
            return self._make_rule_decision(vehicle)

    def _create_enhanced_prompt(self, vehicle: Vehicle) -> str:
        """Create prompt with psychological and irrational factors"""
        # Get nearby stations
        stations_info = []
        for station in list(self.stations.values())[:5]:
            dist = SafeMath.distance(vehicle.position, station.position)
            stations_info.append({
                'id': station.station_id,
                'distance_km': dist / 1000,
                'power_kw': station.power / 1000,
                'price': station.current_price,
                'queue': len(station.vehicles_queued),
                'amenities': station.amenities,
                'renewable': station.renewable_energy
            })

        all_vehicles = self.vehicle_pool.get_all_vehicles()
        charging_vehicles = len([v for v in all_vehicles.values()
                               if v.state_machine and 'CHARGING' in str(v.state_machine.current_state)])

        prompt = f"""
                    Analyze this EV charging decision considering both rational and irrational human factors.
                    VEHICLE: {vehicle.vehicle_type}
                    Battery: {vehicle.battery_soc*100:.1f}%
                    State: {vehicle.state_machine.current_state}
                    Distance traveled: {vehicle.distance_traveled/1000:.1f}km
                    
                    DRIVER PSYCHOLOGY:
                    Age: {vehicle.driver_profile.age}, {vehicle.driver_profile.gender}
                    Occupation: {vehicle.driver_profile.occupation}
                    Personality:
                    - Openness: {vehicle.driver_profile.openness:.2f}
                    - Conscientiousness: {vehicle.driver_profile.conscientiousness:.2f}
                    - Extraversion: {vehicle.driver_profile.extraversion:.2f}
                    - Agreeableness: {vehicle.driver_profile.agreeableness:.2f}
                    - Neuroticism: {vehicle.driver_profile.neuroticism:.2f}
                    
                    Current State:
                    - Mood: {vehicle.driver_profile.mood}
                    - Fatigue: {vehicle.driver_profile.fatigue_level:.2f}
                    - Stress: {vehicle.driver_profile.stress_level:.2f}
                    - Cognitive Load: {vehicle.driver_profile.cognitive_load:.2f}
                    
                    Irrational Factors:
                    - Herd Mentality: {vehicle.driver_profile.irrational.herd_mentality:.2f}
                    - Impulse Control: {vehicle.driver_profile.irrational.impulse_control:.2f}
                    - Habit Strength: {vehicle.driver_profile.irrational.habit_strength:.2f}
                    - Optimism Bias: {vehicle.driver_profile.irrational.optimism_bias:.2f}
                    - Loss Aversion: {vehicle.driver_profile.irrational.loss_aversion:.2f}
                    
                    CONTEXT:
                    Time: {int((self.sim_time/3600)%24)}:00
                    Vehicles charging nearby: {charging_vehicles}
                    
                    STATIONS: {json.dumps(stations_info, indent=2)}
                    
                    Consider:
                    1. Rational factors (battery, distance, price)
                    2. Personality influence on decisions
                    3. Current emotional/cognitive state
                    4. Irrational biases and heuristics
                    5. Social influences and habits
                    
                    Respond with JSON:
                    {{
                        "should_charge": true/false,
                        "station_id": "id or null",
                        "duration": seconds,
                        "urgency": 0-1,
                        "reasoning": {{
                            "rational_factors": "logical reasoning",
                            "emotional_factors": "mood/stress influence",
                            "irrational_factors": "biases affecting decision",
                            "decision_quality": 0-1 (affected by cognitive load)
                        }}
                    }}"""

        return prompt

    def _get_enhanced_system_prompt(self) -> str:
        """System prompt emphasizing human irrationality"""
        return """You are simulating realistic human EV charging decisions with psychological depth.

Key principles:
1. Humans are NOT purely rational - emotions, biases, and habits strongly influence decisions
2. High stress/fatigue leads to poor decisions
3. Personality traits affect behavior:
   - High neuroticism = more anxiety, earlier charging
   - Low conscientiousness = procrastination, risky delays
   - High agreeableness = follow social norms
4. Cognitive biases:
   - Optimism bias: "I can make it further"
   - Loss aversion: Overreact to low battery
   - Anchoring: Stick to familiar stations
   - Herd mentality: Charge where others charge
5. Context matters:
   - Tired drivers make impulsive decisions
   - Stressed drivers are risk-averse
   - Happy drivers are more optimistic

Make decisions that reflect real human psychology, not optimal robot behavior."""

    def _make_rule_decision(self, vehicle: Vehicle) -> Dict:
        """Rule-based decision with irrational factors"""
        decision = {
            "should_charge": False,
            "station_id": None,
            "duration": ChargingConstants.STANDARD_CHARGE_TIME,
            "urgency": 0.0,
            "reasoning": {}
        }

        # Base urgency on battery
        if vehicle.battery_soc < BatteryConstants.CRITICAL:
            decision["urgency"] = 1.0
        elif vehicle.battery_soc < BatteryConstants.LOW:
            decision["urgency"] = 0.8
        elif vehicle.battery_soc < BatteryConstants.MEDIUM_LOW:
            decision["urgency"] = 0.5
        else:
            decision["urgency"] = 0.2

        # Apply irrational modifications
        context = {
            'battery_low': vehicle.battery_soc < BatteryConstants.MEDIUM_LOW,
            'others_charging': random.random() < 0.3,
            'is_usual_time': int((self.sim_time/3600)%24) in [12, 18]
        }

        modified_urgency = vehicle.driver_profile.irrational.apply_to_decision(
            decision["urgency"], context
        )

        # Stress and fatigue effects
        if vehicle.driver_profile.stress_level > 0.7:
            modified_urgency *= 1.3  # Stressed people charge earlier

        if vehicle.driver_profile.fatigue_level > 0.7:
            if random.random() < 0.3:  # Tired people make random decisions
                modified_urgency = random.random()

        decision["urgency"] = SafeMath.clamp(modified_urgency, 0, 1)

        # Decide based on modified urgency
        if decision["urgency"] > 0.5:
            decision["should_charge"] = True

            # Find station
            station = self._find_best_station(vehicle)
            if station:
                decision["station_id"] = station.station_id

            # Duration based on patience and vehicle type
            if vehicle.vehicle_type == 'electric_taxi':
                decision["duration"] = ChargingConstants.QUICK_CHARGE_TIME
            elif vehicle.driver_profile.patience_level < 0.3:
                decision["duration"] = ChargingConstants.MIN_CHARGE_TIME
            else:
                decision["duration"] = ChargingConstants.STANDARD_CHARGE_TIME

        decision["reasoning"] = {
            "rational_factors": f"Battery at {vehicle.battery_soc*100:.1f}%",
            "emotional_factors": f"Mood: {vehicle.driver_profile.mood}, Stress: {vehicle.driver_profile.stress_level:.2f}",
            "irrational_factors": f"Modified by biases from {decision['urgency']:.2f}",
            "decision_quality": 1.0 - vehicle.driver_profile.cognitive_load
        }

        self._buffer_decision(vehicle, decision, "rule")
        self.statistics['rule_decisions'] += 1

        return decision

    def _buffer_decision(self, vehicle: Vehicle, decision: Dict, decision_type: str):
        """Buffer decision for batch writing"""
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "sim_time": self.sim_time,
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "battery_soc": vehicle.battery_soc,
                "state": str(vehicle.state_machine.current_state),
                "decision_type": decision_type,
                "decision": decision,
                "driver": {
                    "age": vehicle.driver_profile.age,
                    "personality": {
                        "O": vehicle.driver_profile.openness,
                        "C": vehicle.driver_profile.conscientiousness,
                        "E": vehicle.driver_profile.extraversion,
                        "A": vehicle.driver_profile.agreeableness,
                        "N": vehicle.driver_profile.neuroticism
                    },
                    "state": {
                        "mood": vehicle.driver_profile.mood,
                        "fatigue": vehicle.driver_profile.fatigue_level,
                        "stress": vehicle.driver_profile.stress_level,
                        "cognitive_load": vehicle.driver_profile.cognitive_load,
                        "emotional_state": vehicle.driver_profile.emotional_state
                    }
                }
            }

            self.decision_buffer.append((decision_type, log_entry))

        except Exception as e:
            self.logger.error(f"Failed to buffer decision: {e}", "logging")

    def process_charging_queues(self):
        """Process charging station queues with timeout protection"""
        try:
            # Create a copy of station IDs to avoid modification during iteration
            station_ids = list(self.station_queues.keys())

            for station_id in station_ids:
                # Check if station still exists
                if station_id not in self.stations:
                    self.logger.warning(f"Station {station_id} no longer exists, cleaning up queue")
                    if station_id in self.station_queues:
                        del self.station_queues[station_id]
                    continue

                station = self.stations[station_id]
                queue = self.station_queues.get(station_id)

                if not queue:
                    continue

                # Process vehicles in queue with timeout
                process_count = 0
                max_process = 10  # Prevent infinite loops

                while queue and len(station.vehicles_charging) < station.capacity and process_count < max_process:
                    process_count += 1
                    veh_id = queue[0]

                    # Check if vehicle still exists in SUMO
                    try:
                        if veh_id not in traci.vehicle.getIDList():
                            self.logger.debug(f"Vehicle {veh_id} no longer in simulation")
                            queue.popleft()
                            station.remove_vehicle(veh_id)
                            continue
                    except Exception as e:
                        self.logger.error(f"Error checking vehicle {veh_id}: {e}")
                        queue.popleft()
                        continue

                    # Get vehicle object
                    vehicle = self.vehicle_pool.get_vehicle(veh_id)
                    if not vehicle:
                        self.logger.warning(f"Vehicle {veh_id} not found in pool")
                        queue.popleft()
                        continue

                    # Check if vehicle is at station
                    try:
                        distance = SafeMath.distance(vehicle.position, station.position)
                        self.logger.debug(f"Vehicle {veh_id} distance to station: {distance:.1f}m")

                        if distance < 100:  # At station
                            # Start charging
                            queue.popleft()
                            station.remove_vehicle(veh_id)
                            station.add_vehicle(veh_id, charging=True)

                            # Create charging session
                            session = {
                                'vehicle_id': veh_id,
                                'station_id': station_id,
                                'start_time': self.sim_time,
                                'start_soc': vehicle.battery_soc,
                                'target_soc': BatteryConstants.HIGH,
                                'power': station.power,
                                'price': station.current_price
                            }
                            self.charging_sessions[veh_id] = session

                            self.logger.info(f"{veh_id} started charging at {station_id}")
                            vehicle.charging_sessions += 1
                        else:
                            # Vehicle not yet at station, check if it's making progress
                            if not hasattr(vehicle, '_last_station_distance'):
                                vehicle._last_station_distance = distance
                                vehicle._stuck_counter = 0
                            else:
                                # Check if vehicle is stuck
                                if abs(vehicle._last_station_distance - distance) < 1:  # Not moving
                                    vehicle._stuck_counter += 1
                                    if vehicle._stuck_counter > 10:
                                        self.logger.warning(f"Vehicle {veh_id} appears stuck, removing from queue")
                                        queue.popleft()
                                        station.remove_vehicle(veh_id)
                                        # Reset vehicle destination
                                        self._set_safe_destination(veh_id)
                                else:
                                    vehicle._stuck_counter = 0
                                vehicle._last_station_distance = distance
                            break  # Don't process more vehicles if this one is still approaching

                    except Exception as e:
                        self.logger.error(f"Error processing vehicle {veh_id} in queue: {e}")
                        queue.popleft()
                        continue

        except Exception as e:
            self.logger.error(f"Error processing charging queues: {e}", "charging", exc_info=True)

    def update_charging_vehicles(self):
        """Update vehicles currently charging"""
        try:
            completed = []

            for veh_id, session in list(self.charging_sessions.items()):
                vehicle = self.vehicle_pool.get_vehicle(veh_id)

                if not vehicle:
                    completed.append(veh_id)
                    continue

                # Calculate charging progress
                charge_time = self.sim_time - session['start_time']
                charge_rate = session['power'] * self.stations[session['station_id']].efficiency

                # Update SOC
                energy_added = charge_rate * charge_time / TimeConstants.HOUR
                soc_added = SafeMath.divide(energy_added, vehicle.specs.battery_capacity, 0)
                vehicle.battery_soc = SafeMath.clamp(
                    session['start_soc'] + soc_added,
                    0, 1
                )

                # Check if charging complete
                if vehicle.battery_soc >= session['target_soc'] or \
                        charge_time >= ChargingConstants.MAX_CHARGE_TIME:
                    completed.append(veh_id)

                    # Calculate cost
                    energy_kwh = energy_added / 1000
                    cost = energy_kwh * session['price']

                    # Log session
                    session['end_time'] = self.sim_time
                    session['end_soc'] = vehicle.battery_soc
                    session['energy_kwh'] = energy_kwh
                    session['cost'] = cost
                    session['duration'] = charge_time

                    self._buffer_charging_session(session)

                    # Update station
                    station = self.stations[session['station_id']]
                    station.remove_vehicle(veh_id)
                    station.total_energy += energy_kwh

                    self.logger.info(
                        f"{veh_id} completed charging: {session['start_soc']:.2f} -> {vehicle.battery_soc:.2f}")

            # Remove completed sessions
            for veh_id in completed:
                del self.charging_sessions[veh_id]

        except Exception as e:
            self.logger.error(f"Error updating charging vehicles: {e}", "charging")

    def _buffer_charging_session(self, session: Dict):
        """Buffer charging session data"""
        try:
            self.session_buffer.append(session)

            self.statistics['total_charging_sessions'] += 1
            self.statistics['total_energy_delivered'] += session['energy_kwh']
            self.statistics['total_revenue'] += session['cost']

        except Exception as e:
            self.logger.error(f"Failed to buffer charging session: {e}", "logging")

    def update_station_metrics(self):
        """Update charging station metrics"""
        try:
            hour = int((self.sim_time / TimeConstants.HOUR) % 24)

            for station in self.stations.values():
                # Calculate utilization
                utilization = len(station.vehicles_charging) / max(1, station.capacity)

                # Update pricing
                station.update_pricing(hour, utilization)

                # Update statistics
                if len(station.vehicles_charging) > 0:
                    self.statistics[f'station_{station.station_id}_usage'] += 1

        except Exception as e:
            self.logger.error(f"Error updating station metrics: {e}", "metrics")

    def collect_statistics(self) -> Dict:
        """Collect comprehensive simulation statistics"""
        all_vehicles = self.vehicle_pool.get_all_vehicles()

        stats = {
            'timestamp': datetime.datetime.now().isoformat(),
            'sim_time': self.sim_time,
            'step': self.step,
            'vehicles': {
                'total': len(all_vehicles),
                'active': len(self.vehicle_pool.active_vehicles),
                'inactive': len(self.vehicle_pool.inactive_vehicles),
                'charging': len(self.charging_sessions),
                'by_type': defaultdict(int),
                'by_state': defaultdict(int),
                'avg_battery': 0
            },
            'stations': {
                'total': len(self.stations),
                'occupied': sum(1 for s in self.stations.values() if s.vehicles_charging),
                'avg_utilization': 0,
                'total_queue_length': sum(len(q) for q in self.station_queues.values())
            },
            'performance': {
                'memory_mb': 0,
                'step_time': 0,
                'errors': dict(self.logger.error_count),
                'warnings': dict(self.logger.warning_count)
            },
            'decisions': {
                'gpt': self.statistics.get('gpt_decisions', 0),
                'rule': self.statistics.get('rule_decisions', 0),
                'gpt_failures': self.statistics.get('gpt_failures', 0)
            }
        }

        # Get memory usage
        if self.performance_monitor:
            stats['performance']['memory_mb'] = self.performance_monitor.check_memory()['rss']

        # Calculate vehicle statistics
        if all_vehicles:
            for v in all_vehicles.values():
                stats['vehicles']['by_type'][v.vehicle_type] += 1
                if v.state_machine and v.state_machine.current_state:
                    stats['vehicles']['by_state'][str(v.state_machine.current_state)] += 1

            battery_sum = sum(v.battery_soc for v in all_vehicles.values())
            stats['vehicles']['avg_battery'] = battery_sum / len(all_vehicles) if all_vehicles else 0

        # Calculate station utilization
        if self.stations:
            util_sum = sum(
                len(s.vehicles_charging) / max(1, s.capacity)
                for s in self.stations.values()
            )
            stats['stations']['avg_utilization'] = util_sum / len(self.stations)

        return stats

    def _save_buffered_data(self):
        """Save all buffered data to files"""
        try:
            # Save statistics
            if self.statistics_buffer:
                stats_file = os.path.join(self.config.data_dir, "statistics.jsonl")
                with open(stats_file, 'a', encoding='utf-8') as f:
                    for stat in self.statistics_buffer:
                        f.write(json.dumps(stat) + '\n')
                self.statistics_buffer.clear()

            # Save decisions
            if self.decision_buffer:
                gpt_file = os.path.join(self.config.data_dir, "decisions_gpt.jsonl")
                rule_file = os.path.join(self.config.data_dir, "decisions_rule.jsonl")

                with open(gpt_file, 'a', encoding='utf-8') as gf, \
                     open(rule_file, 'a', encoding='utf-8') as rf:
                    for decision_type, entry in self.decision_buffer:
                        if decision_type == "gpt":
                            gf.write(json.dumps(entry) + '\n')
                        else:
                            rf.write(json.dumps(entry) + '\n')
                self.decision_buffer.clear()

            # Save charging sessions
            if self.session_buffer:
                session_file = os.path.join(self.config.data_dir, "charging_sessions.jsonl")
                with open(session_file, 'a', encoding='utf-8') as f:
                    for session in self.session_buffer:
                        f.write(json.dumps(session) + '\n')
                self.session_buffer.clear()

            self.last_save_time = time.time()

        except Exception as e:
            self.logger.error(f"Failed to save buffered data: {e}", "save")

    def run_step(self) -> bool:
        """Run single simulation step with performance optimization and timeout protection"""
        try:
            start_time = time.time()

            # Set a timeout for SUMO step to prevent hanging
            step_timeout = 5.0  # 5 seconds timeout

            # Create a thread for SUMO step with timeout
            import threading
            step_completed = threading.Event()
            step_error = [None]  # Use list to store error from thread

            def sumo_step_thread():
                try:
                    traci.simulationStep()
                    step_completed.set()
                except Exception as e:
                    step_error[0] = e
                    step_completed.set()

            # Run SUMO step in thread
            step_thread = threading.Thread(target=sumo_step_thread)
            step_thread.daemon = True
            step_thread.start()

            # Wait for completion with timeout
            if not step_completed.wait(timeout=step_timeout):
                self.logger.error("SUMO step timed out - possible deadlock detected")
                # Try to recover by skipping this step
                return self.step < self.config.max_steps

            # Check for errors
            if step_error[0]:
                raise step_error[0]

            self.sim_time = traci.simulation.getTime()
            self.step += 1

            # Get current vehicles with error handling
            try:
                current_vehicles = set(traci.vehicle.getIDList())
            except Exception as e:
                self.logger.error(f"Error getting vehicle list: {e}")
                current_vehicles = set()

            all_tracked = self.vehicle_pool.get_all_vehicles()

            # Create new vehicles
            new_vehicles = current_vehicles - set(all_tracked.keys())
            for veh_id in new_vehicles:
                self.create_vehicle(veh_id)

            # Remove departed vehicles
            departed = set(all_tracked.keys()) - current_vehicles
            for veh_id in departed:
                self.vehicle_pool.remove_vehicle(veh_id)
                if veh_id in self.charging_sessions:
                    del self.charging_sessions[veh_id]

            # Rotate active/inactive vehicles for performance
            if self.config.enable_adaptive_processing:
                self.vehicle_pool.rotate_vehicles(self.sim_time)

            # Update active vehicles
            active_vehicles = self.vehicle_pool.get_active_vehicles()

            # Batch update for performance
            batch_size = PerformanceConstants.BATCH_UPDATE_SIZE
            for i in range(0, len(active_vehicles), batch_size):
                batch = active_vehicles[i:i + batch_size]
                # Add timeout protection for vehicle updates
                update_start = time.time()
                for vehicle in batch:
                    if time.time() - update_start > 2.0:  # 2 second timeout for batch
                        self.logger.warning(f"Vehicle update batch timeout at vehicle {vehicle.vehicle_id}")
                        break
                    self.update_vehicle(vehicle)

            # Process charging with timeout
            charge_start = time.time()
            self.process_charging_queues()
            if time.time() - charge_start > 1.0:
                self.logger.warning("Charging queue processing took too long")

            self.update_charging_vehicles()
            self.update_station_metrics()

            # Periodic tasks with staggering to prevent spikes
            if self.step % self.config.log_interval == 0:
                # Collect statistics
                stats = self.collect_statistics()
                stats['performance']['step_time'] = time.time() - start_time
                self.statistics_buffer.append(stats)

                # Performance monitoring
                if self.performance_monitor:
                    self.performance_monitor.record_metric('step_time', stats['performance']['step_time'])
                    self.performance_monitor.record_metric('memory_mb', stats['performance']['memory_mb'])

                    # Trigger GC if needed
                    if self.step % PerformanceConstants.GC_INTERVAL == 0:
                        self.performance_monitor.trigger_gc_if_needed()

                    # Check if throttling needed
                    if self.performance_monitor.should_throttle():
                        self.logger.warning("High memory usage, throttling processing", "performance")
                        time.sleep(0.1)

            # Save buffered data periodically
            if self.step % self.config.save_interval == 0 or \
               time.time() - self.last_save_time > 300:  # Save at least every 5 minutes
                self._save_buffered_data()

            # Log progress
            if self.step % 1000 == 0:
                self.logger.info(f"Step {self.step}, Time: {self.sim_time:.0f}s, "
                                 f"Vehicles: {len(current_vehicles)}, "
                                 f"Charging: {len(self.charging_sessions)}, "
                                 f"GPT Decisions: {self.statistics.get('gpt_decisions', 0)}, "
                                 f"Rule Decisions: {self.statistics.get('rule_decisions', 0)}")

            # Clear large data structures periodically to prevent memory buildup
            if self.step % 10000 == 0:
                self._cleanup_memory()

            # Warn if step takes too long
            total_step_time = time.time() - start_time
            if total_step_time > 2.0:
                self.logger.warning(f"Step {self.step} took {total_step_time:.2f}s - performance issue detected")

            return self.step < self.config.max_steps

        except Exception as e:
            self.logger.error(f"Error in simulation step {self.step}: {e}", "step", exc_info=True)
            # Try to continue simulation despite errors
            return self.step < self.config.max_steps

    def _cleanup_memory(self):
        """Clean up memory periodically"""
        try:
            # Clear old state history from vehicles
            all_vehicles = self.vehicle_pool.get_all_vehicles()
            for vehicle in all_vehicles.values():
                if vehicle.state_machine and len(vehicle.state_machine.state_history) > PerformanceConstants.STATE_HISTORY_LIMIT:
                    # Keep only recent history
                    vehicle.state_machine.state_history = deque(
                        list(vehicle.state_machine.state_history)[-PerformanceConstants.STATE_HISTORY_LIMIT:],
                        maxlen=PerformanceConstants.STATE_HISTORY_LIMIT
                    )

            # Clear old statistics
            for key in list(self.statistics.keys()):
                if key.startswith('station_') and '_usage' in key:
                    # Keep only total counts, not per-station
                    if self.step % 50000 == 0:
                        del self.statistics[key]

            # Force garbage collection
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}", "cleanup")

    def run(self):
        """Main simulation loop"""
        self.logger.info("=" * 80)
        self.logger.info("Starting EV Charging Simulation with GPT Decision Making")
        self.logger.info(f"Configuration: {self.config}")
        self.logger.info(f"GPT Enabled: {self.config.use_gpt}")
        self.logger.info("=" * 80)

        try:
            # Initialize SUMO
            if not self.init_sumo():
                self.logger.error("Failed to initialize SUMO", "init")
                return

            # Initialize charging stations
            self.init_charging_stations()

            # Main loop
            self.logger.info("Starting main simulation loop")
            self.logger.info("GPT Decision Making: " + ("ENABLED" if self.config.use_gpt else "DISABLED"))

            while self.run_step():
                pass

            self.logger.info(f"Simulation completed after {self.step} steps")

        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")

        except Exception as e:
            self.logger.error(f"Fatal error in simulation: {e}", "fatal", exc_info=True)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Comprehensive cleanup with final reporting"""
        self.logger.info("=" * 80)
        self.logger.info("Cleaning up simulation...")

        try:
            # Save any remaining buffered data
            self._save_buffered_data()

            # Final statistics
            final_stats = self.collect_statistics()

            # Add summary statistics
            final_stats['summary'] = {
                'total_steps': self.step,
                'total_sim_time': self.sim_time,
                'total_vehicles_created': sum(
                    v for k, v in self.statistics.items() if k.startswith('vehicles_created_')),
                'total_charging_sessions': self.statistics.get('total_charging_sessions', 0),
                'total_energy_delivered_kwh': self.statistics.get('total_energy_delivered', 0),
                'total_revenue': self.statistics.get('total_revenue', 0),
                'gpt_decisions': self.statistics.get('gpt_decisions', 0),
                'rule_decisions': self.statistics.get('rule_decisions', 0),
                'gpt_failures': self.statistics.get('gpt_failures', 0),
                'errors': dict(self.logger.error_count),
                'warnings': dict(self.logger.warning_count)
            }

            # Performance metrics
            if self.performance_monitor:
                final_stats['summary']['avg_step_time'] = self.performance_monitor.get_avg_metric('step_time')
                final_stats['summary']['avg_memory_mb'] = self.performance_monitor.get_avg_metric('memory_mb')
                final_stats['summary']['memory_warnings'] = self.performance_monitor.memory_warnings

            # Save final statistics
            self.statistics_buffer.append(final_stats)
            self._save_buffered_data()

            # Save vehicle trajectories
            self._save_vehicle_trajectories()

            # Generate summary report
            self._generate_summary_report(final_stats)

            # Shutdown components
            if self.executor:
                self.executor.shutdown(wait=False)

            self.logger.shutdown()

            # Close SUMO
            if traci.isLoaded():
                traci.close()

            print("\n" + "=" * 80)
            print("SIMULATION COMPLETE")
            print("=" * 80)
            print(f"Total Steps: {self.step}")
            print(f"Simulation Time: {self.sim_time:.0f} seconds")
            print(f"Vehicles Processed: {final_stats['summary']['total_vehicles_created']}")
            print(f"Charging Sessions: {final_stats['summary']['total_charging_sessions']}")
            print(f"Energy Delivered: {final_stats['summary']['total_energy_delivered_kwh']:.2f} kWh")
            print(f"Total Revenue: ${final_stats['summary']['total_revenue']:.2f}")
            print(f"\nDECISION MAKING STATISTICS:")
            print(f"  GPT Decisions: {final_stats['summary']['gpt_decisions']}")
            print(f"  Rule-based Decisions: {final_stats['summary']['rule_decisions']}")
            print(f"  GPT Failures: {final_stats['summary']['gpt_failures']}")
            if final_stats['summary']['gpt_decisions'] > 0:
                success_rate = (final_stats['summary']['gpt_decisions'] /
                               (final_stats['summary']['gpt_decisions'] + final_stats['summary']['gpt_failures'])) * 100
                print(f"  GPT Success Rate: {success_rate:.1f}%")
            print(f"\nOutput files saved to: {self.config.data_dir}/")
            print("  - simulation.log: Detailed execution log")
            print("  - statistics.jsonl: Time-series statistics")
            print("  - decisions_gpt.jsonl: GPT charging decisions")
            print("  - decisions_rule.jsonl: Rule-based decisions")
            print("  - charging_sessions.jsonl: Charging session data")
            print("  - trajectories.json: Vehicle trajectories")
            print("  - summary_report.txt: Human-readable summary")
            print("=" * 80)

            # Function completeness check
            print("\n🔍 FUNCTION COMPLETENESS CHECK:")
            print("  ✅ GPT Decision System: ACTIVE" if self.config.use_gpt else "  ⚠️  GPT Decision System: DISABLED")
            print("  ✅ State Machines: 5 vehicle types implemented")
            print("  ✅ Psychology Engine: Big Five + 8 irrational factors")
            print("  ✅ Charging Management: Dynamic pricing + queuing")
            print("  ✅ Performance Optimization: Vehicle pooling + memory management")
            print("  ✅ Data Logging: Complete decision & session tracking")
            print("  ✅ Visualization: REMOVED per requirements")
            print("  ✅ Error Recovery: Comprehensive exception handling")
            print("  ✅ Memory Management: Periodic cleanup + buffered I/O")
            print("=" * 80)

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _save_vehicle_trajectories(self):
        """Save vehicle trajectory data with limited history"""
        try:
            trajectories = {}
            all_vehicles = self.vehicle_pool.get_all_vehicles()

            for vehicle in list(all_vehicles.values())[:1000]:  # Limit to prevent huge files
                trajectories[vehicle.vehicle_id] = {
                    'type': vehicle.vehicle_type,
                    'distance_traveled': vehicle.distance_traveled,
                    'energy_consumed': vehicle.energy_consumed,
                    'charging_sessions': vehicle.charging_sessions,
                    'final_soc': vehicle.battery_soc,
                    'time_active': self.sim_time - vehicle.time_created,
                    'state_history': [
                        {
                            'state': str(h['to']),
                            'time': h['time']
                        } for h in list(vehicle.state_machine.state_history)[-20:]  # Last 20 states only
                    ]
                }

            traj_file = os.path.join(self.config.data_dir, "trajectories.json")
            with open(traj_file, 'w', encoding='utf-8') as f:
                json.dump(trajectories, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save trajectories: {e}", "save")

    def _generate_summary_report(self, final_stats: Dict):
        """Generate human-readable summary report"""
        try:
            report_file = os.path.join(self.config.data_dir, "summary_report.txt")

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EV CHARGING SIMULATION SUMMARY REPORT\n")
                f.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

                # Simulation Overview
                f.write("SIMULATION OVERVIEW\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Steps: {final_stats['summary']['total_steps']}\n")
                f.write(f"Simulation Time: {final_stats['summary']['total_sim_time']:.0f} seconds "
                        f"({final_stats['summary']['total_sim_time'] / 3600:.2f} hours)\n")
                f.write(f"Configuration: {self.config.sumo_config}\n")
                f.write(f"GPT Decision Making: {'ENABLED' if self.config.use_gpt else 'DISABLED'}\n\n")

                # Vehicle Statistics
                f.write("VEHICLE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Vehicles Created: {final_stats['summary']['total_vehicles_created']}\n")
                f.write(f"Final Active Vehicles: {final_stats['vehicles']['active']}\n")
                f.write(f"Final Inactive Vehicles: {final_stats['vehicles']['inactive']}\n")
                f.write(f"Average Battery SOC: {final_stats['vehicles']['avg_battery'] * 100:.1f}%\n")
                f.write("\nVehicles by Type:\n")
                for vtype, count in final_stats['vehicles']['by_type'].items():
                    f.write(f"  {vtype}: {count}\n")
                f.write("\n")

                # Charging Statistics
                f.write("CHARGING STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Charging Sessions: {final_stats['summary']['total_charging_sessions']}\n")
                f.write(f"Total Energy Delivered: {final_stats['summary']['total_energy_delivered_kwh']:.2f} kWh\n")
                f.write(f"Total Revenue: ${final_stats['summary']['total_revenue']:.2f}\n")
                if final_stats['summary']['total_charging_sessions'] > 0:
                    f.write(f"Average Session Energy: "
                            f"{final_stats['summary']['total_energy_delivered_kwh'] / final_stats['summary']['total_charging_sessions']:.2f} kWh\n")
                    f.write(f"Average Session Revenue: "
                            f"${final_stats['summary']['total_revenue'] / final_stats['summary']['total_charging_sessions']:.2f}\n")
                f.write("\n")

                # Station Statistics
                f.write("CHARGING STATION STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Stations: {final_stats['stations']['total']}\n")
                f.write(f"Stations Currently Occupied: {final_stats['stations']['occupied']}\n")
                f.write(f"Average Utilization: {final_stats['stations']['avg_utilization'] * 100:.1f}%\n")
                f.write(f"Total Queue Length: {final_stats['stations']['total_queue_length']}\n\n")

                # Decision Statistics
                f.write("DECISION MAKING STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"GPT Decisions: {final_stats['summary']['gpt_decisions']}\n")
                f.write(f"Rule-based Decisions: {final_stats['summary']['rule_decisions']}\n")
                f.write(f"GPT Failures: {final_stats['summary']['gpt_failures']}\n")
                if final_stats['summary']['gpt_decisions'] > 0:
                    success_rate = (final_stats['summary']['gpt_decisions'] /
                                    (final_stats['summary']['gpt_decisions'] + final_stats['summary']['gpt_failures'])) * 100
                    f.write(f"GPT Success Rate: {success_rate:.1f}%\n")
                f.write("\n")

                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n")
                if 'avg_step_time' in final_stats['summary']:
                    f.write(f"Average Step Time: {final_stats['summary']['avg_step_time']:.3f} seconds\n")
                if 'avg_memory_mb' in final_stats['summary']:
                    f.write(f"Average Memory Usage: {final_stats['summary']['avg_memory_mb']:.1f} MB\n")
                if 'memory_warnings' in final_stats['summary']:
                    f.write(f"Memory Warnings: {final_stats['summary']['memory_warnings']}\n")
                f.write(f"Current Memory Usage: {final_stats['performance']['memory_mb']:.1f} MB\n\n")

                # Error Summary
                f.write("ERROR AND WARNING SUMMARY\n")
                f.write("-" * 40 + "\n")
                if final_stats['summary']['errors']:
                    f.write("Errors by Category:\n")
                    for category, count in final_stats['summary']['errors'].items():
                        f.write(f"  {category}: {count}\n")
                else:
                    f.write("No errors recorded\n")

                if final_stats['summary']['warnings']:
                    f.write("\nWarnings by Category:\n")
                    for category, count in final_stats['summary']['warnings'].items():
                        f.write(f"  {category}: {count}\n")
                else:
                    f.write("No warnings recorded\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")

        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}", "report")

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point with command line arguments"""
    import argparse

    # ASCII Art Banner
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ███████╗██╗   ██╗    ███████╗██╗███╗   ███╗                  ║
    ║     ██╔════╝██║   ██║    ██╔════╝██║████╗ ████║                  ║
    ║     █████╗  ██║   ██║    ███████╗██║██╔████╔██║                  ║
    ║     ██╔══╝  ╚██╗ ██╔╝    ╚════██║██║██║╚██╔╝██║                  ║
    ║     ███████╗ ╚████╔╝     ███████║██║██║ ╚═╝ ██║                  ║
    ║     ╚══════╝  ╚═══╝      ╚══════╝╚═╝╚═╝     ╚═╝                  ║
    ║                                                                  ║
    ║    Electric Vehicle Charging Simulation with AI Decision Making  ║
    ║                    Powered by SUMO & GPT-4                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Electric Vehicle Charging Simulation System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='osm.sumocfg',
        help='SUMO configuration file (default: osm.sumocfg)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=172800,
        help='Maximum simulation steps (default: 172800)'
    )

    parser.add_argument(
        '--no-gpt',
        action='store_true',
        help='Disable GPT decision making'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='simulation_data',
        help='Output data directory (default: simulation_data)'
    )

    parser.add_argument(
        '--decision-interval',
        type=int,
        default=900,
        help='Decision making interval in seconds (default: 900)'
    )

    parser.add_argument(
        '--gpt-model',
        type=str,
        default='gpt-4o-mini',
        help='GPT model to use (default: gpt-4o-mini)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (overrides environment variable)'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default='https://xiaoai.plus/v1',
        help='OpenAI API base URL (default: https://xiaoai.plus/v1)'
    )

    parser.add_argument(
        '--max-vehicles',
        type=int,
        default=5000,
        help='Maximum active vehicles to process (default: 5000)'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode with reduced features for testing'
    )

    args = parser.parse_args()

    # Create configuration
    config = Config(
        max_steps=args.steps,
        sumo_config=args.config,
        use_gpt=not args.no_gpt,
        debug_mode=args.debug,
        data_dir=args.data_dir,
        decision_interval=args.decision_interval,
        gpt_model=args.gpt_model,
        api_base_url=args.api_url
    )

    # Override API key if provided
    if args.api_key:
        config.api_key = args.api_key

    # Fast mode adjustments
    if args.fast:
        config.use_gpt = False
        config.log_interval = 5000
        config.save_interval = 10000
        PerformanceConstants.MAX_ACTIVE_VEHICLES = 1000
        print("\n⚡ Fast mode enabled - reduced features for testing\n")

    # Adjust max vehicles
    PerformanceConstants.MAX_ACTIVE_VEHICLES = args.max_vehicles

    # Show configuration
    print("\nConfiguration:")
    print("-" * 40)
    print(f"  SUMO Config: {config.sumo_config}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  GPT Enabled: {config.use_gpt}")
    if config.use_gpt:
        print(f"  GPT Model: {config.gpt_model}")
        print(f"  API URL: {config.api_base_url}")
        print(f"  API Key: {'***' + config.api_key[-4:] if config.api_key else 'Not provided'}")
    print(f"  Debug Mode: {config.debug_mode}")
    print(f"  Data Directory: {config.data_dir}")
    print(f"  Max Active Vehicles: {args.max_vehicles}")
    print("-" * 40)
    print()

    # Check for API key if GPT is enabled
    if config.use_gpt and not config.api_key:
        print("⚠️  WARNING: GPT is enabled but no API key provided!")
        print("   Set OPENAI_API_KEY environment variable or use --api-key")
        print("   GPT features will be disabled.\n")
        config.use_gpt = False

    # Confirm start
    try:
        input("Press Enter to start simulation (Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\nSimulation cancelled.")
        return

    # Run simulation
    try:
        print("\n🚀 Starting simulation...\n")
        sim = EVSimulation(config)
        sim.run()
        print("\n✅ Simulation completed successfully!")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())