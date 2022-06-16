""" Provides constant parameters for the wildfire domain.

Author: Adam Eck
"""
import numpy as np

cimport numpy as np


cdef class WildfireSettings():
    """Stores the constant parameters for the wildfire domain.

    Attributes:
    SETUP -- the specific grid setup to use
    REASONING_TYPE -- the WildfireReasoningType that each agent should follow
    N_CENTRALIZED_MODELING_GROUPS -- the number of groups of neighbors to be sampled by a centralized agent,
        determining who models whom for the agents' reasoning
    N_AGENTS -- total number of agents
    SUPPRESSANT_STATES -- number of possible states of each agent's suppressants
    INITIAL_SUPPRESSANT_DISTRIBUTION -- the probabilities that an arbitrary agent starts with each suppressant level
    FRAMES -- number of agent frames
    AGENT_LOCATIONS -- the (x, y) coordinates of each location where agents reside in the grid
    N_AGENTS_PER_FRAME -- the number of agents belonging to each frame
    N_AGENTS_PER_FRAME_PER_LOCATION -- the number of agents of each frame in each location, indexed first by frame,
        second by location number
    FIRE_STATES -- number of possible states of each fire
    FIRES -- number of fires
    FIRE_LOCATIONS -- the (x, y) coordinates of each location where agents reside in the grid (indexed in the same
        order as FIRE_SIZES)
    FIRE_SIZES -- the sizes of each fire (indexed in the same order as FIRE_LOCATIONS)
    FIRE_POWER_NEEDED -- the amount of joint suppressant needed to start reducing each of the fire sizes (indexed by
        fire size)
    BASE_FIRE_REDUCTION -- smallest amount of fire reduction by the simplest agent frame
    FRAME_POWERS -- the amount of suppressant each frame contributes for putting out fires (indexed by frame)
    DISCHARGE_PROB -- the probability that an agent's suppressant decreases when it fights a fire
    RECHARGE_PROB -- the probability that an agent's suppressant recharges when it is empty
    BASE_SPREAD_RATE -- the base rate (in meters per turn) that a fire spreads to a nearby location
    MAX_SPREAD_RATE -- the fastest rate (in meters per turn) that a fire can spread
    RANDOM_SPREAD_PROB -- the probability that a new fire starts (before calculating spread)
    CELL_SIZE -- the size (in meters) of each cell
    WIND_DIRECTION -- the direction of the wind to use for the simulation (in radians)
    NORTH_IGNITION_PROB -- the probability that a fire spreads to the north
    EAST_IGNITION_PROB -- the probability that a fire spreads to the east
    SOUTH_IGNITION_PROB -- the probability that a fire spreads to the south
    WEST_IGNITION_PROB -- the probability the a fire spreads to the west
    BURNOUT_PROB -- the probability that a fire burns out when it has the highest active fire state
    FIRE_REDUCTION_PROB_PER_EXTRA_AGENT -- the additional probability of fire reduction contributed by each
        BASE_FIRE_REDUCTION of suppressant beyond the amount required (in FIRE_POWER_NEEDED)
    NO_FIRE_PENALTY -- the penalty received for fighting a non-existant fire (or fighting a fire with no suppressant)
    NO_FIRE_REWARDS -- the rewards received by all agents when a fire is put out (indexed by fire size)
    FIRE_CONTRIBUTION_BONUS -- the reward bonus an agent receives for acting on a fire the turn it is put out (currently
        not used)
    NOOP_WITH_SUPPRESSANT_PENALTY -- the penalty for taking a NOOP when the agent has suppressant (currently not used)
    FIRE_BURNOUT_PENALTY -- the penalty each agent receives when a fire burns out
    NO_OBS -- the value to use for the "none" observation
    OBSERVATION_ERROR -- the probability that an incorrect observation is made
    """
    def __cinit__(self, setup=None, reasoning_type=WildfireReasoningType.NOOP):
        """Constructs a WildfireSettings instance."""
        # save the parameters
        if setup is None:
            self.SETUP = 100
        else:
            self.SETUP = setup
        self.REASONING_TYPE = reasoning_type

        if (self.REASONING_TYPE == WildfireReasoningType.POMCPPFComm or
                self.REASONING_TYPE == WildfireReasoningType.IPOMCPPFComm):
            self.COMMUNICATION = True
        else:
            self.COMMUNICATION = False

        # common values for all setups
        self.FIRE_STATES = 5

        self.INITIAL_SUPPRESSANT_DISTRIBUTION = np.array([0.33, 0.33, 0.33], dtype=np.double)
        self.DISCHARGE_PROB = 1.0 / 3 # average of 3 time steps per suppressant level
        self.RECHARGE_PROB = 1.0 / 2  # average of 2 time steps

        self.BASE_SPREAD_RATE = 3.0
        self.MAX_SPREAD_RATE = 67.0
        self.RANDOM_SPREAD_PROB = 0.05
        self.CELL_SIZE = 200.0
        self.WIND_DIRECTION = 0.25 * np.pi # 0 is N, pi is S, 0.5pi is E, 1.5pi is W
        self.NORTH_IGNITION_PROB =  self.BASE_SPREAD_RATE \
                                    / (self.CELL_SIZE * (1 - np.cos(0 - self.WIND_DIRECTION)
                                                         * (1 - self.BASE_SPREAD_RATE / self.MAX_SPREAD_RATE)))
        self.EAST_IGNITION_PROB = self.BASE_SPREAD_RATE \
                                  / (self.CELL_SIZE * (1 - np.cos(0.5 * np.pi - self.WIND_DIRECTION)
                                                       * (1 - self.BASE_SPREAD_RATE / self.MAX_SPREAD_RATE)))
        self.SOUTH_IGNITION_PROB = self.BASE_SPREAD_RATE \
                                   / (self.CELL_SIZE * (1 - np.cos(np.pi - self.WIND_DIRECTION)
                                                        * (1 - self.BASE_SPREAD_RATE / self.MAX_SPREAD_RATE)))
        self.WEST_IGNITION_PROB = self.BASE_SPREAD_RATE \
                                  / (self.CELL_SIZE * (1 - np.cos(1.5 * np.pi - self.WIND_DIRECTION)
                                                       * (1 - self.BASE_SPREAD_RATE / self.MAX_SPREAD_RATE)))
        self.BURNOUT_PROB = 4 * 0.167 * self.MAX_SPREAD_RATE / self.CELL_SIZE

        # print("N:", self.NORTH_IGNITION_PROB, "E:", self.EAST_IGNITION_PROB, "S:", self.SOUTH_IGNITION_PROB,
        #       "W:", self.WEST_IGNITION_PROB, "Start:", self.RANDOM_SPREAD_PROB, "Burn:", self.BURNOUT_PROB)

        self.BASE_FIRE_REDUCTION = 1.0
        self.FIRE_REDUCTION_PROB_PER_EXTRA_AGENT = 0.06
        self.FIRE_POWER_NEEDED = np.array([10.0, 20.0, 30.0], dtype=np.double)
        self.FIRE_POWER_NEEDED_PROBABILITY = 0.6 # likelihood fire decreases if FIRE_POWER_NEEDED is met

        self.NO_FIRE_PENALTY = -100.0
        self.NO_FIRE_REWARDS = np.array([20.0, 45.0, 100.0], dtype=np.double)
        self.FIRE_CONTRIBUTION_BONUS = 0.0
        self.NOOP_WITH_SUPPRESSANT_PENALTY = 0.0
        self.FIRE_BURNOUT_PENALTY = -1.0

        self.NO_OBS = 3 # since the largest change is from 0 to 2
        self.OBSERVATION_ERROR = 0.1

        # Communication parameters
        self.COMMUNICATION_COST = -0.05
        self.HONEST_COMM_PROB = 0.75
        self.HONEST_COMM_REWARD = 0.0

        # set the values depending on the chosen setup
        if self.SETUP >= 100 and self.SETUP < 200:
            self.FRAMES = 2
            self.FRAME_POWERS = np.array([1, 1], dtype=np.double)
            self.AGENT_LOCATIONS = np.array([np.array([0, 0]),
                                             np.array([2, 0])], dtype=np.intc)
            self.FRAME_LOCATIONS = np.array([0, 1], dtype=np.intc)
            self.N_AGENTS_PER_FRAME = np.array([1, 1], dtype=np.intc)

            self.FIRE_LOCATIONS = np.array([np.array([0, 1]),
                                            np.array([1, 1]),
                                            np.array([2, 1])], dtype=np.intc)
            self.FIRE_SIZES = np.array([0, 1, 0], dtype=np.intc)
            self.FIRE_POWER_NEEDED = np.array([1.0, 2.0, 3.0], dtype=np.double)
            self.FIRE_POWER_NEEDED_PROBABILITY = 0.85 # likelihood fire decreases if FIRE_POWER_NEEDED is met
            self.FIRE_REDUCTION_PROB_PER_EXTRA_AGENT = 0.15 * self.BASE_FIRE_REDUCTION

            self.INITIAL_SUPPRESSANT_DISTRIBUTION = np.array([0, 0, 1.0], dtype=np.double)
            self.DISCHARGE_PROB = 1.0 / 4 # average of 4 time steps per suppressant level

            self.NO_FIRE_REWARDS = np.array([20.0, 50.0, 125.0], dtype=np.double)

            self.OBSERVATION_ERROR = 0.0
            self.HONEST_COMM_PROB = 0.95

            comm_version = self.SETUP % 10
            if comm_version == 0:
                self.COMMUNICATION_COST = 0.0
            elif comm_version == 1:
                self.COMMUNICATION_COST = -0.05
            elif comm_version == 2:
                self.COMMUNICATION_COST = -0.1
            elif comm_version == 3:
                self.COMMUNICATION_COST = -0.2
            elif comm_version == 4:
                self.COMMUNICATION_COST = -0.5
            elif comm_version == 5:
                self.COMMUNICATION_COST = -1.0
            

        elif self.SETUP >= 200 and self.SETUP < 300:
            self.FRAMES = 3
            self.FRAME_POWERS = np.array([1, 1, 1], dtype=np.double)
            self.AGENT_LOCATIONS = np.array([np.array([0, 0]),
                                             np.array([1, 0]),
                                             np.array([2, 0])], dtype=np.intc)
            self.FRAME_LOCATIONS = np.array([0, 1, 2], dtype=np.intc)
            self.N_AGENTS_PER_FRAME = np.array([1, 1, 1], dtype=np.intc)

            self.FIRE_LOCATIONS = np.array([np.array([0, 1]),
                                            np.array([1, 1]),
                                            np.array([2, 1])], dtype=np.intc)
            self.FIRE_SIZES = np.array([1, 2, 1], dtype=np.intc)
            self.FIRE_POWER_NEEDED = np.array([1.0, 2.0, 3.0], dtype=np.double)
            self.FIRE_POWER_NEEDED_PROBABILITY = 0.85 # likelihood fire decreases if FIRE_POWER_NEEDED is met
            self.FIRE_REDUCTION_PROB_PER_EXTRA_AGENT = 0.15 * self.BASE_FIRE_REDUCTION

            self.INITIAL_SUPPRESSANT_DISTRIBUTION = np.array([0, 0, 1.0], dtype=np.double)
            self.DISCHARGE_PROB = 1.0 / 4 # average of 4 time steps per suppressant level

            self.NO_FIRE_REWARDS = np.array([20.0, 50.0, 125.0], dtype=np.double)

            self.OBSERVATION_ERROR = 0.0
            self.HONEST_COMM_PROB = 0.95

            comm_version = self.SETUP % 10
            if comm_version == 0:
                self.COMMUNICATION_COST = 0.0
            elif comm_version == 1:
                self.COMMUNICATION_COST = -0.05
            elif comm_version == 2:
                self.COMMUNICATION_COST = -0.1
            elif comm_version == 3:
                self.COMMUNICATION_COST = -0.2
            elif comm_version == 4:
                self.COMMUNICATION_COST = -0.5
            elif comm_version == 5:
                self.COMMUNICATION_COST = -1.0

        elif self.SETUP >= 300 and self.SETUP < 400:
            self.FRAMES = 4
            self.FRAME_POWERS = self.BASE_FIRE_REDUCTION * np.array([1, 2, 1, 2], dtype=np.double)
            self.AGENT_LOCATIONS = np.array([np.array([0, 1]),
                                             np.array([2, 1])], dtype=np.intc)
            self.FRAME_LOCATIONS = np.array([0, 0, 1, 1], dtype=np.intc)
            self.N_AGENTS_PER_FRAME = np.array([1, 1, 1, 1], dtype=np.intc)

            self.FIRE_LOCATIONS = np.array([np.array([0, 0]),
                                            np.array([1, 1]),
                                            np.array([2, 0])], dtype=np.intc)
            self.FIRE_SIZES = np.array([1, 2, 1], dtype=np.intc)
            self.FIRE_POWER_NEEDED = np.array([2.0, 3.0, 5.0], dtype=np.double)
            self.FIRE_POWER_NEEDED_PROBABILITY = 0.85 # likelihood fire decreases if FIRE_POWER_NEEDED is met
            self.FIRE_REDUCTION_PROB_PER_EXTRA_AGENT = 0.15 * self.BASE_FIRE_REDUCTION

            self.INITIAL_SUPPRESSANT_DISTRIBUTION = np.array([0, 0, 1.0], dtype=np.double)
            self.DISCHARGE_PROB = 1.0 / 4 # average of 4 time steps per suppressant level

            self.NO_FIRE_REWARDS = np.array([50.0, 125.0, 300.0], dtype=np.double)

            self.OBSERVATION_ERROR = 0.0
            self.HONEST_COMM_PROB = 0.95

            comm_version = self.SETUP % 10
            if comm_version == 0:
                self.COMMUNICATION_COST = 0.0
            elif comm_version == 1:
                self.COMMUNICATION_COST = -0.05
            elif comm_version == 2:
                self.COMMUNICATION_COST = -0.1
            elif comm_version == 3:
                self.COMMUNICATION_COST = -0.2
            elif comm_version == 4:
                self.COMMUNICATION_COST = -0.5
            elif comm_version == 5:
                self.COMMUNICATION_COST = -1.0            
            

        # sum regardless of configuration
        self.SUPPRESSANT_STATES = len(self.INITIAL_SUPPRESSANT_DISTRIBUTION)
        self.FIRES = len(self.FIRE_SIZES)
        self.N_AGENTS = sum(self.N_AGENTS_PER_FRAME)


