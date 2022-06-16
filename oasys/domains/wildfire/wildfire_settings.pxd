from .wildfire_reasoning import WildfireReasoningType

from .wildfire_reasoning cimport WildfireReasoningType


cdef class WildfireSettings():
    cdef readonly int SETUP
    cdef readonly WildfireReasoningType REASONING_TYPE
    cdef readonly bint COMMUNICATION

    cdef readonly int N_AGENTS
    cdef readonly int SUPPRESSANT_STATES
    cdef readonly double[:] INITIAL_SUPPRESSANT_DISTRIBUTION
    cdef readonly int FRAMES
    cdef readonly int[:,:] AGENT_LOCATIONS
    cdef readonly int[:] FRAME_LOCATIONS
    cdef readonly int[:] N_AGENTS_PER_FRAME
    cdef readonly int FIRE_STATES
    cdef readonly int FIRES
    cdef readonly int[:,:] FIRE_LOCATIONS
    cdef readonly int[:] FIRE_SIZES
    cdef readonly double[:] FIRE_POWER_NEEDED
    cdef readonly double FIRE_POWER_NEEDED_PROBABILITY
    cdef readonly double BASE_FIRE_REDUCTION
    cdef readonly double[:] FRAME_POWERS

    cdef readonly double DISCHARGE_PROB
    cdef readonly double RECHARGE_PROB

    cdef readonly double BASE_SPREAD_RATE
    cdef readonly double MAX_SPREAD_RATE
    cdef readonly double RANDOM_SPREAD_PROB
    cdef readonly double CELL_SIZE
    cdef readonly double WIND_DIRECTION
    cdef readonly double NORTH_IGNITION_PROB
    cdef readonly double EAST_IGNITION_PROB
    cdef readonly double SOUTH_IGNITION_PROB
    cdef readonly double WEST_IGNITION_PROB
    cdef readonly double BURNOUT_PROB

    cdef readonly double FIRE_REDUCTION_PROB_PER_EXTRA_AGENT

    cdef readonly double NO_FIRE_PENALTY
    cdef readonly double[:] NO_FIRE_REWARDS
    cdef readonly double FIRE_CONTRIBUTION_BONUS
    cdef readonly double NOOP_WITH_SUPPRESSANT_PENALTY
    cdef readonly double FIRE_BURNOUT_PENALTY
    cdef readonly double COMMUNICATION_COST

    cdef readonly int NO_OBS
    cdef readonly double OBSERVATION_ERROR
    cdef readonly double HONEST_COMM_PROB
    cdef readonly double HONEST_COMM_REWARD