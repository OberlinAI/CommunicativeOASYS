""" Defines the wildfire domain and its dynamics.

Author: Adam Eck
"""

from oasys.domains.domain import Domain
from oasys.structures.pomdp_structures import Configuration
from .wildfire_settings import WildfireSettings
from .wildfire_structures import WildfireState, WildfireConfiguration, WildfireObservation

from oasys.domains.domain cimport Domain
from oasys.structures.pomdp_structures cimport Configuration
from .wildfire_settings cimport WildfireSettings
from .wildfire_structures cimport WildfireState, WildfireConfiguration, WildfireObservation


cdef class Wildfire(Domain):
    cdef readonly WildfireSettings settings
    cdef readonly double[:,:] spread_probabilities
    cdef readonly double[:,:,:] internal_state_transition_probabilities
    cdef readonly list modeling_neighborhoods
    cdef readonly list modeling_neighborhoods_per_agent

    cdef void prepare_planners(self)
    cdef list centralized_neighbor_sampling(self, double epsilon_p)
    cdef list organize_modeling_neighborhoods_per_agent(self)
    cdef bint can_perform(self, int actionNum, int locNum)
    cdef void create_configuration_helper(self, int index, int summ, int config_n, int agents_n,
                                          int[:] max_configuration, WildfireConfiguration configuration,
                                          list all_configurations)

    cdef double calculate_fire_transition_probability(self, int agent_num, WildfireState state,
                                                      Configuration configuration, int action_index,
                                                      WildfireState next_state)
    cdef double[:,:] calculate_fire_spread_probabilities(self)
    cdef double calculate_internal_state_transition_probability(self, int internal_state, int action_index,
                                                                int next_internal_state)
    cdef double[:,:,:] calculate_internal_state_transition_probabilities(self)
    cdef double calculate_observation_probability(self, WildfireState state, int action_index, WildfireState next_state,
                                                  WildfireObservation observation)
    cdef double reward_single_agent(self, WildfireState state, int internal_state, int action_index, int message,
                                    WildfireState next_state)

    cpdef void validate_transitions(self)
    cpdef void validate_internal_transitions(self)
    cpdef void validate_observations(self)
