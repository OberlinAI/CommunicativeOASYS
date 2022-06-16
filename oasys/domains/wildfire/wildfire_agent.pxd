from oasys.agents.agent import Agent
from oasys.structures.pomdp_structures import Observation, PartialInternalStates, Action
from .wildfire_reasoning import WildfireReasoningType
from .wildfire_structures import WildfireState

from oasys.agents.agent cimport Agent
from oasys.structures.pomdp_structures cimport Observation, PartialInternalStates, Action
from .wildfire_reasoning import WildfireReasoningType
from .wildfire_reasoning cimport WildfireReasoningType
from .wildfire_structures cimport WildfireState


cdef class WildfireAgent(Agent):
    cdef Observation last_observation
    cdef WildfireState current_state
    cdef int current_suppressant
    cdef WildfireReasoningType reasoning_type
    cpdef Action choose_action(self, int[:] message_vector=*)
    cpdef list calculate_q_values(self, int[:] message_vector=*)
    cpdef void set_current_state(self, WildfireState state)
    cpdef void set_current_suppressant(self, int suppressant)

    cdef list sample_agents_for_modeling(self, double epsilon_p, list available_agents,
                                         WildfireReasoningType neighbor_reasoning_type, bint use_models=*)
    cdef list sample_all_agents(self, WildfireReasoningType neighbor_reasoning_type, bint use_models=*)
    cdef void sample_agents_from_list(self, list agents, int n, list chosen_ids)
    cdef bint enough_agents_for_modeling(self, double epsilon_p, list available_agents)
    cdef list organize_into_neighborhoods(self, list agents)
    cpdef list organize_into_neighborhoods_small(self, list agents, int required_neighborhoods)
    cdef list sample_agents_for_modeling_small(self, list available_agents, WildfireReasoningType neighbor_reasoning_type, int required_neighborhoods, bint use_models=*)
