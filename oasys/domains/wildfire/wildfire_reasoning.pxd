from oasys.agents.agent import Agent
from oasys.planning.pomcppf import POMCPPFPlanner
from oasys.planning.pomcppfcomm import POMCPPFCommPlanner, MessageEstimator
from oasys.planning.ipomcppf import IPOMCPPFPlanner
from oasys.planning.ipomcppfcomm import IPOMCPPFCommPlanner
from oasys.structures.pomdp_structures import Action, State, Observation, PartialInternalStates, PartialJointActionComm
from oasys.structures.planning_structures import Particle, NestedParticle, ParticleFilter
import numpy as np
import cython

from oasys.agents.agent cimport Agent
from oasys.planning.pomcppf cimport POMCPPFPlanner
from oasys.planning.pomcppfcomm cimport POMCPPFCommPlanner, MessageEstimator
from oasys.planning.ipomcppf cimport IPOMCPPFPlanner
from oasys.planning.ipomcppfcomm cimport IPOMCPPFCommPlanner
from oasys.planning.reasoning cimport ReasoningModel
from oasys.structures.pomdp_structures cimport Action, State, Observation, PartialInternalStates, PartialJointActionComm
from oasys.structures.planning_structures cimport Particle, NestedParticle, ParticleFilter
cimport numpy as np
cimport cython
from cython cimport view


cpdef public enum WildfireReasoningType:
    NOOP = 0,
    Heuristic = 1,
    Coordination = 2,
    POMCPPF = 3,
    IPOMCPPF = 4,
    IPOMCPPFComm = 5,
    POMCPPFComm = 6


cdef class WildfirePOMCPPFReasoning(ReasoningModel):
    cdef readonly POMCPPFPlanner planner
    cdef int trajectories
    cdef int particles_n

    cdef list calculate_q_values(self, State state, int internal_state)
    cdef void create_planner(self, list possible_next_states=*, list agent_models=*)
    cdef void refresh_particles(self, State state, int internal_state)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)


cdef class WildfirePOMCPPFCommReasoning(ReasoningModel):
    cdef readonly POMCPPFCommPlanner planner
    cdef int trajectories
    cdef int particles_n

    cpdef Action choose_action_with_messages(self, State state, int internal_state, int[:] message_vector)
    cdef list calculate_q_values_with_messages(self, State state, int internal_state, int[:] message_vector)
    cdef void create_planner(self, list agent_neighbors, list possible_next_states=*, bint in_child_process=*)
    cdef void refresh_particles(self, State state, int internal_state)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)


cdef class WildfireIPOMCPPFReasoning(ReasoningModel):
    cdef readonly IPOMCPPFPlanner planner
    cdef int trajectories
    cdef int particles_n
    cdef int level

    cdef void create_planner(self, list possible_next_states=*, list agent_models=*)
    cdef void refresh_particles(self, State state, int internal_state)
    cdef ParticleFilter refresh_nested_particlefilters(self, State state, int internal_state, int agent_num,
                                                       int[:] model_nums, ParticleFilter existing_pf, int level)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)


cdef class WildfireIPOMCPPFCommReasoning(ReasoningModel):
    cdef readonly IPOMCPPFCommPlanner planner
    cdef int trajectories
    cdef int particles_n
    cdef int level
    cdef int[:] last_messages

    cdef void create_planner(self, list agent_neighbors, list possible_next_states=*, bint in_child_process=*)
    cdef void refresh_particles(self, State state, int internal_state)
    cdef ParticleFilter refresh_nested_particlefilters(self, State state, int internal_state, int agent_num,
                                                       int[:] model_nums, ParticleFilter existing_pf, int level)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)


cdef class WildfireHeuristicReasoning(ReasoningModel):
    pass


cdef class WildfireNOOPReasoning(ReasoningModel):
    pass


cdef class WildfireCoordinationReasoning(ReasoningModel):
    cdef list fire_intensities


cdef class WildfireMessageEstimator(MessageEstimator):
    cdef int setup
    cdef int fires
    cdef double honest_comm_prob

