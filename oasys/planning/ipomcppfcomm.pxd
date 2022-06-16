from oasys.agents.agent import Agent
from oasys.domains.domain import Domain
from oasys.planning.pomcppfcomm import POMCPPFCommPlanner, MessageEstimator
from oasys.structures.planning_structures import Particle, NestedParticle, ParticleFilter
from oasys.structures.pomdp_structures import State, PartialInternalStates, Action, ActionComm, PartialJointAction, PartialJointActionComm, Configuration, Observation, ObservationComm

from oasys.agents.agent cimport Agent
from oasys.domains.domain cimport Domain
from oasys.planning.pomcppfcomm cimport POMCPPFCommPlanner, MessageEstimator
from oasys.structures.planning_structures cimport Particle, NestedParticle, ParticleFilter
from oasys.structures.pomdp_structures cimport State, PartialInternalStates, Action, ActionComm, PartialJointAction, PartialJointActionComm, Configuration, Observation, ObservationComm


cdef class IPOMCPPFCommPlanner:
    cdef Domain domain
    cdef Agent agent
    cdef int horizon
    cdef double gamma
    cdef readonly list agent_models
    cdef int trajectories
    cdef int[:] agents_modeled_per_frame
    cdef int[:] available_actions_per_frame
    cdef BeliefNode tree
    cdef double ucb_c
    cdef double q_sensitivity
    cdef ActionNode last_action_node
    cdef MessageEstimator message_estimator
    cdef list possible_next_states
    cdef int level
    cdef bint fully_observable_environment
    cdef list neighbor_planners
    cdef list neighbor_next_full_particle_sets
    cdef int[:] possible_messages_per_agent
    cdef dict last_action_observation_probabilities
    cdef dict global_pomcppfcomm_policy
    cdef dict neighbor_particle_filters
    cdef bint use_premodel
    cdef bint reuse_tree
    
    
    cpdef ActionComm create_plan(self, int[:] messages)
    cdef double update_tree(self, ParticleFilter particle_filter, int[:] messages, int h, BeliefNode belief_node)
    cdef NestedParticle update_nested_beliefs(self, NestedParticle particle, State next_state,
                                              PartialInternalStates next_internal_states,
                                              PartialJointActionComm modeled_joint_action)
    cdef double rollout_all(self, ParticleFilter particle_filter, int h)
    cdef double rollout(self, NestedParticle particle, int h)
    cdef list create_neighbor_planners(self)
    cdef PartialJointActionComm sample_modeled_joint_action(self, NestedParticle particle, int h, int[:] messages)
    cdef PartialJointAction sample_random_joint_action(self, State state, PartialInternalStates internal_states)
    cdef Configuration sample_configuration(self, PartialJointAction joint_action)
    cdef void make_observation(self, State next_state, int next_internal_state, ObservationComm observation)
    cdef ParticleFilter belief_update(self, State next_state, int next_internal_state, ParticleFilter particle_filter,
                                      ActionComm action_comm, ObservationComm observation_comm)
    cdef void reset(self)


cdef class BeliefNode:
    cdef int observation_index
    cdef long message_index
    cdef int internal_state
    cdef int visits
    cdef ParticleFilter particle_filter
    cdef list action_nodes
    cdef ActionNode parent_action_node
    cdef dict message_probabilties

    cdef dict calculate_last_action_observation_probabilities(self, int agent_num, ActionNode action_node)
    cdef void create_action_nodes(self, Agent agent, State state=*, int internal_state=*)
    cdef ActionNode argmax_ucb1(self, double ucb_c, Agent agent, State state=*, int internal_state=*)
    cdef ActionNode argmax_q(self, double q_sensitivity, double ucb_c=*)


cdef class ActionNode:
    cdef ActionComm action
    cdef int available_action_index
    cdef int visits
    cdef double q_value
    cdef dict belief_nodes
    cdef list all_belief_nodes
    cdef BeliefNode parent_belief_node

    cdef BeliefNode get_belief(self, int internal_state, int observation_index, long message_index)
    cdef list all_beliefs(self)
