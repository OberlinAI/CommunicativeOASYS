from oasys.agents.agent import Agent
from oasys.domains.domain import Domain
from oasys.structures.planning_structures import Particle, ParticleFilter
from oasys.structures.pomdp_structures import State, PartialInternalStates, Action, ActionComm, PartialJointAction, PartialJointActionComm, Configuration, Observation, ObservationComm

from oasys.agents.agent cimport Agent
from oasys.domains.domain cimport Domain
from oasys.structures.planning_structures cimport Particle, ParticleFilter
from oasys.structures.pomdp_structures cimport State, PartialInternalStates, Action, ActionComm, PartialJointAction, PartialJointActionComm, Configuration, Observation, ObservationComm


cdef class POMCPPFCommPlanner:
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
    cdef int[:] possible_messages_per_agent
    cdef MessageEstimator message_estimator
    cdef list possible_next_states
    cdef bint fully_observable_environment
    cdef dict last_action_observation_probabilities

    cpdef ActionComm create_plan(self, int[:] messages)
    cdef double update_tree(self, ParticleFilter particle_filter, int[:] messages, int h, BeliefNode belief_node)
    cdef double rollout(self, Particle particle, int h)
    cdef double rollout_all(self, ParticleFilter particle_filter, int h)
    cdef PartialJointAction sample_random_joint_action(self, State state)
    cdef PartialJointActionComm sample_joint_action_with_comm(self, int action_index, int[:] messages)
    cdef Configuration sample_configuration(self, PartialJointAction joint_action)
    cdef void make_observation(self, State next_state, int next_internal_state, ObservationComm observation)
    cdef void reset(self)
    cpdef list root_q_values(self)


cdef class BeliefNode:
    cdef int observation_index
    cdef long message_index
    cdef int internal_state
    cdef int visits
    cdef ParticleFilter particle_filter
    cdef list action_nodes
    cdef ActionNode parent_action_node

    cdef dict calculate_last_action_observation_probabilities(self, int agent_num, ActionNode action_node)
    cdef void create_action_nodes(self, Agent agent, State state=*, int internal_state=*)
    cdef ActionNode argmax_ucb1(self, double ucb_c, Agent agent, State state=*, int internal_state=*)
    cdef ActionNode argmax_q(self, double q_sensitivity, double ucb_c=*)


cdef class ActionNode:
    cdef Action action
    cdef int available_action_index
    cdef int visits
    cdef double q_value
    cdef dict belief_nodes
    cdef list all_belief_nodes
    cdef BeliefNode parent_belief_node

    cdef BeliefNode get_belief(self, int internal_state, int observation_index, long message_index)
    cdef list all_beliefs(self)


cdef class MessageEstimator():
    cdef list internal_state_stack

    cdef void start_internal_state_stack(self, PartialInternalStates internal_states)
    cdef void pop_internal_state_stack(self)
    cdef int estimate_message(self, int internal_state, int possible_messages_n, Agent agent)
    cdef int estimate_action(self, int message, Agent agent)
    cdef void next_messages(self, int[:] messages, PartialJointActionComm joint_action, list agent_models,
                                  int[:] possible_messages_per_agent, Domain domain)
    cdef list update_internal_state_prob(self, int message, int possible_messages_n, list prior_probs)
