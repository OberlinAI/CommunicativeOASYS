from oasys.agents.agent import Agent
from oasys.domains.domain import Domain
from oasys.structures.planning_structures import Particle, ParticleFilter
from oasys.structures.pomdp_structures import State, PartialInternalStates, Action, PartialJointAction, Configuration, Observation

from oasys.agents.agent cimport Agent
from oasys.domains.domain cimport Domain
from oasys.structures.planning_structures cimport Particle, ParticleFilter
from oasys.structures.pomdp_structures cimport State, PartialInternalStates, Action, PartialJointAction, Configuration, Observation


cdef class POMCPPFPlanner:
    cdef Domain domain
    cdef Agent agent
    cdef int horizon
    cdef double gamma
    cdef readonly list agent_models
    cdef int[:] agents_modeled_per_frame
    cdef int[:] available_actions_per_frame
    cdef BeliefNode tree
    cdef double ucb_c
    cdef double q_sensitivity
    cdef ActionNode last_action_node
    cdef list possible_next_states
    cdef bint fully_observable_environment

    cpdef Action create_plan(self, int trajectories)
    cdef double update_tree(self, ParticleFilter particle_filter, int h, BeliefNode belief_node)
    cdef double rollout(self, Particle particle, int h)
    cdef double rollout_all(self, ParticleFilter particle_filter, int h)
    cdef PartialJointAction sample_random_joint_action(self, State state)
    cdef Configuration sample_configuration(self, PartialJointAction joint_action)
    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation)
    cdef void reset(self)
    cpdef list root_q_values(self)


cdef class BeliefNode:
    cdef int observation_index
    cdef int internal_state
    cdef int visits
    cdef ParticleFilter particle_filter
    cdef list action_nodes
    cdef ActionNode parent_action_node

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

    cdef BeliefNode get_belief(self, int observation_index, int internal_state)
    cdef list all_belief_particles(self)
