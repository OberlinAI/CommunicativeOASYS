from oasys.agents.agent import Agent
from oasys.structures.pomdp_structures import State, InternalStates, PartialInternalStates, Action, ActionComm, PartialJointAction, JointAction, JointActionComm, Configuration, Observation

from oasys.agents.agent cimport Agent
from oasys.structures.pomdp_structures cimport State, InternalStates, PartialInternalStates, Action, ActionComm, PartialJointAction, JointAction, JointActionComm, Configuration, Observation

cdef Domain get_domain()
cdef void set_domain(Domain d)

cdef class Domain:
    cdef readonly list states
    cdef readonly list actions
    cdef readonly list actions_with_comm
    cdef readonly list observations
    cdef readonly list frames
    cdef readonly list agents
    cdef int[:] max_configuration_counts
    cdef int[:] configuration_actions
    cdef dict configuration_indices

    cdef list create_states(self)
    cdef list create_actions(self)
    cdef list create_actions_comm(self)
    cdef list create_observations(self)
    cdef list create_frames(self)
    cdef list create_agents(self, dict settings)
    cdef int[:] calculate_configuration_max(self)
    cdef int[:] calculate_configuration_actions(self)
    cdef dict calculate_configuration_indices(self)
    cpdef list create_all_configurations(self, Agent agent=*)

    cdef double transition_probability(self, int agent_num, State state, InternalStates internal_states,
                                       JointAction joint_action, State next_state, InternalStates next_internal_states)
    cdef double state_transition_probability(self, int agent_num, State state, JointAction joint_action,
                                             State next_state)
    cdef double state_transition_probability_configuration(self, int agent_num, State state,
                                                           Configuration configuration, Action action, State next_state)
    cdef double internal_states_transition_probability(self, int agent_num, InternalStates internal_states,
                                                       JointAction joint_action, InternalStates next_internal_states)
    cdef double single_internal_state_transition_probability(self, int agent_num, int internal_state,
                                                             Configuration configuration, Action action,
                                                             int next_internal_state)
    cdef double observation_probability(self, int agent_num, State state, JointAction joint_action, State next_state,
                                        Observation observation)
    cdef double observation_probability_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state, Observation observation)
    cdef double reward(self, int agent_num, State state, InternalStates internal_states, JointAction joint_action,
                       State next_state, InternalStates next_internal_states)
    cpdef double reward_configuration(self, int agent_num, State state, InternalStates internal_states,
                                     Configuration configuration, Action action, State next_state,
                                     InternalStates next_internal_states)
    cdef double reward_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                 JointActionComm joint_action, State next_state, InternalStates next_internal_states)
    cpdef double reward_configuration_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                                Configuration configuration, ActionComm action, State next_state,
                                                InternalStates next_internal_states)

    cdef Configuration create_configuration(self, int agent_num, JointAction joint_action,
                                            bint leave_out_subject_agent=*)
    cdef Configuration create_configuration_from_partial(self, int agent_num, PartialJointAction joint_action,
                                            bint leave_out_subject_agent=*)
    cdef State sample_next_state(self, int agent_num, State state, JointAction joint_action)
    cdef State sample_next_state_configuration(self, int agent_num, State state, Configuration configuration,
                                               Action action)
    cdef State sample_next_state_configuration_from_possible(self, int agent_num, State state,
                                                             Configuration configuration, Action action,
                                                             list possible_next_states)
    cdef InternalStates sample_next_internal_states(self, int agent_num, InternalStates internal_states,
                                                    JointAction joint_action)
    cdef PartialInternalStates sample_next_partial_internal_states(self, PartialInternalStates internal_states,
                                                                   PartialJointAction joint_action)
    cdef int sample_single_next_internal_state(self, int agent_num, int internal_state, Configuration configuration,
                                               Action action)
    cdef Observation sample_observation(self, int agent_num, State state, JointAction joint_action, State next_state)
    cdef Observation sample_observation_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state)
    cpdef State generate_start_state(self)
    cpdef InternalStates generate_start_internal_states(self)

    cdef int num_agents(self)
    cdef int[:] num_agents_per_frame(self)
    cdef double max_reward(self)

    cdef list cache_possible_next_state_transitions(self)
    cdef list possible_next_states(self, int state_i)
