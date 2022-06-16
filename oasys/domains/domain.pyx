from oasys.agents.agent import Agent
from oasys.agents.frame import Frame
from oasys.structures.pomdp_structures import State, InternalStates, PartialInternalStates, Action, PartialJointAction, JointAction, Configuration, Observation
import numpy as np

from oasys.agents.agent cimport Agent
from oasys.agents.frame cimport Frame
from oasys.structures.pomdp_structures cimport State, InternalStates, PartialInternalStates, Action, PartialJointAction, JointAction, Configuration, Observation
cimport numpy as np

''' A singleton instance of Domain to simplify access'''
cdef Domain domain


cdef Domain get_domain():
    return domain


cdef void set_domain(Domain d):
    global domain
    domain = d


cdef class Domain():
    def __cinit__(self):
        pass


    cdef list create_states(self):
        pass


    cdef list create_actions(self):
        pass


    cdef list create_actions_comm(self):
        pass


    cdef list create_observations(self):
        pass


    cdef list create_frames(self):
        pass


    cdef list create_agents(self, dict settings):
        pass


    cdef int[:] calculate_configuration_max(self):
        cdef int frames_n = self.settings.FRAMES
        cdef int config_n = 0
        cdef int frame_i, actions_n
        cdef Frame frame

        for frame_i in range(frames_n):
            frame = <Frame> self.frames[frame_i]
            actions_n = len(frame.get_all_available_actions())
            config_n = config_n + actions_n

        cdef int[:] counts = np.zeros((config_n,), dtype=np.intc)
        cdef int frame_offset = 0
        cdef int action_i, config_i, agents_n
        for frame_i in range(frames_n):
            frame = <Frame> self.frames[frame_i]
            actions_n = len(frame.get_all_available_actions())
            agents_n = self.settings.N_AGENTS_PER_FRAME[frame_i]

            for action_i in range(actions_n):
                config_i = frame_offset + action_i
                counts[config_i] = agents_n

            frame_offset = frame_offset + actions_n

        return counts


    cdef int[:] calculate_configuration_actions(self):
        cdef int frames_n = self.settings.FRAMES
        cdef int config_n = len(self.max_configuration_counts)
        cdef int frame_offset = 0
        cdef int frame_i, actions_n, action_i, config_i
        cdef int[:] available_actions
        cdef Frame frame

        cdef int[:] config_actions = np.empty((config_n,), dtype=np.intc)
        for frame_i in range(frames_n):
            frame = <Frame> self.frames[frame_i]
            available_actions = frame.get_all_available_actions()
            actions_n = len(available_actions)

            for action_i in range(actions_n):
                config_i = frame_offset + action_i
                config_actions[config_i] = available_actions[action_i]

            frame_offset = frame_offset + actions_n

        return config_actions


    cdef dict calculate_configuration_indices(self):
        cdef dict configuration_indices = dict()

        cdef int config_n = len(self.configuration_actions)
        cdef int frames_n = len(self.frames)
        cdef int frame_offset = 0
        cdef int frame_i, actions_n, action_i, config_i, action_index
        cdef Frame frame
        cdef tuple t

        for frame_i in range(frames_n):
            frame = <Frame> self.frames[frame_i]
            actions_n = len(frame.get_all_available_actions())

            for action_i in range(actions_n):
                config_i = frame_offset + action_i
                action_index = self.configuration_actions[config_i]
                t = tuple((frame_i, action_index))
                configuration_indices[t] = config_i

            frame_offset += actions_n

        return configuration_indices


    cpdef list create_all_configurations(self, Agent agent=None):
        pass


    cdef double transition_probability(self, int agent_num, State state, InternalStates internal_states,
                                       JointAction joint_action, State next_state, InternalStates next_internal_states):
        pass


    cdef double state_transition_probability(self, int agent_num, State state, JointAction joint_action,
                                             State next_state):
        pass


    cdef double state_transition_probability_configuration(self, int agent_num, State state,
                                                           Configuration configuration, Action action,
                                                           State next_state):
        pass


    cdef double internal_states_transition_probability(self, int agent_num, InternalStates internal_states,
                                                       JointAction joint_action, InternalStates next_internal_states):
        pass


    cdef double single_internal_state_transition_probability(self, int agent_num, int internal_state,
                                                             Configuration configuration, Action action,
                                                             int next_internal_state):
        pass


    cdef double observation_probability(self, int agent_num, State state, JointAction joint_action, State next_state,
                                        Observation observation):
        pass


    cdef double observation_probability_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state, Observation observation):
        pass


    cdef double reward(self, int agent_num, State state, InternalStates internal_states, JointAction joint_action,
                       State next_state, InternalStates next_internal_states):
        pass


    cpdef double reward_configuration(self, int agent_num, State state, InternalStates internal_states,
                                     Configuration configuration, Action action, State next_state,
                                     InternalStates next_internal_states):
        pass


    cdef double reward_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                 JointActionComm joint_action, State next_state, InternalStates next_internal_states):
        pass


    cpdef double reward_configuration_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                                Configuration configuration, ActionComm action, State next_state,
                                                InternalStates next_internal_states):
        pass


    cdef State sample_next_state(self, int agent_num, State state, JointAction joint_action):
        pass


    cdef State sample_next_state_configuration(self, int agent_num, State state, Configuration configuration,
                                               Action action):
        pass


    cdef State sample_next_state_configuration_from_possible(self, int agent_num, State state,
                                                             Configuration configuration, Action action,
                                                             list possible_next_states):
        pass


    cdef InternalStates sample_next_internal_states(self, int agent_num, InternalStates internal_states,
                                                    JointAction joint_action):
        pass


    cdef PartialInternalStates sample_next_partial_internal_states(self, PartialInternalStates internal_states,
                                                                   PartialJointAction joint_action):
        pass


    cdef int sample_single_next_internal_state(self, int agent_num, int internal_state, Configuration configuration,
                                               Action action):
        pass


    cdef Observation sample_observation(self, int agent_num, State state, JointAction joint_action, State next_state):
        pass


    cdef Observation sample_observation_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state):
        pass


    cpdef State generate_start_state(self):
        pass


    cpdef InternalStates generate_start_internal_states(self):
        pass


    cdef Configuration create_configuration(self, int agent_num, JointAction joint_action,
                              bint leave_out_subject_agent=True):
        pass


    cdef Configuration create_configuration_from_partial(self, int agent_num, PartialJointAction joint_action,
                                            bint leave_out_subject_agent=True):
        pass


    cdef int num_agents(self):
        pass


    cdef int[:] num_agents_per_frame(self):
        pass


    cdef double max_reward(self):
        pass


    cdef list cache_possible_next_state_transitions(self):
        pass


    cdef list possible_next_states(self, int state_i):
        pass
