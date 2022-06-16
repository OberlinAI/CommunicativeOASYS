from oasys.structures.pomdp_structures import State
from .wildfire_structures import WildfireActionComm
import numpy as np

from oasys.structures.pomdp_structures cimport State
from .wildfire_structures cimport WildfireState, WildfireActionComm
cimport numpy as np

cdef class WildfireFrame(Frame):
    """Representation of an agent frame in the Wildfire domain.

    Attributes:
    index -- the index of the frame (from the Frame parent class)
    loc_num -- the location in the grid of agents with this Frame
    fire_reduction -- the amount of reduction in a fire produced by agents of this frame
    """
    def __cinit__(self, int index, int loc_num, double fire_reduction, Wildfire domain):
        """Constructs a new WildfireFrame."""
        self.index = index
        self.loc_num = loc_num
        self.fire_reduction = fire_reduction
        self.domain = domain

        self.possible_internal_states = self.find_possible_internal_states()
        self.all_available_actions = self.find_available_actions()
        self.all_available_actions_comm = self.find_available_actions_comm()
        self.available_actions_per_state = self.find_available_actions_per_state()
        self.available_actions_comm_per_state = self.find_available_actions_comm_per_state()
        self.noop_actions_comm = self.find_noop_actions_comm()


    cdef int[:] find_possible_internal_states(self):
        return np.array(list(range(self.domain.settings.SUPPRESSANT_STATES)),
                        dtype=np.int32)


    cdef int[:] find_available_actions(self):
        cdef int fires_n = self.domain.settings.FIRES
        cdef list actions = []
        cdef int x = self.domain.settings.AGENT_LOCATIONS[self.loc_num][0]
        cdef int y = self.domain.settings.AGENT_LOCATIONS[self.loc_num][1]
        cdef int fire_i, fire_x, fire_y, x_diff, y_diff

        # check which fires are within a range of 1 step
        for fire_i in range(fires_n):
            fire_x = self.domain.settings.FIRE_LOCATIONS[fire_i][0]
            fire_y = self.domain.settings.FIRE_LOCATIONS[fire_i][1]

            x_diff = x - fire_x
            y_diff = y - fire_y

            if x_diff < 2 and y_diff < 2 and x_diff > -2 and y_diff > -2:
                actions.append(fire_i)

        # add the noop action
        actions.append(fires_n)

        # convert to an array
        return np.array(actions, dtype=np.int32)


    cdef int[:] find_available_actions_comm(self):
        cdef list actions = list(self.all_available_actions)

        cdef list actions_comm = []
        cdef WildfireActionComm action
        cdef int actions_n = len(self.domain.actions_with_comm)
        cdef int action_i
        cdef int action_num

        for action_i in range(actions_n):
            action = <WildfireActionComm> self.domain.actions_with_comm[action_i]

            if action.index in actions:
                actions_comm.append(action_i)

        return np.array(actions_comm, dtype=np.int32)


    cdef list find_available_actions_per_state(self):
        cdef list available_actions_per_state = list()

        cdef int fires_n = self.domain.settings.FIRES
        cdef int burned_out = self.domain.settings.FIRE_STATES - 1
        cdef int states_n = len(self.domain.states)
        cdef int available_actions_n = len(self.all_available_actions)
        cdef int state_i, available_action_i, action_i, fire_value
        cdef WildfireState state
        cdef list state_actions

        for state_i in range(states_n):
            state = <WildfireState> self.domain.states[state_i]
            state_actions = list()

            for available_action_i in range(available_actions_n):
                action_i = self.all_available_actions[available_action_i]

                if action_i == fires_n:
                    # NOOP is always available
                    state_actions.append(action_i)
                else:
                    fire_value = state.values[action_i]

                    # can we fight this fire?
                    if fire_value > 0 and fire_value < burned_out:
                        state_actions.append(action_i)

            available_actions_per_state.append(np.array(state_actions, dtype=np.intc))

        return available_actions_per_state


    cdef list find_available_actions_comm_per_state(self):
        cdef list available_actions_per_state = list()

        cdef int fires_n = self.domain.settings.FIRES
        cdef int burned_out = self.domain.settings.FIRE_STATES - 1
        cdef int states_n = len(self.domain.states)
        cdef int available_actions_n = len(self.all_available_actions_comm)
        cdef int state_i, available_action_i, action_i, action_index, fire_value
        cdef WildfireState state
        cdef WildfireActionComm action_comm
        cdef list state_actions

        for state_i in range(states_n):
            state = <WildfireState> self.domain.states[state_i]
            state_actions = list()

            for available_action_i in range(available_actions_n):
                action_i = self.all_available_actions_comm[available_action_i]
                action_comm = <WildfireActionComm> self.domain.actions_with_comm[action_i]
                action_index = action_comm.index

                if action_index == fires_n:
                    # NOOP is always available
                    state_actions.append(action_i)
                else:
                    fire_value = state.values[action_index]

                    # can we fight this fire?
                    if fire_value > 0 and fire_value < burned_out:
                        state_actions.append(action_i)

            available_actions_per_state.append(np.array(state_actions, dtype=np.intc))

        return available_actions_per_state


    cdef int[:] find_noop_actions_comm(self):
        cdef int noop = self.domain.settings.FIRES
        cdef int available_actions_n = len(self.all_available_actions_comm)
        cdef int available_action_i, action_i
        cdef WildfireActionComm action_comm

        cdef list action_indices = list()
        for available_action_i in range(available_actions_n):
            action_i = self.all_available_actions_comm[available_action_i]
            action_comm = <WildfireActionComm> self.domain.actions_with_comm[action_i]

            if action_comm.index == noop:
                action_indices.append(action_i)

        return np.asarray(action_indices, dtype=np.intc)


    cdef int[:] get_all_available_actions(self):
        return self.all_available_actions


    cdef int[:] get_available_actions(self, int internal_state):
        # WildfireAgents can take the same actions in all internal states
        return self.all_available_actions


    cdef int[:] get_all_available_actions_comm(self):
        return self.all_available_actions_comm


    cdef int[:] get_available_actions_per_state(self, State state, int internal_state):
        if internal_state == 0:
            # TODO avoid calling np each return
            return np.array([self.domain.settings.FIRES], dtype=np.intc)
        else:
            return self.available_actions_per_state[state.index]


    cdef int[:] get_available_actions_comm_per_state(self, State state, int internal_state):
        if internal_state == 0:
            return self.noop_actions_comm
        else:
            return self.available_actions_comm_per_state[state.index]


    cdef int[:] get_available_actions_comm(self, int internal_state):
        # WildfireAgents can take the same actions in all internal states
        return self.all_available_actions_comm
