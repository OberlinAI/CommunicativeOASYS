import numpy as np

cimport numpy as np
from cython cimport view

"""Defines the data structures uses in POMDPs.

Author: Adam Eck
"""

cdef class State():
    """Representation of an environment state.

    Attributes:

    index -- the unique index of the state
    """
    pass


cdef class InternalStates():
    """Representation of an collection of internal states.

    Attributes:

    values -- the values of each factored state variable
    """
    def __cinit__(self, int agents_n, *argv):
        """Constructs a new InternalStates.


        """
        self.values = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")


cdef class PartialInternalStates(InternalStates):
    """Representation of an collection of internal states for some agents.

    Attributes:

    values -- the values of each factored state variable
    agent_num -- the corresponding agent numbers
    """
    def __cinit__(self, int agents_n, *argv):
        """Constructs a new PartialInternalStates.


        """
        # NOTE: values is automatically created by InternalStates constructor since Cython calls that first
        self.agent_nums = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")


    def __reduce__(self):
        return (rebuild_partial_internal_states, (np.asarray(self.values), np.asarray(self.agent_nums)))


def rebuild_partial_internal_states(values, agent_nums):
    partial_internal_states = PartialInternalStates(len(values))

    partial_internal_states.values = values
    partial_internal_states.agent_nums = agent_nums

    return partial_internal_states


cdef class NestedPartialInternalStates(PartialInternalStates):
    """Representation of a nested collection of internal states for some agents.

    Attributes:

    values -- the values of each factored state variable
    agent_num -- the corresponding agent numbers
    nested_internal_states -- a 2D list representing a nested particle filter of neighbor beliefs
                              the first index is a modeled neighbor,
                              the second index is one of their NestedPartialInternalStates (a nested particle)
    """
    def __cinit__(self, int agents_n, int level):
        """Constructs a new NestedPartialInternalStates.


        """
        # NOTE: values and agent_nums are automatically created by the parent constructors since Cython calls those first
        self.level = level
        self.nested_internal_states = []


    def __reduce__(self):
        return (rebuild_nested_partial_internal_states, (np.asarray(self.values), np.asarray(self.agent_nums),
                                                         self.level, self.nested_internal_states))


def rebuild_nested_partial_internal_states(values, agent_nums, level, nested_internal_states):
    nested_partial_internal_states = NestedPartialInternalStates(len(values), level)

    nested_partial_internal_states.values = values
    nested_partial_internal_states.agent_nums = agent_nums
    nested_partial_internal_states.nested_internal_states = nested_internal_states

    return nested_partial_internal_states



cdef class FactoredState(State):
    """Factored representation of an environment state

    index -- the unique index of the state (from the state parent class)
    values -- the values of each factored state variable
    """
    def __cinit__(self, int state_vars_n):
        self.values = view.array(shape=(state_vars_n,), itemsize=sizeof(int), format="i")


cdef class Action():
    """Representation of an action.

    Attributes:

    index -- the unique index of the action
    """
    def __cinit__(self, int index, *argv):
        self.index = index


    def __reduce__(self):
        return (rebuild_action, (self.index,))


def rebuild_action(index):
    return Action(index)


cdef class ActionComm():
    """Representation of an action with communication.

    Attributes:

    index -- the unique index of the action
    message -- the message shared by the agent
    """
    def __cinit__(self, int index, int message, *argv):
        self.index = index
        self.message = message


    def __reduce__(self):
        return (rebuild_actioncomm, (self.index, self.message))


def rebuild_actioncomm(index, message):
    return ActionComm(index, message)


cdef class JointAction():
    """Representation of a joint action by multiple agents.

    Attributes:

    actions -- the actions for each agent in the system
    """
    def __cinit__(self, int agents_n):
        """Constructs a new JointAction.


        """
        self.actions = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")


cdef class JointActionComm(JointAction):
    """Representation of a joint action by multiple agents with communication.

    Attributes:

    actions -- the actions for each agent in the system
    messages -- the messages sent by each agent in the system
    """
    def __cinit__(self, int agents_n):
        """Constructs a new JointAction.


        """
        self.messages = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")

        cdef int i
        for i in range(agents_n):
            self.messages[i] = -1 # by default, agents send nothing


cdef class PartialJointAction(JointAction):
    """Representation of a joint action by some (but maybe not all) agents.

    Attributes:

    actions -- the actions for each modeled agent
    agent_nums -- the corresponding agent numbers
    """
    def __cinit__(self, int agents_n):
        """Constructs a new PartialJointAction.


        """
        self.agent_nums = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")


cdef class PartialJointActionComm(PartialJointAction):
    """Representation of a joint action by some (but maybe not all) agents with communication.

    Attributes:

    actions -- the actions for each modeled agent
    agent_nums -- the corresponding agent numbers
    messages -- the messages sent by each modeled agent
    """
    def __cinit__(self, int agents_n):
        """Constructs a new PartialJointAction.


        """
        self.messages = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")

        cdef int i
        for i in range(agents_n):
            self.messages[i] = -1 # by default, agents send nothing


cdef class Configuration():
    """Representation of a configuration of actions by multiple agents.

    Attributes:

    actions -- the counts for each action/frame combination
    """
    def __cinit__(self, int config_n):
        """Constructs a new Configuration.


        """
        self.actions = view.array(shape=(config_n,), itemsize=sizeof(int), format="i")

        # start each count at 0
        cdef int i
        for i in range(config_n):
            self.actions[i] = 0


    def __lt__(self, other):
        # NOTE: for the ability to sort (in case of tie in NestedVI)
        return True


cdef class Observation():
    """Representation of an environment observation.

    Attributes:

    index -- the unique index of the observation
    """
    pass


cdef class ObservationComm(Observation):
    """Representation of an environment observation.

    Attributes:

    index -- the unique index of the observation
    """
    def __cinit__(self, int index, int agents_n, *argv):
        self.index = index
        self.messages = view.array(shape=(agents_n,), itemsize=sizeof(int), format="i")

        cdef int i
        for i in range(agents_n):
            self.messages[i] = -1 # by default, agents send nothing


    cdef long calculate_message_index(self, int[:] possible_messages):
        cdef long total_index = 0
        cdef long offset = 1

        # calculate the index from the messages
        cdef long messages_n = self.messages.shape[0]
        cdef long message_i
        cdef long agent_possible_messages_n
        for message_i in range(messages_n):
            agent_possible_messages_n = possible_messages[message_i + 1] # add 1 since first spot is self agent
            message = self.messages[message_i] + 1 # add 1 since no message = -1

            total_index = total_index + offset * message
            offset *= agent_possible_messages_n

        return total_index
