"""Defines the data structures used by the wildfire domain.

Author: Adam Eck
"""

import oasys.structures.pomdp_structures as pomdp_structures
from .wildfire_settings import WildfireSettings
import numpy as np

cimport oasys.structures.pomdp_structures as pomdp_structures
from .wildfire_settings cimport WildfireSettings
cimport numpy as np
from cython cimport view


cdef class WildfireState(pomdp_structures.FactoredState):
    """Factored representation of environment states in the Wildfire domain.

    Attributes:
    index -- the unique index of the state (from the State parent class)
    values -- the values of each factored state variable (from the State parent class)
        Each state variable represents a different fire.
    """
    def __cinit__(self, int num_fires):
        pass


    cpdef void calculate_index(self, WildfireSettings settings):
        """Calculates self.index based on self.values."""
        self.index = 0

        cdef int i, val
        cdef int m = 1
        cdef values_n = len(self.values)
        cdef fire_states_n = settings.FIRE_STATES

        for i in range(values_n):
            val = self.values[i]
            self.index += val * m
            m *= fire_states_n


    def __reduce__(self):
        return (rebuild_wildfire_state, (np.asarray(self.values), self.index))


def rebuild_wildfire_state(values, index):
    state = WildfireState(len(values))

    state.values = values
    state.index = index

    return state


cdef class WildfireInternalStates(pomdp_structures.InternalStates):
    """Representation of internal states in the Wildfire domain.

    Attributes:
    values-- the array of internal state values for the agents (from the InternalStates parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfireInternalStates.


        """
        pass


    def __reduce__(self):
        return (rebuild_wildfire_internal_states, (np.asarray(self.values), len(self.values)))

def rebuild_wildfire_internal_states(values, length):
    wildfire_internal_states = WildfireInternalStates(length)
    wildfire_internal_states.values = values
    return wildfire_internal_states

cdef class WildfirePartialInternalStates(pomdp_structures.PartialInternalStates):
    """Representation of internal states in the Wildfire domain.

    Attributes:
    values-- the array of internal state values for the agents (from the PartialInternalStates parent class)
    agent_nums-- the array of unique number ids for the agents (from the PartialInternalStates parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfirePartialInternalStates.


        """
        pass


cdef class WildfireAction(pomdp_structures.Action):
    """Representation of actions in the Wildfire domain.

    Attributes:
    index -- the index of the actions chosen (from the Action parent class)
        Values 0 up to wildfire_settings.FIRES - 1 indicates a fire to fight, wildfire_settings.FIRES is NOOP
    """
    def __cinit__(self, int fire):
        """Constructs a new WildfireAction."""
        self.index = fire


cdef class WildfireActionComm(pomdp_structures.ActionComm):
    """Representation of actions with communication in the Wildfire domain.

    Attributes:
    index -- the index of the actions chosen (from the Action parent class)
        Values 0 up to wildfire_settings.FIRES - 1 indicates a fire to fight, wildfire_settings.FIRES is NOOP
    message -- the message sent by the agent
    """
    def __cinit__(self, int fire, int message):
        """Constructs a new WildfireActionComm."""
        self.index = fire
        self.message = message


cdef class WildfireJointAction(pomdp_structures.JointAction):
    """Representation of a joint action in the Wildfire domain.

    Attributes:
    actions -- the array of WildfireActions (as indices) chosen by the agents (from the JointAction parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfireJointAction.


        """
        pass

    def __reduce__(self):
        return (rebuild_wildfire_joint_action, (np.asarray(self.actions), len(self.actions)))

def rebuild_wildfire_joint_action(actions, length):
    wildfire_joint_action = WildfireJointAction(length)
    wildfire_joint_action.actions = actions
    return wildfire_joint_action


cdef class WildfireJointActionComm(pomdp_structures.JointActionComm):
    """Representation of a joint action with communication in the Wildfire domain.

    Attributes:
    actions -- the array of WildfireActions (as indices) chosen by the agents (from the JointAction parent class)
    messages -- the array of corresponding messages from the agents (from the PartialJointAction parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfireJointAction.


        """
        pass

    def __reduce__(self):
        return (rebuild_wildfire_joint_action_comm, (np.asarray(self.actions), np.asarray(self.messages)))

def rebuild_wildfire_joint_action_comm(actions, messages):
    wildfire_joint_action_comm = WildfireJointActionComm(len(actions))
    wildfire_joint_action_comm.actions = actions
    wildfire_joint_action_comm.messages = messages
    return wildfire_joint_action_comm


cdef class PartialWildfireJointAction(pomdp_structures.PartialJointAction):
    """Representation of a partial joint action in the Wildfire domain.

    Attributes:
    actions -- the array of WildfireActions (as indices) chosen by the agents (from the PartialJointAction parent class)
    agent_nums -- the array of corresponding number ids of the agents (from the PartialJointAction parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfirePartialJointAction.


        """
        pass

    def __reduce__(self):
        return (rebuild_partial_wildfire_joint_action, (np.asarray(self.actions), np.asarray(self.agent_nums)))

def rebuild_partial_wildfire_joint_action(actions, agent_nums):
    partial_wildfire_joint_action = PartialWildfireJointAction(len(actions))
    partial_wildfire_joint_action.actions = actions
    partial_wildfire_joint_action.agent_nums = agent_nums
    return partial_wildfire_joint_action


cdef class PartialWildfireJointActionComm(pomdp_structures.PartialJointActionComm):
    """Representation of a partial joint action with communication in the Wildfire domain.

    Attributes:
    actions -- the array of WildfireActions (as indices) chosen by the agents (from the PartialJointAction parent class)
    agent_nums -- the array of corresponding number ids of the agents (from the PartialJointAction parent class)
    messages -- the array of corresponding messages from the agents (from the PartialJointAction parent class)
    """
    def __cinit__(self, int agents_n):
        """Constructs a new WildfirePartialJointActionComm.


        """
        pass

    def __reduce__(self):
        return (rebuild_partial_wildfire_joint_action_comm, (np.asarray(self.actions), np.asarray(self.agent_nums), np.asarray(self.messages)))

def rebuild_partial_wildfire_joint_action_comm(actions, agent_nums, messages):
    partial_wildfire_joint_action_comm = PartialWildfireJointActionComm(len(actions))
    partial_wildfire_joint_action_comm.actions = actions
    partial_wildfire_joint_action_comm.agent_nums = agent_nums
    partial_wildfire_joint_action_comm.messages = messages
    return partial_wildfire_joint_action_comm


cdef class WildfireConfiguration(pomdp_structures.Configuration):
    """Representation of a configuration of actions by multiple agents in the Wildfire domain.

    Attributes:

    index -- the unique index of the joint action
    actions -- the counts for each action/frame combination
    """
    def __cinit__(self, int config_n):
        """Constructs a new WildfireConfiguration.


        """
        pass


cdef class WildfireObservation(pomdp_structures.Observation):
    """Representation of observations in the Wildfire domain.

    Attributes:
    index -- the index of the observation made (from the Observation parent class)
    fire_change -- the amount the fire being fought changed
    """
    def __cinit__(self, int fire_change):
        """Constructs a new WildfireObservation."""
        self.fire_change = fire_change
        self.index = fire_change + 1 # since smallest observation is -1


    def __str__(self):
        return str(self.fire_change)
       
    def __reduce__(self):
        return (rebuild_wildfire_observation, (self.index, self.fire_change))


def rebuild_wildfire_observation(index, fire_change):
    return WildfireObservation(fire_change)


cdef class WildfireObservationComm(pomdp_structures.ObservationComm):
    """Representation of observations in the Wildfire domain.

    Attributes:
    index -- the index of the observation made (from the Observation parent class)
    fire_change -- the amount the fire being fought changed
    """
    def __cinit__(self, int fire_change, int agents_n):
        """Constructs a new WildfireObservationComm."""
        self.fire_change = fire_change
        self.index = fire_change + 1 # since smallest observation is -1

    def __str__(self):
        return str(self.fire_change)
        
    def __reduce__(self):
        return (rebuild_wildfire_observationcomm, (self.fire_change, np.asarray(self.messages)))


def rebuild_wildfire_observationcomm(fire_change, messages):
    observation_comm =  WildfireObservationComm(fire_change, len(messages))
    observation_comm.messages = messages
    return observation_comm
