from .wildfire_settings import WildfireSettings
import oasys.structures.pomdp_structures as pomdp_structures

from .wildfire_settings cimport WildfireSettings
cimport oasys.structures.pomdp_structures as pomdp_structures


cdef class WildfireState(pomdp_structures.FactoredState):
    cpdef void calculate_index(self, WildfireSettings settings)


cdef class WildfireInternalStates(pomdp_structures.InternalStates):
    pass


cdef class WildfirePartialInternalStates(pomdp_structures.PartialInternalStates):
    pass


cdef class WildfireAction(pomdp_structures.Action):
    pass


cdef class WildfireActionComm(pomdp_structures.ActionComm):
    pass


cdef class WildfireJointAction(pomdp_structures.JointAction):
    pass


cdef class WildfireJointActionComm(pomdp_structures.JointActionComm):
    pass


cdef class WildfirePartialJointAction(pomdp_structures.PartialJointAction):
    pass


cdef class WildfirePartialJointActionComm(pomdp_structures.PartialJointActionComm):
    pass


cdef class WildfireConfiguration(pomdp_structures.Configuration):
    pass


cdef class WildfireObservation(pomdp_structures.Observation):
    cdef int fire_change


cdef class WildfireObservationComm(pomdp_structures.ObservationComm):
    cdef int fire_change
