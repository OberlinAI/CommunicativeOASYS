from oasys.domains.domain import Domain
from oasys.structures.pomdp_structures import State, PartialInternalStates, Observation

from oasys.domains.domain cimport Domain
from oasys.structures.pomdp_structures cimport State, PartialInternalStates, Observation


cdef class Particle:
    cdef readonly State state
    cdef readonly PartialInternalStates partial_internal_states


cdef class NestedParticle(Particle):
    cdef readonly int level
    cdef readonly list nested_particles


cdef class ParticleFilter:
    cdef readonly list particles
    cdef readonly list weights
    cdef void resample_particlefilter(self, int pf_size=*)
    cdef void normalize(self)


cdef class AlphaVector:
    cdef readonly int action_index
    cdef readonly dict values
    cdef readonly dict counts
    cdef readonly dict sum_squares
    cdef readonly dict lower_bounds

    cdef void calculate_lower_bounds(self)
