from scipy.stats import t as t_dist
import numpy as np
import cython

cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
cdef extern from "math.h":
    double sqrt(double m)


cdef class Particle:
    def __cinit__(self, State state, PartialInternalStates partial_internal_states, *argv):
        self.state = state
        self.partial_internal_states = partial_internal_states


cdef class ParticleFilter:
    """ A weighted particle filter

    Attributes:
        particles(list):
            list of particles
        weights(list):
            list of weights for each particle
    """
    def __cinit__(self):
        self.particles = list()
        self.weights = list()


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.profile(True)
    cdef void resample_particlefilter(self, int pf_size = -1):
        """ Resamples the particle filter of given size according to the weights

        Attributes:
            pf_size(int):
                Size of the final particle filter
        """
        # sampling based on weights
        cdef int particles_n = len(self.particles)
        cdef double weight_sum = 0.0
        cdef list sampled_particles = list()
        cdef list sampled_weights = list()
        cdef int i, j, last
        cdef double rand_val, weight


        # if pf_size is not mentioned take the present size
        if pf_size == -1:
            pf_size = particles_n

        # perform resampling only if there are some particles in the particle filter
        if particles_n == 0:
            print('Particle Filter is empty')
        for i in range(pf_size):
            #print(pf_size, particles_n)
            # randomly sample an existing particle
            rand_val = rand() / (RAND_MAX + 1.0)
            for j in range(particles_n):
                weight = <double> self.weights[j]
                last = j

                if weight > rand_val:
                    break
                else:
                    rand_val -= weight

            sampled_particles.append(<Particle> self.particles[last])
            sampled_weights.append(weight)
            weight_sum += weight

        # Normalizing the weights
        for i in range(pf_size):
            sampled_weights[i] = (<double> sampled_weights[i]) / weight_sum

        self.particles = sampled_particles
        self.weights = sampled_weights


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.profile(True)
    cdef void normalize(self):
        cdef double weight_sum = 0.0
        cdef int particles_n = len(self.particles)
        cdef int i

        for i in range(particles_n):
            weight_sum += <double> self.weights[i]

        if weight_sum > 1.001 or weight_sum < 0.999:
            for i in range(particles_n):
                self.weights[i] = (<double> self.weights[i]) / weight_sum


cdef class NestedParticle:
    def __cinit__(self, State state, PartialInternalStates partial_internal_states, int level):
        # saving the state and partial_internal_states is done by the parent constructor
        self.level = level
        self.nested_particles = list()


cdef class AlphaVector:
    def __cinit__(self, int action_index):
        self.action_index = action_index
        self.values = dict()
        self.counts = dict()
        self.sum_squares = dict()
        self.lower_bounds = dict()

    cdef void calculate_lower_bounds(self):
        cdef int state_i, count
        cdef double mean, sum_squared, var, se, t_alpha
        for state_i in self.counts:
            count = <int> self.counts[state_i]
            mean = <double> self.values[state_i]
            sum_squared = <double> self.sum_squares[state_i]

            if count == 1:
                var = 10 * mean # TODO
            else:
                var = (sum_squared - count * mean * mean) / (count - 1)
            se = sqrt(var / count)
            t_alpha = t_dist.interval(0.95, count-1)[1]

            self.lower_bounds[state_i] = mean - t_alpha * se
