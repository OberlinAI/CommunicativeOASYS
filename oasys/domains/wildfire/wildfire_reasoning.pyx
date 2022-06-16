from oasys.planning.pomcppf import BeliefNode as BeliefNode_pf
from oasys.planning.pomcppfcomm import BeliefNode as BeliefNode_pfcomm
from oasys.planning.ipomcppf import BeliefNode as BeliefNode_ipf
from oasys.planning.ipomcppfcomm import BeliefNode as BeliefNode_ipfcomm
from oasys.structures.planning_structures import Particle, NestedParticle, ParticleFilter
from oasys.structures.pomdp_structures import Action, PartialInternalStates, NestedPartialInternalStates
from .wildfire_agent import WildfireAgent
from .wildfire_settings import WildfireSettings
from .wildfire_structures import WildfireInternalStates, WildfirePartialInternalStates, WildfireAction, WildfireObservationComm
import numpy as np
import cython

from oasys.planning.pomcppf cimport BeliefNode as BeliefNode_pf
from oasys.planning.pomcppfcomm cimport BeliefNode as BeliefNode_pfcomm
from oasys.planning.ipomcppf cimport BeliefNode as BeliefNode_ipf
from oasys.planning.ipomcppfcomm cimport BeliefNode as BeliefNode_ipfcomm
from oasys.planning.reasoning cimport ReasoningModel
from oasys.structures.planning_structures cimport Particle, NestedParticle,  ParticleFilter
from oasys.structures.pomdp_structures cimport Action, PartialInternalStates, NestedPartialInternalStates
from .wildfire_agent cimport WildfireAgent
from .wildfire_settings cimport WildfireSettings
from .wildfire_structures cimport WildfireInternalStates, WildfirePartialInternalStates, WildfireAction, WildfireObservationComm
cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX


cdef class WildfirePOMCPPFReasoning(ReasoningModel):
    def __cinit__(self, WildfireAgent agent):
        self.agent = agent
        self.trajectories = agent.settings["trajectories"]
        self.particles_n = agent.settings["particles_n"]


    cdef void create_planner(self, list possible_next_states=None, list agent_models=None):
        cdef WildfireAgent agent = <WildfireAgent> self.agent

        cdef WildfireReasoningType neighbor_reasoning_type = WildfireReasoningType.Heuristic

        cdef double epsilon_p = agent.settings["epsilon_p"]
        if agent_models is None:
            if epsilon_p == 0.0:
                agent_models = agent.sample_all_agents(neighbor_reasoning_type)
            else:
                if not agent.enough_agents_for_modeling(epsilon_p, agent.domain.agents):
                    agent_models = agent.sample_agents_for_modeling_small(agent.domain.agents, neighbor_reasoning_type, agent.settings["cliques_n"])
                else:
                    agent_models = agent.sample_agents_for_modeling(epsilon_p, agent.domain.agents, neighbor_reasoning_type)

        # create the planner
        self.planner = POMCPPFPlanner(agent, self.agent.settings["horizon"], 0.99,
                                      agent.settings["ucb_c"], agent_models, True, possible_next_states)


    @cython.boundscheck(True)
    @cython.cdivision(True)
    cdef void refresh_particles(self, State state, int internal_state):
        cdef BeliefNode_pf old_tree = self.planner.tree

        # restart the tree
        cdef BeliefNode_pf tree = BeliefNode_pf(-1, -1)
        self.planner.tree = tree

        # only use the known internal state (we don't model anyone else!)
        cdef WildfirePartialInternalStates partial_internal_states = WildfirePartialInternalStates(1)
        partial_internal_states.values[0] = internal_state
        partial_internal_states.agent_nums[0] = self.agent.agent_num

        # also only use the known State
        cdef Particle particle = Particle(state, partial_internal_states)
        cdef double weight = <double> 1/self.particles_n
        cdef int particle_i
        for particle_i in range(self.particles_n):
            tree.particle_filter.particles.append(particle)
            tree.particle_filter.weights.append(weight)


    cdef Action choose_action(self, State state, int internal_state):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        return self.planner.create_plan(self.trajectories)

    cdef list calculate_q_values(self, State state, int internal_state):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        self.planner.create_plan(self.trajectories)
        return self.planner.root_q_values()

    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        self.planner.make_observation(next_state, next_internal_state, observation)


cdef class WildfirePOMCPPFCommReasoning(ReasoningModel):
    def __cinit__(self, WildfireAgent agent):
        self.agent = agent
        self.trajectories = agent.settings["trajectories"]
        self.particles_n = agent.settings["particles_n"]


    cdef void create_planner(self, list agent_neighbors, list possible_next_states=None, bint in_child_process=False):
        cdef WildfireAgent agent = <WildfireAgent> self.agent
        cdef WildfireAgent neighbor

        cdef int neighbors_n = len(agent_neighbors)
        cdef int[:] possible_messages = np.zeros((neighbors_n + 1,), dtype=np.intc)
        possible_messages[0] = len(self.agent.frame.possible_internal_states) + 1

        cdef WildfireReasoningType neighbor_reasoning_type = WildfireReasoningType.Heuristic

        cdef int neighbor_i
        cdef list agent_models = []
        for neighbor_i in range(neighbors_n):
            neighbor = <WildfireAgent> agent_neighbors[neighbor_i]
            possible_messages[neighbor_i + 1] = len(neighbor.frame.possible_internal_states) + 1
            agent_models.append(neighbor.replicate_as_model())

        # create the planner
        self.planner = POMCPPFCommPlanner(agent, self.agent.settings["horizon"], 0.99,
                                          agent.settings["ucb_c"], agent_models, self.trajectories, True,
                                          possible_messages,
                                          WildfireMessageEstimator(self.agent.domain.settings.SETUP,
                                                                   self.agent.domain.settings.FIRES,
                                                                   self.agent.domain.settings.HONEST_COMM_PROB),
                                          possible_next_states)


    @cython.boundscheck(True)
    @cython.cdivision(True)
    cdef void refresh_particles(self, State state, int internal_state):
        cdef BeliefNode_pfcomm old_tree = self.planner.tree

        # restart the tree
        cdef BeliefNode_pfcomm tree = BeliefNode_pfcomm(-1, -1, -1)
        self.planner.tree = tree

        # only use the known internal state (we don't model anyone else!)
        cdef WildfirePartialInternalStates partial_internal_states = WildfirePartialInternalStates(1)
        partial_internal_states.values[0] = internal_state
        partial_internal_states.agent_nums[0] = self.agent.agent_num

        # also only use the known State
        cdef Particle particle = Particle(state, partial_internal_states)
        cdef double weight = <double> 1/self.particles_n
        cdef int particle_i
        for particle_i in range(self.particles_n):
            tree.particle_filter.particles.append(particle)
            tree.particle_filter.weights.append(weight)


    cpdef Action choose_action_with_messages(self, State state, int internal_state, int[:] message_vector):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        # message estimator stack is empty
        cdef int models_n = len(self.planner.agent_models)
        cdef int model_i, neighbor_i
        cdef int message, neighbor_internal_state
        cdef Agent neighbor

        cdef PartialInternalStates partial_internal_states = PartialInternalStates(models_n + 1)
        partial_internal_states.values[0] = internal_state
        partial_internal_states.agent_nums[0] = self.agent.agent_num

        for model_i in range(models_n):
            neighbor = <Agent> self.planner.agent_models[model_i]
            neighbor_i = model_i + 1  # self agent is first in the messages and internal states

            message = message_vector[neighbor_i]
            if message == -1:
                # Assume a full suppressant so that agents optimistically
                # collaborate at the beginning since they will lean towards neighbors being around
                # Eg: message_vector = [-1,1,0,2,2,-1,0], the partial_internal_states = [2,1,0,2,2,2,0]
                neighbor_internal_state = 2
            else:
                neighbor_internal_state = message

            partial_internal_states.values[neighbor_i] = neighbor_internal_state
            partial_internal_states.agent_nums[neighbor_i] = neighbor.agent_num

        self.planner.message_estimator.start_internal_state_stack(partial_internal_states)

        return self.planner.create_plan(message_vector)


    cdef list calculate_q_values_with_messages(self, State state, int internal_state, int[:] message_vector):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        # message estimator stack is empty
        cdef int models_n = len(self.planner.agent_models)
        cdef int model_i, neighbor_i
        cdef int message, neighbor_internal_state
        cdef Agent neighbor

        cdef PartialInternalStates partial_internal_states = PartialInternalStates(models_n + 1)
        partial_internal_states.values[0] = internal_state
        partial_internal_states.agent_nums[0] = self.agent.agent_num

        for model_i in range(models_n):
            neighbor = <Agent> self.planner.agent_models[model_i]
            neighbor_i = model_i + 1  # self agent is first in the messages and internal states

            message = message_vector[neighbor_i]
            if message == -1:
                # Assume a full suppressant so that agents optimistically
                # collaborate at the beginning since they will lean towards neighbors being around
                # Eg: message_vector = [-1,1,0,2,2,-1,0], the partial_internal_states = [2,1,0,2,2,2,0]
                neighbor_internal_state = 2
            else:
                neighbor_internal_state = message

            partial_internal_states.values[neighbor_i] = neighbor_internal_state
            partial_internal_states.agent_nums[neighbor_i] = neighbor.agent_num

        self.planner.message_estimator.start_internal_state_stack(partial_internal_states)

        self.planner.create_plan(message_vector)
        return self.planner.root_q_values()


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        self.planner.make_observation(next_state, next_internal_state, observation)


cdef class WildfireIPOMCPPFReasoning(ReasoningModel):
    def __cinit__(self, WildfireAgent agent):
        self.agent = agent
        self.trajectories = agent.settings["trajectories"]
        self.particles_n = agent.settings["particles_n"]
        self.level = agent.settings["level"]


    cdef void create_planner(self, list possible_next_states=None, list agent_models=None):
        cdef WildfireAgent agent = <WildfireAgent> self.agent

        cdef WildfireReasoningType neighbor_reasoning_type = WildfireReasoningType.IPOMCPPF
        if self.level == 1:
            neighbor_reasoning_type = WildfireReasoningType.POMCPPF

        cdef double epsilon_p = agent.settings["epsilon_p"]
        if agent_models is None:
            if epsilon_p == 0.0:
                agent_models = agent.sample_all_agents(neighbor_reasoning_type)
            else:
                if not agent.enough_agents_for_modeling(epsilon_p, agent.domain.agents):
                    agent_models = agent.sample_agents_for_modeling_small(agent.domain.agents, neighbor_reasoning_type, agent.settings["cliques_n"])
                else:
                    agent_models = agent.sample_agents_for_modeling(epsilon_p, agent.domain.agents, neighbor_reasoning_type)

        # create the planner
        self.planner = IPOMCPPFPlanner(agent, self.level, self.agent.settings["horizon"], 0.99,
                                       agent.settings["ucb_c"], agent_models, self.trajectories, True, {}, {},
                                       possible_next_states)


    @cython.boundscheck(True)
    @cython.cdivision(True)
    cdef void refresh_particles(self, State state, int internal_state):
        cdef BeliefNode_ipf old_tree = self.planner.tree

        # restart the tree
        cdef BeliefNode_ipf tree = BeliefNode_ipf(-1, -1)
        self.planner.tree = tree

        # get the ids of the modeled neighbors
        cdef int models_n = len(self.planner.agent_models)
        cdef int[:] model_nums = np.zeros((models_n,), np.intc)
        cdef int model_i

        cdef Agent model
        for model_i in range(models_n):
            model = <Agent> self.planner.agent_models[model_i]
            model_nums[model_i] = model.agent_num

        # refresh the nested belief states about neighbor internal states
        tree.particle_filter = self.refresh_nested_particlefilters(state, internal_state, self.agent.agent_num,
                                                                   model_nums, old_tree.particle_filter, self.level)

    @cython.cdivision(True)
    cdef ParticleFilter refresh_nested_particlefilters(self, State state, int internal_state, int agent_num,
                                                       int[:] model_nums, ParticleFilter existing_pf, int level):
        cdef bint is_new = (len(existing_pf.particles) == 0)
        cdef int modeled_agents_n = len(model_nums)
        cdef int particle_i, particle_j, neighbor_num, model_i, nested_model_i, nested_internal_state_value
        cdef double uniform_weight = 1.0 / self.particles_n
        cdef int[:] nested_model_nums
        cdef NestedParticle particle
        cdef Particle neighbor_l0_particle
        cdef WildfireInternalStates internal_states
        cdef PartialInternalStates partial_internal_states, neighbor_partial_internal_states
        cdef ParticleFilter new_particle_filter, neighbor_particle_filter

        # do we need to make a new particle filter?
        if is_new:
            # make a new particle filter
            new_particle_filter = ParticleFilter()

            # create the particles
            for particle_i in range(self.particles_n):
                # this needs to remain random!
                internal_states = self.agent.domain.generate_start_internal_states()

                # save the internal states of our self (fully observable) and neighbors
                partial_internal_states = PartialInternalStates(modeled_agents_n + 1)
                partial_internal_states.values[0] = internal_state
                partial_internal_states.agent_nums[0] = agent_num

                for model_i in range(modeled_agents_n):
                    neighbor_num = model_nums[model_i]
                    partial_internal_states.values[model_i + 1] = internal_states.values[neighbor_num]
                    partial_internal_states.agent_nums[model_i + 1] = neighbor_num

                # create the particle
                particle = NestedParticle(state, partial_internal_states, level)
                new_particle_filter.particles.append(particle)
                new_particle_filter.weights.append(uniform_weight)
        else:
            # make sure the particle filter is normalized
            existing_pf.normalize()

            # resample the correct number of particles
            if len(existing_pf.particles) != self.particles_n:
                existing_pf.resample_particlefilter(self.particles_n)

            # use the existing one as our new particle filter
            new_particle_filter = existing_pf

        # update the particles with fully observable information
        for particle_i in range(self.particles_n):
            particle = <NestedParticle> new_particle_filter.particles[particle_i]

            # save the fully observable information (already done if we are making a new particle filter)
            if not is_new:
                particle.state = state
                particle.partial_internal_states.values[0] = internal_state

                # should we clear out the L0 particles since they will depend on particle.partial_internal_states?
                if level == 1:
                    particle.nested_particles.clear()

            # update the lower levels
            for model_i in range(modeled_agents_n):
                neighbor_num = model_nums[model_i]
                nested_internal_state_value = particle.partial_internal_states.values[model_i + 1]

                if level > 1:
                    # create the list of neighbors modeled by this neighbor
                    nested_model_nums = np.zeros((modeled_agents_n,), np.intc)
                    for nested_model_i in range(modeled_agents_n):
                        # the neighbor models the subject agent instead of itself
                        if nested_model_i == model_i:
                            nested_model_nums[nested_model_i] = agent_num
                        else:
                            nested_model_nums[nested_model_i] = model_nums[nested_model_i]

                    # recursively build this neighbor's particle filter
                    if is_new:
                        # we will need all new values
                        neighbor_particle_filter = ParticleFilter()
                    else:
                        neighbor_particle_filter = <ParticleFilter> particle.nested_particles[model_i]

                    neighbor_particle_filter = self.refresh_nested_particlefilters(state, nested_internal_state_value,
                                                                                   neighbor_num, nested_model_nums,
                                                                                   neighbor_particle_filter, level - 1)
                elif level == 1:
                    neighbor_particle_filter = ParticleFilter()

                    neighbor_partial_internal_states = PartialInternalStates(1)
                    neighbor_partial_internal_states.values[0] = nested_internal_state_value
                    neighbor_partial_internal_states.agent_nums[0] = neighbor_num

                    # also only use the known State
                    neighbor_l0_particle = Particle(state, neighbor_partial_internal_states)
                    for particle_j in range(self.particles_n):
                        neighbor_particle_filter.particles.append(neighbor_l0_particle)
                        neighbor_particle_filter.weights.append(uniform_weight)

                # save the neighbor's particle filter
                if is_new or level == 1:
                    particle.nested_particles.append(neighbor_particle_filter)
                else:
                    particle.nested_particles[model_i] = neighbor_particle_filter

        return new_particle_filter


    cdef Action choose_action(self, State state, int internal_state):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        return self.planner.create_plan()


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        self.planner.make_observation(next_state, next_internal_state, observation)


cdef class WildfireIPOMCPPFCommReasoning(ReasoningModel):
    def __cinit__(self, WildfireAgent agent):
        self.agent = agent
        self.trajectories = agent.settings["trajectories"]
        self.particles_n = agent.settings["particles_n"]
        self.level = agent.settings["level"]


    cdef void create_planner(self, list agent_neighbors, list possible_next_states=None, bint in_child_process=False):
        cdef WildfireAgent agent = <WildfireAgent> self.agent
        cdef WildfireAgent neighbor

        cdef int neighbors_n = len(agent_neighbors)
        cdef int[:] possible_messages = np.zeros((neighbors_n + 1,), dtype=np.intc)
        possible_messages[0] = len(self.agent.frame.possible_internal_states) + 1


        cdef WildfireReasoningType neighbor_reasoning_type = WildfireReasoningType.IPOMCPPFComm
        
        # if self.level == 1:
        #     neighbor_reasoning_type = WildfireReasoningType.POMCPPFComm

        cdef int neighbor_i
        cdef list agent_models = []
        for neighbor_i in range(neighbors_n):
            neighbor = <WildfireAgent> agent_neighbors[neighbor_i]
            possible_messages[neighbor_i + 1] = len(neighbor.frame.possible_internal_states) + 1
            agent_models.append(neighbor.replicate_as_model())

        # create the initial message received by the agent
        self.last_messages = -1 * np.ones((neighbors_n + 1,), dtype=np.intc)

        # create the planner
        self.planner = IPOMCPPFCommPlanner(agent, self.level, self.agent.settings["horizon"], 0.99,
                                           agent.settings["ucb_c"], agent_models, self.trajectories, True,
                                           possible_messages,
                                           WildfireMessageEstimator(self.agent.domain.settings.SETUP,
                                                                    self.agent.domain.settings.FIRES,
                                                                    self.agent.domain.settings.HONEST_COMM_PROB),
                                           possible_next_states)


    @cython.boundscheck(True)
    @cython.cdivision(True)
    cdef void refresh_particles(self, State state, int internal_state):
        cdef BeliefNode_ipfcomm old_tree = self.planner.tree

        # restart the tree
        cdef BeliefNode_ipfcomm tree = BeliefNode_ipfcomm(-1, -1, -1)
        self.planner.tree = tree

        # get the ids of the modeled neighbors
        cdef int models_n = len(self.planner.agent_models)
        cdef int[:] model_nums = np.zeros((models_n,), np.intc)
        cdef int model_i

        cdef Agent model
        for model_i in range(models_n):
            model = <Agent> self.planner.agent_models[model_i]
            model_nums[model_i] = model.agent_num

        # refresh the nested belief states about neighbor internal states
        tree.particle_filter = self.refresh_nested_particlefilters(state, internal_state, self.agent.agent_num,
                                                                   model_nums, old_tree.particle_filter, self.level)


    @cython.cdivision(True)
    cdef ParticleFilter refresh_nested_particlefilters(self, State state, int internal_state, int agent_num,
                                                       int[:] model_nums, ParticleFilter existing_pf, int level):
        cdef bint is_new = (len(existing_pf.particles) == 0)
        cdef int modeled_agents_n = len(model_nums)
        cdef int particle_i, particle_j, neighbor_num, model_i, nested_model_i, nested_internal_state_value
        cdef double uniform_weight = 1.0 / self.particles_n
        cdef int[:] nested_model_nums
        cdef NestedParticle particle
        cdef Particle neighbor_l0_particle
        cdef WildfireInternalStates internal_states
        cdef PartialInternalStates partial_internal_states, neighbor_partial_internal_states
        cdef ParticleFilter new_particle_filter, neighbor_particle_filter

        # do we need to make a new particle filter?
        if is_new:
            # make a new particle filter
            new_particle_filter = ParticleFilter()

            # create the particles
            for particle_i in range(self.particles_n):
                # this needs to remain random!
                internal_states = self.agent.domain.generate_start_internal_states()

                # save the internal states of our self (fully observable) and neighbors
                partial_internal_states = PartialInternalStates(modeled_agents_n + 1)
                partial_internal_states.values[0] = internal_state
                partial_internal_states.agent_nums[0] = agent_num

                for model_i in range(modeled_agents_n):
                    neighbor_num = model_nums[model_i]
                    partial_internal_states.values[model_i + 1] = internal_states.values[neighbor_num]
                    partial_internal_states.agent_nums[model_i + 1] = neighbor_num

                # create the particle
                particle = NestedParticle(state, partial_internal_states, level)
                new_particle_filter.particles.append(particle)
                new_particle_filter.weights.append(uniform_weight)
        else:
            # make sure the particle filter is normalized
            existing_pf.normalize()

            # resample the correct number of particles
            existing_pf.resample_particlefilter(self.particles_n)

            # use the existing one as our new particle filter
            new_particle_filter = existing_pf

        # update the particles with fully observable information
        for particle_i in range(self.particles_n):
            particle = <NestedParticle> new_particle_filter.particles[particle_i]

            # save the fully observable information (already done if we are making a new particle filter)
            if not is_new:
                particle.state = state
                particle.partial_internal_states.values[0] = internal_state

                # should we clear out the L0 particles since they will depend on particle.partial_internal_states?
                if level == 1:
                    particle.nested_particles.clear()

            # update the lower levels
            for model_i in range(modeled_agents_n):
                neighbor_num = model_nums[model_i]
                nested_internal_state_value = particle.partial_internal_states.values[model_i + 1]

                if level > 1:
                    # create the list of neighbors modeled by this neighbor
                    nested_model_nums = np.zeros((modeled_agents_n,), np.intc)
                    for nested_model_i in range(modeled_agents_n):
                        # the neighbor models the subject agent instead of itself
                        if nested_model_i == model_i:
                            nested_model_nums[nested_model_i] = agent_num
                        else:
                            nested_model_nums[nested_model_i] = model_nums[nested_model_i]

                    # recursively build this neighbor's particle filter
                    if is_new:
                        # we will need all new values
                        neighbor_particle_filter = ParticleFilter()
                    else:
                        neighbor_particle_filter = <ParticleFilter> particle.nested_particles[model_i]

                    neighbor_particle_filter = self.refresh_nested_particlefilters(state, nested_internal_state_value,
                                                                                   neighbor_num, nested_model_nums,
                                                                                   neighbor_particle_filter, level - 1)
                elif level == 1:
                    neighbor_particle_filter = ParticleFilter()

                    neighbor_partial_internal_states = PartialInternalStates(1)
                    neighbor_partial_internal_states.values[0] = nested_internal_state_value
                    neighbor_partial_internal_states.agent_nums[0] = neighbor_num

                    # also only use the known State
                    neighbor_l0_particle = Particle(state, neighbor_partial_internal_states)
                    for particle_j in range(self.particles_n):
                        neighbor_particle_filter.particles.append(neighbor_l0_particle)
                        neighbor_particle_filter.weights.append(uniform_weight)

                # save the neighbor's particle filter
                if is_new or level == 1:
                    particle.nested_particles.append(neighbor_particle_filter)
                else:
                    particle.nested_particles[model_i] = neighbor_particle_filter

        return new_particle_filter


    cdef Action choose_action(self, State state, int internal_state):
        # first refresh the particle filter
        self.refresh_particles(state, internal_state)

        return self.planner.create_plan(self.last_messages)


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        # save the last messages received
        cdef WildfireObservationComm observation_comm = <WildfireObservationComm> observation
        self.last_messages[0] = self.planner.last_action_node.action.message

        cdef int models_n = len(observation_comm.messages)
        cdef int model_i
        for model_i in range(models_n):
            self.last_messages[model_i + 1] = observation_comm.messages[model_i]

        self.planner.make_observation(next_state, next_internal_state, observation_comm)


cdef class WildfireHeuristicReasoning(ReasoningModel):
    def __cinit__(self, agent):
        self.agent = agent


    cdef Action choose_action(self, State state, int suppressant):
        return self.agent.sample_action(state, suppressant)


cdef class WildfireNOOPReasoning(ReasoningModel):
    def __cinit__(self, agent):
        self.agent = agent


    cdef Action choose_action(self, State state, int suppressant):
        cdef int noop = self.agent.domain.settings.FIRES
        return WildfireAction(noop)


cdef class WildfireCoordinationReasoning(ReasoningModel):
    """ A class for the co-ordination reasoning for the agents
    extending the ReasoningModel class.
    """
    def __cinit__(self, WildfireAgent agent):
        """Initialises the WildfireCoordinationReasoning class.
        Arguments:
            agent (object):
                An object of WildfireAgent class declared
                seperately for every agent
        """
        self.agent = agent
        # self.fire_intensities always contain fire intensities around an
        # agent at previous time step
        self.fire_intensities = [0]*(len(self.agent.frame.get_all_available_actions())-1)

    cdef Action choose_action(self, State state, int suppressant):
        """Performs the Co-ordination Reasoning among the agents.
        Arguments:
            state (State):
                An object of State class denoting the current state of
                the wildfire domain
            suppressant (int):
                A present level of the suppressants for the agent.
        Returns:
            Action (WildfireAction):
                An action that an agent takes
        """
        cdef int fires_n = self.agent.domain.settings.FIRES
        cdef int noop = fires_n
        cdef int burned = self.agent.domain.settings.FIRE_STATES - 1
        cdef int fire_i, f, a
        cdef list fires
        cdef int[:] available_actions

        # do we have any suppressant
        if suppressant == 0:
            # we must NOOP
            return WildfireAction(noop)
        else:
            # which fires can we fight?
            # temp history
            available_actions = self.agent.frame.get_available_actions(suppressant)
            fires, reduced_fires, present_intensities = [], [], []
            for fire_i in range(len(available_actions)):
                a = available_actions[fire_i]
                if a == noop:
                    continue

                # is this fire fightable?
                f = state.values[a]
                present_intensities.append(f)
                if f != 0 and f != burned:
                    # check the change in intensity
                    if self.fire_intensities[fire_i] - f > 0:
                        reduced_fires.append(a)
                    else:
                        fires.append(a)
            # Updating intensities in-order to use in next time step.
            self.fire_intensities = present_intensities[:]
            # Co-ordination Block
            if reduced_fires:
                # If there are any fires that reduced, fight that fire
                # or fight randomly among such fires
                a = rand() % len(reduced_fires)
                return WildfireAction(reduced_fires[a])

            elif len(fires) > 0:
                # pick an active fire at random
                a = rand() % len(fires)
                return WildfireAction(fires[a])

            else:
                # we have nothing to do, so NOOP
                return WildfireAction(noop)


cdef class WildfireMessageEstimator(MessageEstimator):
    def __cinit__(self, int setup, int fires, double honest_comm_prob = 1.00):
        self.setup = setup
        self.fires = fires
        self.honest_comm_prob = honest_comm_prob

    def __reduce__(self):
        return (rebuild_wildfiremessageestimator, (self.setup, self.fires, self.honest_comm_prob))


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef int estimate_message(self, int internal_state, int possible_messages_n, Agent agent):
        # Assume that agent's are are truthful honest_comm_prob times
        # and random for other messages rest of the times
        cdef double prob, other_comm_prob = (1.0 - self.honest_comm_prob) / (possible_messages_n - 1)
        cdef int message

        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        for message in range(-1, possible_messages_n - 1):  # -1 = no message
            if message == internal_state:
                prob = self.honest_comm_prob
            else:
                prob = other_comm_prob

            if rand_val < prob:
                break
            else:
                rand_val = rand_val - prob

        return message


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef int estimate_action(self, int message, Agent agent):
        cdef double expected_prob = 1.0
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        if message == 0:
            # they said they are out of suppressant, so they will take a noop
            return self.fires
        if self.setup >= 100 and self.setup < 200:
            if message == 2:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 0
                else:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 2
            elif message == 1:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 0
                    else:
                        return 1
                else:
                    if rand_val < expected_prob:
                        return 2
                    else:
                        return 1
            elif message == -1:
                if agent.frame.loc_num == 0:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 0
                    else:
                        return 1
                else:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 2
                    else:
                        return 1
        elif self.setup >= 200 and self.setup < 300:
            if message == 2:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 0
                elif agent.frame.loc_num == 1:
                    if rand_val < expected_prob:
                        return 1
                    elif rand_val < expected_prob + (1.0 - expected_prob) / 2:
                        return 0
                    else:
                        return 2
                else:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 2
            elif message == 1:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 0
                    else:
                        return 1
                elif agent.frame.loc_num == 1:
                    if rand_val < expected_prob / 2:
                        return 0
                    elif rand_val < expected_prob:
                        return 2
                    else:
                        return 1
                else:
                    if rand_val < expected_prob:
                        return 2
                    else:
                        return 1
            elif message == -1:
                if agent.frame.loc_num == 0:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 0
                    else:
                        return 1
                elif agent.frame.loc_num == 1:
                    if rand_val < 0.25:
                        return self.fires
                    elif rand_val < 0.50:
                        return 0
                    elif rand_val < 0.75:
                        return 1
                    else:
                        return 2
                else:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 2
                    else:
                        return 1
        elif self.setup >= 300 and self.setup < 400:
            if message == 2:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 0
                else:
                    if rand_val < expected_prob:
                        return 1
                    else:
                        return 2
            elif message == 1:
                if agent.frame.loc_num == 0:
                    if rand_val < expected_prob:
                        return 0
                    else:
                        return 1
                else:
                    if rand_val < expected_prob:
                        return 2
                    else:
                        return 1
            elif message == -1:
                if agent.frame.loc_num == 0:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 0
                    else:
                        return 1
                else:
                    if rand_val < 0.3333:
                        return self.fires
                    elif rand_val < 0.6667:
                        return 2
                    else:
                        return 1
        else:
            return self.fires


    cdef list update_internal_state_prob(self, int message, int possible_messages_n, list prior_probs):
        # no message contains no information, so the posterior equals the prior probs
        if message == -1:
            return prior_probs

        cdef double other_comm_prob = (1.0 - self.honest_comm_prob) / (possible_messages_n - 1)
        cdef double prob_m = 0.0
        cdef double prob_m_s, prob_s, numerator
        cdef int internal_states_n = len(prior_probs)
        cdef int internal_state
        cdef list posterior_probs = list()

        # calculate each numerator of the Bayesian update
        for internal_state in range(internal_states_n):
            if internal_state == message:
                prob_m_s = self.honest_comm_prob
            else:
                prob_m_s = other_comm_prob

            prob_s = <double> prior_probs[internal_state]

            numerator = prob_m_s * prob_s
            posterior_probs.append(numerator)
            prob_m = prob_m + numerator

        # update based on the total prob_m
        for internal_state in range(internal_states_n):
            numerator = <double> posterior_probs[internal_state]
            posterior_probs[internal_state] = numerator / prob_m

        return posterior_probs


def rebuild_wildfiremessageestimator(setup, fires, honest_comm_prob):
    return WildfireMessageEstimator(setup, fires, honest_comm_prob)
