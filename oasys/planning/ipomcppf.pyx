from oasys.agents.frame import Frame
from oasys.planning.pomcppf import BeliefNode as BeliefNode_pf, ActionNode as ActionNode_pf
from oasys.structures.planning_structures import Particle, ParticleFilter, NestedParticle
import numpy as np
import time

from oasys.agents.frame cimport Frame
from oasys.domains.domain cimport get_domain
from oasys.planning.pomcppf cimport BeliefNode as BeliefNode_pf, ActionNode as ActionNode_pf
from oasys.structures.planning_structures cimport Particle, ParticleFilter, NestedParticle
cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX
cdef extern from "math.h":
    double sqrt(double m)
    double log(double m)


global filename

def set_filename(run_number):
    global filename
    if run_number:
        filename = "ipomcppf_logger_run"+str(run_number)+".txt"
    else:
        filename = "ipomcppf_logger.txt"


def log_info(data, log_time=True):
    """Logs the data into a file
    Args:
        data:
            The information that needs to be logged into the file
        log_time:(bool)
            Logs the time of the information if true else just logs the
            information
    """
    global filename

    with open(filename, "a") as file1:
        sec = time.time()
        local_time = time.ctime(sec)
        content = f'{data}\n\n'
        if log_time:
            content = f'{local_time}\n' + content
        # Writing data to a file
        file1.write(content)


cdef class IPOMCPPFPlanner:
    def __cinit__(self, Agent agent, int level, int horizon, double gamma, double ucb_c,
                  list agent_models, int trajectories, bint fully_observable_environment, dict pomcp_policy,
                  dict cached_neighbor_next_full_particle_sets, list possible_next_states=None,
                  int[:,:,:,:,:] global_pomcppf_policy=None):
        self.agent = agent
        self.domain = get_domain()
        self.horizon = horizon
        self.gamma = gamma
        self.agent_models = agent_models
        self.models_n = len(agent_models)
        self.ucb_c = ucb_c
        self.q_sensitivity = 0.001
        self.last_action_node = None
        self.possible_next_states = possible_next_states
        self.trajectories = trajectories
        self.level = level
        self.fully_observable_environment = fully_observable_environment
        self.pomcp_policy = pomcp_policy
        self.cached_neighbor_next_full_particle_sets = cached_neighbor_next_full_particle_sets
        self.reuse_tree = False

        self.use_premodel = self.agent.settings["premodel"]
        self.cached_level0_neighbors_particle_filter = dict()

        if global_pomcppf_policy == None:
            self.global_pomcppf_policy = self.agent.settings["POMCPPF_global_policy"]
        else:
            self.global_pomcppf_policy = global_pomcppf_policy

        self.neighbor_planners = self.create_neighbor_planners()

        self.neighbor_next_full_particle_sets = list()
        self.tree = BeliefNode(-1, -1)

        set_filename(self.agent.settings['run'])

        # count the number of agents modeled per each frame
        cdef Agent neighbor
        cdef int[:] total_agents_per_frame = self.domain.num_agents_per_frame()
        cdef int frames_n = len(total_agents_per_frame)
        cdef int modeled_agents_n = len(agent_models)
        cdef int agent_i

        self.agents_modeled_per_frame = np.zeros((frames_n,), dtype=np.int32)
        for agent_i in range(modeled_agents_n):
            neighbor = agent_models[agent_i]
            self.agents_modeled_per_frame[neighbor.frame.index] += 1

        cdef Frame frame
        cdef int frame_i
        self.available_actions_per_frame = np.zeros((frames_n,), dtype=np.intc)
        for frame_i in range(frames_n):
            frame = <Frame> self.domain.frames[frame_i]
            self.available_actions_per_frame[frame_i] = len(frame.get_all_available_actions())


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cpdef Action create_plan(self):
        cdef int trajectories = self.trajectories
        cdef int traj
        cdef bint print_traj = not self.agent.settings["premodel"] or \
                               (self.agent.settings["level"] > 1 and self.level == self.agent.settings["level"])

        for traj in range(trajectories):
            self.update_tree(self.tree.particle_filter, 0, self.tree)

            if print_traj and traj % 100 == 0:
                print("Agent", self.agent.agent_num, "Level:", self.level,"Trajectories:", traj)

        # choose the best action
        cdef ActionNode best_action_node
        if self.level == self.agent.settings["level"]:
            print("Agent:", self.agent.agent_num, "Iterations:", trajectories)
            log_info(f"Agent:{self.agent.agent_num}, Iterations: {trajectories}")
            best_action_node = self.tree.argmax_q(self.q_sensitivity, self.ucb_c)
        else:
            best_action_node = self.tree.argmax_q(self.q_sensitivity)

        # save the action taken
        self.last_action_node = best_action_node
        return best_action_node.action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double update_tree(self, ParticleFilter particle_filter, int h, BeliefNode belief_node):
        # update the particle filter
        if h == 1 or (self.reuse_tree and h > 0):
            belief_node.particle_filter.particles.extend(particle_filter.particles)
            belief_node.particle_filter.weights.extend(particle_filter.weights)

        # have we reached the end of the tree?
        if h >= self.horizon:
            return 0.0

        # is belief_node a new leaf?
        if belief_node.visits == 0:
            belief_node.visits += 1  # the sum of weights in the particle filter is 1 (see note at end of function)
            return self.rollout_all(particle_filter, h)

        cdef bint first_sample = False
        cdef NestedParticle particle
        if len(belief_node.action_nodes) == 0:
            if self.fully_observable_environment:
                particle = <NestedParticle> particle_filter.particles[0]
                belief_node.create_action_nodes(self.agent, particle.state, particle.partial_internal_states.values[0])
            else:
                belief_node.create_action_nodes(self.agent)
                first_sample = True

        cdef int agent_num = self.agent.agent_num
        cdef ActionNode action_node
        if self.fully_observable_environment:
            particle = <NestedParticle> particle_filter.particles[0]
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent, particle.state,
                                               particle.partial_internal_states.values[0])
        else:
            action_node = belief_node.argmax_ucb1(self.ucb_c, self.agent)

        cdef Action action = action_node.action
        cdef list observations_particle_filters = list()
        cdef list observation_weights = list()
        cdef list all_observations = self.domain.observations
        cdef int internal_states_n = len(self.agent.frame.possible_internal_states)
        cdef int obs_n = len(all_observations)
        cdef int obs_i, internal_state
        for internal_state in range(internal_states_n):
            for obs_i in range(obs_n):
                observations_particle_filters.append(ParticleFilter())
                observation_weights.append(0)

        cdef int particles_n = len(particle_filter.particles)
        cdef double reward = 0.0
        cdef double obs_weight_sum = 0.0
        cdef int particle_i, next_internal_state, index, index_base
        cdef Particle new_particle
        cdef double weight, prob
        cdef State state, next_state
        cdef PartialInternalStates internal_states, next_internal_states
        cdef PartialJointAction modeled_joint_action
        cdef Configuration configuration
        cdef Observation observation
        cdef ParticleFilter new_particle_filter

        for particle_i in range(particles_n):
            particle = <NestedParticle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]

            state = particle.state
            internal_states = particle.partial_internal_states
            modeled_joint_action = self.sample_modeled_joint_action(particle, h)
            configuration = self.sample_configuration(modeled_joint_action)

            # self.agent is the first spot in modeled_joint_action since its partial
            modeled_joint_action.actions[0] = action.index
            modeled_joint_action.agent_nums[0] = agent_num

            next_state = None
            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states, modeled_joint_action)
            new_particle = self.update_nested_beliefs(particle, configuration, modeled_joint_action, next_state,
                                                      next_internal_states)

            reward = reward + weight * self.domain.reward_configuration(agent_num, state, internal_states,
                                                                        configuration, action, next_state,
                                                                        next_internal_states)
            next_internal_state = next_internal_states.values[0]
            index_base = next_internal_state * obs_n
            for obs_i in range(obs_n):
                observation = <Observation> all_observations[obs_i]
                index = index_base + obs_i
                new_particle_filter = <ParticleFilter> observations_particle_filters[index]

                prob = weight * self.domain.observation_probability_configuration(agent_num, state, configuration,
                                                                                  action, next_state, observation)
                new_particle_filter.particles.append(new_particle)
                new_particle_filter.weights.append(prob)
                observation_weights[index] = (<double> observation_weights[index]) + prob
                obs_weight_sum += prob

        # Normalizing observation weights
        for internal_state in range(internal_states_n):
            index_base = internal_state * obs_n
            for obs_i in range(obs_n):
                index = index_base + obs_i
                observation_weights[index] = (<double> observation_weights[index]) / obs_weight_sum

        # sampling an observation/internal state pair
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        cdef int last
        cdef int indices = internal_states_n * obs_n
        for index in range(indices):
            prob = <double> observation_weights[index]
            last = index

            if prob > rand_val:
                break
            else:
                rand_val -= prob

        obs_i = last % obs_n
        next_internal_state = last // obs_n
        observation = <Observation> all_observations[obs_i]

        # grab the corresponding particle filter and reweight it
        new_particle_filter = <ParticleFilter> observations_particle_filters[last]
        new_particle_filter.normalize()
        new_particle_filter.resample_particlefilter()

        cdef BeliefNode next_belief_node = action_node.get_belief(observation.index, next_internal_state)
        cdef double R = reward + self.gamma * self.update_tree(new_particle_filter, h+1, next_belief_node)
        belief_node.visits += 1 # as long as reward is a weighted sum above, this needs to be a 1 (total weight is 1)
        action_node.visits += 1 # so if these are not 1, then the average below for Q is broken

        # update action_node's Q value
        action_node.q_value += (R - action_node.q_value) / action_node.visits
        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef NestedParticle update_nested_beliefs(self, NestedParticle particle, Configuration configuration,
                                              PartialJointAction modeled_joint_action, State next_state,
                                              PartialInternalStates next_internal_states):
        cdef NestedParticle next_particle = NestedParticle(next_state, next_internal_states, particle.level)

        cdef int models_n = self.models_n
        cdef int desired_particles_n = <int> self.agent.settings["particles_n"]
        cdef int model_i, next_internal_state, particles_n, particle_i, list_n, valid_obs_n, valid_obs_i, obs_i
        cdef int action_i
        cdef double prob, total_prob
        cdef list all_neighbor_next_particles, neighbor_filters, valid_obs, probs
        cdef ParticleFilter neighbor_filter, other_filter
        cdef NestedParticle neighbor_particle
        cdef Observation obs
        cdef Agent neighbor
        cdef Action action
        cdef int next_state_index = <int> next_state.index
        cdef tuple tup

        cdef Particle level_0_particle
        cdef PartialInternalStates neighbor_partial_internal_states
        cdef int neighbor_agent_num
        cdef ParticleFilter neighbor_particle_filter

        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_agent_num = neighbor.agent_num
            # get the neighbor's new belief particles
            next_internal_state = next_internal_states.values[model_i + 1]
            if self.level==1: # and self.use_premodel:
                if self.fully_observable_environment:
                    tup = tuple((next_state_index, neighbor_agent_num, next_internal_state))

                    if tup in self.cached_level0_neighbors_particle_filter:
                        neighbor_particle_filter = <ParticleFilter> self.cached_level0_neighbors_particle_filter[tup]
                    else:
                        # Creating a particle filter with single particle with single partial internal states
                        neighbor_partial_internal_states = PartialInternalStates(1)
                        neighbor_partial_internal_states.agent_nums[0] = <int> neighbor_agent_num
                        neighbor_partial_internal_states.values[0] = <int> next_internal_state

                        level_0_particle = Particle(next_state, neighbor_partial_internal_states)
                        neighbor_particle_filter = ParticleFilter()
                        neighbor_particle_filter.particles.append(level_0_particle)
                        neighbor_particle_filter.weights.append(<double> 1.0)

                        # Cache this particle filter
                        self.cached_level0_neighbors_particle_filter[tup] = <ParticleFilter> neighbor_particle_filter
                    next_particle.nested_particles.append(neighbor_particle_filter)
                else:
                    # Code for a partially observable environment
                    pass

            else:
                all_neighbor_next_particles = <list> self.neighbor_next_full_particle_sets[model_i]
                neighbor_filters = <list> all_neighbor_next_particles[next_internal_state]

                # create the expected weights of all particles for the given next internal state for this neighbor
                list_n = len(neighbor_filters)
                valid_obs = <list> neighbor_filters[list_n - 1]
                valid_obs_n = len(valid_obs)

                if valid_obs_n > 0:
                    neighbor = <Agent> self.agent_models[model_i]
                    action_i = modeled_joint_action.actions[model_i + 1]
                    action = <Action> self.domain.actions[action_i]

                    probs = list()
                    total_prob = 0.0
                    for valid_obs_i in range(valid_obs_n):
                        obs_i = <int> valid_obs[valid_obs_i]
                        obs = <Observation> self.domain.observations[obs_i]

                        # TODO remove action from configuration? not needed for Wildfire or CyberSec...
                        prob = self.domain.observation_probability_configuration(neighbor.agent_num, particle.state,
                                                                                configuration, action, next_state, obs)
                        probs.append(prob)
                        total_prob = total_prob + prob
                        # print(prob, total_prob, flush=True)

                    # use the first particle filter for the new particle
                    obs_i = <int> valid_obs[0]
                    neighbor_filter = <ParticleFilter> neighbor_filters[obs_i]

                    if valid_obs_n > 1:
                        # update the weights of the first particle filter
                        prob = (<double> probs[0]) / total_prob
                        particles_n = len(neighbor_filter.weights)
                        for particle_i in range(particles_n):
                            neighbor_filter.weights[particle_i] = (<double> neighbor_filter.weights[particle_i]) * prob

                        # add the particles from the other observations with updated weights
                        for valid_obs_i in range(1, valid_obs_n):
                            prob = (<double> probs[valid_obs_i]) / total_prob
                            obs_i = valid_obs[valid_obs_i]
                            other_filter = <ParticleFilter> neighbor_filters[obs_i]

                            particles_n = len(other_filter.weights)
                            for particle_i in range(particles_n):
                                weight = (<double> other_filter.weights[particle_i]) * prob

                                neighbor_filter.particles.append(other_filter.particles[particle_i])
                                neighbor_filter.weights.append(weight)

                    # normalize and resample the particle filter
                    neighbor_filter.normalize()
                    neighbor_filter.resample_particlefilter(desired_particles_n)
                    next_particle.nested_particles.append(neighbor_filter)
                else:
                    print("No next particle filter for neighbor", model_i, "from agent", self.agent.agent_num)
                    next_particle.nested_particles.append(ParticleFilter())

        return next_particle


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double rollout_all(self, ParticleFilter particle_filter, int h):
        cdef double R = 0.0
        cdef int particles_n = len(particle_filter.particles)

        cdef Particle particle
        cdef double weight
        cdef int particle_i

        for particle_i in range(particles_n):
            particle = <Particle> particle_filter.particles[particle_i]
            weight = <double> particle_filter.weights[particle_i]
            R = R + weight * self.rollout(particle, h)

        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef double rollout(self, NestedParticle particle, int h):
        cdef Configuration configuration
        cdef Action action
        cdef State next_state
        cdef PartialInternalStates next_internal_states
        cdef PartialJointAction random_joint_action
        cdef int agent_num = self.agent.agent_num
        cdef double R = 0.0
        cdef double running_gamma = 1.0

        cdef State state = particle.state
        cdef PartialInternalStates internal_states = particle.partial_internal_states

        for h in range(h, self.horizon):
            # simulate the environment
            random_joint_action = self.sample_random_joint_action(state, internal_states)
            configuration = self.sample_configuration(random_joint_action)
            action = self.agent.sample_action(state, internal_states.values[0])

            # self.agent is the first spot in random_joint_action since its partial
            random_joint_action.actions[0] = action.index
            random_joint_action.agent_nums[0] = agent_num

            if self.possible_next_states is not None:
                next_state = self.domain.sample_next_state_configuration_from_possible(agent_num, state, configuration,
                                                                                       action,
                                                                                       self.possible_next_states[state.index])
            else:
                next_state = self.domain.sample_next_state_configuration(agent_num, state, configuration, action)

            next_internal_states = self.domain.sample_next_partial_internal_states(internal_states,
                                                                                   random_joint_action)

            # self.agent is the first spot in internal_states since its partial
            reward = self.domain.reward_configuration(agent_num, state, internal_states, configuration, action,
                                                      next_state, next_internal_states)

            # update the cumulative reward
            R = R + running_gamma * reward

            # prepare for the next iteration
            state = next_state
            internal_states = next_internal_states
            running_gamma = running_gamma * self.gamma

        return R


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef list create_neighbor_planners(self):
        cdef list neighbor_planners = list()

        cdef Agent model = self.agent.replicate_as_model()
        cdef Agent neighbor

        cdef int models_n = self.models_n
        cdef int model_i
        cdef list neighbor_models

        cdef IPOMCPPFPlanner ipomcppf_planner
        cdef POMCPPFPlanner pomcppf_planner
        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_models = self.agent_models[:] # make a deep copy so we can edit it below

            # deleting the model of neighbor and creating the IPOMCPNMCTS model for the agent
            neighbor_models[model_i] = model
            #print(type(self.global_pomcppf_policy), self.global_pomcppf_policy)

            if self.level > 1:
                ipomcppf_planner = IPOMCPPFPlanner(neighbor, self.level - 1, self.horizon, self.gamma,
                                                   self.ucb_c, neighbor_models, self.trajectories,
                                                   self.fully_observable_environment, self.pomcp_policy,
                                                   self.cached_neighbor_next_full_particle_sets,
                                                   self.possible_next_states, self.global_pomcppf_policy)
                neighbor_planners.append(ipomcppf_planner)
            else:
                pomcppf_planner = POMCPPFPlanner(neighbor, self.horizon, self.gamma, self.ucb_c, neighbor_models,
                                                 self.fully_observable_environment, self.possible_next_states)
                neighbor_planners.append(pomcppf_planner)

        return neighbor_planners


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.profile(True)
    cdef PartialJointAction sample_modeled_joint_action(self, NestedParticle particle, int h):
        cdef int models_n = self.models_n

        # clear the particle sets found for our neighbors in this step
        if self.level > 1 or not self.use_premodel:
            self.neighbor_next_full_particle_sets.clear()

        # creating a partial joint action
        cdef PartialJointAction modeled_joint_action = PartialJointAction(models_n + 1)
        modeled_joint_action.agent_nums[0] = self.agent.agent_num

        cdef Action neighbor_action
        cdef int model_i, neighbor_i, best_actions_n, rand_i

        # plan actions for our neighbors using lower level reasoning
        cdef IPOMCPPFPlanner ipomcppf_planner
        cdef POMCPPFPlanner pomcppf_planner
        cdef ActionNode action_node
        cdef int neighbor_horizon, state_index
        cdef Agent neighbor
        cdef ActionNode_pf action_node_pf
        cdef int internal_state, frame_index, global_policy_action_index
        cdef tuple pomcp_state
        cdef list next_particle_set

        state_index = particle.state.index
        # Synchronized horizon for the neighbor planners
        neighbor_horizon = self.horizon - h


        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1 # self.agent is always first
            internal_state = particle.partial_internal_states.values[neighbor_i]
            frame_index = neighbor.frame.index

            if self.level > 1:
                ipomcppf_planner = <IPOMCPPFPlanner> self.neighbor_planners[model_i]
                ipomcppf_planner.horizon = neighbor_horizon
                ipomcppf_planner.tree.particle_filter = particle.nested_particles[model_i]
                ipomcppf_planner.tree.particle_filter.resample_particlefilter(self.agent.settings["particles_n"]//2)

                # get the neighbor's action
                neighbor_action = ipomcppf_planner.create_plan()

                # get the neighbor's new belief particle set
                action_node = ipomcppf_planner.last_action_node
                next_particle_set = action_node.all_belief_particles()
                self.neighbor_next_full_particle_sets.append(next_particle_set)
                ipomcppf_planner.reset()
            else:
                # have we already cached this neighbor's action?
                # including the neighbor horizon as an additional key due to synchronized horizon
                if self.use_premodel:
                    # Using a global offline policy
                    best_actions_n = self.global_pomcppf_policy[frame_index][state_index][internal_state][neighbor_horizon-1][0]

                    if best_actions_n == 1:
                        global_policy_action_index = self.global_pomcppf_policy[frame_index][state_index][internal_state][neighbor_horizon-1][1]
                    else:
                        rand_i = rand() % best_actions_n
                        global_policy_action_index = self.global_pomcppf_policy[frame_index][state_index][internal_state][neighbor_horizon - 1][rand_i + 1]
                    # global_policy_action_index = self.global_pomcppf_policy[frame_index][state_index][internal_state][neighbor_horizon-1]
                    # global_policy_action_index = self.global_pomcppf_policy[pomcp_state]
                    modeled_joint_action.actions[neighbor_i] = global_policy_action_index
                    modeled_joint_action.agent_nums[neighbor_i] = neighbor.agent_num
                    continue
                else:
                    pomcp_state = tuple((state_index, internal_state,frame_index, neighbor_horizon))

                    if self.fully_observable_environment and pomcp_state in self.pomcp_policy:
                        neighbor_action = <Action> self.pomcp_policy[pomcp_state]
                        next_particle_set = <list> self.cached_neighbor_next_full_particle_sets[pomcp_state]
                        self.neighbor_next_full_particle_sets.append(next_particle_set)

                    else:
                        pomcppf_planner = <POMCPPFPlanner> self.neighbor_planners[model_i]
                        pomcppf_planner.horizon = neighbor_horizon
                        pomcppf_planner.tree.particle_filter = particle.nested_particles[model_i]
                        pomcppf_planner.tree.particle_filter.resample_particlefilter(1)

                        # get the neighbor's action
                        neighbor_action = pomcppf_planner.create_plan(self.trajectories)

                        # get the neighbor's new belief particle set
                        action_node_pf = pomcppf_planner.last_action_node
                        next_particle_set = action_node_pf.all_belief_particles()
                        self.neighbor_next_full_particle_sets.append(next_particle_set)
                        pomcppf_planner.reset()

                        # cache this action
                        if self.fully_observable_environment:
                            self.pomcp_policy[pomcp_state] = neighbor_action
                            self.cached_neighbor_next_full_particle_sets[pomcp_state] = next_particle_set

            modeled_joint_action.actions[neighbor_i] = neighbor_action.index
            modeled_joint_action.agent_nums[neighbor_i] = neighbor.agent_num

        return modeled_joint_action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef PartialJointAction sample_random_joint_action(self, State state, PartialInternalStates internal_states):
        # creating a partial joint action
        cdef int models_n = self.models_n
        cdef PartialJointAction random_joint_action = PartialJointAction(models_n + 1)
        random_joint_action.agent_nums[0] = self.agent.agent_num

        cdef Agent neighbor
        cdef int[:] available_actions
        cdef int model_i, available_actions_n, agent_i, rand_i, random_action, neighbor_i

        # pick a random action for each agent that is modeled at higher level of reasoning
        for model_i in range(models_n):
            neighbor = <Agent> self.agent_models[model_i]
            neighbor_i = model_i + 1 # self.agent is always first

            internal_state = internal_states.values[neighbor_i]
            available_actions = neighbor.frame.get_available_actions(internal_state)
            available_actions_n = len(available_actions)

            # pick a random available action for that agent
            rand_i = rand() % available_actions_n
            random_action = available_actions[rand_i]

            random_joint_action.actions[neighbor_i] = random_action
            random_joint_action.agent_nums[neighbor_i] = neighbor.agent_num

        return random_joint_action


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef Configuration sample_configuration(self, PartialJointAction joint_action):
        cdef Configuration small_configuration = self.domain.create_configuration_from_partial(self.agent.agent_num,
                                                                                               joint_action)
        if len(joint_action.actions) == self.domain.num_agents():
            return small_configuration

        # calculate the proportions for each frame/action
        cdef int frames_n = len(self.agents_modeled_per_frame)
        cdef int config_n = len(small_configuration.actions)
        cdef list prop_per_action = list()
        cdef int frame_offset = 0
        cdef int actions_n, config_i, frame_i, action_i
        cdef double prob, frame_size

        for frame_i in range(frames_n):
            frame_size = self.agents_modeled_per_frame[frame_i]
            actions_n = self.available_actions_per_frame[frame_i]

            for action_i in range(actions_n):
                config_i = action_i  + frame_offset
                prob = small_configuration.actions[config_i] / frame_size
                prop_per_action.append(prob)

            frame_offset += actions_n

        # print(np.array(prop_per_action), np.sum(prop_per_action), np.sum(small_configuration.actions))

        # randomly sample a configuration based on those proportions
        cdef Configuration configuration = Configuration(config_n)

        cdef int[:] agents_per_frame = self.domain.num_agents_per_frame()
        cdef int agent_i, last, agents_n
        cdef double rand_val
        frame_offset = 0
        for frame_i in range(frames_n):
            actions_n = self.available_actions_per_frame[frame_i]

            agents_n = agents_per_frame[frame_i]
            if frame_i == self.agent.frame.index:
                # don't include ourselves in the configuration
                agents_n -= 1

            # randomly sample an action for each agent of this frame
            for agent_i in range(agents_n):
                rand_val = rand() / (RAND_MAX + 1.0)

                for action_i in range(actions_n):
                    config_i = action_i  + frame_offset
                    prob = <double> prop_per_action[config_i]

                    if prob > 0.0:
                        last = config_i

                        if prob > rand_val:
                            break
                        else:
                            rand_val -= prob

                # increase the count of the chosen action
                configuration.actions[last] += 1

            frame_offset += actions_n

        return configuration


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        cdef int next_state_index = 0
        if next_state is not None:
            next_state_index = next_state.index

        # update the root of the tree
        self.tree = self.last_action_node.get_belief(observation.index, next_internal_state)
        self.tree.parent_action_node = None # for garbage collection
        self.last_action_node = None


    cdef void reset(self):
        self.tree = BeliefNode(-1, -1)
        self.last_action_node = None


cdef class BeliefNode:
    def __cinit__(self, int observation_index, int internal_state, ActionNode parent_action_node=None):
        self.observation_index = observation_index
        self.internal_state = internal_state
        self.visits = 0
        self.particle_filter = ParticleFilter()
        self.action_nodes = list()
        self.parent_action_node = parent_action_node


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef void create_action_nodes(self, Agent agent, State state=None, int internal_state=0):
        cdef int[:] available_actions
        if state is not None:
            available_actions = agent.frame.get_available_actions_per_state(state, internal_state)
        else:
            available_actions = agent.frame.get_all_available_actions()

        cdef int actions_n = len(available_actions)
        cdef int actions_i
        cdef ActionNode action_node
        cdef Action action

        for action_i in range(actions_n):
            action = <Action> agent.domain.actions[available_actions[action_i]]
            action_node = ActionNode(action, action_i, self)
            self.action_nodes.append(action_node)


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef NestedParticle sample_particle(self):
        cdef int particles_n = len(self.particle_set)
        cdef int rand_i = rand() % particles_n
        return self.particle_set[rand_i]


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef ActionNode argmax_ucb1(self, double ucb_c, Agent agent, State state = None, int internal_state = -1):
        # find the ActionNode with the highest UCB-1 value
        cdef ActionNode best_action_node = None
        cdef ActionNode action_node
        cdef int[:] available_actions

        if state is not None:
            available_actions = agent.frame.get_available_actions_per_state(state, internal_state)
        else:
            available_actions = agent.frame.get_all_available_actions()

        cdef int n_visits = self.visits
        cdef int actions_n = len(self.action_nodes)
        cdef int available_actions_n = len(available_actions)
        cdef int action_i, available_action_i, rand_i

        cdef double ucb1
        cdef double best_ucb1 = -99999999
        cdef bint available

        cdef list best_action_nodes = []
        cdef int best_action_nodes_n = 0

        for action_i in range(actions_n):
            action_node = <ActionNode> self.action_nodes[action_i]

            # is this action possible in (state, internal_state)?
            available = False
            for available_action_i in range(available_actions_n):
                if action_node.action.index == available_actions[available_action_i]:
                    available = True
                    break

            if available:
                if action_node.visits > 0:
                    ucb1 = action_node.q_value + ucb_c * sqrt(log(n_visits) / action_node.visits)
                else:
                    ucb1 = 10000000

                # print("A:", action_node.action.index, "V:", action_node.visits, "Q:", action_node.q_value, "UCB:", ucb1)

                if ucb1 > best_ucb1:
                    best_ucb1 = ucb1
                    best_action_node = action_node
                    best_action_nodes_n = 1
                elif ucb1 == best_ucb1:
                    if best_action_nodes_n == 1:
                        best_action_nodes = list()
                        best_action_nodes.append(best_action_node)

                    best_action_nodes.append(action_node)
                    best_action_nodes_n = best_action_nodes_n + 1

        if best_action_nodes_n == 1:
            return best_action_node
        else:
            rand_i = rand() % best_action_nodes_n
            return best_action_nodes[rand_i]


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    cdef ActionNode argmax_q(self, double q_sensitivity, double ucb_c=0.0):
        # find the ActionNode with the highest Q value
        cdef ActionNode action_node
        cdef list best_action_nodes = []

        cdef int best_action_nodes_n = 0
        cdef int actions_n = len(self.action_nodes)
        cdef int action_i

        cdef double q, ucb1
        cdef double best_q = -99999999

        for action_i in range(actions_n):
            action_node = <ActionNode> self.action_nodes[action_i]
            q = action_node.q_value

            if action_node.visits > 0:
                if q > best_q + q_sensitivity:
                    best_q = q
                    best_action_nodes = [action_node]
                    best_action_nodes_n = 1
                elif q > best_q - q_sensitivity:
                    best_action_nodes.append(action_node)
                    best_action_nodes_n += 1

            if ucb_c > 0.0:
                if action_node.visits == 0:
                    ucb1 = 10000000
                else:
                    ucb1 = q + ucb_c * sqrt(log(self.visits) / action_node.visits)
                print("Action:", action_node.action.index, "q:", action_node.q_value,
                      "visits:", action_node.visits, "ucb1:", ucb1)
                log_info(f"Action:{ action_node.action.index}, q:{action_node.q_value},visits:{action_node.visits}, ucb1:{ucb1}", False)


        # randomly break ties, if needed
        cdef int rand_i
        if best_action_nodes_n == 1:
            return best_action_nodes[0]
        else:
            rand_i = rand() % best_action_nodes_n
            return best_action_nodes[rand_i]


cdef class ActionNode:
    def __cinit__(self, Action action, int available_action_i, BeliefNode parent_belief_node):
        self.action = action
        self.available_action_index = available_action_i
        self.visits = 0
        self.q_value = 0.0
        self.belief_nodes = {}
        self.all_belief_nodes = list() # for iterating over quickly in all_belief_particles
        self.parent_belief_node = parent_belief_node


    cdef BeliefNode get_belief(self, int observation_index, int internal_state):
        cdef tuple t = tuple((internal_state, observation_index))
        cdef BeliefNode belief_node
        if t in self.belief_nodes:
            return self.belief_nodes[t]
        else:
            belief_node = BeliefNode(observation_index, internal_state, self)
            self.belief_nodes[t] = belief_node
            self.all_belief_nodes.append(belief_node)
            return belief_node


    cdef list all_belief_particles(self):
        cdef list all_particles = list()
        cdef list internal_list

        cdef int children_n = len(self.all_belief_nodes)
        cdef int child_i, particles_n, particle_i, internal_state, obs_i
        cdef int max_internal_state = -1
        cdef int max_obs = -1
        cdef BeliefNode node
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            if node.internal_state > max_internal_state:
                max_internal_state = node.internal_state
            if node.observation_index > max_obs:
                max_obs = node.observation_index

        for internal_state in range(max_internal_state + 1):
            internal_list = list()
            all_particles.append(internal_list)

            # these will be replaced by particle filters
            for obs_i in range(max_obs + 1):
                internal_list.append(None)

            # the eventual list for valid obs_i for this internal_state
            internal_list.append(list())

        cdef int list_i = max_obs + 1
        cdef list valid_obs_i
        for child_i in range(children_n):
            node = <BeliefNode> self.all_belief_nodes[child_i]
            internal_list = <list> all_particles[node.internal_state]
            internal_list[node.observation_index] = node.particle_filter

            valid_obs_i = <list> internal_list[list_i]
            valid_obs_i.append(node.observation_index)

        return all_particles
