""" Defines the wildfire domain and its dynamics.

Author: Adam Eck
"""


from oasys.agents.agent import Agent
from oasys.structures.pomdp_structures import (State, InternalStates,
                                               PartialInternalStates,
                                               Action, ActionComm,
                                               JointAction, PartialJointAction,
                                               JointActionComm, Observation)
from .wildfire_agent import WildfireAgent
from .wildfire_frame import WildfireFrame
from .wildfire_reasoning import (WildfireReasoningType,
                                 WildfirePOMCPPFReasoning,
                                 WildfirePOMCPPFCommReasoning,
                                 WildfireIPOMCPPFReasoning,
                                 WildfireIPOMCPPFCommReasoning)
from .wildfire_structures import WildfireInternalStates, WildfirePartialInternalStates, WildfireAction, WildfireActionComm
import numpy as np

from oasys.agents.agent cimport Agent
from oasys.domains.domain cimport set_domain
from oasys.structures.pomdp_structures cimport (State, InternalStates,
                                                PartialInternalStates,
                                                Action, ActionComm,
                                                JointAction, PartialJointAction,
                                                JointActionComm, Observation)
from .wildfire_agent cimport WildfireAgent
from .wildfire_frame cimport WildfireFrame
from .wildfire_reasoning cimport (WildfireReasoningType,
                                  WildfirePOMCPPFReasoning,
                                  WildfirePOMCPPFCommReasoning,
                                  WildfireIPOMCPPFReasoning,
                                  WildfireIPOMCPPFCommReasoning)
from .wildfire_structures cimport WildfireInternalStates, WildfirePartialInternalStates, WildfireAction, WildfireActionComm
cimport numpy as np
cimport cython

from libc.stdlib cimport rand, RAND_MAX


cdef class Wildfire(Domain):
    """ A class representing the parameters and dynamics of the Wildfire domain.

    Attributes:
    settings -- An instance of the WildfireSettings class
    states -- A list of all possible WildfireStates
    actions -- A list of all possible WildfireActions
    observations -- A list of all possible WildfireActions
    frames -- A list of all possible WildfireFrames
    agents -- A list of all WildfireAgents in the domain
    max_configuration_counts -- The maximum count for each position in a WildfireConfiguration
    spread_probabilities -- The likelihoods of fires spreading between two fire locations
    modeling_neighborhoods -- The modeling neighborhoods agents should use (if centralized determined)
    messages -- The messages communicated by agents in the current time step

    Methods:
    prepare_ipomcp -- creates the information needed by I-POMCP reasoning
    centralized_neighbor_sampling -- randomly samples modeling neighborhoods for the agents
    create_states -- creates a list of all possible WildfireStates
    create_actions -- creates a list of all possible WildfireActions
    create_observations -- creates a list of all possible WildfireObservations
    create_frames -- creates a list of all possible WildfireFrames
    create_agents -- creates a list of all WildfireAgents in the domain
    create_all_configurations -- creates a list of all possible WildfireConfigurations
    create_configuration_helper -- a recursive method for enumerating all possible WildfireConfigurations
    calculate_configuration_max -- calculates the maximum of each position in a WildfireConfiguration
    can_perform -- determines whether an agent in a given location can perform a given action
    transition_probability -- Calculates the likelihood of both state and internal state transitions
    state_transition_probability -- calculates the likelihood of a State transition based on a JointAction
    state_transition_probability_configuration -- calculates the likelihood of a State transition based on
            a Configuration
    calculate_fire_transition_probability -- calculates the likelihood of transitioning between two WildfireStates
    calculate_fire_spread_probabilities -- calculates the likelihoods of fires spreading between locations
    internal_states_transition_probability --
    process_messages -- communicates all sent WildfireMessages to WildfireAgents
    send_message -- adds a WildfireMessage to be communicated (when process_messages is called)
    """

    def __cinit__(self, setup=None, reasoning_type=WildfireReasoningType.NOOP, dict settings={}):
        """Constructs a new Wildfire instance.

        Keyword arguments:

        setup The number of setup to use
        reasoning_type The type of reasoning that WildfireAgents should follow
        settings A dictionary of reasoning-specific settings
        """
        # save this new domain as the global singleton
        set_domain(self)

        self.settings = WildfireSettings(setup, reasoning_type)
        self.states = self.create_states()
        self.actions = self.create_actions()
        self.actions_with_comm = self.create_actions_comm()
        self.observations = self.create_observations()
        self.frames = self.create_frames()
        self.agents = self.create_agents(settings)
        self.max_configuration_counts = self.calculate_configuration_max()
        self.configuration_actions = self.calculate_configuration_actions()
        self.configuration_indices = self.calculate_configuration_indices()
        self.spread_probabilities = self.calculate_fire_spread_probabilities()
        self.internal_state_transition_probabilities = self.calculate_internal_state_transition_probabilities()

        cdef double epsilon_p
        if self.settings.COMMUNICATION:
            epsilon_p = <double> settings["epsilon_p"]
            self.modeling_neighborhoods = self.centralized_neighbor_sampling(epsilon_p)
            self.modeling_neighborhoods_per_agent = self.organize_modeling_neighborhoods_per_agent()
        else:
            self.modeling_neighborhoods = []

        self.prepare_planners()


    cdef void prepare_planners(self):
        cdef list next_states = self.cache_possible_next_state_transitions()
        cdef list agent_neighbors

        # create each agent's planner
        cdef int agents_n = len(self.agents)
        cdef int agent_i, neighbor_i
        cdef WildfireAgent agent
        for agent_i in range(agents_n):
            agent = <WildfireAgent> self.agents[agent_i]

            if agent.reasoning_type == WildfireReasoningType.POMCPPF:
                (<WildfirePOMCPPFReasoning> agent.reasoning).create_planner(next_states)
            elif agent.reasoning_type == WildfireReasoningType.POMCPPFComm:
                # get the list of agents to model
                agent_neighbors = <list> self.modeling_neighborhoods_per_agent[agent.agent_num]

                # create the planner
                (<WildfirePOMCPPFCommReasoning> agent.reasoning).create_planner(agent_neighbors, next_states)
            elif agent.reasoning_type == WildfireReasoningType.IPOMCPPF:
                (<WildfireIPOMCPPFReasoning> agent.reasoning).create_planner(next_states)
            elif agent.reasoning_type == WildfireReasoningType.IPOMCPPFComm:
                # get the list of agents to model
                agent_neighbors = <list> self.modeling_neighborhoods_per_agent[agent.agent_num]

                # create the planner
                (<WildfireIPOMCPPFCommReasoning> agent.reasoning).create_planner(agent_neighbors, next_states)


    cdef list centralized_neighbor_sampling(self, double epsilon_p):
        """Randomly samples modeling neighborhoods (all cliques) for the agents to use.
        
        Returns: a list of modeling neighborhoods"""
        cdef list neighborhoods = []

        cdef list available_agents = self.agents[:]
        cdef int available_agents_n = len(available_agents)
        cdef int rand_i, chosen_i, agent_i, chosen_n
        cdef WildfireAgent agent, neighbor, chosen_agent
        cdef bint found
        cdef list chosen
        cdef WildfireReasoningType neighbor_reasoning_type_pf = WildfireReasoningType.IPOMCPPFComm

        cdef int level = (<WildfireAgent> self.agents[0]).settings["level"]
        if level == 1:
            neighbor_reasoning_type_pf = WildfireReasoningType.POMCPPF
        elif level == 0:
            neighbor_reasoning_type = WildfireReasoningType.Heuristic
            neighbor_reasoning_type_pf = WildfireReasoningType.Heuristic

        while len(available_agents) > 0:
            # pick an agent at random
            rand_i = 0 #rand() % available_agents_n
            agent = <WildfireAgent> available_agents[rand_i]

            # remove this agent from the list of available agents
            new_available_agents = []
            for agent_i in range(available_agents_n):
                if rand_i != agent_i:
                    neighbor = <WildfireAgent> available_agents[agent_i]
                    new_available_agents.append(neighbor)

            # does the agent have enough neighbors to model?
            if not agent.enough_agents_for_modeling(epsilon_p, new_available_agents):
                if "cliques_n" not in (<WildfireAgent> self.agents[0]).settings:
                    break
                else:
                    chosen = agent.sample_agents_for_modeling_small(available_agents, neighbor_reasoning_type_pf, (<WildfireAgent> self.agents[0]).settings["cliques_n"])

            # save these chosen agents as a neighborhood
            neighborhoods.append([agent] + chosen)

            # remove the chosen agents from the available list
            available_agents = []
            chosen_n = len(chosen)
            for agent_i in range(available_agents_n - 1): # minus 1 since we already removed agent from the new_available_agents
                neighbor = new_available_agents[agent_i]

                # was this neighbor already chosen
                found = False
                for chosen_i in range(chosen_n):
                    chosen_agent = <WildfireAgent> chosen[chosen_i]

                    if chosen_agent.agent_num == neighbor.agent_num:
                        found = True
                        break

                if not found:
                    available_agents.append(neighbor)

            # update the count of remaining available agents
            available_agents_n = len(available_agents)

        # double check that we have at least one neighborhood (in case we need to put everyone together)
        if len(neighborhoods) == 0:
            neighborhoods.append([])

        # handle any left over agents
        cdef int[:] remaining_indices
        cdef int neighborhoods_n, neighborhood_i
        if available_agents_n > 0:
            # randomize where the remaining agents go
            remaining_indices = np.arange(available_agents_n, dtype=np.intc)
            np.random.shuffle(remaining_indices.base)

            neighborhoods_n = len(neighborhoods)
            for agent_i in range(available_agents_n):
                # what is the next neighborhood to put them in (so that everything is balanced)?
                neighborhood_i = agent_i % neighborhoods_n
                neighbor = <WildfireAgent> available_agents[remaining_indices[agent_i]]

                neighborhoods[neighborhood_i].append(neighbor)

        return neighborhoods


    cdef list organize_modeling_neighborhoods_per_agent(self):
        cdef list organized_neighborhoods = []

        # create placeholders for the organized neighborhoods
        cdef int agent_i
        for agent_i in range(self.num_agents()):
            organized_neighborhoods.append([])

        # organize the neighborhoods for each agent
        cdef int neighborhoods_n = len(self.modeling_neighborhoods)
        cdef int neighborhood_i, models_n, model_i
        cdef list neighborhood, local_neighborhood
        cdef Agent agent, neighbor

        for neighborhood_i in range(neighborhoods_n):
            neighborhood = <list> self.modeling_neighborhoods[neighborhood_i]
            models_n = len(neighborhood)

            for agent_i in range(models_n):
                agent = <Agent> neighborhood[agent_i]

                # fill in this agent's local neighborhood
                local_neighborhood = <list> organized_neighborhoods[agent.agent_num]

                for model_i in range(models_n):
                    # an agent doesn't model itself
                    if agent_i != model_i:
                        neighbor = <Agent> neighborhood[model_i]
                        local_neighborhood.append(neighbor)

        return organized_neighborhoods


    cdef list create_states(self):
        """Creates and returns all possible WildfireStates.
        
        Returns: a list of all possible WildfireStates
        """
        cdef int i, s
        cdef int fire_states = self.settings.FIRE_STATES
        cdef int fires_n = self.settings.FIRES
        cdef int states_n = fire_states ** fires_n
        cdef WildfireState state
        cdef list states = []

        for i in range(states_n):
            s = i

            state = WildfireState(fires_n)
            states.append(state)
            for f in range(fires_n):
                state.values[f] = s % fire_states
                s /= fire_states
            state.calculate_index(self.settings)

        return states


    cdef list create_actions(self):
        """Creates and returns all possible WildfireActions.
        
        Returns: a list of all possible WildfireActions
        """
        # fill the list of actions
        cdef int i
        cdef list actions = []
        for i in range(self.settings.FIRES + 1):
            actions.append(WildfireAction(i))

        return actions


    cdef list create_actions_comm(self):
        """Creates and returns all possible WildfireActionComms.
        
        Returns: a list of all possible WildfireActionComms
        """
        # fill the list of actions
        cdef list actions = []

        cdef int supp_n = self.settings.SUPPRESSANT_STATES
        cdef int fire_i, message

        for fire_i in range(self.settings.FIRES + 1):  # include NOOP
            for message in range(-1, supp_n):  # include no message (-1)
                actions.append(WildfireActionComm(fire_i, message))

        return actions


    cdef list create_observations(self):
        """Creates and returns all possible WildfireObservations.
        
        Returns: a list of all possible WildfireObservations
        """
        cdef int fire_reduction
        cdef list observations = []
        cdef WildfireObservation observation

        for fire_change in range(-1, self.settings.NO_OBS + 1):
            observation = WildfireObservation(fire_change)
            observations.append(observation)

        return observations


    cdef list create_frames(self):
        """Creates and returns all possible WildfireFrames.
        
        Returns: a list of all possible WildfireFrames
        """
        cdef int i
        cdef list frames = []
        for i in range(self.settings.FRAMES):
            frames.append(WildfireFrame(i, self.settings.FRAME_LOCATIONS[i], self.settings.FRAME_POWERS[i], self))

        return frames


    cdef list create_agents(self, dict settings):
        """Creates and returns the WildfireAgents in the Wildfire domain.
        
        Returns: a list of new WildfireAgents to populate the Wildfire domain
        """
        cdef int frame_num, i
        cdef int id = 0
        cdef int frames_n = self.settings.FRAMES
        cdef WildfireAgent agent
        cdef WildfireFrame frame
        cdef list agents = []

        for frame_num in range(frames_n):
            frame = self.frames[frame_num]

            for i in range(self.settings.N_AGENTS_PER_FRAME[frame_num]):
                agent = WildfireAgent(id, frame_num, self.settings.REASONING_TYPE, settings)
                agents.append(agent)
                id += 1

        return agents


    cpdef list create_all_configurations(self, Agent agent=None):
        """Creates and returns all possible WildfireConfigurations.
        
        Returns: a list of possible WildfireConfigurations
        """
        cdef int frames_n = self.settings.FRAMES
        cdef int config_n = len(self.max_configuration_counts)

        cdef int[:] max_configuration_counts = np.empty((config_n,), dtype=np.intc)
        cdef int config_i, frame_i
        for config_i in range(config_n):
            max_configuration_counts[config_i] = self.max_configuration_counts[config_i]

        cdef int[:] frame_sizes = np.empty((frames_n,), dtype=np.intc)
        for frame_i in range(frames_n):
            frame_sizes[frame_i] = self.settings.N_AGENTS_PER_FRAME[frame_i]

        cdef int frame_offset = 0
        cdef int actions_n, action_i
        cdef int[:] all_available_actions
        cdef WildfireFrame frame
        if agent is not None:
            for frame_i in range(frames_n):
                frame = <WildfireFrame> self.frames[frame_i]
                actions_n = len(frame.all_available_actions)

                if agent.frame.index == frame.index:
                    frame_sizes[frame_i] -= 1

                    for action_i in range(actions_n):
                        config_i = frame_offset + action_i
                        max_configuration_counts[config_i] -= 1
                    break

                frame_offset = frame_offset + actions_n

        cdef int agents_n = self.num_agents()
        cdef int total_configs = 1
        cdef int start_i = 0
        cdef int end_i
        cdef list all_frames_configurations = []
        cdef list frame_configurations

        for frame_i in range(frames_n):
            frame = <WildfireFrame> self.frames[frame_i]
            actions_n = len(frame.all_available_actions)
            end_i = start_i + actions_n

            frame_configurations = []
            self.create_configuration_helper(0, 0, actions_n, frame_sizes[frame_i],
                                             max_configuration_counts[start_i:end_i],
                                             WildfireConfiguration(actions_n), frame_configurations)

            all_frames_configurations.append(frame_configurations)
            total_configs *= len(frame_configurations)

            start_i += actions_n

        # we are done if there is only one frame
        if frames_n == 1:
            return frame_configurations

        # concatenate all combinations of frames
        cdef list all_configurations = []
        cdef int i, m
        cdef WildfireConfiguration config, partial
        for i in range(total_configs):
            config = WildfireConfiguration(config_n)

            frame_offset = 0
            for frame_i in range(frames_n):
                frame_configurations = all_frames_configurations[frame_i]

                m = len(frame_configurations)
                config_i = i % m
                partial = <WildfireConfiguration> frame_configurations[config_i]
                actions_n = len(partial.actions)

                for action_i in range(actions_n):
                    config.actions[frame_offset + action_i] = partial.actions[action_i]

                i = i // m
                frame_offset += actions_n

            all_configurations.append(config)

        return all_configurations


    cdef void create_configuration_helper(self, int index, int summ, int config_n, int agents_n,
                                          int[:] max_configuration, WildfireConfiguration configuration,
                                          list all_configurations):
        """A recursive method for assisting with determining all possible WildfireConfigurations."""
        cdef int i, num, frame
        if index == config_n:
            if summ == agents_n:
                all_configurations.append(configuration)
        else:
            for num in range(max_configuration[index] + 1):
                if summ + num <= agents_n:
                    # deep copy the configuration
                    new_configuration = WildfireConfiguration(config_n)
                    for i in range(index):
                        new_configuration.actions[i] = configuration.actions[i]

                    # add this number
                    new_configuration.actions[index] = num

                    self.create_configuration_helper(index + 1, summ + num, config_n, agents_n, max_configuration,
                                                     new_configuration, all_configurations)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint can_perform(self, int actionNum, int locNum):
        if self.settings.FIRES == actionNum:
            return True

        cdef int locX = self.settings.AGENT_LOCATIONS[locNum][0]
        cdef int locY = self.settings.AGENT_LOCATIONS[locNum][1]
        cdef int fireX = self.settings.FIRE_LOCATIONS[actionNum][0]
        cdef int fireY = self.settings.FIRE_LOCATIONS[actionNum][1]

        cdef int xDiff = locX - fireX
        cdef int yDiff = locY - fireY

        return xDiff > -2 and xDiff < 2 and yDiff > -2 and yDiff < 2


    cdef double transition_probability(self, int agent_num, State state, InternalStates internal_states,
                                       JointAction joint_action, State next_state, InternalStates next_internal_states):
        # first find the state transition portion
        cdef WildfireConfiguration configuration = self.create_configuration(agent_num, joint_action)
        cdef double state_transition = self.calculate_fire_transition_probability(agent_num, state, configuration,
                                                                                  joint_action.actions[agent_num],
                                                                                  next_state)

        # now find the internal state transition probabilities
        cdef double internal_transitions = self.internal_states_transition_probability(agent_num, internal_states,
                                                                                      joint_action,
                                                                                      next_internal_states)

        return state_transition * internal_transitions


    cdef double state_transition_probability(self, int agent_num, State state, JointAction joint_action,
                                             State next_state):
        cdef WildfireConfiguration configuration = self.create_configuration(agent_num, joint_action)
        return self.calculate_fire_transition_probability(agent_num, state, configuration,
                                                          joint_action.actions[agent_num], next_state)


    cdef double state_transition_probability_configuration(self, int agent_num, State state,
                                                           Configuration configuration, Action action,
                                                           State next_state):
        # handle the none action case
        cdef int action_index
        if action is None:
            action_index = -1
        else:
            action_index = action.index

        return self.calculate_fire_transition_probability(agent_num, state, configuration, action_index, next_state)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.nonecheck(False)
    # @cython.profile(True)
    cdef double calculate_fire_transition_probability(self, int agent_num, WildfireState state,
                                                      Configuration configuration, int action_index,
                                                      WildfireState next_state):
        """Calculates the likelihood of the environment transition between two WildfireStates, given a 
        Configuration.
        
        Returns: the likelihood of a transition from state to next_state given configuration
        """
        cdef int fires_n = self.settings.FIRES
        cdef int frames_n = self.settings.FRAMES
        cdef int state_i = state.index
        cdef int next_state_i = next_state.index
        cdef int burned = self.settings.FIRE_STATES - 1
        cdef int almost_burned = burned - 1
        cdef int new_fire = min(2, almost_burned)
        cdef int config_n = len(configuration.actions)
        cdef int fire_num, frame_num, s, ns, size, num, actions_n
        cdef double prob = 1.0
        cdef double burned_prob = self.settings.BURNOUT_PROB
        # cdef double reduction_prob_per_extra = self.settings.FIRE_REDUCTION_PROB_PER_EXTRA_AGENT
        cdef double probability = self.settings.FIRE_POWER_NEEDED_PROBABILITY
        cdef double base_reduction = self.settings.BASE_FIRE_REDUCTION
        cdef double total_reduction, spread, p, needed
        cdef int[:] fire_sizes = self.settings.FIRE_SIZES
        cdef double[:] power_needed = self.settings.FIRE_POWER_NEEDED

        # calculate how much suppressant power was used on each fire
        # list is actually faster here than building an array...
        cdef list fire_powers = list()
        for fire_num in range(fires_n):
            fire_powers.append(0.0)

        cdef WildfireFrame frame
        cdef int config_i = 0
        for frame_num in range(frames_n):
            frame = <WildfireFrame> self.frames[frame_num]
            actions_n = len(frame.all_available_actions)

            for action_i in range(actions_n):
                num = configuration.actions[config_i]
                if num > 0:
                    fire_num = self.configuration_actions[config_i]
                    if fire_num != fires_n:
                        fire_powers[fire_num] = (<double> fire_powers[fire_num]) + num * frame.fire_reduction
                config_i = config_i + 1

        cdef WildfireAgent agent
        for fire_num in range(fires_n):
            s = state.values[fire_num]
            ns = next_state.values[fire_num]

            # if its burned out, it will remain burned out
            if s == burned:
                if ns == burned:
                    continue # this is like multiplying by 1
                else:
                    return 0.0

            # if its not on fire, a neighboring fire could spread
            if s == 0:
                spread = self.spread_probabilities[state_i][fire_num]
                if ns == new_fire:
                    prob = prob * spread
                    continue  # skip to the next fire
                elif ns == s:
                    prob = prob * (1.0 - spread)
                    continue  # skip to the next fire
                else:
                    return 0.0

            # otherwise a fire cannot increase by more than 1
            if ns > s + 1:
                return 0.0

            # a fire cannot decrease by more than 1
            if ns < s - 1:
                return 0.0

            # how much reduction did the agents create?
            total_reduction = <double> fire_powers[fire_num]

            if agent_num > -1 and action_index == fire_num:
                agent = <WildfireAgent> self.agents[agent_num]
                frame = <WildfireFrame> agent.frame
                total_reduction += frame.fire_reduction

            # did they reduce the fire by enough?
            size = fire_sizes[fire_num]
            needed = power_needed[size]
            if total_reduction > needed - 0.01: # handle rounding errors
                # p = total_reduction / base_reduction * reduction_prob_per_extra
                p = probability * total_reduction / needed

                if p > 1.0:
                    p = 1.0

                if ns == s - 1:
                    prob = prob * p
                elif ns == s:
                    prob = prob * (1.0 - p)
                else:
                    return 0.0
            else:
                # if its almost burned out, then there is a special chance of burning out
                if s == almost_burned:
                    if ns == burned:
                        prob = prob * burned_prob
                    elif ns == almost_burned:
                        prob = prob * (1 - burned_prob)
                    else:
                        return 0.0
                # otherwise fires increase deterministically
                elif ns == s + 1:
                    continue # this is like multiplying by 1
                else:
                    return 0.0

        return prob


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:,:] calculate_fire_spread_probabilities(self):
        # determine which fires are neighbors
        cdef int fires_n = self.settings.FIRES
        cdef int fire_i, fire_j, i_x, i_y, j_x, j_y, diff_x, diff_y
        cdef int[:,:] neighbors = np.zeros((fires_n, fires_n), dtype=np.intc)

        for fire_i in range(fires_n):
            i_x = self.settings.FIRE_LOCATIONS[fire_i][0]
            i_y = self.settings.FIRE_LOCATIONS[fire_i][1]

            for fire_j in range(fires_n):
                if fire_i == fire_j:
                    continue  # a fire is not its own neighbor

                j_x = self.settings.FIRE_LOCATIONS[fire_j][0]
                j_y = self.settings.FIRE_LOCATIONS[fire_j][1]

                diff_x = i_x - j_x
                diff_y = i_y - j_y

                # N = 1, E  2, S = 3, W = 4
                if diff_x == 0 and diff_y == 1:
                    # fire would move NORTH
                    neighbors[fire_i][fire_j] = 1
                elif diff_x == 1 and diff_y == 0:
                    # fire would move EAST
                    neighbors[fire_i][fire_j] = 2
                elif diff_x == 0 and diff_y == -1:
                    # fire would move SOUTH
                    neighbors[fire_i][fire_j] = 3
                elif diff_x == -1 and diff_y == 0:
                    # fire would move WEST
                    neighbors[fire_i][fire_j] = 4

        cdef int states_n = len(self.states)
        cdef int state_i
        cdef WildfireState state

        cdef double[:,:] spread_probabilities = np.zeros((states_n, fires_n), dtype=np.double)
        cdef double default_spread_prob = self.settings.RANDOM_SPREAD_PROB
        cdef int burned = self.settings.FIRE_STATES - 1
        cdef double prob
        cdef int direction, fire_j_value
        for state_i in range(states_n):
            state = self.states[state_i]

            for fire_i in range(fires_n):
                # can this fire be started?
                if state.values[fire_i] != 0:
                    continue # already on fire or burned

                # every spot can randomly start on fire
                prob = default_spread_prob

                # look for spread
                for fire_j in range(fires_n):
                    if fire_i == fire_j:
                        continue

                    fire_j_value = state.values[fire_j]
                    if fire_j_value != 0 and fire_j_value != burned:
                        direction = neighbors[fire_i][fire_j]

                        if direction == 1:
                            prob += self.settings.NORTH_IGNITION_PROB
                        elif direction == 2:
                            prob += self.settings.EAST_IGNITION_PROB
                        elif direction == 3:
                            prob += self.settings.SOUTH_IGNITION_PROB
                        elif direction == 4:
                            prob += self.settings.WEST_IGNITION_PROB

                # save the spread probability
                spread_probabilities[state_i][fire_i] = prob

        return spread_probabilities


    cdef double calculate_internal_state_transition_probability(self, int internal_state, int action_index,
                                                                int next_internal_state):
        cdef int noop = self.settings.FIRES
        cdef int max_suppressant = self.settings.SUPPRESSANT_STATES - 1
        cdef double discharge_prob = self.settings.DISCHARGE_PROB
        cdef double recharge_prob = self.settings.RECHARGE_PROB

        # was this a NOOP?
        if action_index == noop:
            if internal_state > 0:
                # they don't lose suppressant when doing nothing
                if next_internal_state == internal_state:
                    return 1.0
                else:
                    return 0.0
            else:
                # they might have recharged
                if next_internal_state == max_suppressant:
                    return recharge_prob
                elif next_internal_state == 0:
                    return 1.0 - recharge_prob
                else:
                    return 0.0

        # otherwise they fought a fire
        else:
            # we need to discharge
            if internal_state == 0:
                if next_internal_state == 0:
                    return 1.0
                else:
                    return 0.0
            else:
                if next_internal_state == internal_state - 1:
                    return discharge_prob
                elif next_internal_state == internal_state:
                    return 1.0 - discharge_prob
                else:
                    return 0.0


    cdef double[:,:,:] calculate_internal_state_transition_probabilities(self):
        cdef int supp_n = self.settings.SUPPRESSANT_STATES
        cdef int actions_n = len(self.actions)
        cdef double[:,:,:] transition_probabilities = np.zeros((supp_n, actions_n, supp_n), dtype=np.double)

        cdef int supp_i, action_i, next_supp_i
        for supp_i in range(supp_n):
            for action_i in range(actions_n):
                for next_supp_i in range(supp_n):
                    transition_probabilities[supp_i, action_i, next_supp_i] = self.calculate_internal_state_transition_probability(supp_i, action_i, next_supp_i)

        return transition_probabilities


    cdef double internal_states_transition_probability(self, int agent_num, InternalStates internal_states,
                                                       JointAction joint_action, InternalStates next_internal_states):
        cdef int agents_n = len(joint_action.actions)
        cdef int state_value, next_state_value, action_index, transition_agent_num
        cdef double prob = 1.0, p

        for transition_agent_num in range(agents_n):
            action_index = joint_action.actions[transition_agent_num]
            state_value = internal_states.values[transition_agent_num]
            next_state_value = next_internal_states.values[transition_agent_num]

            p = self.internal_state_transition_probabilities[state_value, action_index, next_state_value]
            if p == 0.0:
                return 0.0
            else:
                prob = prob * p

        return prob


    @cython.profile(True)
    cdef double single_internal_state_transition_probability(self, int agent_num, int state_value,
                                                             Configuration configuration, Action action,
                                                             int next_state_value):
        cdef int action_index = action.index

        return self.internal_state_transition_probabilities[state_value, action_index, next_state_value]


    cdef double observation_probability(self, int agent_num, State state, JointAction joint_action, State next_state,
                                        Observation observation):
        return self.calculate_observation_probability(state, joint_action.actions[agent_num], next_state, observation)


    cdef double observation_probability_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state, Observation observation):
        # handle the none action case
        cdef int action_index
        if action is None:
            action_index = -1
        else:
            action_index = action.index

        return self.calculate_observation_probability(state, action_index, next_state, observation)


    cdef double calculate_observation_probability(self, WildfireState state, int action_index, WildfireState next_state,
                                                  WildfireObservation observation):
        cdef int noop = self.settings.FIRES
        cdef int noobs = self.settings.NO_OBS
        cdef int o = observation.fire_change

        # check for NO_OBS and NOOP
        if action_index == noop and o == noobs:
            return 1.0
        elif action_index == noop or o == noobs:
            return 0.0

        cdef int diff = next_state.values[action_index] - state.values[action_index]
        cdef double error = self.settings.OBSERVATION_ERROR
        if diff == o:
            return 1.0 - error
        else:
            return error / (len(self.observations) - 2) # subtract for NO_OBS and correct observation


    cdef double reward(self, int agent_num, State state, InternalStates internal_states, JointAction joint_action,
                       State next_state, InternalStates next_internal_states):
        return self.reward_single_agent(state, internal_states.values[agent_num], joint_action.actions[agent_num], -1,
                                        next_state)


    cpdef double reward_configuration(self, int agent_num, State state, InternalStates internal_states,
                                     Configuration configuration, Action action, State next_state,
                                     InternalStates next_internal_states):
        '''
        
        :param agent_num: The index into internal_states for the agent taking the specified action 
        :param state: 
        :param internal_states: 
        :param configuration: 
        :param action: 
        :param next_state: 
        :param next_internal_states: 
        :return: 
        '''

        # handle the none action case
        cdef int action_index
        if action is None:
            action_index = -1
        else:
            action_index = action.index

        if agent_num != 0 and isinstance(internal_states, PartialInternalStates):
            agent_num = 0

        return self.reward_single_agent(state, internal_states.values[agent_num], action_index, -1, next_state)


    cdef double reward_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                 JointActionComm joint_action, State next_state, InternalStates next_internal_states):
        return self.reward_single_agent(state, internal_states.values[agent_num], joint_action.actions[agent_num],
                                        joint_action.messages[agent_num], next_state)


    cpdef double reward_configuration_with_comm(self, int agent_num, State state, InternalStates internal_states,
                                                Configuration configuration, ActionComm action, State next_state,
                                                InternalStates next_internal_states):
        '''
        
        :param agent_num: The index into internal_states for the agent taking the specified action 
        :param state: 
        :param internal_states: 
        :param configuration: 
        :param action: 
        :param next_state: 
        :param next_internal_states: 
        :return: 
        '''

        # handle the none action case
        cdef int action_index, message
        if action is None:
            action_index = -1
            message = -1
        else:
            action_index = action.index
            message = action.message

        if agent_num != 0 and isinstance(internal_states, PartialInternalStates):
            agent_num = 0

        return self.reward_single_agent(state, internal_states.values[agent_num], action_index, message, next_state)


    cdef double reward_single_agent(self, WildfireState state, int internal_state, int action_index, int message,
                                    WildfireState next_state):
        cdef double r = 0.0

        cdef int fire, s, ns
        cdef int fires_n = self.settings.FIRES
        cdef int burned = self.settings.FIRE_STATES - 1
        cdef int[:] fire_sizes = self.settings.FIRE_SIZES
        cdef double[:] rewards = self.settings.NO_FIRE_REWARDS
        cdef double burned_penalty = self.settings.FIRE_BURNOUT_PENALTY

        # calculate the shared rewards/penalty
        for fire in range(fires_n):
            s = state.values[fire]
            ns = next_state.values[fire]

            if ns == 0 and s > 0:
                r += rewards[fire_sizes[fire]]
                # if action_index == fire and internal_state > 0:
                #     r += self.settings.FIRE_CONTRIBUTION_BONUS
            elif ns == burned and s < burned:
                r += burned_penalty

        # calculate individual penalty if not taking a NOOP
        if action_index < fires_n:
            s = state.values[action_index]

            # agents shouldn't act when out of suppressant or fight empty/burned out fires
            if s == 0 or s == burned or internal_state == 0:
                r += self.settings.NO_FIRE_PENALTY
        # elif internal_state > 0:
        #     # here the agent chose NOOP but had supressant
        #     r += self.settings.NOOP_WITH_SUPPRESSANT_PENALTY

        # did the agent send a message?
        if message > -1:
            r += self.settings.COMMUNICATION_COST

        if message == internal_state:
            r += self.settings.HONEST_COMM_REWARD

        return r


    cdef State sample_next_state(self, int agent_num, State state, JointAction joint_action):
        cdef WildfireConfiguration configuration = self.create_configuration(agent_num, joint_action)
        return self.sample_next_state_configuration(agent_num, state, configuration,
                                                    self.actions[joint_action.actions[agent_num]])


    @cython.cdivision(True)
    @cython.profile(True)
    cdef State sample_next_state_configuration(self, int agent_num, State state, Configuration configuration,
                                               Action action):
        cdef int i
        cdef int states_n = len(self.states)
        cdef WildfireState possible
        cdef WildfireState last = None

        # use roulette wheel selection
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        for i in range(states_n):
            possible = self.states[i]

            # how likely is this transition?
            prob = self.state_transition_probability_configuration(agent_num, state, configuration, action, possible)
            if prob > 0.0:
                last = possible

                if prob > rand_val:
                    # we found our next state, so stop looking
                    break
                else:
                    rand_val -= prob

        return last


    @cython.cdivision(True)
    @cython.profile(True)
    cdef State sample_next_state_configuration_from_possible(self, int agent_num, State state,
                                                             Configuration configuration, Action action,
                                                             list possible_next_states):
        cdef int i, possible_index
        cdef int states_n = len(possible_next_states)
        cdef WildfireState possible
        cdef WildfireState last = None

        # use roulette wheel selection
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        for i in range(states_n):
            possible_index = <int> possible_next_states[i]
            possible = <WildfireState> self.states[possible_index]

            # how likely is this transition?
            prob = self.state_transition_probability_configuration(agent_num, state, configuration, action, possible)
            if prob > 0.0:
                last = possible

                if prob > rand_val:
                    # we found our next state, so stop looking
                    break
                else:
                    rand_val -= prob

        return last


    @cython.cdivision(True)
    @cython.profile(True)
    cdef InternalStates sample_next_internal_states(self, int agent_num, InternalStates internal_states,
                                                    JointAction joint_action):
        cdef int agents_n = len(internal_states.values)
        cdef int suppressants_n = self.settings.SUPPRESSANT_STATES
        cdef int agent_i, supp, a, possible, last
        cdef double rand_val, prob

        # create the next internal states object
        cdef WildfireInternalStates next_internal_states = WildfireInternalStates(agents_n)

        # randomly sample next internal states for each agent
        for agent_i in range(agents_n):
            supp = internal_states.values[agent_i]
            a = joint_action.actions[agent_i]

            # use roulette wheel selection
            rand_val = rand() / (RAND_MAX + 1.0)
            for possible in range(suppressants_n):
                # how likely is this transition?
                prob = self.internal_state_transition_probabilities[supp, a, possible]
                if prob > 0.0:
                    last = possible

                    if prob > rand_val:
                        # we found our next state, so stop looking
                        break
                    else:
                        rand_val -= prob

            # save the chosen state
            next_internal_states.values[agent_i] = last

        return next_internal_states


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    @cython.profile(True)
    @cython.cdivision(True)
    cdef PartialInternalStates sample_next_partial_internal_states(self, PartialInternalStates internal_states,
                                                            PartialJointAction joint_action):
        cdef int agents_n = len(internal_states.values)
        cdef int suppressants_n = self.settings.SUPPRESSANT_STATES
        cdef int agent_i, agent_num, supp, a, possible, last
        cdef double rand_val, prob

        # create the next internal states object
        cdef WildfirePartialInternalStates next_internal_states = WildfirePartialInternalStates(agents_n)

        # randomly sample next internal states for each agent
        for agent_i in range(agents_n):
            supp = internal_states.values[agent_i]
            a = joint_action.actions[agent_i]
            agent_num = internal_states.agent_nums[agent_i]

            # use roulette wheel selection
            rand_val = rand() / (RAND_MAX + 1.0)
            for possible in range(suppressants_n):
                # how likely is this transition?
                prob = self.internal_state_transition_probabilities[supp, a, possible]
                if prob > 0.0:
                    last = possible

                    if prob > rand_val:
                        # we found our next state, so stop looking
                        break
                    else:
                        rand_val -= prob

            # save the chosen state
            next_internal_states.values[agent_i] = last
            next_internal_states.agent_nums[agent_i] = agent_num

        return next_internal_states


    @cython.cdivision(True)
    @cython.profile(True)
    cdef int sample_single_next_internal_state(self, int agent_num, int internal_state, Configuration configuration,
                                               Action action):
        cdef int suppressants_n = self.settings.SUPPRESSANT_STATES
        cdef int possible, last
        cdef double rand_val, prob

        # use roulette wheel selection
        rand_val = rand() / (RAND_MAX + 1.0)
        for possible in range(suppressants_n):
            # how likely is this transition?
            prob = self.internal_state_transition_probabilities[internal_state, action.index, possible]
            if prob > 0.0:
                last = possible

                if prob > rand_val:
                    # we found our next state, so stop looking
                    break
                else:
                    rand_val -= prob

        return last


    cdef Observation sample_observation(self, int agent_num, State state, JointAction joint_action, State next_state):
        cdef WildfireConfiguration configuration = self.create_configuration(agent_num, joint_action)
        return self.sample_observation_configuration(agent_num, state, configuration,
                                                     self.actions[joint_action.actions[agent_num]], next_state)


    @cython.cdivision(True)
    @cython.profile(True)
    cdef Observation sample_observation_configuration(self, int agent_num, State state, Configuration configuration,
                                                      Action action, State next_state):
        cdef int i
        cdef int observations_n = len(self.observations)
        cdef WildfireObservation possible
        cdef WildfireObservation last = None

        # use roulette wheel selection
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        for i in range(observations_n):
            possible = self.observations[i]

            # how likely is this transition?
            prob = self.observation_probability_configuration(agent_num, state, configuration, action, next_state,
                                                              possible)
            if prob > 0.0:
                last = possible

                if prob > rand_val:
                    # we found our observation, so stop looking
                    break
                else:
                    rand_val -= prob

        return last


    cpdef State generate_start_state(self):
        cdef int fires_n = self.settings.FIRES
        cdef WildfireState state = WildfireState(fires_n)

        cdef int fire_start = 2
        cdef int i

        for i in range(fires_n):
            state.values[i] = fire_start
        state.calculate_index(self.settings)

        return state


    @cython.cdivision(True)
    @cython.profile(True)
    cpdef InternalStates generate_start_internal_states(self):
        cdef int agents_n = self.settings.N_AGENTS
        cdef WildfireInternalStates internal_states = WildfireInternalStates(agents_n)

        cdef int suppressants_n = self.settings.SUPPRESSANT_STATES
        cdef int suppressant_start = suppressants_n - 1
        cdef int agent_i, possible, last
        cdef double rand_val, prob

        for agent_i in range(agents_n):
            # use roulette wheel selection
            rand_val = rand() / (RAND_MAX + 1.0)

            for possible in range(suppressants_n):
                prob = self.settings.INITIAL_SUPPRESSANT_DISTRIBUTION[possible]

                if prob > 0.0:
                    last = possible

                    if rand_val < prob:
                        # we found our suppressant, so stop looking
                        break
                    else:
                        rand_val -= prob

            internal_states.values[agent_i] = last

        return internal_states


    cdef Configuration create_configuration(self, int agent_num, JointAction joint_action,
                                            bint leave_out_subject_agent=True):
        # TODO try to speedup?
        
        cdef int config_n = len(self.max_configuration_counts)
        cdef WildfireConfiguration configuration = WildfireConfiguration(config_n)

        cdef int agents_n = len(self.agents)
        cdef tuple t
        cdef int agent_i, frame_index, action_num, config_i
        cdef WildfireAgent agent
        for agent_i in range(agents_n):
            if leave_out_subject_agent and agent_i == agent_num:
                continue

            agent = <WildfireAgent> self.agents[agent_i]
            frame_index = agent.frame.index
            action_num = joint_action.actions[agent_i]

            t = tuple((frame_index, action_num))
            config_i = self.configuration_indices[t]
            configuration.actions[config_i] += 1

        return configuration


    @cython.profile(True)
    cdef Configuration create_configuration_from_partial(self, int agent_num, PartialJointAction joint_action,
                                            bint leave_out_subject_agent=True):
        cdef int config_n = len(self.max_configuration_counts)
        cdef WildfireConfiguration configuration = WildfireConfiguration(config_n)

        cdef int agents_n = len(joint_action.actions)
        cdef tuple t
        cdef int agent_i, loop_agent_num, frame_index, action_num, config_i
        cdef WildfireAgent agent
        for agent_i in range(agents_n):
            loop_agent_num = joint_action.agent_nums[agent_i]

            if leave_out_subject_agent and loop_agent_num == agent_num:
                continue

            agent = <WildfireAgent> self.agents[loop_agent_num]
            frame_index = agent.frame.index
            action_num = joint_action.actions[agent_i]

            t = tuple((frame_index, action_num))
            config_i = self.configuration_indices[t]
            configuration.actions[config_i] += 1

        return configuration


    cdef int num_agents(self):
        return self.settings.N_AGENTS


    cdef int[:] num_agents_per_frame(self):
        return self.settings.N_AGENTS_PER_FRAME


    cdef double max_reward(self):
        cdef int fires_n = self.settings.FIRES
        cdef int fire_i, size
        cdef double max_fire_reward = 0.0

        for fire_i in range(fires_n):
            size = self.settings.FIRE_SIZES[fire_i]
            max_fire_reward += self.settings.NO_FIRE_REWARDS[size]
        max_fire_reward += self.settings.FIRE_CONTRIBUTION_BONUS

        return max([
            -self.settings.FIRE_BURNOUT_PENALTY,
            -self.settings.NO_FIRE_PENALTY,
            max_fire_reward
        ])


    #@cython.profile(True)
    cpdef void validate_transitions(self):
        """Validates the transition function by checking every state-configuration-state combination.
        
        Prints any state-configuration combination whose marginal probability is in error."""
        cdef list configs = self.create_all_configurations()

        cdef int state_i, next_state_i, config_i
        cdef int states_n = len(self.states)
        cdef int configs_n = len(configs)
        cdef double total_prob = 0.0
        cdef double prob, p
        cdef State state, next_state
        cdef Configuration config

        # commented out since this slows things way down, but useful for tracing errors
        # cdef str s

        for state_i in range(states_n):
            state = self.states[state_i]

            for config_i in range(configs_n):
                config = configs[config_i]

                prob = 0.0
                # s = ""
                for next_state_i in range(states_n):
                    next_state = self.states[next_state_i]

                    # in the Wildfire domain, agent_num and action are irrelevant
                    #p = self.transition_probability_configuration(-1, state, config, None, next_state)
                    p = self.calculate_fire_transition_probability(-1, state, config, -1, next_state)
                    prob += p
                    total_prob += p

                    # if p > 0.0:
                    #     s += "{0} {1} {2} {3}\n".format(np.array(state.values), np.array(config.actions), np.array(next_state.values), p)

                if prob > 1.001 or prob < 0.999:
                    print("###################################################")
                    print(np.array(state.values), np.array(config.actions), prob)
                    # print(s)

                    for next_state_i in range(states_n):
                        next_state = self.states[next_state_i]

                        # in the Wildfire domain, agent_num and action are irrelevant
                        p = self.calculate_fire_transition_probability(-1, state, config, -1, next_state)
                        print(np.array(next_state.values), p)


        print(configs_n, total_prob, configs_n * states_n)


    cpdef void validate_internal_transitions(self):
        # create the list of possible internal states
        cdef list internal_states_list = []
        cdef int i
        cdef int suppressants_n = self.settings.SUPPRESSANT_STATES
        cdef WildfireInternalStates internal_states, next_internal_states

        for i in range(suppressants_n):
            internal_states = WildfireInternalStates(1)
            internal_states.values[0] = i
            internal_states_list.append(internal_states)

        cdef int state_i, next_state_i, action_i
        cdef int actions_n = self.settings.FIRES + 1
        cdef double total_prob = 0.0
        cdef double prob, p
        cdef WildfireAction action

        # commented out since this slows things way down, but useful for tracing errors
        # cdef str s

        for state_i in range(suppressants_n):
            state = internal_states_list[state_i]

            for action_i in range(actions_n):
                action = self.actions[action_i]

                prob = 0.0
                # s = ""
                for next_state_i in range(suppressants_n):
                    next_internal_states = internal_states_list[next_state_i]

                    # in the Wildfire domain, agent_num and action are irrelevant
                    p = self.internal_state_transition_probabilities[state_i, action_i, next_state_i]
                    prob += p
                    total_prob += p

                    # if p > 0.0:
                    #     s += "{0} {1} {2} {3}\n".format(np.array(state.values), np.array(config.actions), np.array(next_state.values), p)

                if prob > 1.001 or prob < 0.999:
                    print("###################################################")
                    print(np.array(internal_states.values), np.array(action.index), prob)
                    # print(s)

        print(total_prob, suppressants_n * actions_n)


    cpdef void validate_observations(self):
        """Validates the observation function by checking every state-action-state-observation combination.
        
        Prints any state-action-state combination whose marginal probability is in error."""
        cdef int state_i, next_state_i, action_i, observation_i, fire_i, s, ns
        cdef int states_n = len(self.states)
        cdef int actions_n = len(self.actions)
        cdef int observations_n = len(self.observations)
        cdef int fires_n = self.settings.FIRES
        cdef int burned = self.settings.FIRE_STATES - 1
        cdef int count = 0
        cdef double total_prob = 0.0
        cdef double prob, p
        cdef bint valid
        cdef WildfireState state, next_state
        cdef WildfireAction action
        cdef WildfireObservation observation

        # commented out since this slows things way down, but useful for tracing errors
        # cdef str st

        for state_i in range(states_n):
            state = self.states[state_i]

            for next_state_i in range(states_n):
                next_state = self.states[next_state_i]

                # is this a valid next state?
                valid = True
                for fire_i in range(fires_n):
                    s = state.values[fire_i]
                    ns = next_state.values[fire_i]

                    if s == burned and ns != burned:
                        valid = False
                    elif s == 0 and ns != 0 and ns != 2:
                        valid = False
                    elif ns < s - 1 or ns > s + 1:
                        valid = False

                if valid:
                    for action_i in range(actions_n):
                        action = self.actions[action_i]

                        count += 1

                        prob = 0.0
                        # st = ""

                        for observation_i in range(observations_n):
                            observation = self.observations[observation_i]

                            p = self.calculate_observation_probability(state, action.index, next_state, observation)
                            prob += p
                            total_prob += p

                            # if p > 0.0:
                            #     st += "{0} {1} {2} {3} {4}\n".format(np.array(state.values), action.index, np.array(next_state.values), observation.fire_change, p)

                        if prob > 1.001 or prob < 0.999:
                            print("###################################################")
                            print(np.array(state.values), action.index, np.array(next_state.values), prob)
                            # print(st)

        print(total_prob, count)


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    #@cython.profile(True)
    cdef list cache_possible_next_state_transitions(self):
        cdef int states_n = len(self.states)
        cdef list all_possible_transitions = []
        cdef int state_i

        for state_i in range(states_n):
            all_possible_transitions.append(self.possible_next_states(state_i))

        return all_possible_transitions


    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    #@cython.profile(True)
    cdef list possible_next_states(self, int state_i):
        cdef State state = self.states[state_i]
        cdef int fires_n = self.settings.FIRES
        cdef int burned = self.settings.FIRE_STATES - 1
        cdef int almost_burned = burned - 1
        cdef int new_fire = min(2, almost_burned)
        cdef int fire_i, fire_value
        cdef list possible_next_fire_combinations = []
        cdef list possible_next_fire_values

        for fire_i in range(fires_n):
            fire_value = state.values[fire_i]

            if fire_value == 0:
                possible_next_fire_values = [0, new_fire]
            elif fire_value == burned:
                possible_next_fire_values = [burned]
            else:
                possible_next_fire_values = [fire_value - 1, fire_value, fire_value + 1]

            possible_next_fire_combinations.append(possible_next_fire_values)

        cdef int states_n = len(self.states)
        cdef int next_state_i
        cdef State next_state
        cdef bint possible
        cdef list all_possible_next_states = []
        for next_state_i in range(states_n):
            next_state = self.states[next_state_i]
            possible = True

            for fire_i in range(fires_n):
                if next_state.values[fire_i] not in possible_next_fire_combinations[fire_i]:
                    possible = False
                    break

            if possible:
                all_possible_next_states.append(next_state_i)

        return all_possible_next_states
