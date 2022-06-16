from oasys.structures.pomdp_structures import State, Action, Configuration
from .wildfire import Wildfire
from .wildfire_frame import WildfireFrame
from .wildfire_reasoning import (WildfireNOOPReasoning,
                                 WildfireHeuristicReasoning,
                                 WildfireCoordinationReasoning,
                                 WildfirePOMCPPFReasoning,
                                 WildfirePOMCPPFCommReasoning,
                                 WildfireIPOMCPPFReasoning,
                                 WildfireIPOMCPPFCommReasoning)
from .wildfire_structures import WildfireAction
import math
import numpy as np
from scipy.stats import t as t_dist

from oasys.domains.domain cimport get_domain
from oasys.structures.pomdp_structures cimport State, Action, Configuration
from .wildfire cimport Wildfire
from .wildfire_frame cimport WildfireFrame
from .wildfire_reasoning cimport (WildfireNOOPReasoning,
                                  WildfireHeuristicReasoning,
                                  WildfireCoordinationReasoning,
                                  WildfirePOMCPPFReasoning,
                                  WildfirePOMCPPFCommReasoning,
                                  WildfireIPOMCPPFReasoning,
                                  WildfireIPOMCPPFCommReasoning)
from .wildfire_structures cimport WildfireAction
cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX


def rebuild(agent_num, frame_num, reasoning_type, settings, reasoning, last_observation, current_state,
                current_suppressant):
        agent = WildfireAgent(agent_num, frame_num, reasoning_type, settings, True)

        agent.reasoning = reasoning
        agent.last_observation = last_observation
        agent.current_state = current_state
        agent.current_suppressant = current_suppressant

        return agent


cdef class WildfireAgent(Agent):
    def __cinit__(self, int agent_num, int frame_index, WildfireReasoningType reasoning_type, dict settings,
                  bint is_model=False):
        self.agent_num = agent_num
        self.domain = <Wildfire> get_domain()
        self.frame = self.domain.frames[frame_index]
        self.reasoning_type = reasoning_type
        self.settings = settings

        self.last_observation = None
        self.current_state = self.domain.generate_start_state()
        self.current_suppressant = -1  # randomly determined in the simulator

        if not is_model:
            if reasoning_type == WildfireReasoningType.NOOP:
                self.reasoning = WildfireNOOPReasoning(self)
            elif reasoning_type == WildfireReasoningType.Heuristic:
                self.reasoning = WildfireHeuristicReasoning(self)
            elif reasoning_type == WildfireReasoningType.Coordination:
                self.reasoning = WildfireCoordinationReasoning(self)
            elif reasoning_type == WildfireReasoningType.POMCPPF:
                self.reasoning = WildfirePOMCPPFReasoning(self)
            elif reasoning_type == WildfireReasoningType.POMCPPFComm:
                self.reasoning = WildfirePOMCPPFCommReasoning(self)
            elif reasoning_type == WildfireReasoningType.IPOMCPPF:
                self.reasoning = WildfireIPOMCPPFReasoning(self)
            elif reasoning_type == WildfireReasoningType.IPOMCPPFComm:
                self.reasoning = WildfireIPOMCPPFCommReasoning(self)

    def __reduce__(self):
        return (rebuild, (self.agent_num, self.frame.index, self.reasoning_type, self.settings, self.reasoning,
                          self.last_observation, self.current_state, self.current_suppressant))

    cpdef Action choose_action(self, int[:] message_vector=None):
        if self.reasoning_type == WildfireReasoningType.POMCPPFComm:
            return self.reasoning.choose_action_with_messages(self.current_state, self.current_suppressant,
                                                              message_vector)
        else:
            return self.reasoning.choose_action(self.current_state, self.current_suppressant)


    cpdef list calculate_q_values(self, int[:] message_vector=None):
        if self.reasoning_type == WildfireReasoningType.POMCPPFComm:
            return (<WildfirePOMCPPFCommReasoning> self.reasoning).calculate_q_values_with_messages(self.current_state,
                                                                                                    self.current_suppressant,
                                                                                                    message_vector)
        elif self.reasoning_type == WildfireReasoningType.POMCPPF:
            return (<WildfirePOMCPPFReasoning> self.reasoning).calculate_q_values(self.current_state,
                                                                                  self.current_suppressant)


    cdef void make_observation(self, State next_state, int next_internal_state, Observation observation):
        self.current_state = next_state
        self.current_suppressant = next_internal_state
        self.last_observation = observation

        if self.reasoning_type == WildfireReasoningType.POMCPPF:
            (<WildfirePOMCPPFReasoning> self.reasoning).make_observation(next_state, next_internal_state, observation)
        elif self.reasoning_type == WildfireReasoningType.POMCPPFComm:
            (<WildfirePOMCPPFCommReasoning> self.reasoning).make_observation(next_state, next_internal_state, observation)
        elif self.reasoning_type == WildfireReasoningType.IPOMCPPF:
            (<WildfireIPOMCPPFReasoning> self.reasoning).make_observation(next_state, next_internal_state, observation)
        elif self.reasoning_type == WildfireReasoningType.IPOMCPPFComm:
            (<WildfireIPOMCPPFCommReasoning> self.reasoning).make_observation(next_state, next_internal_state, observation)


    cdef Action sample_action(self, State state, int internal_state):
        cdef int fires_n = self.domain.settings.FIRES
        cdef int noop = fires_n
        cdef int burned = self.domain.settings.FIRE_STATES - 1
        cdef int fire_i, f, a
        cdef list fires

        # do we have any suppressant
        if internal_state == 0:
            # we must NOOP
            return WildfireAction(noop)
        else:
            # which fires can we fight?
            fires = []
            for fire_i in range(len(self.frame.all_available_actions)):
                a = self.frame.all_available_actions[fire_i]
                if a == noop:
                    continue

                # is this fire fightable?
                f = state.values[a]
                if f != 0 and f != burned:
                    fires.append(a)

            if len(fires) > 0:
                # pick an active fire at random
                a = rand() % len(fires)
                return WildfireAction(fires[a])
            else:
                # we have nothing to do, so NOOP
                return WildfireAction(noop)


    cdef Action sample_action_with_configuration(self, State state, int internal_state, Configuration configuration):
        cdef Wildfire domain = <Wildfire> self.domain
        cdef int[:] all_available_actions = self.frame.all_available_actions
        cdef list actions = domain.actions
        cdef WildfireAction action
        cdef int all_available_actions_n = len(all_available_actions)
        cdef int agent_num = self.agent_num
        cdef int available_action_i, action_i
        cdef double[:] rewards = np.zeros((all_available_actions_n,), np.double)
        cdef double total_sum = 0.0
        cdef double reward_exp

        for available_action_i in range(all_available_actions_n):
            action_i = <int> all_available_actions[available_action_i]
            action = <WildfireAction> actions[action_i]

            # randomly sample a next state
            next_state = domain.sample_next_state_configuration(agent_num, state, configuration, action)

            # save the sampled reward
            reward_exp = np.exp(domain.reward_single_agent(state, internal_state, action, -1, next_state))
            rewards[available_action_i] = reward_exp
            total_sum = total_sum + reward_exp

        # pick an action using softmax (temperature = 1)
        cdef int last_action_i
        cdef double rand_val = rand() / (RAND_MAX + 1.0)
        for available_action_i in range(all_available_actions_n):
            last_action_i = all_available_actions[available_action_i]
            if rand_val < rewards[available_action_i] / total_sum:
                break

        return actions[last_action_i]


    cdef Agent replicate_as_model(self):
        return WildfireAgent(self.agent_num, self.frame.index, self.reasoning_type, self.settings, True)


    cpdef void set_current_state(self, WildfireState state):
        self.current_state = state


    cpdef void set_current_suppressant(self, int suppresssant):
        self.current_suppressant = suppresssant


    cdef list sample_agents_for_modeling(self, double epsilon_p, list available_agents,
                                         WildfireReasoningType neighbor_reasoning_type, bint use_models=True):
        # organize the neighborhoods by frame/action
        cdef list available_neighborhoods = self.organize_into_neighborhoods(available_agents)
        cdef list full_neighborhoods = self.organize_into_neighborhoods(self.domain.agents)

        # sort the neighborhoods by size
        cdef list available_action_neighborhood, full_action_neighborhood
        cdef tuple tup
        cdef int neighborhood_i
        cdef int neighborhoods_n = len(available_neighborhoods)
        for neighborhood_i in range(neighborhoods_n):
            available_action_neighborhood = <list> available_neighborhoods[neighborhood_i]
            full_action_neighborhood = <list> full_neighborhoods[neighborhood_i]
            tup = tuple((len(available_action_neighborhood), len(full_action_neighborhood),
                         available_action_neighborhood, full_action_neighborhood))
            available_neighborhoods[neighborhood_i] = tup
        available_neighborhoods.sort(reverse=True)

        # sample from the neighborhoods
        cdef list chosen_ids = []
        cdef int N, n
        cdef double t_alpha, part
        for neighborhood_i in range(neighborhoods_n):
            tup = available_neighborhoods[neighborhood_i]
            N_available = <int> tup[0]
            N = <int> tup[1]

            if N > 0:
                for n in range(2, N+1):
                    t_alpha = t_dist.interval(0.95, n-1)[1]
                    part = t_alpha / (2 * epsilon_p)
                    part = part * part

                    if n > math.ceil((N * part) / (N - 1 + part)):
                        break

                self.sample_agents_from_list(tup[2], n, chosen_ids)

        # create the chosen agent models
        cdef int chosen_n = len(chosen_ids)
        cdef int chosen_i, agent_i
        cdef list chosen = []
        cdef WildfireAgent neighbor
        for chosen_i in range(chosen_n):
            agent_i = chosen_ids[chosen_i]
            neighbor = <WildfireAgent> self.domain.agents[agent_i]

            # treat them as a Heuristic agent
            chosen.append(neighbor.replicate_as_model())

        return chosen


    cdef list sample_all_agents(self, WildfireReasoningType neighbor_reasoning_type, bint use_models=True):
        cdef int agents_n = self.domain.num_agents()
        cdef int agent_i
        cdef WildfireAgent model, neighbor
        cdef list chosen = []
        for agent_i in range(agents_n):
            if agent_i != self.agent_num:
                neighbor = self.domain.agents[agent_i]

                # treat them as a Heuristic agent
                model = WildfireAgent(neighbor.agent_num, neighbor.frame.index, neighbor_reasoning_type,
                                      self.settings, use_models)
                chosen.append(model)

        return chosen


    cdef void sample_agents_from_list(self, list agents, int n, list chosen_ids):
        cdef int agents_n = len(agents)
        cdef int chosen_n = len(chosen_ids)
        cdef int agent_i, chosen_i, rand_i
        cdef bint found
        cdef Agent agent

        cdef list available = []
        cdef int modeled_n = 0
        for agent_i in range(agents_n):
            found = False
            agent = agents[agent_i]

            for chosen_i in range(chosen_n):
                if agent.agent_num == chosen_ids[chosen_i]:
                    found = True
                    modeled_n += 1
                    break

            if not found:
                available.append(agent)

        cdef int available_n = len(available)
        while modeled_n < n:
            rand_i = rand() % available_n
            agent = available.pop(rand_i)

            chosen_ids.append(agent.agent_num)
            modeled_n += 1
            available_n -= 1


    cdef bint enough_agents_for_modeling(self, double epsilon_p, list available_agents):
        # organize the neighborhoods by frame/action
        cdef list available_neighborhoods = self.organize_into_neighborhoods(available_agents)
        cdef list full_neighborhoods = self.organize_into_neighborhoods(self.domain.agents)

        # sort the neighborhoods by size
        cdef list available_action_neighborhood, full_action_neighborhood
        cdef tuple tup
        cdef int neighborhood_i
        cdef int neighborhoods_n = len(available_neighborhoods)
        for neighborhood_i in range(neighborhoods_n):
            available_action_neighborhood = <list> available_neighborhoods[neighborhood_i]
            full_action_neighborhood = <list> full_neighborhoods[neighborhood_i]
            tup = tuple((len(available_action_neighborhood), len(full_action_neighborhood),
                         available_action_neighborhood, full_action_neighborhood))
            available_neighborhoods[neighborhood_i] = tup
        available_neighborhoods.sort(reverse=True)

        # are there enough agents in each neighborhood?
        cdef int N, n, N_available
        cdef double t_alpha, part
        for neighborhood_i in range(neighborhoods_n):
            tup = available_neighborhoods[neighborhood_i]
            N_available = <int> tup[0]
            N = <int> tup[1]

            if N > 0:
                for n in range(2, N+2):
                    t_alpha = t_dist.interval(0.95, n-1)[1]
                    part = t_alpha / (2 * epsilon_p)
                    part = part * part

                    if n > math.ceil((N * part) / (N - 1 + part)):
                        break

                # were there enough agents in the neighborhood to model?
                if n > N_available:
                    return False

        # every neighborhood checked out okay
        return True


    cdef list organize_into_neighborhoods(self, list agents):
        # organize the neighborhoods by frame/action
        cdef list neighborhoods = []
        cdef int frames_n = self.domain.settings.FRAMES
        cdef int actions_n = len(self.domain.actions)
        cdef int neighborhoods_n = frames_n * actions_n
        cdef list action_neighborhood

        # create initial empty lists for all neighborhoods
        cdef int neighborhood_i
        for neighborhood_i in range(neighborhoods_n):
            action_neighborhood = []
            neighborhoods.append(action_neighborhood)

        # organize the agents into neighborhoods
        cdef int agents_n = len(agents)
        cdef int agent_i, neighbor_actions_n, neighbor_action_i, frame_offset
        cdef WildfireAgent neighbor
        cdef int[:] all_available_actions

        for agent_i in range(agents_n):
            neighbor = <WildfireAgent> agents[agent_i]

            # do not model ourselves!
            if neighbor.agent_num == self.agent_num:
                continue

            all_available_actions = neighbor.frame.get_all_available_actions()
            frame_offset = neighbor.frame.index * actions_n

            neighbor_actions_n = len(all_available_actions)
            for neighbor_action_i in range(neighbor_actions_n):
                neighborhood_i = frame_offset + all_available_actions[neighbor_action_i]

                action_neighborhood = neighborhoods[neighborhood_i]
                action_neighborhood.append(neighbor)

        return neighborhoods

    cpdef list organize_into_neighborhoods_small(self, list agents, int required_neighborhoods):
        """ Splits the agents in particular setup into required number of neighborhoods
        Args:
            agents(list):
                list of all agents in the setup
            required_neighborhoods(int):
                Number of neighborhoods/ cliques

        Returns:
            neighborhoods(list):
                List of neighborhoods with agent numbers
        """
        cdef int frames_n, frame_i, frame_offset, agents_n, agents_in_frame ,agent_i,neighborhoods_i
        #cdef list agents_per_frame = list(self.domain.settings.N_AGENTS_PER_FRAME)
        cdef list agent_nums_in_frames = []
        cdef list neighborhoods = []
        frames_n = self.domain.settings.FRAMES
        agents_n = len(agents)
        frame_offset = 0
        for neighborhood_i in range(required_neighborhoods):
            neighborhoods.append([])

        # Creating the list of agent numbers per frame
        for frame_i in range(frames_n):
            agent_nums_in_frames.append([])
            agents_in_frame = self.domain.settings.N_AGENTS_PER_FRAME[frame_i]
            #print(agents_in_frame)

            for agent_i in range(agents_in_frame):
                agent_nums_in_frames[frame_i].append(frame_offset + agent_i)

            frame_offset = frame_offset + agents_in_frame

        # finding neighborhoods
        if agents_n % required_neighborhoods !=0:
            return list(range(agents_n))
        else:
            for frame_i in range(frames_n):
                agent_nums = agent_nums_in_frames[frame_i]
                agents_split = np.array_split(agent_nums,required_neighborhoods)
                for i in range(len(neighborhoods)):
                    neighborhoods[i].extend(agents_split[i])
            return neighborhoods

    cdef list sample_agents_for_modeling_small(self, list available_agents,
                                              WildfireReasoningType neighbor_reasoning_type,
                                              int required_neighborhoods, bint use_models=True):
        """ Samples agents when there are very few agents and returns

        Returns:
            chosen(list):
                List of neighbors from the neighborhood of subject agent
        """
        cdef list neighborhoods = self.organize_into_neighborhoods_small(available_agents, required_neighborhoods)
        # cdef int agents_n = self.domain.num_agents()
        cdef int agent_i, neighbor_num, neighborhood_i
        cdef WildfireAgent model, neighbor
        cdef list chosen = []
        cdef list neighborhood_list, agent_neighborhood
        # cdef int agent_num = self.agent_num

        # find the neighborhood of subject agent
        for neighborhood_i in range(len(neighborhoods)):
            neighborhood_list  = neighborhoods[neighborhood_i]
            for neighborhood_i in range(len(neighborhood_list)):
                neighbor_num = neighborhood_list[neighborhood_i]
                if neighbor_num  == self.agent_num:
                    agent_neighborhood = neighborhood_list
                    break

        for agent_i in range(len(agent_neighborhood)):
            agent_num = agent_neighborhood[agent_i]
            if agent_num != self.agent_num:
                neighbor = self.domain.agents[agent_num]

                # treat them as a Heuristic agent
                model = WildfireAgent(neighbor.agent_num, neighbor.frame.index, neighbor_reasoning_type,
                                      self.settings, use_models)
                chosen.append(model)
        return chosen
