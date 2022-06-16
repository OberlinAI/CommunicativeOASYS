import sys; sys.path.append("../..")
import time

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

import pstats
import cProfile
import numpy as np


def main():
    setup = 100

    settings = {}
    settings["epsilon_p"] = 0.0
    settings["trajectories"] = 3000
    settings["horizon"] = 10
    settings["ucb_c"] = 80
    settings["level"] = 0
    settings["particles_n"] = 1

    start = time.time()
    wildfire = Wildfire(setup, WildfireReasoningType.POMCPPFComm, settings)
    stop = time.time()
    print("Constructor:", (stop-start))

    print(len(wildfire.modeling_neighborhoods), len(wildfire.modeling_neighborhoods[0]))

    agent_num = 0

    state = wildfire.generate_start_state()
    internal_states = wildfire.generate_start_internal_states()
    internal_states.values[agent_num] = 2

    agent = wildfire.agents[agent_num]
    agent.set_current_state(state)
    agent.set_current_suppressant(internal_states.values[agent_num])

    message_vector = np.asarray([-1, 2, 2], dtype=np.intc)
    cProfile.runctx("agent.choose_action(message_vector)", globals(), locals(), "POMCPPFComm_Profile.prof")

    s = pstats.Stats("POMCPPFComm_Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


if __name__ == "__main__":
    main()
