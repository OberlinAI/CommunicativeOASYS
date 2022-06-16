import sys; sys.path.append("../..")
import time

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

import pstats
import cProfile
import numpy as np
import pickle


def main():
    setup = 100

    settings = {}
    settings["epsilon_p"] = 0.0
    settings["trajectories"] = 5000
    settings["horizon"] = 10
    settings["ucb_c"] = 25.0
    settings["level"] = 1
    settings["particles_n"] = 100
    settings["cliques_n"] = 1
    settings["run"] = 1
    settings["premodel"] = True

    if settings["premodel"]:
        # Policy setup numbers are either 1, 2, or 3.
        # They correspond to actual setup numbers 1XX, 2XX, and 3XX, respectively
        setup_number = setup // 100
        filename = f'POMCPPFComm_policy_Setup{setup_number}.pkl'
        with open(filename, "rb") as f:
            settings["POMCPPFComm_global_policy"] = pickle.load(f)
    else:
        settings["POMCPPFComm_global_policy"] = dict()

    start = time.time()
    wildfire = Wildfire(setup, WildfireReasoningType.IPOMCPPFComm, settings)
    stop = time.time()
    print("Constructor:", (stop-start))

    print(len(wildfire.modeling_neighborhoods), len(wildfire.modeling_neighborhoods[0]))
    # sys.exit(-1)

    agent_num = 0

    state = wildfire.generate_start_state()
    internal_states = wildfire.generate_start_internal_states()
    internal_states.values[agent_num] = 2

    agent = wildfire.agents[agent_num]
    agent.set_current_state(state)
    agent.set_current_suppressant(internal_states.values[agent_num])

    cProfile.runctx("agent.choose_action()", globals(), locals(), "IPOMCPPFComm_Profile.prof")

    s = pstats.Stats("IPOMCPPFComm_Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


if __name__ == "__main__":
    main()
