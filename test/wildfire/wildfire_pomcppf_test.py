import sys; sys.path.append("../..")
import time

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType

import pstats
import cProfile


def main():
    setup = 100

    settings = {}
    settings["epsilon_p"] = 0.0
    settings["trajectories"] = 1000
    settings["horizon"] = 10
    settings["ucb_c"] = 25
    settings["particles_n"] = 100
    settings["level"] = 0

    start = time.time()
    wildfire = Wildfire(setup, WildfireReasoningType.POMCPPF, settings)
    stop = time.time()
    print("Constructor:", (stop-start))

    state = wildfire.generate_start_state()
    internal_states = wildfire.generate_start_internal_states()
    internal_states.values[0] = 2
    agent = wildfire.agents[0]
    agent.set_current_state(state)
    agent.set_current_suppressant(internal_states.values[0])

    cProfile.runctx("agent.choose_action()", globals(), locals(), "POMCPPF_Profile.prof")

    s = pstats.Stats("POMCPPF_Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()


if __name__ == "__main__":
    main()
