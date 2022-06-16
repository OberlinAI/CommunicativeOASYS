import sys; sys.path.append("../..")

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType


REASONING_TYPE = WildfireReasoningType.POMCPPF
STEPS = 15

START_RUN = 1
STOP_RUN = 100


def main():
    # parse the command line arguments
    setup = int(sys.argv[1])
    start_run = int(sys.argv[2])
    stop_run = int(sys.argv[3])

    settings = {}
    settings["epsilon_p"] = float(sys.argv[4])
    settings["trajectories"] = int(sys.argv[5])
    settings["horizon"] = int(sys.argv[6])
    settings["ucb_c"] = float(sys.argv[7])
    settings["particles_n"] = int(sys.argv[8])
    settings["level"] = 0
    settings["cliques_n"] = int(sys.argv[9])

    for run in range(start_run, stop_run + 1):
        simulator = WildfireSimulation(run, setup, REASONING_TYPE, settings)
        simulator.run(STEPS)


if __name__ == "__main__":
    main()
