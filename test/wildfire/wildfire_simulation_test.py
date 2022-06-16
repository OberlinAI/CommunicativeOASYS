import sys; sys.path.append("../..")

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_simulation import WildfireSimulation
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType


SETUP = 100
REASONING_TYPE = WildfireReasoningType.Heuristic
STEPS = 15

START_RUN = 1
STOP_RUN = 1


def main():
    settings = {}
    settings["trajectories"] = 0

    for run in range(START_RUN, STOP_RUN + 1):
        simulator = WildfireSimulation(run, SETUP, REASONING_TYPE, settings)
        simulator.run(STEPS)


if __name__ == "__main__":
    main()
