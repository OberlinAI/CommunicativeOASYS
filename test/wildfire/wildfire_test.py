import sys; sys.path.append("../..")
import time

from oasys.domains.wildfire.wildfire import Wildfire
from oasys.domains.wildfire.wildfire_reasoning import WildfireReasoningType


def main():
    start = time.time()
    wildfire = Wildfire(100, WildfireReasoningType.NOOP)
    stop = time.time()
    print("Constructor:", (stop-start))

    start = time.time()
    configurations = wildfire.create_all_configurations()
    stop = time.time()
    print("Create all configurations:", (stop - start), len(configurations))

    start = time.time()
    wildfire.validate_transitions()
    stop = time.time()
    print("Validate transitions:", (stop - start))

    start = time.time()
    wildfire.validate_internal_transitions()
    stop = time.time()
    print("Validate internal transitions:", (stop - start))

    start = time.time()
    wildfire.validate_observations()
    stop = time.time()
    print("Validate observations:", (stop - start))


if __name__ == "__main__":
    main()
