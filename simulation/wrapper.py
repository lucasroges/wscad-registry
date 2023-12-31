from .strategies import p2p_registry_provisioning, p2p_enhanced_registry_provisioning
from .utils import follow_user

import random

def algorithm_wrapper(parameters: dict):
    """Wrapper function to store random state for different datasets while the algorithm runs.

    Args:
        parameters (dict): Strategy parameters
    """
    # Saving the random state to restore it later because the code below uses random and may variate between different strategies of container registry provisioning and/or network scheduling
    random_state = random.getstate()

    # Running the custom algorithm
    try:
        eval(f"{parameters['algorithm']}_registry_provisioning")(parameters=parameters)
    except NameError:
        print(f"{parameters['algorithm']} strategy does not require a custom algorithm.")

    # Running the service reallocation algorithm
    follow_user()

    # Restoring the random state
    random.setstate(random_state)