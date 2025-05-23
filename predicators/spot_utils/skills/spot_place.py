"""Interface for spot placing skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_stow_arm import stow_arm


def place_at_relative_position(robot: Robot,
                               body_tform_goal: math_helpers.SE3Pose) -> None:
    """Assuming something is held, place is at the given pose.

    The position is relative to the robot's body. It is the responsibility
    of the user of this method to specify a pose that makes sense, e.g.,
    one with an angle facing downward to facilitate the place.

    Placing is always straight ahead of the robot, angled down.
    """
    # wait for user approval
    print("Place the object at the given pose {}".format(body_tform_goal))
    input("Press Enter to place the object...")
    move_hand_to_relative_pose(robot, body_tform_goal)
    # NOTE: short sleep necessary because without it, the robot
    # sometimes opens the gripper before the arm has fully
    # arrived at its position.
    time.sleep(1.0)
    # Open the hand.
    open_gripper(robot)
    # Stow the arm.
    time.sleep(1.0)
    stow_arm(robot)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # To run this test, the robot should already be holding something and at
    # the location where it wants to place.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.utils import verify_estop, DEFAULT_DUMPED_TF

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip

        sdk = create_standard_sdk('PlaceSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        robot.time_sync.wait_for_sync()
        # open_gripper(robot)
        # # Stow the arm.
        # time.sleep(1.0)
        # stow_arm(robot)
        # relative to body
        target_pos = DEFAULT_DUMPED_TF
        place_at_relative_position(robot, target_pos)

    _run_manual_test()
