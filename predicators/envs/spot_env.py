"""Basic environment for the Boston Dynamics Spot Robot."""
import abc
import functools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Collection, Dict, Iterator, List, \
    Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import open3d as o3d
from bosdyn.client import RetryableRpcError, create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate, setup_logging
from gym.spaces import Box
from scipy.spatial import Delaunay

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    AprilTagObjectDetectionID, KnownStaticObjectDetectionID, \
    LanguageObjectDetectionID, ObjectDetectionID, detect_objects, \
    visualize_all_artifacts
from predicators.spot_utils.perception.object_specific_grasp_selection import \
    brush_prompt, bucket_prompt, football_prompt, yogurt_prompt
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import \
    init_search_for_objects
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_absolute_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import _base_object_type, _container_type, \
    _immovable_object_type, _movable_object_type, _robot_type, \
    _robot_hand_type, get_allowed_map_regions, get_graph_nav_dir, \
    get_robot_gripper_open_percentage, get_spot_home_pose, \
    load_spot_metadata, object_to_top_down_geom, verify_estop, \
    DEFAULT_STAIR2HAND_TF
from predicators.structs import Action, EnvironmentTask, GoalDescription, \
    GroundAtom, LiftedAtom, Object, Observation, Predicate, State, \
    STRIPSOperator, Type, Variable

###############################################################################
#                                Base Class                                   #
###############################################################################


@dataclass(frozen=True)
class _SpotObservation:
    """An observation for a SpotEnv."""
    # Camera name to image
    images: Dict[str, RGBDImageWithContext]
    # Objects that are seen in the current image and their positions in world
    objects_in_view: Dict[Object, math_helpers.SE3Pose]
    # Objects seen only by the hand camera
    objects_in_hand_view: Set[Object]
    # Objects seen by any camera except the back camera
    objects_in_any_view_except_back: Set[Object]
    # Expose the robot object.
    robot: Object
    # Status of the robot gripper.
    gripper_open_percentage: float
    # Robot SE3 Pose
    robot_pos: math_helpers.SE3Pose
    # Ground atoms without ground-truth classifiers
    # A placeholder until all predicates have classifiers
    nonpercept_atoms: Set[GroundAtom]
    nonpercept_predicates: Set[Predicate]

@dataclass(frozen=True)
class _SpotArmObservation:
    """An observation for a SpotEnv."""
    # Camera name to image
    images: Dict[str, RGBDImageWithContext]
    # Objects that are seen in the current image and their global information
    # including at least SE3 pose, and possibly other features.
    objects_in_view: Dict[Object, Dict]
    # Objects seen only by the hand camera
    objects_in_hand_view: Set[Object]
    # Objects seen by any camera except the back camera
    objects_in_any_view_except_back: Set[Object]
    object_in_hand: Optional[Object]
    # Expose the robot object.
    robot: Object
    # Status of the robot gripper.
    gripper_open_percentage: float
    # Robot SE3 Pose
    robot_pos: math_helpers.SE3Pose
    hand_pos: math_helpers.SE3Pose
    # Other attributes
    calibrated: bool
    calibration_id: int
    # Ground atoms without ground-truth classifiers
    # A placeholder until all predicates have classifiers
    nonpercept_atoms: Set[GroundAtom]
    nonpercept_predicates: Set[Predicate]

class _AugmentPerceptionState(State):
    """Some continuous object features, and ground atoms in simulator_state.

    The main idea here is that we have some predicates with actual
    classifiers implemented, but not all.

    NOTE: these states are only created in the perceiver, but they are used
    in the classifier definitions for the dummy predicates
    """
    def __init__(self, data: Dict[Object, tuple],
                 simulator_state: Optional[Dict[str, Set]] = None,
                augmented_state: Optional[Dict[str, tuple]] = None):  
        super().__init__(data, simulator_state=simulator_state)
        self.augmented_state = augmented_state

    def copy(self) -> State:
        state_copy = {o: self._copy_state_value(self.data[o]) for o in self}
        augmented_state_copy = self.augmented_state.copy()
        return _AugmentPerceptionState(state_copy,
                                       augmented_state=augmented_state_copy)

class _PartialPerceptionState(State):
    """Some continuous object features, and ground atoms in simulator_state.

    The main idea here is that we have some predicates with actual
    classifiers implemented, but not all.

    NOTE: these states are only created in the perceiver, but they are used
    in the classifier definitions for the dummy predicates
    """

    @property
    def _simulator_state_predicates(self) -> Set[Predicate]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["predicates"]

    @property
    def _simulator_state_atoms(self) -> Set[GroundAtom]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["atoms"]

    def simulator_state_atom_holds(self, atom: GroundAtom) -> bool:
        """Check whether an atom holds in the simulator state."""
        assert atom.predicate in self._simulator_state_predicates
        return atom in self._simulator_state_atoms

    def allclose(self, other: State) -> bool:
        if self.simulator_state != other.simulator_state:
            return False
        return self._allclose(other)

    def copy(self) -> State:
        state_copy = {o: self._copy_state_value(self.data[o]) for o in self}
        sim_state_copy = {
            "predicates": self._simulator_state_predicates.copy(),
            "atoms": self._simulator_state_atoms.copy()
        }
        return _PartialPerceptionState(state_copy,
                                       simulator_state=sim_state_copy)


def _create_dummy_predicate_classifier(
        pred: Predicate) -> Callable[[State, Sequence[Object]], bool]:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        assert isinstance(s, _PartialPerceptionState)
        atom = GroundAtom(pred, objs)
        return s.simulator_state_atom_holds(atom)

    return _classifier


@functools.lru_cache(maxsize=None)
def get_robot(
) -> Tuple[Optional[Robot], Optional[SpotLocalizer], Optional[LeaseClient]]:
    """Create the robot only once.

    If we are doing a dry run, return dummy Nones for each component.
    """
    if CFG.spot_run_dry:
        return None, None, None
    setup_logging(False)
    hostname = CFG.spot_robot_ip
    path = get_graph_nav_dir()
    sdk = create_standard_sdk("PredicatorsClient-")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client,
                                     must_acquire=True,
                                     return_at_exit=True)
    assert path.exists()
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    return robot, localizer, lease_client


@functools.lru_cache(maxsize=None)
def get_detection_id_for_object(obj: Object) -> ObjectDetectionID:
    """Exposed for wrapper and options."""
    # Avoid circular import issues.
    from predicators.envs import \
        get_or_create_env  # pylint: disable=import-outside-toplevel
    env = get_or_create_env(CFG.env)
    assert isinstance(env, SpotRearrangementEnv)
    detection_id_to_obj = env._detection_id_to_obj  # pylint: disable=protected-access
    obj_to_detection_id = {o: d for d, o in detection_id_to_obj.items()}
    return obj_to_detection_id[obj]


def get_known_immovable_objects() -> Dict[Object, math_helpers.SE3Pose]:
    """Load known immovable object poses from metadata."""
    known_immovables = load_spot_metadata()["known-immovable-objects"]
    obj_to_pose: Dict[Object, math_helpers.SE3Pose] = {}
    for obj_name, obj_pos in known_immovables.items():
        obj = Object(obj_name, _immovable_object_type)
        yaw = obj_pos.get("yaw", 0.0)
        rot = math_helpers.Quat.from_yaw(yaw)
        pose = math_helpers.SE3Pose(obj_pos["x"],
                                    obj_pos["y"],
                                    obj_pos["z"],
                                    rot=rot)
        obj_to_pose[obj] = pose
    return obj_to_pose


class SpotRearrangementEnv(BaseEnv):
    """An environment containing tasks for a real Spot robot to execute.

    This is a base class that specific sub-classes that define actual
    tasks should inherit from.
    """

    render_x_lb: ClassVar[float] = -1.0
    render_x_ub: ClassVar[float] = 5.0
    render_y_lb: ClassVar[float] = -3.0
    render_y_ub: ClassVar[float] = 3.0

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        assert "spot_wrapper" in CFG.approach or \
               "spot_wrapper" in CFG.approach_wrapper, \
            "Must use spot wrapper in spot envs!"
        robot, localizer, lease_client = get_robot()
        self._robot = robot
        self._localizer = localizer
        self._lease_client = lease_client
        # Note that we need to include the operators in this
        # class because they're used to update the symbolic
        # parts of the state during execution.
        self._strips_operators: Set[STRIPSOperator] = set()
        self._current_task_goal_reached = False
        self._last_action: Optional[Action] = None

        # Create constant objects.
        self._spot_object = Object("robot", _robot_type)

        # For noisy simulation in dry runs.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # For object detection.
        self._allowed_regions: Collection[Delaunay] = get_allowed_map_regions()

        # Used for the move-related hacks in step().
        self._last_known_object_poses: Dict[Object, math_helpers.SE3Pose] = {}

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    @property
    def types(self) -> Set[Type]:
        return set(_ALL_TYPES)

    @property
    def predicates(self) -> Set[Predicate]:
        return set(_ALL_PREDICATES)

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set(_ALL_PREDICATES)

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return self.predicates - _NONPERCEPT_PREDICATES

    @property
    def action_space(self) -> Box:
        # The action space is effectively empty because only the extra info
        # part of actions are used.
        return Box(0, 1, (0, ))

    @abc.abstractmethod
    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        """Environment-specific task generation for spot dry runs."""

    def _get_next_dry_observation(
            self, action: Action,
            nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
        """Step-like function for spot dry runs."""
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, _, action_args = action.extra_info
        obs = self._current_observation
        assert isinstance(obs, _SpotObservation)

        if action_name == "MoveToHandViewObject":
            _, target_obj = action_objs
            robot_rel_se2_pose = action_args[1]
            return _dry_simulate_move_to_view_hand(obs, target_obj,
                                                   robot_rel_se2_pose,
                                                   nonpercept_atoms)

        if action_name == "PickObjectFromTop":
            _, target_obj, _ = action_objs
            pixel = action_args[2]
            return _dry_simulate_pick_from_top(obs, target_obj, pixel,
                                               nonpercept_atoms)

        if action_name == "PickObjectToDrag":
            _, target_obj = action_objs
            pixel = action_args[2]
            return _dry_simulate_pick_from_top(obs, target_obj, pixel,
                                               nonpercept_atoms)

        if action_name in [
                "MoveToReachObject", "MoveToReadySweep", "MoveToBodyViewObject"
        ]:
            robot_rel_se2_pose = action_args[1]
            return _dry_simulate_move_to_reach_obj(obs, robot_rel_se2_pose,
                                                   nonpercept_atoms)

        if action_name == "PlaceObjectOnTop":
            _, held_obj, target_surface = action_objs
            if len(action_args) == 1:
                if not CFG.spot_run_dry:
                    assert target_surface.name == "floor"
                place_offset = math_helpers.Vec3(0.0, 0.0, 0.0)
            else:
                place_offset = action_args[1]
            return _dry_simulate_place_on_top(obs, held_obj, target_surface,
                                              place_offset, nonpercept_atoms)

        if action_name == "DropObjectInside":
            _, held_obj, container_obj = action_objs
            drop_offset = action_args[1]
            return _dry_simulate_drop_inside(obs, held_obj, container_obj,
                                             drop_offset, nonpercept_atoms)

        if action_name == "PrepareContainerForSweeping":
            _, container_obj, _, _ = action_objs
            _, _, new_robot_se2_pose, _, _ = action_args
            return _dry_simulate_prepare_container_for_sweeping(
                obs, container_obj, new_robot_se2_pose, nonpercept_atoms)

        if action_name == "SweepIntoContainer":
            _, _, target, _, container = action_objs
            _, _, _, _, _, duration = action_args
            return _dry_simulate_sweep_into_container(obs, {target},
                                                      container,
                                                      nonpercept_atoms,
                                                      duration=duration,
                                                      rng=self._noise_rng)

        if action_name == "SweepTwoObjectsIntoContainer":
            _, _, target1, target2, _, container = action_objs
            _, _, _, _, _, duration = action_args
            return _dry_simulate_sweep_into_container(obs, {target1, target2},
                                                      container,
                                                      nonpercept_atoms,
                                                      duration=duration,
                                                      rng=self._noise_rng)

        if action_name in ["DragToUnblockObject", "DragToBlockObject"]:
            _, blocker, _ = action_objs
            _, robot_rel_se2_pose = action_args
            return _dry_simulate_drag(obs, blocker, robot_rel_se2_pose,
                                      nonpercept_atoms)

        if action_name in ["PickAndDumpCup", "PickAndDumpContainer"]:
            objs_inside = {action_objs[3]}
            return _dry_simulate_pick_and_dump_container(
                obs, objs_inside, nonpercept_atoms, self._noise_rng)

        if action_name == "PickAndDumpTwoFromContainer":
            objs_inside = {action_objs[3], action_objs[4]}
            return _dry_simulate_pick_and_dump_container(
                obs, objs_inside, nonpercept_atoms, self._noise_rng)

        if action_name == "DropNotPlaceableObject":
            return _dry_simulate_drop_not_placeable_object(
                obs, nonpercept_atoms)

        if action_name in ["find-objects", "stow-arm"]:
            return _dry_simulate_noop(obs, nonpercept_atoms)

        raise NotImplementedError("Dry simulation not implemented for action "
                                  f"{action_name}")

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        # NOTE: task_idx and train_or_test ignored unless loading from JSON!
        if CFG.test_task_json_dir is not None and train_or_test == "test":
            self._current_task = self._test_tasks[task_idx]
        elif CFG.spot_run_dry:
            self._current_task = self._get_dry_task(train_or_test, task_idx)
        elif self._current_observation is not None and train_or_test == "train":
            # For the real spot environment, only actively construct the state
            # once, at the very beginning (or on loading, if needed).
            goal_description = self._generate_goal_description()
            self._current_task = EnvironmentTask(self._current_observation,
                                                 goal_description)
        else:
            prompt = f"Please set up {train_or_test} task {task_idx}!"
            utils.prompt_user(prompt)
            assert self._lease_client is not None
            # Automatically retry if a retryable error is encountered.
            while True:
                try:
                    self._lease_client.take()
                    self._current_task = self._actively_construct_env_task()
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")
        self._current_observation = self._current_task.init_obs
        self._current_task_goal_reached = False
        self._last_action = None
        return self._current_task.init_obs

    def step(self, action: Action) -> Observation:
        """Override step() because simulate() is not implemented."""
        assert isinstance(action.extra_info, (list, tuple))
        action_name, action_objs, action_fn, action_fn_args = action.extra_info
        self._last_action = action
        # The extra info is (action name, objects, function, function args).
        # The action name is either an operator name (for use with nonpercept
        # predicates) or a special name. See below for the special names.

        obs = self._current_observation
        assert isinstance(obs, _SpotObservation)
        assert self.action_space.contains(action.arr)

        # Special case: the action is "done", indicating that the robot
        # believes it has finished the task. Used for goal checking.
        if action_name == "done":

            # During a dry run, trust that the goal is accomplished if the
            # done action is returned, since we don't want a human in the loop.
            if CFG.spot_run_dry:
                self._current_task_goal_reached = True
                return self._current_observation

            while True:
                goal_description = self._current_task.goal_description
                logging.info(f"The goal is: {goal_description}")
                prompt = "Is the goal accomplished? Answer y or n. "
                response = utils.prompt_user(prompt).strip()
                if response == "y":
                    self._current_task_goal_reached = True
                    break
                if response == "n":
                    self._current_task_goal_reached = False
                    break
                logging.info("Invalid input, must be either 'y' or 'n'")
            return self._current_observation

        # Otherwise, the action is either an operator to execute or a special
        # action. The only difference between the two is that operators update
        # the non-perfect states.

        operator_names = {o.name for o in self._strips_operators}

        # The action corresponds to an operator finishing.
        if action_name in operator_names:
            # Update the non-percepts.
            operator_names = {o.name for o in self._strips_operators}
            next_nonpercept = self._get_next_nonpercept_atoms(obs, action)
        else:
            next_nonpercept = obs.nonpercept_atoms

        if CFG.spot_run_dry:
            # Simulate the effect of the action.
            next_obs = self._get_next_dry_observation(action, next_nonpercept)

        else:
            # Execute the action in the real environment. Automatically retry
            # if a retryable error is encountered.
            while True:
                try:
                    action_fn(*action_fn_args)  # type: ignore
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")

            # Get the new observation. Again, automatically retry if needed.
            while True:
                try:
                    next_obs = self._build_observation(next_nonpercept)
                    break
                except RetryableRpcError as e:
                    logging.warning("WARNING: the following retryable error "
                                    f"was encountered. Trying again.\n{e}")

            # Very hacky optimization to force viewing/reaching to work.
            if action_name in [
                    "MoveToHandViewObject", "MoveToBodyViewObject",
                    "MoveToReachObject"
            ]:
                _, target_obj = action_objs
                # Retry if each of the types of moving failed in their own way.
                if action_name == "MoveToHandViewObject":
                    need_retry = target_obj not in \
                        next_obs.objects_in_hand_view
                elif action_name == "MoveToBodyViewObject":
                    need_retry = target_obj not in \
                        next_obs.objects_in_any_view_except_back
                else:
                    assert action_name == "MoveToReachObject"
                    obj_pose = self._last_known_object_poses[target_obj]
                    obj_position = math_helpers.Vec3(x=obj_pose.x,
                                                     y=obj_pose.y,
                                                     z=obj_pose.z)
                    need_retry = not _obj_reachable_from_spot_pose(
                        next_obs.robot_pos, obj_position)
                if need_retry:
                    logging.warning(f"WARNING: retrying {action_name} because "
                                    f"{target_obj} was not seen/reached.")
                    prompt = (
                        "Hit 'c' to have the robot do a random movement "
                        "or take control and move the robot accordingly. "
                        "Hit the 'Enter' key when you're done!")
                    user_pref = input(prompt)
                    assert self._lease_client is not None
                    self._lease_client.take()
                    angle = 0.0
                    if user_pref == "c":
                        # Do a small random movement to get a new view.
                        assert isinstance(action_fn_args[1],
                                          math_helpers.SE2Pose)
                        angle = self._noise_rng.uniform(-np.pi / 6, np.pi / 6)
                    rel_pose = math_helpers.SE2Pose(0, 0, angle)
                    new_action_args = action_fn_args[0:1] + (rel_pose, ) + \
                        action_fn_args[2:]
                    new_action = utils.create_spot_env_action(
                        action_name,
                        action_objs,
                        action_fn,
                        new_action_args,
                    )
                    return self.step(new_action)

        self._current_observation = next_obs
        return self._current_observation

    def get_observation(self) -> Observation:
        return self._current_observation

    def goal_reached(self) -> bool:
        return self._current_task_goal_reached

    def _build_observation(self,
                           ground_atoms: Set[GroundAtom]) -> _SpotObservation:
        """Helper for building a new _SpotObservation().

        This is an environment method because the nonpercept predicates
        may vary per environment.
        """
        # Make sure the robot pose is up to date.
        assert self._robot is not None
        assert self._localizer is not None
        self._localizer.localize()
        # Get the universe of all object detections.
        all_object_detection_ids = set(self._detection_id_to_obj)
        # Get the camera images.
        time.sleep(0.5)
        rgbds = capture_images(self._robot, self._localizer)
        all_detections, all_artifacts = detect_objects(
            all_object_detection_ids, rgbds, self._allowed_regions)

        if CFG.spot_render_perception_outputs:
            outdir = Path(CFG.spot_perception_outdir)
            time_str = time.strftime("%Y%m%d-%H%M%S")
            detections_outfile = outdir / f"detections_{time_str}.png"
            no_detections_outfile = outdir / f"no_detections_{time_str}.png"
            visualize_all_artifacts(all_artifacts, detections_outfile,
                                    no_detections_outfile)

        # Separately, get detections for the hand in particular.
        hand_rgbd = {
            k: v
            for (k, v) in rgbds.items() if k == "hand_color_image"
        }
        hand_detections, hand_artifacts = detect_objects(
            all_object_detection_ids, hand_rgbd, self._allowed_regions)

        if CFG.spot_render_perception_outputs:
            detections_outfile = outdir / f"hand_detections_{time_str}.png"
            no_detect_outfile = outdir / f"hand_no_detections_{time_str}.png"
            visualize_all_artifacts(hand_artifacts, detections_outfile,
                                    no_detect_outfile)

        # Also, get detections that every camera except the back camera can
        # see. This is important for our 'InView' predicate.
        non_back_camera_rgbds = {
            k: v
            for (k, v) in rgbds.items() if k in [
                "hand_color_image", "frontleft_fisheye_image",
                "frontright_fisheye_image"
            ]
        }
        non_back_detections, _ = detect_objects(all_object_detection_ids,
                                                non_back_camera_rgbds,
                                                self._allowed_regions)

        # Now construct a dict of all objects in view, as well as a set
        # of objects that the hand can see, and that all cameras except
        # the back can see.
        all_objects_in_view = {
            self._detection_id_to_obj[det_id]: val
            for (det_id, val) in all_detections.items()
        }
        self._last_known_object_poses.update(all_objects_in_view)
        objects_in_hand_view = set(self._detection_id_to_obj[det_id]
                                   for det_id in hand_detections)
        objects_in_any_view_except_back = set(
            self._detection_id_to_obj[det_id]
            for det_id in non_back_detections)
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        robot_pos = self._localizer.get_last_robot_pose()

        # Hack to deal with the difficulty of reliably seeing objects after
        # sweeping them. If we just finished sweeping some object(s) but we
        # don't immediately see them afterwards, then ask the user to indicate
        # whether each object ended up in the container. If so, then update
        # the observation so that the swept object is inside the container.
        # Otherwise do nothing and let the lost object dance proceed.
        if self._last_action is not None:
            assert isinstance(self._last_action.extra_info, (list, tuple))
            op_name, op_objects, _, _ = self._last_action.extra_info
            if op_name == "SweepTwoObjectsIntoContainer":
                swept_objects: Set[Object] = set(op_objects[2:4])
                container: Optional[Object] = op_objects[-1]
            elif op_name == "SweepIntoContainer":
                swept_objects = {op_objects[2]}
                container = op_objects[-1]
            else:
                swept_objects = set()
                container = None
            static_feats = load_spot_metadata()["static-object-features"]
            for swept_object in swept_objects:
                if swept_object not in all_objects_in_view:
                    assert container is not None
                    assert container in all_objects_in_view
                    while True:
                        msg = (f"\nATTENTION! The {swept_object.name} was not "
                               "seen after sweeping. Is it now in the "
                               f"{container.name}? [y/n]\n")
                        response = utils.prompt_user(msg)
                        if response == "y":
                            # Update the pose to be inside the container.
                            container_pose = all_objects_in_view[container]
                            # Calculate the z pose of the swept object.
                            height = static_feats[swept_object.name]["height"]
                            swept_object_z = container_pose.z + height / 2
                            swept_pose = math_helpers.SE3Pose(
                                x=container_pose.x,
                                y=container_pose.y,
                                z=swept_object_z,
                                rot=container_pose.rot)
                            all_objects_in_view[swept_object] = swept_pose
                            objects_in_any_view_except_back.add(swept_object)
                            break
                        if response == "n":
                            break

        # Prepare the non-percepts.
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in ground_atoms)
        obs = _SpotObservation(rgbds, all_objects_in_view,
                               objects_in_hand_view,
                               objects_in_any_view_except_back,
                               self._spot_object, gripper_open_percentage,
                               robot_pos, ground_atoms, nonpercept_preds)

        return obs

    def _get_next_nonpercept_atoms(self, obs: _SpotObservation,
                                   action: Action) -> Set[GroundAtom]:
        """Helper for step().

        This should be deprecated eventually.
        """
        assert isinstance(action.extra_info, (list, tuple))
        op_name, op_objects, _, _ = action.extra_info
        op_name_to_op = {o.name: o for o in self._strips_operators}
        op = op_name_to_op[op_name]
        ground_op = op.ground(tuple(op_objects))
        # Update the atoms using the operator.
        next_ground_atoms = utils.apply_operator(ground_op,
                                                 obs.nonpercept_atoms)
        # Return only the atoms for the non-percept predicates.
        return {
            a
            for a in next_ground_atoms
            if a.predicate not in self.percept_predicates
        }

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for SpotEnv.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_goal_description()  # currently just one goal
        return [
            EnvironmentTask(None, goal) for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_goal_description()  # currently just one goal
        return [EnvironmentTask(None, goal) for _ in range(CFG.num_test_tasks)]

    def _actively_construct_env_task(self) -> EnvironmentTask:
        # Have the spot walk around the environment once to construct
        # an initial observation.
        assert self._robot is not None
        assert self._localizer is not None
        objects_in_view = self._actively_construct_initial_object_views()
        rgbd_images = capture_images(self._robot, self._localizer)
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        self._localizer.localize()
        robot_pos = self._localizer.get_last_robot_pose()
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in nonpercept_atoms)
        obs = _SpotObservation(rgbd_images, objects_in_view, set(), set(),
                               self._spot_object, gripper_open_percentage,
                               robot_pos, nonpercept_atoms, nonpercept_preds)
        goal_description = self._generate_goal_description()
        task = EnvironmentTask(obs, goal_description)
        # Save the task for future use.
        json_objects = {o.name: o.type.name for o in objects_in_view}
        json_objects[self._spot_object.name] = self._spot_object.type.name
        init_json_dict = {
            o.name: {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "qw": pose.rot.w,
                "qx": pose.rot.x,
                "qy": pose.rot.y,
                "qz": pose.rot.z,
            }
            for o, pose in objects_in_view.items()
        }
        # Add static object features.
        metadata = load_spot_metadata()
        static_object_features = metadata.get("static-object-features", {})
        for obj_name, obj_feats in static_object_features.items():
            if obj_name in init_json_dict:
                init_json_dict[obj_name].update(obj_feats)
        for obj in objects_in_view:
            if "lost" in obj.type.feature_names:
                init_json_dict[obj.name]["lost"] = 0.0
            if "in_hand_view" in obj.type.feature_names:
                init_json_dict[obj.name]["in_hand_view"] = 1.0
            if "in_view" in obj.type.feature_names:
                init_json_dict[obj.name]["in_view"] = 1.0
            if "held" in obj.type.feature_names:
                init_json_dict[obj.name]["held"] = 0.0
        init_json_dict[self._spot_object.name] = {
            "gripper_open_percentage": gripper_open_percentage,
            "x": robot_pos.x,
            "y": robot_pos.y,
            "z": robot_pos.z,
            "qw": robot_pos.rot.w,
            "qx": robot_pos.rot.x,
            "qy": robot_pos.rot.y,
            "qz": robot_pos.rot.z,
        }
        json_dict = {
            "objects": json_objects,
            "init": init_json_dict,
            "goal_description": goal_description,
        }
        outfile = utils.get_env_asset_path("task_jsons/spot/last.json",
                                           assert_exists=False)
        outpath = Path(outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)
        logging.info(f"Dumped task to {outfile}. Rename it to save it.")
        return task

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        # Use the BaseEnv default code for loading from JSON, which will
        # create a State as an observation. We'll then convert that State
        # into a _SpotObservation instead.
        base_env_task = super()._load_task_from_json(json_file)
        init = base_env_task.init
        # Images not currently saved or used.
        images: Dict[str, RGBDImageWithContext] = {}
        objects_in_view: Dict[Object, math_helpers.SE3Pose] = {}
        known_objects = set(
            self._detection_id_to_obj.values()) | {self._spot_object}
        robot: Optional[Object] = None
        for obj in init:
            assert obj in known_objects
            if obj.is_instance(_robot_type):
                robot = obj
                continue
            pos = math_helpers.SE3Pose(
                init.get(obj, "x"), init.get(obj, "y"), init.get(obj, "z"),
                math_helpers.Quat(init.get(obj, "qw"), init.get(obj, "qx"),
                                  init.get(obj, "qy"), init.get(obj, "qz")))
            objects_in_view[obj] = pos
        assert robot is not None
        assert robot == self._spot_object
        gripper_open_percentage = init.get(robot, "gripper_open_percentage")
        robot_pos = math_helpers.SE3Pose(
            init.get(robot, "x"), init.get(robot, "y"), init.get(robot, "z"),
            math_helpers.Quat(init.get(robot, "qw"), init.get(robot, "qx"),
                              init.get(robot, "qy"), init.get(robot, "qz")))

        # Reset the robot to the given position.
        if self._robot is not None:
            assert self._localizer is not None
            self._localizer.localize()
            navigate_to_absolute_pose(self._robot, self._localizer,
                                      robot_pos.get_closest_se2_transform())

        # Prepare the non-percepts.
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        init_obs = _SpotObservation(
            images,
            objects_in_view,
            set(),
            set(),
            robot,
            gripper_open_percentage,
            robot_pos,
            nonpercept_atoms,
            nonpercept_preds,
        )
        # The goal can remain the same.
        goal = base_env_task.goal_description
        return EnvironmentTask(init_obs, goal)

    def _actively_construct_initial_object_views(
            self) -> Dict[Object, math_helpers.SE3Pose]:
        assert self._robot is not None
        assert self._localizer is not None
        stow_arm(self._robot)
        go_home(self._robot, self._localizer)
        self._localizer.localize()
        detection_ids = self._detection_id_to_obj.keys()
        detections = self._run_init_search_for_objects(set(detection_ids))
        stow_arm(self._robot)
        obj_to_se3_pose = {
            self._detection_id_to_obj[det_id]: val
            for (det_id, val) in detections.items()
        }
        self._last_known_object_poses.update(obj_to_se3_pose)
        return obj_to_se3_pose

    def _run_init_search_for_objects(
        self, detection_ids: Set[ObjectDetectionID]
    ) -> Dict[ObjectDetectionID, math_helpers.SE3Pose]:
        """Have the hand look down from high up at first."""
        assert self._robot is not None
        assert self._localizer is not None
        hand_pose = math_helpers.SE3Pose(x=0.80,
                                         y=0.0,
                                         z=0.75,
                                         rot=math_helpers.Quat.from_pitch(
                                             np.pi / 3))
        move_hand_to_relative_pose(self._robot, hand_pose)
        detections, artifacts = init_search_for_objects(
            self._robot,
            self._localizer,
            detection_ids,
            allowed_regions=self._allowed_regions)
        if CFG.spot_render_perception_outputs:
            outdir = Path(CFG.spot_perception_outdir)
            time_str = time.strftime("%Y%m%d-%H%M%S")
            detections_outfile = outdir / f"detections_{time_str}.png"
            no_detections_outfile = outdir / f"no_detections_{time_str}.png"
            visualize_all_artifacts(artifacts, detections_outfile,
                                    no_detections_outfile)
        return detections

    @property
    @abc.abstractmethod
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:
        """Get an object from a perception detection ID."""

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        """Get the initial atoms for nonpercept predicates."""
        return set()

    @abc.abstractmethod
    def _generate_goal_description(self) -> GoalDescription:
        """For now, we assume that there's only one goal per environment."""


###############################################################################
#                   Shared Types, Predicates, Operators                       #
###############################################################################

## Constants
# This was 2.5, not sure if it affects future tasks.
HANDEMPTY_GRIPPER_THRESHOLD = 0.5  # made public for use in perceiver
_ONTOP_Z_THRESHOLD = 0.2
_INSIDE_Z_THRESHOLD = 0.4
_ONTOP_SURFACE_BUFFER = 0.48
_INSIDE_SURFACE_BUFFER = 0.1
_FITS_IN_XY_BUFFER = 0.1
_REACHABLE_THRESHOLD = 0.5  # very close
_VIEWABLE_THRESHOLD = 1.1  # slightly less than half length of arm
_REACHABLE_YAW_THRESHOLD = 0.1  # higher better
_CONTAINER_SWEEP_READY_BUFFER = 0.5
_ROBOT_SWEEP_READY_TOL = 0.25

## Types
_ALL_TYPES = {
    _robot_type,
    _base_object_type,
    _movable_object_type,
    _immovable_object_type,
    _container_type,
}


## Predicates
def _neq_classifier(state: State, objects: Sequence[Object]) -> bool:
    del state  # not used
    obj0, obj1 = objects
    return obj0 != obj1


def _handempty_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot = objects[0]
    gripper_open_percentage = state.get(spot, "gripper_open_percentage")
    return gripper_open_percentage >= HANDEMPTY_GRIPPER_THRESHOLD


def _holding_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    if _handempty_classifier(state, [spot]):
        return False
    return state.get(obj, "held") > 0.5


def _not_holding_classifier(state: State, objects: Sequence[Object]) -> bool:
    _, obj = objects
    if not obj.is_instance(_movable_object_type):
        return True
    return not _holding_classifier(state, objects)


def _object_in_xy_classifier(state: State,
                             obj1: Object,
                             obj2: Object,
                             buffer: float = 0.0) -> bool:
    """Helper for _on_classifier and _inside_classifier."""
    if obj1 == obj2:
        return False

    spot, = state.get_objects(_robot_type)
    if obj1.is_instance(_movable_object_type) and \
        _is_placeable_classifier(state, [obj1]) and \
        _holding_classifier(state, [spot, obj1]):
        return False

    # Check that the center of the object is contained within the surface in
    # the xy plane. Add a size buffer to the surface to compensate for small
    # errors in perception.
    surface_geom = object_to_top_down_geom(obj2, state, size_buffer=buffer)
    center_x = state.get(obj1, "x")
    center_y = state.get(obj1, "y")
    ret_val = surface_geom.contains_point(center_x, center_y)

    return ret_val


def _on_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj_on, obj_surface = objects

    # Check that the bottom of the object is close to the top of the surface.
    expect = state.get(obj_surface, "z") + state.get(obj_surface, "height") / 2
    actual = state.get(obj_on, "z") - state.get(obj_on, "height") / 2
    classification_val = abs(actual - expect) < _ONTOP_Z_THRESHOLD

    # If so, check that the object is within the bounds of the surface.
    if not _object_in_xy_classifier(
            state, obj_on, obj_surface, buffer=_ONTOP_SURFACE_BUFFER):
        return False

    return classification_val


def _top_above_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj1, obj2 = objects

    top1 = state.get(obj1, "z") + state.get(obj1, "height") / 2
    top2 = state.get(obj2, "z") + state.get(obj2, "height") / 2

    return top1 > top2


def _inside_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj_in, obj_container = objects

    if not _object_in_xy_classifier(
            state, obj_in, obj_container, buffer=_INSIDE_SURFACE_BUFFER):
        return False

    obj_z = state.get(obj_in, "z")
    obj_half_height = state.get(obj_in, "height") / 2
    obj_bottom = obj_z - obj_half_height
    obj_top = obj_z + obj_half_height

    container_z = state.get(obj_container, "z")
    container_half_height = state.get(obj_container, "height") / 2
    container_bottom = container_z - container_half_height
    container_top = container_z + container_half_height

    # Check that the bottom is "above" the bottom of the container.
    if obj_bottom < container_bottom - _INSIDE_Z_THRESHOLD:
        return False

    # Check that the top is "below" the top of the container.
    return obj_top < container_top + _INSIDE_Z_THRESHOLD


def _not_inside_any_container_classifier(state: State,
                                         objects: Sequence[Object]) -> bool:
    obj_in, = objects
    for container in state.get_objects(_container_type):
        if _inside_classifier(state, [obj_in, container]):
            return False
    return True


def _fits_in_xy_classifier(state: State, objects: Sequence[Object]) -> bool:
    # Just look in the xy plane and use a conservative approximation.
    contained, container = objects
    obj_to_radius: Dict[Object, float] = {}
    for obj in objects:
        obj_geom = object_to_top_down_geom(obj, state)
        if isinstance(obj_geom, utils.Rectangle):
            if obj is contained:
                radius = max(obj_geom.width / 2, obj_geom.height / 2)
            else:
                assert obj is container
                radius = min(obj_geom.width / 2, obj_geom.height / 2)
        else:
            assert isinstance(obj_geom, utils.Circle)
            radius = obj_geom.radius
        obj_to_radius[obj] = radius
    contained_radius = obj_to_radius[contained]
    container_radius = obj_to_radius[container]
    return contained_radius + _FITS_IN_XY_BUFFER < container_radius


def in_hand_view_classifier(state: State, objects: Sequence[Object]) -> bool:
    """Made public for perceiver."""
    _, tool = objects
    return state.get(tool, "in_hand_view") > 0.5


def in_general_view_classifier(state: State,
                               objects: Sequence[Object]) -> bool:
    """Made public for perceiver."""
    _, tool = objects
    return state.get(tool, "in_view") > 0.5


def _obj_reachable_from_spot_pose(spot_pose: math_helpers.SE3Pose,
                                  obj_pose: math_helpers.SE3Pose) -> bool:
    # Check if the obj is not on the ground
    average_stair_height = (CFG.viewplan_stair_height_range[0] +
                            CFG.viewplan_stair_height_range[1]) / 2
    if obj_pose.z > CFG.viewplan_ground_z + average_stair_height + \
        2 * CFG.viewplan_trans_noise:
        logging.info("Not reachable because of z")
        return False
    body_pose_mat = spot_pose.to_matrix()
    # hand pose
    hand_pose = obj_pose.mult(DEFAULT_STAIR2HAND_TF)
    hand_pose_mat = hand_pose.to_matrix()
    # Check if IK has a solution.
    robot = SpotArmFK()
    suc, _ = robot.compute_ik(body_pose_mat, hand_pose_mat)
    spot_yaw = spot_pose.get_closest_se2_transform().angle
    obj_yaw = obj_pose.get_closest_se2_transform().angle
    yaw_difference = abs(abs(spot_yaw - obj_yaw) - np.pi)
    is_yaw_good = yaw_difference < CFG.reach_stair_yaw_tol
    
    distance_2d = np.sqrt(
        (spot_pose.x - obj_pose.x)**2 +
        (spot_pose.y - obj_pose.y)**2)
    is_distance_good = ((distance_2d <= CFG.reach_stair_distance[1])
                and (distance_2d >= CFG.reach_stair_distance[0]))

    return (suc and is_yaw_good)

def _obj_viewable_from_spot_pose(spot_pose: math_helpers.SE3Pose,
                                 obj_pose: math_helpers.SE3Pose) -> bool:
    # first check if the object is within the robot's Viewable threshold
    # if not, the object is not viewable
    is_xy_near = np.sqrt(
        (spot_pose.x - obj_pose.x)**2 +
        (spot_pose.y - obj_pose.y)**2) <= (CFG.cam2obj_distance_tol_trivial[1] 
                                           + CFG.spot_arm_length)
    
    # Compute angle between spot's forward direction and the line from
    # spot to the object.
    spot_yaw = spot_pose.get_closest_se2_transform().angle
    obj_yaw = obj_pose.get_closest_se2_transform().angle

    yaw_difference = abs(spot_yaw - obj_yaw)

    is_yaw_near = yaw_difference > np.pi / 2

    return is_xy_near and is_yaw_near

def _obj_viewable_from_spot_pose_arm(spot_pose: math_helpers.SE3Pose,
                                     obj_pose: math_helpers.SE3Pose) -> bool:
    spot_kine_model = SpotArmFK()
    # hand pose
    tgt_x_axis = obj_pose.to_matrix()[:3, 0]
    tgt_z_axis = obj_pose.to_matrix()[:3, 2]
    tgt_position = np.array([obj_pose.x, 
                            obj_pose.y,
                            obj_pose.z])
    hand_tgt_dist = (CFG.cam2obj_distance_tol[0] + CFG.cam2obj_distance_tol[1]) / 2
    hand_position = tgt_position + hand_tgt_dist * tgt_x_axis
    hand_x_axis = -tgt_x_axis
    hand_z_axis = tgt_z_axis
    hand_y_axis = np.cross(hand_z_axis, hand_x_axis)
    hand_rot = math_helpers.Quat.from_matrix(np.array([hand_x_axis, hand_y_axis, hand_z_axis]).T)
    hand_pose = math_helpers.SE3Pose(x=hand_position[0], y=hand_position[1], z=hand_position[2], rot=hand_rot)
    hand_pose_mat = hand_pose.to_matrix()
    spot_pose_mat = spot_pose.to_matrix()
    # Check if IK has a solution.
    suc, _ = spot_kine_model.compute_ik(spot_pose_mat, hand_pose_mat)
    if not suc:
        return False
    return True

def _hand_sees_obj(spot_hand_pose: math_helpers.SE3Pose,
                   obj_pose: math_helpers.SE3Pose) -> bool:
    """
    Classifies whether the object is visible from the arm camera based on hand pose.

    Args:
        spot_hand_pose: SE3 pose of the arm's hand (relative to the robot base).
        obj_pose: SE3 pose of the object.

    Returns:
        bool: True if the object is visible and within criteria, False otherwise.
    """
    # Pre-defined constants (tolerances and camera intrinsics)
    distance_range = CFG.cam2obj_distance_tol_trivial                # Allowed distance range (min, max)
    noise_tolerance_angle = np.deg2rad(CFG.cam2obj_angle_tol)  # Noise tolerance in alignment
    
    # Get the relative pose from the hand to the object
    spot_hand_x_axis = spot_hand_pose.rot.to_matrix()[:, 0]
    obj_x_axis = obj_pose.rot.to_matrix()[:, 0]
    mis_match = np.pi - np.arccos(np.dot(spot_hand_x_axis, obj_x_axis))
    view_angle_ok = mis_match < noise_tolerance_angle
    
    # Check if the hand’s x-axis (camera x-axis) points to the object within tolerance
    relative_pos = np.array([
        obj_pose.x - spot_hand_pose.x,
        obj_pose.y - spot_hand_pose.y,
        obj_pose.z - spot_hand_pose.z
    ])
    obj_distance = np.linalg.norm(relative_pos)
    distance_ok = distance_range[0] <= obj_distance <= distance_range[1]
    
    if view_angle_ok and distance_ok:
        return True
    return False

def _hand_sees_hard_obj(spot_hand_pose: math_helpers.SE3Pose,
                   obj_pose: math_helpers.SE3Pose) -> bool:
    """
    Classifies whether the object is visible from the arm camera based on hand pose.

    Args:
        spot_hand_pose: SE3 pose of the arm's hand (relative to the robot base).
        obj_pose: SE3 pose of the object.

    Returns:
        bool: True if the object is visible and within criteria, False otherwise.
    """
    # Pre-defined constants (tolerances and camera intrinsics)
    distance_range = CFG.cam2obj_distance_tol                # Allowed distance range (min, max)
    noise_tolerance_angle = np.deg2rad(CFG.cam2obj_angle_tol_hard)  # Noise tolerance in alignment
    
    # Get the relative pose from the hand to the object
    spot_hand_x_axis = spot_hand_pose.rot.to_matrix()[:, 0]
    obj_x_axis = obj_pose.rot.to_matrix()[:, 0]
    mis_match = np.pi - np.arccos(np.dot(spot_hand_x_axis, obj_x_axis))
    view_angle_ok = mis_match < noise_tolerance_angle
    
    # Check if the hand’s x-axis (camera x-axis) points to the object within tolerance
    relative_pos = np.array([
        obj_pose.x - spot_hand_pose.x,
        obj_pose.y - spot_hand_pose.y,
        obj_pose.z - spot_hand_pose.z
    ])
    obj_distance = np.linalg.norm(relative_pos)
    distance_ok = distance_range[0] <= obj_distance <= distance_range[1]
    
    if view_angle_ok and distance_ok:
        return True
    return False

def _reachable_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    spot_pose = utils.get_se3_pose_from_state(state, spot)
    obj_pose = utils.get_se3_pose_from_state(state, obj)
    return _obj_reachable_from_spot_pose(spot_pose, obj_pose)

def _viewable_classifier(state: State, objects: Sequence[Object]) -> bool:
    # viewable also checks orientation of the object
    # this in general is more strict than reachable
    spot, obj = objects
    spot_pose = utils.get_se3_pose_from_state(state, spot)
    obj_pose = utils.get_se3_pose_from_state(state, obj)
    return _obj_viewable_from_spot_pose(spot_pose, obj_pose)

def _viewable_arm_classifier(state: State, objects: Sequence[Object]) -> bool:
    # viewable also checks orientation of the object
    # this in general is more strict than reachable
    spot, obj = objects
    spot_pose = utils.get_se3_pose_from_state(state, spot)
    obj_pose = utils.get_se3_pose_from_state(state, obj)
    return _obj_viewable_from_spot_pose_arm(spot_pose, obj_pose)

def _hand_sees_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    spot_hand_pose = utils.get_se3_hand_pose_from_state(state, spot)
    obj_pose = utils.get_se3_pose_from_state(state, obj)
    return _hand_sees_obj(spot_hand_pose, obj_pose)

def _hand_sees_hard_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    spot_hand_pose = utils.get_se3_hand_pose_from_state(state, spot)
    obj_pose = utils.get_se3_pose_from_state(state, obj)
    return _hand_sees_hard_obj(spot_hand_pose, obj_pose)

def _blocking_classifier(state: State, objects: Sequence[Object]) -> bool:
    """This is not a very good classifier for blocking because it only looks at
    the relationship between the objects and the robot's home pose.

    It's possible that objects will appear blocked under this
    classifier, but could actually be accessed from another angle. Doing
    this in a better way is hard, but hopefully something we can do in
    the future.
    """
    blocker_obj, blocked_obj = objects

    if blocker_obj == blocked_obj:
        return False

    # Only consider draggable (non-placeable, movable) objects to be blockers.
    if not blocker_obj.is_instance(_movable_object_type):
        return False
    if _is_placeable_classifier(state, [blocker_obj]):
        return False

    if _object_in_xy_classifier(state,
                                blocked_obj,
                                blocker_obj,
                                buffer=_ONTOP_SURFACE_BUFFER):
        return False

    if _object_in_xy_classifier(state,
                                blocker_obj,
                                blocked_obj,
                                buffer=_ONTOP_SURFACE_BUFFER):
        return False

    spot, = state.get_objects(_robot_type)
    if blocked_obj.is_instance(_movable_object_type) and \
        _holding_classifier(state, [spot, blocked_obj]):
        return False

    # Draw a line between blocked and the robot’s current pose.
    # Check if blocker intersects that line.
    robot_x = state.get(spot, "x")
    robot_y = state.get(spot, "y")

    blocked_x = state.get(blocked_obj, "x")
    blocked_y = state.get(blocked_obj, "y")

    blocked_robot_line = utils.LineSegment(robot_x, robot_y, blocked_x,
                                           blocked_y)

    # Don't put the blocker on the robot, even if it's held, because we don't
    # want to consider the blocker to be unblocked until it's actually moved
    # out of the way and released by the robot. Otherwise the robot might just
    # pick something up and put it back down, thinking it's unblocked. Also,
    # use a relatively large size buffer so that we're conservative in terms
    # of what's blocking versus not blocking.
    size_buffer = 0.25
    blocker_geom = object_to_top_down_geom(blocker_obj,
                                           state,
                                           size_buffer=size_buffer,
                                           put_on_robot_if_held=False)

    return blocker_geom.intersects(blocked_robot_line)


def _not_blocked_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj, = objects
    blocker_type, _ = _Blocking.types
    for blocker in state.get_objects(blocker_type):
        if _blocking_classifier(state, [blocker, obj]):
            return False
    return True


def _get_highest_surface_object_is_on(obj: Object,
                                      state: State) -> Optional[Object]:
    highest_surface: Optional[Object] = None
    highest_surface_z = -np.inf
    for other_obj in state.get_objects(_immovable_object_type):
        if _on_classifier(state, [obj, other_obj]):
            other_obj_z = state.get(other_obj, "z")
            if other_obj_z > highest_surface_z:
                highest_surface_z = other_obj_z
                highest_surface = other_obj
    return highest_surface


def _container_adjacent_to_surface_for_sweeping(container: Object,
                                                surface: Object,
                                                state: State) -> bool:

    surface_x = state.get(surface, "x")
    surface_y = state.get(surface, "y")

    # This is the location for spot to go to before placing. We need to convert
    # it into an expected location for the container.
    param_dict = load_spot_metadata()["prepare_container_relative_xy"]
    dx, dy, angle = param_dict["dx"], param_dict["dy"], param_dict["angle"]
    place_distance = 0.65
    expected_x = surface_x + dx + place_distance * np.cos(angle)
    expected_y = surface_y + dy + place_distance * np.sin(angle)

    container_x = state.get(container, "x")
    container_y = state.get(container, "y")

    return np.sqrt(
        (expected_x - container_x)**2 +
        (expected_y - container_y)**2) <= _CONTAINER_SWEEP_READY_BUFFER


def _container_ready_for_sweeping_classifier(
        state: State, objects: Sequence[Object]) -> bool:
    container, target = objects

    # Compute the expected x, y position based on the parameters for placing
    # next to the object that the target is on.
    surface = _get_highest_surface_object_is_on(target, state)
    if surface is None:
        return False
    return _container_adjacent_to_surface_for_sweeping(container, surface,
                                                       state)


def _is_placeable_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj, = objects
    return state.get(obj, "placeable") > 0.5


def _is_not_placeable_classifier(state: State,
                                 objects: Sequence[Object]) -> bool:
    return not _is_placeable_classifier(state, objects)


def _is_sweeper_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj, = objects
    return state.get(obj, "is_sweeper") > 0.5


def _has_flat_top_surface_classifier(state: State,
                                     objects: Sequence[Object]) -> bool:
    obj, = objects
    return state.get(obj, "flat_top_surface") > 0.5


def _robot_ready_for_sweeping_classifier(state: State,
                                         objects: Sequence[Object]) -> bool:
    robot, target = objects

    # Check if the target is where we expect it to be relative to the robot.
    robot_pose = utils.get_se3_pose_from_state(state, robot)
    robot_se2 = robot_pose.get_closest_se2_transform()
    sweep_distance = 0.8
    expected_xy_vec = robot_se2 * math_helpers.Vec2(sweep_distance, 0.0)
    expected_xy = np.array([expected_xy_vec.x, expected_xy_vec.y])
    target_xy = np.array([state.get(target, "x"), state.get(target, "y")])
    return np.allclose(expected_xy, target_xy, atol=_ROBOT_SWEEP_READY_TOL)


def _is_semantically_greater_than_classifier(
        state: State, objects: Sequence[Object]) -> bool:
    del state  # unused
    obj1, obj2 = objects
    # Check if the name of object 1 is greater (in a Pythonic sense) than
    # that of object 2.
    return obj1.name > obj2.name


def _get_sweeping_surface_for_container(container: Object,
                                        state: State) -> Optional[Object]:
    if container.is_instance(_container_type):
        for surface in state.get_objects(_immovable_object_type):
            if _container_adjacent_to_surface_for_sweeping(
                    container, surface, state):
                return surface
    return None


_NEq = Predicate("NEq", [_base_object_type, _base_object_type],
                 _neq_classifier)
_On = Predicate("On", [_movable_object_type, _base_object_type],
                _on_classifier)
_TopAbove = Predicate("TopAbove", [_base_object_type, _base_object_type],
                      _top_above_classifier)
_Inside = Predicate("Inside", [_movable_object_type, _container_type],
                    _inside_classifier)
_FitsInXY = Predicate("FitsInXY", [_movable_object_type, _base_object_type],
                      _fits_in_xy_classifier)
# NOTE: use this predicate instead if you want to disable inside checking.
_FakeInside = Predicate(_Inside.name, _Inside.types,
                        _create_dummy_predicate_classifier(_Inside))
_NotInsideAnyContainer = Predicate("NotInsideAnyContainer",
                                   [_movable_object_type],
                                   _not_inside_any_container_classifier)
_HandEmpty = Predicate("HandEmpty", [_robot_type], _handempty_classifier)
_Holding = Predicate("Holding", [_robot_type, _movable_object_type],
                     _holding_classifier)
_NotHolding = Predicate("NotHolding", [_robot_type, _base_object_type],
                        _not_holding_classifier)
_InHandView = Predicate("InHandView", [_robot_type, _movable_object_type],
                        in_hand_view_classifier)
_InView = Predicate("InView", [_robot_type, _movable_object_type],
                    in_general_view_classifier)
_Reachable = Predicate("Reachable", [_robot_type, _base_object_type],
                       _reachable_classifier)
_Viewable = Predicate("Viewable", [_robot_hand_type, _base_object_type],
                       _viewable_classifier)
_Viewable_Arm = Predicate("ViewableArm", [_robot_hand_type, _base_object_type],
                       _viewable_arm_classifier)
_HandSees = Predicate("HandSees", [_robot_hand_type, _base_object_type], 
                     _hand_sees_classifier)
_Blocking = Predicate("Blocking", [_base_object_type, _base_object_type],
                      _blocking_classifier)
_NotBlocked = Predicate("NotBlocked", [_base_object_type],
                        _not_blocked_classifier)
_ContainerReadyForSweeping = Predicate(
    "ContainerReadyForSweeping", [_container_type, _movable_object_type],
    _container_ready_for_sweeping_classifier)
_IsPlaceable = Predicate("IsPlaceable", [_movable_object_type],
                         _is_placeable_classifier)
_IsSweeper = Predicate("IsSweeper", [_movable_object_type],
                       _is_sweeper_classifier)
_IsNotPlaceable = Predicate("IsNotPlaceable", [_movable_object_type],
                            _is_not_placeable_classifier)
_HasFlatTopSurface = Predicate("HasFlatTopSurface", [_immovable_object_type],
                               _has_flat_top_surface_classifier)
_RobotReadyForSweeping = Predicate("RobotReadyForSweeping",
                                   [_robot_type, _movable_object_type],
                                   _robot_ready_for_sweeping_classifier)
_IsSemanticallyGreaterThan = Predicate(
    "_IsSemanticallyGreaterThan", [_base_object_type, _base_object_type],
    _is_semantically_greater_than_classifier)
_ALL_PREDICATES = {
    _NEq, _On, _TopAbove, _Inside, _NotInsideAnyContainer, _FitsInXY,
    _HandEmpty, _Holding, _NotHolding, _InHandView, _InView, _Reachable,
    _Blocking, _NotBlocked, _ContainerReadyForSweeping, _IsPlaceable,
    _IsNotPlaceable, _IsSweeper, _HasFlatTopSurface, _RobotReadyForSweeping,
    _IsSemanticallyGreaterThan
}
_NONPERCEPT_PREDICATES: Set[Predicate] = set()


## Operators (needed in the environment for non-percept atom hack)
def _create_operators() -> Iterator[STRIPSOperator]:
    """Inside a function to avoid scoping issues."""

    # MoveToReachObject
    robot = Variable("?robot", _robot_type)
    obj = Variable("?object", _base_object_type)
    parameters = [robot, obj]
    preconds = {
        LiftedAtom(_NotBlocked, [obj]),
        LiftedAtom(_NotHolding, [robot, obj]),
    }
    add_effs = {LiftedAtom(_Reachable, [robot, obj])}
    del_effs: Set[LiftedAtom] = set()
    ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
    yield STRIPSOperator("MoveToReachObject", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # MoveToHandViewObject
    robot = Variable("?robot", _robot_type)
    obj = Variable("?object", _movable_object_type)
    parameters = [robot, obj]
    preconds = {
        LiftedAtom(_NotBlocked, [obj]),
        LiftedAtom(_HandEmpty, [robot])
    }
    add_effs = {LiftedAtom(_InHandView, [robot, obj])}
    del_effs = set()
    ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
    yield STRIPSOperator("MoveToHandViewObject", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # MoveToBodyViewObject
    robot = Variable("?robot", _robot_type)
    obj = Variable("?object", _movable_object_type)
    parameters = [robot, obj]
    preconds = {
        LiftedAtom(_NotBlocked, [obj]),
        LiftedAtom(_NotHolding, [robot, obj])
    }
    add_effs = {
        LiftedAtom(_InView, [robot, obj]),
    }
    del_effs = set()
    ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
    yield STRIPSOperator("MoveToBodyViewObject", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # PickObjectFromTop
    robot = Variable("?robot", _robot_type)
    obj = Variable("?object", _movable_object_type)
    surface = Variable("?surface", _immovable_object_type)
    parameters = [robot, obj, surface]
    preconds = {
        LiftedAtom(_On, [obj, surface]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, obj]),
        LiftedAtom(_NotInsideAnyContainer, [obj]),
        LiftedAtom(_IsPlaceable, [obj]),
        LiftedAtom(_HasFlatTopSurface, [surface]),
    }
    add_effs = {
        LiftedAtom(_Holding, [robot, obj]),
    }
    del_effs = {
        LiftedAtom(_On, [obj, surface]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, obj]),
        LiftedAtom(_NotHolding, [robot, obj]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PickObjectFromTop", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # PickObjectToDrag
    robot = Variable("?robot", _robot_type)
    obj = Variable("?object", _movable_object_type)
    parameters = [robot, obj]
    preconds = {
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, obj]),
        LiftedAtom(_IsNotPlaceable, [obj]),
    }
    add_effs = {
        LiftedAtom(_Holding, [robot, obj]),
    }
    # Importantly, does not include On as a delete effect.
    del_effs = {
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, obj]),
        LiftedAtom(_NotHolding, [robot, obj]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PickObjectToDrag", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # PlaceObjectOnTop
    robot = Variable("?robot", _robot_type)
    held = Variable("?held", _movable_object_type)
    surface = Variable("?surface", _immovable_object_type)
    parameters = [robot, held, surface]
    preconds = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_Reachable, [robot, surface]),
        LiftedAtom(_NEq, [held, surface]),
        LiftedAtom(_IsPlaceable, [held]),
        LiftedAtom(_HasFlatTopSurface, [surface]),
        LiftedAtom(_FitsInXY, [held, surface]),
    }
    add_effs = {
        LiftedAtom(_On, [held, surface]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_NotHolding, [robot, held]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, held]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PlaceObjectOnTop", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # DropNotPlaceableObject
    robot = Variable("?robot", _robot_type)
    held = Variable("?held", _movable_object_type)
    parameters = [robot, held]
    preconds = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_IsNotPlaceable, [held]),
    }
    add_effs = {
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_NotHolding, [robot, held]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, held]),
    }
    ignore_effs = set()
    yield STRIPSOperator("DropNotPlaceableObject", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # DropObjectInside
    robot = Variable("?robot", _robot_type)
    held = Variable("?held", _movable_object_type)
    container = Variable("?container", _container_type)
    parameters = [robot, held, container]
    preconds = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_Reachable, [robot, container]),
        LiftedAtom(_IsPlaceable, [held]),
        LiftedAtom(_FitsInXY, [held, container]),
    }
    add_effs = {
        LiftedAtom(_Inside, [held, container]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_NotHolding, [robot, held]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_NotInsideAnyContainer, [held])
    }
    ignore_effs = set()
    yield STRIPSOperator("DropObjectInside", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # DropObjectInsideContainerOnTop
    robot = Variable("?robot", _robot_type)
    held = Variable("?held", _movable_object_type)
    container = Variable("?container", _container_type)
    surface = Variable("?surface", _immovable_object_type)
    parameters = [robot, held, container, surface]
    preconds = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_InView, [robot, container]),
        LiftedAtom(_On, [container, surface]),
        LiftedAtom(_IsPlaceable, [held]),
    }
    add_effs = {
        LiftedAtom(_Inside, [held, container]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_On, [held, surface]),
        LiftedAtom(_NotHolding, [robot, held]),
        LiftedAtom(_FitsInXY, [held, container]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, held]),
        LiftedAtom(_NotInsideAnyContainer, [held])
    }
    ignore_effs = set()
    yield STRIPSOperator("DropObjectInsideContainerOnTop", parameters,
                         preconds, add_effs, del_effs, ignore_effs)

    # DragToUnblockObject
    robot = Variable("?robot", _robot_type)
    blocked = Variable("?blocked", _base_object_type)
    blocker = Variable("?blocker", _movable_object_type)
    parameters = [robot, blocker, blocked]
    preconds = {
        LiftedAtom(_Blocking, [blocker, blocked]),
        LiftedAtom(_Holding, [robot, blocker]),
    }
    add_effs = {
        LiftedAtom(_NotBlocked, [blocked]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_NotHolding, [robot, blocker]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, blocker]),
    }
    ignore_effs = {_InHandView, _Reachable, _RobotReadyForSweeping, _Blocking}
    yield STRIPSOperator("DragToUnblockObject", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # DragToBlockObject
    robot = Variable("?robot", _robot_type)
    blocked = Variable("?blocked", _base_object_type)
    blocker = Variable("?blocker", _movable_object_type)
    parameters = [robot, blocker, blocked]
    preconds = {
        LiftedAtom(_NotBlocked, [blocked]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_Holding, [robot, blocker]),
    }
    add_effs = {
        LiftedAtom(_Blocking, [blocker, blocked]),
        LiftedAtom(_NotHolding, [robot, blocker]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, blocker]),
    }
    ignore_effs = {_InHandView, _Reachable, _RobotReadyForSweeping, _Blocking}
    yield STRIPSOperator("DragToBlockObject", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # MoveToReadySweep
    robot = Variable("?robot", _robot_type)
    container = Variable("?container", _container_type)
    target = Variable("?target", _movable_object_type)
    parameters = [robot, container, target]
    preconds = {
        LiftedAtom(_ContainerReadyForSweeping, [container, target]),
    }
    add_effs = {
        LiftedAtom(_RobotReadyForSweeping, [robot, target]),
    }
    del_effs = set()
    ignore_effs = {_Reachable, _InView, _InHandView}
    yield STRIPSOperator("MoveToReadySweep", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # SweepTwoObjectsIntoContainer
    robot = Variable("?robot", _robot_type)
    sweeper = Variable("?sweeper", _movable_object_type)
    target1 = Variable("?target1", _movable_object_type)
    target2 = Variable("?target2", _movable_object_type)
    surface = Variable("?surface", _immovable_object_type)
    container = Variable("?container", _container_type)
    parameters = [robot, sweeper, target1, target2, surface, container]
    preconds = {
        # Arbitrarily pick one of the targets to be the one not blocked,
        # to prevent the robot 'dragging to unblock' twice.
        LiftedAtom(_NotBlocked, [target1]),
        LiftedAtom(_Holding, [robot, sweeper]),
        LiftedAtom(_On, [target1, surface]),
        LiftedAtom(_On, [target2, surface]),
        # This `IsSemanticallyGreaterThan` predicate just serves to
        # provide a canonical grounding for this operator (we don't want
        # Sweep(yogurt, football) to be separate from Sweep(football, yogurt)).
        LiftedAtom(_IsSemanticallyGreaterThan, [target1, target2]),
        # Arbitrarily pick one of the targets to be the one ready for sweeping,
        # to prevent the robot 'moving to get ready for sweeping' twice.
        LiftedAtom(_RobotReadyForSweeping, [robot, target1]),
        # Same idea.
        LiftedAtom(_ContainerReadyForSweeping, [container, target1]),
        LiftedAtom(_IsPlaceable, [target1]),
        LiftedAtom(_IsPlaceable, [target2]),
        LiftedAtom(_HasFlatTopSurface, [surface]),
        LiftedAtom(_TopAbove, [surface, container]),
        LiftedAtom(_IsSweeper, [sweeper]),
        LiftedAtom(_FitsInXY, [target1, container]),
        LiftedAtom(_FitsInXY, [target2, container]),
    }
    add_effs = {
        LiftedAtom(_Inside, [target1, container]),
        LiftedAtom(_Inside, [target2, container]),
    }
    del_effs = {
        LiftedAtom(_On, [target1, surface]),
        LiftedAtom(_On, [target2, surface]),
        LiftedAtom(_ContainerReadyForSweeping, [container, target1]),
        LiftedAtom(_ContainerReadyForSweeping, [container, target2]),
        LiftedAtom(_RobotReadyForSweeping, [robot, target1]),
        LiftedAtom(_RobotReadyForSweeping, [robot, target2]),
        LiftedAtom(_Reachable, [robot, target1]),
        LiftedAtom(_Reachable, [robot, target2]),
        LiftedAtom(_NotInsideAnyContainer, [target1]),
        LiftedAtom(_NotInsideAnyContainer, [target2]),
    }
    ignore_effs = set()
    yield STRIPSOperator("SweepTwoObjectsIntoContainer", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # SweepIntoContainer
    robot = Variable("?robot", _robot_type)
    sweeper = Variable("?sweeper", _movable_object_type)
    target = Variable("?target", _movable_object_type)
    surface = Variable("?surface", _immovable_object_type)
    container = Variable("?container", _container_type)
    parameters = [robot, sweeper, target, surface, container]
    preconds = {
        LiftedAtom(_NotBlocked, [target]),
        LiftedAtom(_Holding, [robot, sweeper]),
        LiftedAtom(_On, [target, surface]),
        LiftedAtom(_RobotReadyForSweeping, [robot, target]),
        LiftedAtom(_ContainerReadyForSweeping, [container, target]),
        LiftedAtom(_IsPlaceable, [target]),
        LiftedAtom(_HasFlatTopSurface, [surface]),
        LiftedAtom(_TopAbove, [surface, container]),
        LiftedAtom(_IsSweeper, [sweeper]),
        LiftedAtom(_FitsInXY, [target, container]),
    }
    add_effs = {
        LiftedAtom(_Inside, [target, container]),
    }
    del_effs = {
        LiftedAtom(_On, [target, surface]),
        LiftedAtom(_ContainerReadyForSweeping, [container, target]),
        LiftedAtom(_RobotReadyForSweeping, [robot, target]),
        LiftedAtom(_Reachable, [robot, target]),
        LiftedAtom(_NotInsideAnyContainer, [target])
    }
    ignore_effs = set()
    yield STRIPSOperator("SweepIntoContainer", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # PrepareContainerForSweeping
    robot = Variable("?robot", _robot_type)
    target = Variable("?target", _movable_object_type)
    container = Variable("?container", _container_type)
    surface = Variable("?surface", _immovable_object_type)
    parameters = [robot, container, target, surface]
    preconds = {
        LiftedAtom(_Holding, [robot, container]),
        LiftedAtom(_On, [target, surface]),
        LiftedAtom(_TopAbove, [surface, container]),
        LiftedAtom(_NEq, [surface, container]),
    }
    add_effs = {
        LiftedAtom(_ContainerReadyForSweeping, [container, target]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_NotHolding, [robot, container]),
    }
    del_effs = {
        LiftedAtom(_Holding, [robot, container]),
    }
    ignore_effs = {_Reachable, _InHandView, _RobotReadyForSweeping}
    yield STRIPSOperator("PrepareContainerForSweeping", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # PickAndDumpCup
    robot = Variable("?robot", _robot_type)
    container = Variable("?container", _container_type)
    surface = Variable("?surface", _base_object_type)
    obj_inside = Variable("?object", _movable_object_type)
    parameters = [robot, container, surface, obj_inside]
    preconds = {
        LiftedAtom(_On, [container, surface]),
        LiftedAtom(_Inside, [obj_inside, container]),
        LiftedAtom(_On, [obj_inside, surface]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, container])
    }
    add_effs = {
        LiftedAtom(_Holding, [robot, container]),
        LiftedAtom(_NotInsideAnyContainer, [obj_inside])
    }
    del_effs = {
        LiftedAtom(_Inside, [obj_inside, container]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, container]),
        LiftedAtom(_On, [container, surface]),
        LiftedAtom(_NotHolding, [robot, container]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PickAndDumpCup", parameters, preconds, add_effs,
                         del_effs, ignore_effs)

    # PickAndDumpContainer (puts the container back down)
    robot = Variable("?robot", _robot_type)
    container = Variable("?container", _container_type)
    surface = Variable("?surface", _base_object_type)
    obj_inside = Variable("?object", _movable_object_type)
    parameters = [robot, container, surface, obj_inside]
    preconds = {
        LiftedAtom(_On, [container, surface]),
        LiftedAtom(_Inside, [obj_inside, container]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, container])
    }
    add_effs = {LiftedAtom(_NotInsideAnyContainer, [obj_inside])}
    del_effs = {
        LiftedAtom(_Inside, [obj_inside, container]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PickAndDumpContainer", parameters, preconds,
                         add_effs, del_effs, ignore_effs)

    # PickAndDumpTwoFromContainer (puts the container back down)
    robot = Variable("?robot", _robot_type)
    container = Variable("?container", _container_type)
    surface = Variable("?surface", _base_object_type)
    obj_inside1 = Variable("?object1", _movable_object_type)
    obj_inside2 = Variable("?object2", _movable_object_type)
    parameters = [robot, container, surface, obj_inside1, obj_inside2]
    preconds = {
        LiftedAtom(_On, [container, surface]),
        LiftedAtom(_Inside, [obj_inside1, container]),
        LiftedAtom(_Inside, [obj_inside2, container]),
        LiftedAtom(_HandEmpty, [robot]),
        LiftedAtom(_InHandView, [robot, container])
    }
    add_effs = {
        LiftedAtom(_NotInsideAnyContainer, [obj_inside1]),
        LiftedAtom(_NotInsideAnyContainer, [obj_inside2]),
    }
    del_effs = {
        LiftedAtom(_Inside, [obj_inside1, container]),
        LiftedAtom(_Inside, [obj_inside2, container]),
    }
    ignore_effs = set()
    yield STRIPSOperator("PickAndDumpTwoFromContainer", parameters, preconds,
                         add_effs, del_effs, ignore_effs)


###############################################################################
#                  Shared Utilities for Dry Run Simulation                    #
###############################################################################


def _dry_simulate_move_to_view_hand(
        last_obs: _SpotObservation, target_obj: Object,
        robot_rel_se2_pose: math_helpers.SE2Pose,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    gripper_open_percentage = last_obs.gripper_open_percentage
    robot_pose = last_obs.robot_pos

    # Add the target object to the set of objects in hand view.
    objects_in_hand_view.add(target_obj)

    # Update the robot position to be looking at the object, roughly.
    current_robot_se2_pose = robot_pose.get_closest_se2_transform()
    new_robot_se2_pose = current_robot_se2_pose * robot_rel_se2_pose
    robot_pose = new_robot_se2_pose.get_closest_se3_transform()

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_move_to_reach_obj(
        last_obs: _SpotObservation, robot_rel_se2_pose: math_helpers.SE2Pose,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    gripper_open_percentage = last_obs.gripper_open_percentage
    robot_pose = last_obs.robot_pos

    # Update the robot position to be looking at the object, roughly.
    current_robot_se2_pose = robot_pose.get_closest_se2_transform()
    new_robot_se2_pose = current_robot_se2_pose * robot_rel_se2_pose
    robot_pose = new_robot_se2_pose.get_closest_se3_transform()

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_pick_from_top(
        last_obs: _SpotObservation, target_obj: Object, pixel: Tuple[int, int],
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos

    # Check if the grasp is valid.
    target_pose = objects_in_view[target_obj]
    if _dry_grasp_is_valid(target_obj, target_pose, pixel):
        # Can't see anything in the hand because it's occluded now.
        objects_in_hand_view: Set[Object] = set()
        # Gripper is now closed.
        gripper_open_percentage = 100.0
        objects_in_any_view_except_back -= set(last_obs.objects_in_hand_view)
    # If the grasp failed, don't update the state.
    else:
        objects_in_hand_view = set(last_obs.objects_in_hand_view)
        gripper_open_percentage = last_obs.gripper_open_percentage

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_grasp_is_valid(target_obj: Object, target_pose: math_helpers.SE3Pose,
                        pixel: Tuple[int, int]) -> bool:
    """Helper for _dry_simulate_pick_from_top()."""
    # For now, we're assuming that the image is already oriented consistently
    # with respect to the object. But in the future, we might want to use the
    # pose of the target object (and the pose of the camera) to re-orient the
    # pixel before checking it in the grasp map.
    del target_pose
    # Load the top-down grasp map for this object.
    grasp_map_filename = f"grasp_maps/{target_obj.name}-grasps.npy"
    grasp_map_path = utils.get_env_asset_path(grasp_map_filename)
    grasp_map = np.load(grasp_map_path)
    is_valid = grasp_map[pixel[0], pixel[1]]

    # Uncomment for debugging.
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.imshow(grasp_map)
    # plt.plot([pixel[1]], [pixel[0]], marker="*", markersize=3, color="red")
    # valid_str = "VALID" if is_valid else "NOT valid"
    # plt.title(f"Grasp for {target_obj.name} is {valid_str}")
    # plt.savefig("grasp_debug.png")
    # import ipdb; ipdb.set_trace()

    return is_valid


def _dry_simulate_place_on_top(
        last_obs: _SpotObservation, held_obj: Object, target_surface: Object,
        place_offset: math_helpers.Vec3,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:

    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos

    static_feats = load_spot_metadata()["static-object-features"]
    surface_height = static_feats[target_surface.name]["height"]
    held_height = static_feats[held_obj.name]["height"]
    surface_pose = objects_in_view[target_surface]
    place_pose = robot_pose.transform_vec3(place_offset)
    x = place_pose.x
    y = place_pose.y
    # Place offset z ignored; gravity.
    z = surface_pose.z + surface_height / 2 + held_height
    held_obj_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
    # We want to ensure the held_obj doesn't get lost after dragging!
    objects_in_view[held_obj] = held_obj_pose
    objects_in_any_view_except_back.add(held_obj)

    # Gripper is now empty.
    gripper_open_percentage = 0.0

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_drop_inside(
        last_obs: _SpotObservation, held_obj: Object, container_obj: Object,
        drop_offset: math_helpers.Vec3,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:

    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos

    static_feats = load_spot_metadata()["static-object-features"]
    container_height = static_feats[container_obj.name]["height"]
    container_pose = objects_in_view[container_obj]
    drop_pose = robot_pose.transform_vec3(drop_offset)
    x = drop_pose.x
    y = drop_pose.y
    # Drop offset z ignored; gravity.
    z = container_pose.z - container_height / 2
    held_obj_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
    # We want to ensure the held_obj doesn't get lost after dropping!
    objects_in_view[held_obj] = held_obj_pose
    objects_in_any_view_except_back.add(held_obj)

    # Gripper is now empty.
    gripper_open_percentage = 0.0

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_drag(last_obs: _SpotObservation, held_obj: Object,
                       robot_rel_se2_pose: math_helpers.SE2Pose,
                       nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:

    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos

    # Update the robot pose.
    old_robot_se2_pose = robot_pose.get_closest_se2_transform()
    new_robot_se2_pose = old_robot_se2_pose * robot_rel_se2_pose
    robot_pose = new_robot_se2_pose.get_closest_se3_transform()

    # Now update the held object relative to the robot.
    old_held_pose = objects_in_view[held_obj]
    robot_length = 0.8
    robot_yaw = new_robot_se2_pose.angle
    x = robot_pose.x + robot_length * np.cos(robot_yaw)
    y = robot_pose.y + robot_length * np.sin(robot_yaw)
    z = old_held_pose.z
    held_obj_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
    # We want to ensure the held_obj doesn't get lost after dragging!
    objects_in_view[held_obj] = held_obj_pose
    objects_in_any_view_except_back.add(held_obj)

    # Gripper is now empty.
    gripper_open_percentage = 0.0

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_prepare_container_for_sweeping(
        last_obs: _SpotObservation, container_obj: Object,
        new_robot_se2_pose: math_helpers.SE2Pose,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:

    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos

    # Place the held container next to the target object, on the floor.
    # Also move the robot accordingly.
    static_object_feats = load_spot_metadata()["static-object-features"]
    container_height = static_object_feats[container_obj.name]["height"]
    floor_obj = next(o for o in objects_in_view if o.name == "floor")
    floor_pose = objects_in_view[floor_obj]
    # First update the robot.
    robot_pose = new_robot_se2_pose.get_closest_se3_transform()
    # Now update the container relative to the robot.
    robot_length = 0.8
    robot_yaw = new_robot_se2_pose.angle
    x = robot_pose.x + robot_length * np.cos(robot_yaw)
    y = robot_pose.y + robot_length * np.sin(robot_yaw)
    z = floor_pose.z + container_height / 2
    container_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
    # We want to ensure the container doesn't get lost after placing!
    objects_in_view[container_obj] = container_pose
    objects_in_any_view_except_back.add(container_obj)

    # Gripper is now empty.
    gripper_open_percentage = 0.0

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_sweep_into_container(
        last_obs: _SpotObservation, swept_objs: Set[Object], container: Object,
        nonpercept_atoms: Set[GroundAtom], duration: float,
        rng: np.random.Generator) -> _SpotObservation:

    # Initialize values based on the last observation.
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos
    gripper_open_percentage = last_obs.gripper_open_percentage

    static_feats = load_spot_metadata()["static-object-features"]
    container_pose = objects_in_view[container]
    container_radius = static_feats[container.name]["width"] / 2
    container_xy = np.array([container_pose.x, container_pose.y])

    # The optimal sweeping velocity depends on the initial positions of the
    # objects being swept. Objects farther from the container need a higher
    # velocity, closer need a lower velocity. Assume that what matters is the
    # FARTHEST object's position, since that object will collide with the
    # other objects during sweeping.
    farthest_swept_obj_distance = 0.0
    for swept_obj in swept_objs:
        swept_obj_pose = objects_in_view[swept_obj]
        swept_xy = np.array([swept_obj_pose.x, swept_obj_pose.y])
        dist = np.sum(np.square(np.subtract(swept_xy, container_xy)))
        farthest_swept_obj_distance = max(farthest_swept_obj_distance, dist)
    # Simply say that the optimal velocity is proportional to the distance.
    optimal_velocity = np.clip(farthest_swept_obj_distance, 0.1, 1.0)
    velocity = 1. / duration
    # If the given velocity is close enough to the optimal velocity, sweep all
    # objects successfully; otherwise, have the objects fall randomly.
    thresh = 0.15
    for swept_obj in swept_objs:
        swept_obj_height = static_feats[swept_obj.name]["height"]
        swept_obj_radius = static_feats[swept_obj.name]["width"] / 2
        # Successful sweep.
        if abs(optimal_velocity - velocity) < thresh:
            x = container_pose.x
            y = container_pose.y
            z = container_pose.z + swept_obj_height / 2
        # Failure: the object fails randomly somewhere around the container.
        else:
            angle = rng.uniform(0, 2 * np.pi)
            distance = (container_radius + swept_obj_radius) + rng.uniform(
                1.1 * _INSIDE_SURFACE_BUFFER, 2 * _INSIDE_SURFACE_BUFFER)
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            x = container_pose.x + dx
            y = container_pose.y + dy
            z = container_pose.z
            dist_to_container = (dx**2 + dy**2)**0.5
            assert dist_to_container > (container_radius +
                                        _INSIDE_SURFACE_BUFFER)

        swept_obj_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
        # We want to make sure the object(s) don't get lost after sweeping!
        objects_in_view[swept_obj] = swept_obj_pose
        objects_in_any_view_except_back.add(swept_obj)

    # Finalize the next observation.
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )

    return next_obs


def _dry_simulate_drop_not_placeable_object(
        last_obs: _SpotObservation,
        nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
    # Simply open the gripper.
    gripper_open_percentage = 0.0

    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )
    return next_obs


def _dry_simulate_noop(last_obs: _SpotObservation,
                       nonpercept_atoms: Set[GroundAtom]) -> _SpotObservation:
    objects_in_view = last_obs.objects_in_view.copy()
    objects_in_hand_view = set(last_obs.objects_in_hand_view)
    objects_in_any_view_except_back = set(
        last_obs.objects_in_any_view_except_back)
    robot_pose = last_obs.robot_pos
    gripper_open_percentage = last_obs.gripper_open_percentage
    next_obs = _SpotObservation(
        images={},
        objects_in_view=objects_in_view,
        objects_in_hand_view=objects_in_hand_view,
        objects_in_any_view_except_back=objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=gripper_open_percentage,
        robot_pos=robot_pose,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )
    return next_obs


def _dry_simulate_pick_and_dump_container(
        last_obs: _SpotObservation, objs_inside: Set[Object],
        nonpercept_atoms: Set[GroundAtom],
        rng: np.random.Generator) -> _SpotObservation:

    # Picking succeeded; dump the object on the floor.
    floor = next(o for o in last_obs.objects_in_view if o.name == "floor")

    # Randomize dropping on the floor.
    dx, dy = rng.uniform(-0.5, 0.5, size=2)
    place_offset = math_helpers.Vec3(dx, dy, 0)
    obs = last_obs
    for obj in objs_inside:
        obs = _dry_simulate_place_on_top(obs, obj, floor, place_offset,
                                         nonpercept_atoms)
    next_obs = _SpotObservation(
        images={},
        objects_in_view=obs.objects_in_view,
        objects_in_hand_view=obs.objects_in_hand_view,
        objects_in_any_view_except_back=obs.objects_in_any_view_except_back,
        robot=last_obs.robot,
        gripper_open_percentage=obs.gripper_open_percentage,
        robot_pos=obs.robot_pos,
        nonpercept_atoms=nonpercept_atoms,
        nonpercept_predicates=last_obs.nonpercept_predicates,
    )
    return next_obs


###############################################################################
#                                Cube Table Env                               #
###############################################################################


class SpotCubeEnv(SpotRearrangementEnv):
    """An environment corresponding to the spot cube task where the robot
    attempts to place an April Tag cube onto a particular table."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_cube_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        cube = Object("cube", _movable_object_type)
        cube_detection = AprilTagObjectDetectionID(410)
        detection_id_to_obj[cube_detection] = cube

        smooth_table = Object("smooth_table", _immovable_object_type)
        smooth_table_detection = AprilTagObjectDetectionID(408)
        detection_id_to_obj[smooth_table_detection] = smooth_table

        sticky_table = Object("sticky_table", _immovable_object_type)
        sticky_table_detection = AprilTagObjectDetectionID(409)
        detection_id_to_obj[sticky_table_detection] = sticky_table

        for obj, pose in get_known_immovable_objects().items():
            # Only keep the floor.
            if obj.name == "floor":
                detection = KnownStaticObjectDetectionID(obj.name, pose=pose)
                detection_id_to_obj[detection] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the cube on the sticky table"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        del train_or_test, task_idx  # task always the same for this simple env

        # Create the objects and their initial poses.
        objects_in_view: Dict[Object, math_helpers.SE3Pose] = {}

        # Make up some poses for the cube and tables, with the cube starting
        # on the smooth table.
        static_object_feats = load_spot_metadata()["static-object-features"]
        cube = Object("cube", _movable_object_type)
        table_height = static_object_feats["smooth_table"]["height"]
        cube_height = static_object_feats["cube"]["height"]
        x = self.render_x_ub - (self.render_x_ub - self.render_x_lb) / 5.0
        y = self.render_y_ub - (self.render_y_ub - self.render_y_lb) / 5.0
        z = table_height + cube_height / 2
        cube_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
        objects_in_view[cube] = cube_pose

        smooth_table = Object("smooth_table", _immovable_object_type)
        r = static_object_feats["smooth_table"]["length"] / 2
        x = x - r / 2
        y = y - r / 2
        z = table_height / 2
        smooth_table_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
        objects_in_view[smooth_table] = smooth_table_pose

        sticky_table = Object("sticky_table", _immovable_object_type)
        y = y - _REACHABLE_THRESHOLD / 2
        sticky_table_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
        objects_in_view[sticky_table] = sticky_table_pose

        # The floor is loaded directly from metadata.
        floor = Object("floor", _immovable_object_type)
        floor_feats = load_spot_metadata()["known-immovable-objects"]["floor"]
        x = floor_feats["x"]
        y = floor_feats["y"]
        z = floor_feats["z"]
        floor_pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
        objects_in_view[floor] = floor_pose

        # Create robot pose.
        robot_se2 = get_spot_home_pose()
        robot_pose = robot_se2.get_closest_se3_transform()

        # Create the initial observation.
        init_obs = _SpotObservation(
            images={},
            objects_in_view=objects_in_view,
            objects_in_hand_view=set(),
            objects_in_any_view_except_back=set(),
            robot=self._spot_object,
            gripper_open_percentage=0.0,
            robot_pos=robot_pose,
            nonpercept_atoms=self._get_initial_nonpercept_atoms(),
            nonpercept_predicates=(self.predicates - self.percept_predicates),
        )

        # Finish the task.
        goal_description = self._generate_goal_description()
        return EnvironmentTask(init_obs, goal_description)


###############################################################################
#                                Soda Table Env                               #
###############################################################################


class SpotSodaTableEnv(SpotRearrangementEnv):
    """An environment where a soda can needs to be moved from a white table to
    the side tables."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_soda_table_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        smooth_table = Object("smooth_table", _immovable_object_type)
        smooth_table_detection = AprilTagObjectDetectionID(408)
        detection_id_to_obj[smooth_table_detection] = smooth_table

        soda_can = Object("soda_can", _movable_object_type)
        soda_can_detection = LanguageObjectDetectionID("soda can")
        detection_id_to_obj[soda_can_detection] = soda_can

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the soda on the smooth table"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        raise NotImplementedError("Dry task generation not implemented.")


###############################################################################
#                               Soda Bucket Env                               #
###############################################################################


class SpotSodaBucketEnv(SpotRearrangementEnv):
    """An environment where a soda can needs to be put in a bucket."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
            "DropObjectInside",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_soda_bucket_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        soda_can = Object("soda_can", _movable_object_type)
        soda_can_detection = LanguageObjectDetectionID("soda can")
        detection_id_to_obj[soda_can_detection] = soda_can

        bucket = Object("bucket", _container_type)
        bucket_detection = LanguageObjectDetectionID("bucket")
        detection_id_to_obj[bucket_detection] = bucket

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the soda in the bucket"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        raise NotImplementedError("Dry task generation not implemented.")


###############################################################################
#                               Soda Chair Env                                #
###############################################################################


class SpotSodaChairEnv(SpotRearrangementEnv):
    """An environment where a soda can needs to be grasped, and a chair might
    need to be moved out of the way in order to grasp it."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
            "DropObjectInside",
            "DragToUnblockObject",
            "DragToBlockObject",
            "PickObjectToDrag",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_soda_chair_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        soda_can = Object("soda_can", _movable_object_type)
        soda_can_detection = LanguageObjectDetectionID("soda can")
        detection_id_to_obj[soda_can_detection] = soda_can

        chair = Object("chair", _movable_object_type)
        chair_detection = LanguageObjectDetectionID("chair")
        detection_id_to_obj[chair_detection] = chair

        bucket = Object("bucket", _container_type)
        bucket_detection = LanguageObjectDetectionID("bucket")
        detection_id_to_obj[bucket_detection] = bucket

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the soda in the bucket"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        raise NotImplementedError("Dry task generation not implemented.")


###############################################################################
#                               Main Sweep Env                                #
###############################################################################


class SpotMainSweepEnv(SpotRearrangementEnv):
    """An environment where objects need to be swept into a bucket."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "MoveToBodyViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
            "DragToUnblockObject",
            "DragToBlockObject",
            "SweepIntoContainer",
            "SweepTwoObjectsIntoContainer",
            "PrepareContainerForSweeping",
            "PickAndDumpContainer",
            "PickAndDumpTwoFromContainer",
            "DropNotPlaceableObject",
            "MoveToReadySweep",
            "PickObjectToDrag",
            "DropObjectInside",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_main_sweep_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        yogurt = Object("yogurt", _movable_object_type)
        yogurt_detection = LanguageObjectDetectionID(yogurt_prompt)
        detection_id_to_obj[yogurt_detection] = yogurt

        football = Object("football", _movable_object_type)
        football_detection = LanguageObjectDetectionID(football_prompt)
        detection_id_to_obj[football_detection] = football

        brush = Object("brush", _movable_object_type)
        brush_detection = LanguageObjectDetectionID(brush_prompt)
        detection_id_to_obj[brush_detection] = brush

        chair = Object("chair", _movable_object_type)
        chair_detection = LanguageObjectDetectionID("chair")
        detection_id_to_obj[chair_detection] = chair

        bucket = Object("bucket", _container_type)
        bucket_detection = LanguageObjectDetectionID(bucket_prompt)
        detection_id_to_obj[bucket_detection] = bucket

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return CFG.spot_sweep_env_goal_description

    @functools.lru_cache(maxsize=None)
    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        del task_idx  # lru_cache takes care of this
        rng = self._train_rng if train_or_test == "train" else self._test_rng

        # Create the objects and their initial poses.
        objects_in_view: Dict[Object, math_helpers.SE3Pose] = {}

        # Make up some poses for the objects, with the soda can starting on the
        # table, and the bucket, chair, and brush starting on the floor.
        metadata = load_spot_metadata()
        static_object_feats = metadata["static-object-features"]
        known_immovables = metadata["known-immovable-objects"]
        table_height = static_object_feats["black_table"]["height"]
        table_width = static_object_feats["black_table"]["width"]
        table_length = static_object_feats["black_table"]["length"]
        yogurt_height = static_object_feats["yogurt"]["height"]
        football_height = static_object_feats["football"]["height"]
        brush_height = static_object_feats["brush"]["height"]
        chair_height = static_object_feats["chair"]["height"]
        chair_width = static_object_feats["chair"]["width"]
        bucket_height = static_object_feats["bucket"]["height"]
        floor_z = known_immovables["floor"]["z"]
        table_x = known_immovables["black_table"]["x"]
        table_y = known_immovables["black_table"]["y"]

        # Create immovable objects.
        for obj, pose in get_known_immovable_objects().items():
            objects_in_view[obj] = pose

        # Create robot pose.
        robot_se2 = get_spot_home_pose()
        robot_pose = robot_se2.get_closest_se3_transform()

        # Create movable objects.
        obj_to_xyz: Dict[Object, Tuple[float, float, float]] = {}

        # Sample positions for the yogurt and football on top of the table.
        table = next(o for o in objects_in_view if o.name == "black_table")
        table_pose = objects_in_view[table]
        angle = table_pose.rot.to_yaw()
        table_rect = utils.Rectangle.from_center(table_pose.x, table_pose.y,
                                                 table_width, table_length,
                                                 angle)

        # Yogurt.
        yogurt = Object("yogurt", _movable_object_type)
        yogurt_x, yogurt_y = table_rect.sample_random_point(rng, 0.13)
        yogurt_z = floor_z + table_height + yogurt_height / 2
        obj_to_xyz[yogurt] = (yogurt_x, yogurt_y, yogurt_z)

        # Football.
        football = Object("football", _movable_object_type)
        football_x, football_y = table_rect.sample_random_point(rng, 0.13)
        football_z = floor_z + table_height + football_height / 2
        obj_to_xyz[football] = (football_x, football_y, football_z)

        if CFG.spot_graph_nav_map == "floor8-v2":

            # Brush.
            brush = Object("brush", _movable_object_type)
            brush_x = table_x
            brush_y = self.render_y_ub - (self.render_y_ub -
                                          self.render_y_lb) / 5
            brush_z = floor_z + brush_height / 2
            obj_to_xyz[brush] = (brush_x, brush_y, brush_z)

            # Chair.
            chair = Object("chair", _movable_object_type)
            chair_x = table_x - 1.5 * chair_width
            chair_y = table_y
            chair_z = floor_z + chair_height / 2
            obj_to_xyz[chair] = (chair_x, chair_y, chair_z)

            # Bucket.
            bucket = Object("bucket", _container_type)
            bucket_x = table_x
            bucket_y = self.render_y_lb + (self.render_y_ub -
                                           self.render_y_lb) / 5
            bucket_z = floor_z + bucket_height / 2
            obj_to_xyz[bucket] = (bucket_x, bucket_y, bucket_z)

        elif CFG.spot_graph_nav_map == "floor8-sweeping":

            # Brush.
            brush = Object("brush", _movable_object_type)
            brush_x = robot_pose.x + 1.0
            brush_y = robot_pose.y - 0.5
            brush_z = floor_z + brush_height / 2
            obj_to_xyz[brush] = (brush_x, brush_y, brush_z)

            # Chair.
            chair = Object("chair", _movable_object_type)
            chair_x = table_x
            chair_y = table_y + 1.5 * chair_width
            chair_z = floor_z + chair_height / 2
            obj_to_xyz[chair] = (chair_x, chair_y, chair_z)

            # Bucket. Note that this starts already next to the table
            # to be conducive to sweeping.
            bucket = Object("bucket", _container_type)
            bucket_x = table_x + 1.5
            bucket_y = table_y + 2.0
            bucket_z = floor_z + bucket_height / 2
            obj_to_xyz[bucket] = (bucket_x, bucket_y, bucket_z)

        else:
            raise NotImplementedError("Dry task generation not implemented "
                                      f"for map {CFG.spot_graph_nav_map}")

        for obj, (x, y, z) in obj_to_xyz.items():
            pose = math_helpers.SE3Pose(x, y, z, math_helpers.Quat())
            objects_in_view[obj] = pose

        # Create the initial observation.
        init_obs = _SpotObservation(
            images={},
            objects_in_view=objects_in_view,
            objects_in_hand_view=set(),
            objects_in_any_view_except_back=set(),
            robot=self._spot_object,
            gripper_open_percentage=0.0,
            robot_pos=robot_pose,
            nonpercept_atoms=self._get_initial_nonpercept_atoms(),
            nonpercept_predicates=(self.predicates - self.percept_predicates),
        )

        # Finish the task.
        goal_description = self._generate_goal_description()
        return EnvironmentTask(init_obs, goal_description)


###############################################################################
#                               Brush Shelf Env                               #
###############################################################################


class SpotBrushShelfEnv(SpotRearrangementEnv):
    """An environment where a brush needs to be moved from the table into one
    of the lower shelves."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_brush_shelf_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        brush = Object("brush", _movable_object_type)
        brush_detection = LanguageObjectDetectionID("yellow brush")
        detection_id_to_obj[brush_detection] = brush

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the brush in the second shelf"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        raise NotImplementedError("Dry task generation not implemented.")


###############################################################################
#                Real-World Ball and Cup Sticky Table Env                     #
###############################################################################

# Specific type for sticky table.
_drafting_table_type = Type(
    "drafting_table",
    list(_immovable_object_type.feature_names) +
    ["sticky-region-x", "sticky-region-y"], _immovable_object_type)


class SpotBallAndCupStickyTableEnv(SpotRearrangementEnv):
    """A real-world version of the ball and cup sticky table environment."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToReachObject", "MoveToHandViewObject",
            "MoveToBodyViewObject", "PickObjectFromTop", "PlaceObjectOnTop",
            "DropObjectInsideContainerOnTop", "PickAndDumpCup"
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @property
    def types(self) -> Set[Type]:
        return set(_ALL_TYPES) | set([_drafting_table_type])

    @classmethod
    def get_name(cls) -> str:
        return "spot_ball_and_cup_sticky_table_env"

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        ball = Object("ball", _movable_object_type)
        ball_prompt = "/".join([
            "small white ball", "ping-pong ball", "snowball", "cotton ball",
            "white button"
        ])
        ball_detection = LanguageObjectDetectionID(ball_prompt)
        detection_id_to_obj[ball_detection] = ball

        cup = Object("cup", _container_type)
        cup_detection = LanguageObjectDetectionID(
            "yellow hoop toy/yellow donut")
        detection_id_to_obj[cup_detection] = cup

        for obj, pose in get_known_immovable_objects().items():
            detection_id = KnownStaticObjectDetectionID(obj.name, pose)
            if obj.name == "drafting_table":
                drafting_table_obj = Object("drafting_table",
                                            _drafting_table_type)
                detection_id_to_obj[detection_id] = drafting_table_obj
            else:
                detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _generate_goal_description(self) -> GoalDescription:
        return "put the ball on the table"

    def _get_dry_task(self, train_or_test: str,
                      task_idx: int) -> EnvironmentTask:
        raise NotImplementedError("Dry task generation not implemented.")