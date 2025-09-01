"""Ground-truth options for the clean table real environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class CleanTableRealGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the clean table real environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"clean-table-real"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        toy_type = types["toy"]
        wiper_type = types["wiper"]
        box_type = types["box"]
        table_type = types["table"]

        # PickToyFromTable: robot + toy -> action_type=0, x, y
        PickToyFromTable = utils.SingletonParameterizedOption(
            "PickToyFromTable",
            types=[robot_type, toy_type],
            params_space=Box(0, 10, (3,)),  # action_type, x, y
            policy=cls._create_pick_toy_from_table_policy())

        # PlaceToyToBox: robot + toy + box -> action_type=1
        PlaceToyToBox = utils.SingletonParameterizedOption(
            "PlaceToyToBox",
            types=[robot_type, toy_type, box_type],
            policy=cls._create_place_toy_to_box_policy())

        # PickWiperFromBox: robot + wiper + box -> action_type=2
        PickWiperFromBox = utils.SingletonParameterizedOption(
            "PickWiperFromBox",
            types=[robot_type, wiper_type, box_type],
            policy=cls._create_pick_wiper_from_box_policy())

        # PickWiperFromTable: robot + wiper -> action_type=3, x, y
        PickWiperFromTable = utils.SingletonParameterizedOption(
            "PickWiperFromTable",
            types=[robot_type, wiper_type],
            params_space=Box(0, 10, (3,)),  # action_type, x, y
            policy=cls._create_pick_wiper_from_table_policy())

        # PlaceWiperAtTable: robot + wiper -> action_type=4, x, y
        PlaceWiperAtTable = utils.SingletonParameterizedOption(
            "PlaceWiperAtTable",
            types=[robot_type, wiper_type],
            params_space=Box(-10, 10, (3,)),  # action_type, x, y
            policy=cls._create_place_wiper_at_table_policy())

        # PlaceWiperToBox: robot + wiper + box -> action_type=5
        PlaceWiperToBox = utils.SingletonParameterizedOption(
            "PlaceWiperToBox",
            types=[robot_type, wiper_type, box_type],
            policy=cls._create_place_wiper_to_box_policy())

        # PushBoxOut: robot + box -> action_type=6
        PushBoxOut = utils.SingletonParameterizedOption(
            "PushBoxOut",
            types=[robot_type, box_type],
            policy=cls._create_push_box_out_policy())

        # PullBoxIn: robot + box -> action_type=7
        PullBoxIn = utils.SingletonParameterizedOption(
            "PullBoxIn",
            types=[robot_type, box_type],
            policy=cls._create_pull_box_in_policy())

        # WipeTable: robot + wiper + box + table -> action_type=8
        WipeTable = utils.SingletonParameterizedOption(
            "WipeTable",
            types=[robot_type, wiper_type, box_type, table_type],
            policy=cls._create_wipe_table_policy())

        # AchieveGoal: robot + table -> action_type=9
        AchieveGoal = utils.SingletonParameterizedOption(
            "AchieveGoal",
            types=[robot_type, table_type],
            policy=cls._create_achieve_goal_policy())

        return {PickToyFromTable, PlaceToyToBox, PickWiperFromBox, PickWiperFromTable,
                PlaceWiperAtTable, PlaceWiperToBox, PushBoxOut, PullBoxIn, WipeTable, AchieveGoal}

    @classmethod
    def _create_pick_toy_from_table_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, toy = objects
            action_type, x, y = params
            return Action(np.array([0, x, y], dtype=np.float32))
        return policy

    @classmethod
    def _create_place_toy_to_box_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, toy, box = objects
            return Action(np.array([1, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_pick_wiper_from_box_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, wiper, box = objects
            return Action(np.array([2, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_pick_wiper_from_table_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, wiper = objects
            action_type, x, y = params
            return Action(np.array([3, x, y], dtype=np.float32))
        return policy

    @classmethod
    def _create_place_wiper_at_table_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, wiper = objects
            action_type, x, y = params
            return Action(np.array([4, x, y], dtype=np.float32))
        return policy

    @classmethod
    def _create_place_wiper_to_box_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, wiper, box = objects
            return Action(np.array([5, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_push_box_out_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, box = objects
            return Action(np.array([6, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_pull_box_in_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, box = objects
            return Action(np.array([7, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_wipe_table_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, wiper, box, table = objects
            return Action(np.array([8, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy

    @classmethod
    def _create_achieve_goal_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, table = objects
            return Action(np.array([9, 0, 0], dtype=np.float32))  # x, y don't matter for this action
        return policy