"""Ground-truth options for the tools environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.tools_pcd import ToolsPCDEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class ToolsPCDGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the tools environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"tools-pcd"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        screw_type = types["screw"]
        screwdriver_type = types["screwdriver"]
        nail_type = types["nail"]
        hammer_type = types["hammer"]
        bolt_type = types["bolt"]
        wrench_type = types["wrench"]
        contraption_type = types["contraption"]



        AdjustScrewdriverShape = utils.SingletonParameterizedOption(
            # variables: [robot, screwdriver]
            # params: []
            "AdjustScrewdriverShape",
            cls._create_adjust_screwdriver_shape_policy(),
            types=[robot_type, screwdriver_type,screw_type])
        PickScrew = utils.SingletonParameterizedOption(
            # variables: [robot, screw to pick]
            # params: []
            "PickScrew",
            cls._create_pick_policy(),
            types=[robot_type, screw_type])

        PickScrewdriver = utils.SingletonParameterizedOption(
            # variables: [robot, screwdriver to pick]
            # params: []
            "PickScrewdriver",
            cls._create_pick_policy(),
            types=[robot_type, screwdriver_type])

        PickNail = utils.SingletonParameterizedOption(
            # variables: [robot, nail to pick]
            # params: []
            "PickNail",
            cls._create_pick_policy(),
            types=[robot_type, nail_type])

        PickHammer = utils.SingletonParameterizedOption(
            # variables: [robot, hammer to pick]
            # params: []
            "PickHammer",
            cls._create_pick_policy(),
            types=[robot_type, hammer_type])

        PickBolt = utils.SingletonParameterizedOption(
            # variables: [robot, bolt to pick]
            # params: []
            "PickBolt",
            cls._create_pick_policy(),
            types=[robot_type, bolt_type])

        PickWrench = utils.SingletonParameterizedOption(
            # variables: [robot, wrench to pick]
            # params: []
            "PickWrench",
            cls._create_pick_policy(),
            types=[robot_type, wrench_type])
        


        # PlaceScrewdriverBack
        PlaceScrewdriverBack =  utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceScrewdriverBack",
            policy=cls._create_place_policy(),
            types=[robot_type, screwdriver_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )
        # PlaceHammerBack
        PlaceHammerBack = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceHammerBack",
            policy=cls._create_place_policy(),
            types=[robot_type, hammer_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )


        # PlaceWrenchBack
        PlaceWrenchBack = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceWrenchBack",
            policy=cls._create_place_policy(),
            types=[robot_type, wrench_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )

        # PlaceScrewOnContraption
        PlaceScrewOnContraption = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceScrewOnContraption",
            policy=cls._create_place_policy(),
            types=[robot_type, screw_type, contraption_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )
        # PlaceNailOnContraption
        PlaceNailOnContraption = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceNailOnContraption",
            policy=cls._create_place_policy(),
            types=[robot_type, nail_type, contraption_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )
        # PlaceBoltOnContraption
        PlaceBoltOnContraption = utils.SingletonParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y]
            "PlaceBoltOnContraption",
            policy=cls._create_place_policy(),
            types=[robot_type, bolt_type, contraption_type],
            params_space=Box(
                np.array([ToolsPCDEnv.table_lx, ToolsPCDEnv.table_ly],
                         dtype=np.float32),
                np.array([ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy],
                         dtype=np.float32)),
        )

        FastenScrewWithScrewdriver = utils.SingletonParameterizedOption(
            # variables: [robot, screw, screwdriver, contraption]
            # params: []
            "FastenScrewWithScrewdriver",
            policy=cls._create_fasten_policy(),
            types=[robot_type, screw_type, screwdriver_type, contraption_type])

        # FastenScrewByHand = utils.SingletonParameterizedOption(
        #     # variables: [robot, screw, contraption]
        #     # params: []
        #     "FastenScrewByHand",
        #     policy=cls._create_fasten_policy(),
        #     types=[robot_type, screw_type, contraption_type])

        FastenNailWithHammer = utils.SingletonParameterizedOption(
            # variables: [robot, nail, hammer, contraption]
            # params: []
            "FastenNailWithHammer",
            policy=cls._create_fasten_policy(),
            types=[robot_type, nail_type, hammer_type, contraption_type])

        FastenBoltWithWrench = utils.SingletonParameterizedOption(
            # variables: [robot, bolt, wrench, contraption]
            # params: []
            "FastenBoltWithWrench",
            policy=cls._create_fasten_policy(),
            types=[robot_type, bolt_type, wrench_type, contraption_type])

        return {
            AdjustScrewdriverShape,PickScrew, PickScrewdriver, PickNail, PickHammer, PickBolt,
            PickWrench, PlaceScrewdriverBack,PlaceHammerBack,PlaceWrenchBack,PlaceScrewOnContraption,
            PlaceNailOnContraption,PlaceBoltOnContraption,FastenScrewWithScrewdriver,
            FastenNailWithHammer, FastenBoltWithWrench
        }

    @classmethod
    def _create_pick_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            assert not params
            _, item_or_tool = objects
            pose_x = state.get(item_or_tool, "pose_x")
            pose_y = state.get(item_or_tool, "pose_y")
            arr = np.array([pose_x, pose_y, 1.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        return policy

    @classmethod
    def _create_place_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(np.r_[params, 0.0, 1.0, 0.0])

        return policy

    @classmethod
    def _create_fasten_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            assert not params
            if len(objects) == 3:
                # Note that the FastenScrewByHand option has only 3 parameters,
                # while all other Fasten options have 4 parameters.
                _, item, contraption = objects
                # For fastening by hand, we don't want to be holding any tool.
                tool_is_correct = (ToolsPCDEnv.get_held_item_or_tool(state) is
                                   None)
            else:
                _, item, tool, contraption = objects
                # For fastening with a tool, we should be holding it.
                tool_is_correct = (state.get(tool, "is_held") > 0.5)
            assert ToolsPCDEnv.is_item(item)
            pose_x = state.get(item, "pose_x")
            pose_y = state.get(item, "pose_y")
            contraption_is_correct = ToolsPCDEnv.is_pose_on_contraption(
                state, pose_x, pose_y, contraption)
            if not tool_is_correct or not contraption_is_correct:
                # Simulate a noop by fastening at poses where there is
                # guaranteed to be no contraption. We don't use an initiable()
                # function here because we want replay data to be able to try
                # this, in order to discover good operator preconditions.
                pose_x, pose_y = ToolsPCDEnv.table_ux, ToolsPCDEnv.table_uy
            arr = np.array([pose_x, pose_y, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        return policy
    @classmethod
    def _create_adjust_screwdriver_shape_policy(cls) -> ParameterizedPolicy:
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects
            arr = np.array([0, 0, 0.0, 0.0, 1.0], dtype=np.float32)
            return Action(arr)
        
        return policy
