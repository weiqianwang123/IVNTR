"""Dummy NSRTs for the clean table real environment.
It only has pre-condition predicates and no effects.
It is used by our baselines if we assume GT sampler is provided.
"""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.clean_table_real import TableCleanEnv
from predicators.ground_truth_models import DummyNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class CleanTableRealDummyNSRTFactory(DummyNSRTFactory):
    """Dummy NSRTs for the clean table real environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"clean-table-real"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        toy_type = types["toy"]
        wiper_type = types["wiper"]
        box_type = types["box"]
        table_type = types["table"]

        # Predicates
        ToyOnTable = predicates["toy_on_table"]
        HandEmpty = predicates["handempty"]
        HoldingToy = predicates["holdingToy"]
        ToyInBox = predicates["toy_in_box"]
        WiperInBox = predicates["wiper_in_box"]
        WiperOnTable = predicates["wiper_on_table"]
        HoldingWiper = predicates["holdingWiper"]
        BoxAtCenter = predicates["box_at_center"]
        BoxAtSide = predicates["box_at_side"]
        NoToyAtTable = predicates["No_toy_at_table"]
        TableClean = predicates["table_clean"]
        GoalAchieved = predicates["goalAchieved"]

        # Options
        PickToyFromTable = options["PickToyFromTable"]
        PlaceToyToBox = options["PlaceToyToBox"]
        PickWiperFromBox = options["PickWiperFromBox"]
        PickWiperFromTable = options["PickWiperFromTable"]
        PlaceWiperAtTable = options["PlaceWiperAtTable"]
        # PlaceWiperToBox = options["PlaceWiperToBox"]
        PushBoxOut = options["PushBoxOut"]
        PullBoxIn = options["PullBoxIn"]
        WipeTable = options["WipeTable"]
        AchieveGoal = options["AchieveGoal"]

        nsrts = set()

        # PickToyFromTable - dummy version with only preconditions
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        table = Variable("?table", table_type)
        parameters = [robot, toy, table]
        option_vars = [robot, toy, table]
        option = PickToyFromTable
        preconditions = {
        }
        add_effects = set()  # No effects for dummy
        delete_effects = set()
        ignore_effects = set()

        def pick_toy_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            robot, toy = objs
            toy_x = state.get(toy, "pose_x")
            toy_y = state.get(toy, "pose_y")
            # Add small noise around toy position
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            return np.array([0, toy_x + noise_x, toy_y + noise_y], dtype=np.float32)

        pick_toy_nsrt = NSRT("PickToyFromTable", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           pick_toy_sampler)
        nsrts.add(pick_toy_nsrt)

        # PlaceToyToBox - dummy version
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        box = Variable("?box", box_type)
        parameters = [robot, toy, box]
        option_vars = [robot, toy, box]
        option = PlaceToyToBox
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        place_toy_nsrt = NSRT("PlaceToyToBox", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option, option_vars,
                            null_sampler)
        nsrts.add(place_toy_nsrt)

        # PickWiperFromBox - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        parameters = [robot, wiper, box]
        option_vars = [robot, wiper, box]
        option = PickWiperFromBox
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        pick_wiper_box_nsrt = NSRT("PickWiperFromBox", parameters, preconditions, add_effects,
                                 delete_effects, ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(pick_wiper_box_nsrt)

        # PickWiperFromTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = PickWiperFromTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        def pick_wiper_table_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del goal  # unused
            robot, wiper = objs
            wiper_x = state.get(wiper, "pose_x")
            wiper_y = state.get(wiper, "pose_y")
            # Add small noise around wiper position
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            return np.array([3, wiper_x + noise_x, wiper_y + noise_y], dtype=np.float32)

        pick_wiper_table_nsrt = NSRT("PickWiperFromTable", parameters, preconditions, add_effects,
                                   delete_effects, ignore_effects, option, option_vars,
                                   pick_wiper_table_sampler)
        nsrts.add(pick_wiper_table_nsrt)

        # PlaceWiperAtTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = PlaceWiperAtTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        def place_wiper_table_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
            del goal  # unused
            # Place wiper at a random location on table
            x = rng.uniform(TableCleanEnv.table_lx, TableCleanEnv.table_ux)
            y = rng.uniform(TableCleanEnv.table_ly, TableCleanEnv.table_uy)
            return np.array([4, x, y], dtype=np.float32)

        place_wiper_table_nsrt = NSRT("PlaceWiperAtTable", parameters, preconditions, add_effects,
                                    delete_effects, ignore_effects, option, option_vars,
                                    place_wiper_table_sampler)
        nsrts.add(place_wiper_table_nsrt)

        # # PlaceWiperToBox - dummy version
        # robot = Variable("?robot", robot_type)
        # wiper = Variable("?wiper", wiper_type)
        # box = Variable("?box", box_type)
        # parameters = [robot, wiper, box]
        # option_vars = [robot, wiper, box]
        # option = PlaceWiperToBox
        # preconditions = {
        # }
        # add_effects = set()
        # delete_effects = set()
        # ignore_effects = set()

        # place_wiper_box_nsrt = NSRT("PlaceWiperToBox", parameters, preconditions, add_effects,
        #                           delete_effects, ignore_effects, option, option_vars,
        #                           null_sampler)
        # nsrts.add(place_wiper_box_nsrt)

        # PushBoxOut - dummy version
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box, table]
        option = PushBoxOut
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        push_box_nsrt = NSRT("PushBoxOut", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(push_box_nsrt)

        # PullBoxIn - dummy version
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box, table]
        option = PullBoxIn
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        pull_box_nsrt = NSRT("PullBoxIn", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(pull_box_nsrt)

        # WipeTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = WipeTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        wipe_table_nsrt = NSRT("WipeTable", parameters, preconditions, add_effects,
                             delete_effects, ignore_effects, option, option_vars,
                             null_sampler)
        nsrts.add(wipe_table_nsrt)

        # AchieveGoal - dummy version
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        parameters = [robot, table]
        option_vars = [robot, table]
        option = AchieveGoal
        preconditions = {
        }
        add_effects = set(LiftedAtom(GoalAchieved, [table]),)
        delete_effects = set()
        ignore_effects = set()

        achieve_goal_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                               delete_effects, ignore_effects, option, option_vars,
                               null_sampler)
        nsrts.add(achieve_goal_nsrt)

        return nsrts


class CleanTableRealRealDummyNSRTFactory(DummyNSRTFactory):
    """Dummy NSRTs for the clean table real environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"clean-table-real-real"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        toy_type = types["toy"]
        wiper_type = types["wiper"]
        box_type = types["box"]
        table_type = types["table"]

        # Predicates
        ToyOnTable = predicates["toy_on_table"]
        HandEmpty = predicates["handempty"]
        HoldingToy = predicates["holdingToy"]
        ToyInBox = predicates["toy_in_box"]
        WiperInBox = predicates["wiper_in_box"]
        WiperOnTable = predicates["wiper_on_table"]
        HoldingWiper = predicates["holdingWiper"]
        BoxAtCenter = predicates["box_at_center"]
        BoxAtSide = predicates["box_at_side"]
        NoToyAtTable = predicates["No_toy_at_table"]
        TableClean = predicates["table_clean"]
        GoalAchieved = predicates["goalAchieved"]

        # Options
        PickToyFromTable = options["PickToyFromTable"]
        PlaceToyToBox = options["PlaceToyToBox"]
        PickWiperFromBox = options["PickWiperFromBox"]
        PickWiperFromTable = options["PickWiperFromTable"]
        PlaceWiperAtTable = options["PlaceWiperAtTable"]
        PlaceWiperToBox = options["PlaceWiperToBox"]
        PushBoxOut = options["PushBoxOut"]
        PullBoxIn = options["PullBoxIn"]
        WipeTable = options["WipeTable"]
        AchieveGoal = options["AchieveGoal"]

        nsrts = set()

        # PickToyFromTable - dummy version with only preconditions
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        parameters = [robot, toy]
        option_vars = [robot, toy]
        option = PickToyFromTable
        preconditions = {
        }
        add_effects = set()  # No effects for dummy
        delete_effects = set()
        ignore_effects = set()

        def pick_toy_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            robot, toy = objs
            toy_x = state.get(toy, "pose_x")
            toy_y = state.get(toy, "pose_y")
            # Add small noise around toy position
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            return np.array([0, toy_x + noise_x, toy_y + noise_y], dtype=np.float32)

        pick_toy_nsrt = NSRT("PickToyFromTable", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           pick_toy_sampler)
        nsrts.add(pick_toy_nsrt)

        # PlaceToyToBox - dummy version
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        box = Variable("?box", box_type)
        parameters = [robot, toy, box]
        option_vars = [robot, toy, box]
        option = PlaceToyToBox
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        place_toy_nsrt = NSRT("PlaceToyToBox", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option, option_vars,
                            null_sampler)
        nsrts.add(place_toy_nsrt)

        # PickWiperFromBox - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        parameters = [robot, wiper, box]
        option_vars = [robot, wiper, box]
        option = PickWiperFromBox
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        pick_wiper_box_nsrt = NSRT("PickWiperFromBox", parameters, preconditions, add_effects,
                                 delete_effects, ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(pick_wiper_box_nsrt)

        # PickWiperFromTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        parameters = [robot, wiper]
        option_vars = [robot, wiper]
        option = PickWiperFromTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        def pick_wiper_table_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del goal  # unused
            robot, wiper = objs
            wiper_x = state.get(wiper, "pose_x")
            wiper_y = state.get(wiper, "pose_y")
            # Add small noise around wiper position
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            return np.array([3, wiper_x + noise_x, wiper_y + noise_y], dtype=np.float32)

        pick_wiper_table_nsrt = NSRT("PickWiperFromTable", parameters, preconditions, add_effects,
                                   delete_effects, ignore_effects, option, option_vars,
                                   pick_wiper_table_sampler)
        nsrts.add(pick_wiper_table_nsrt)

        # PlaceWiperAtTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        parameters = [robot, wiper]
        option_vars = [robot, wiper]
        option = PlaceWiperAtTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        def place_wiper_table_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
            del goal  # unused
            # Place wiper at a random location on table
            x = rng.uniform(TableCleanEnv.table_lx, TableCleanEnv.table_ux)
            y = rng.uniform(TableCleanEnv.table_ly, TableCleanEnv.table_uy)
            return np.array([4, x, y], dtype=np.float32)

        place_wiper_table_nsrt = NSRT("PlaceWiperAtTable", parameters, preconditions, add_effects,
                                    delete_effects, ignore_effects, option, option_vars,
                                    place_wiper_table_sampler)
        nsrts.add(place_wiper_table_nsrt)

        # PlaceWiperToBox - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        parameters = [robot, wiper, box]
        option_vars = [robot, wiper, box]
        option = PlaceWiperToBox
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        place_wiper_box_nsrt = NSRT("PlaceWiperToBox", parameters, preconditions, add_effects,
                                  delete_effects, ignore_effects, option, option_vars,
                                  null_sampler)
        nsrts.add(place_wiper_box_nsrt)

        # PushBoxOut - dummy version
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        parameters = [robot, box]
        option_vars = [robot, box]
        option = PushBoxOut
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        push_box_nsrt = NSRT("PushBoxOut", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(push_box_nsrt)

        # PullBoxIn - dummy version
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        parameters = [robot, box]
        option_vars = [robot, box]
        option = PullBoxIn
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        pull_box_nsrt = NSRT("PullBoxIn", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(pull_box_nsrt)

        # WipeTable - dummy version
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = WipeTable
        preconditions = {
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()

        wipe_table_nsrt = NSRT("WipeTable", parameters, preconditions, add_effects,
                             delete_effects, ignore_effects, option, option_vars,
                             null_sampler)
        nsrts.add(wipe_table_nsrt)

        # AchieveGoal - dummy version
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)
        parameters = [robot, table]
        option_vars = [robot, table]
        option = AchieveGoal
        preconditions = {
        }
        add_effects = set(LiftedAtom(GoalAchieved, [table]),)
        delete_effects = set()
        ignore_effects = set()

        achieve_goal_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                               delete_effects, ignore_effects, option, option_vars,
                               null_sampler)
        nsrts.add(achieve_goal_nsrt)

        return nsrts