"""Ground-truth NSRTs for the clean table real environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.clean_table_real import TableCleanEnv,TableCleanRealEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler

def _default_xy_sampler(state, goal, rng, objs):
    """Returns default action parameters [action_type, x, y]"""
    # Note: action_type will be overridden by the specific option
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


class CleanTableRealGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the clean table real environment."""

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

        # PickToyFromTable
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        table = Variable("?table", table_type)
        parameters = [robot, toy, table]
        option_vars = [robot, toy]
        option = PickToyFromTable
        preconditions = {
            LiftedAtom(ToyOnTable, [toy,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        delete_effects = {
            LiftedAtom(ToyOnTable, [toy,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        ignore_effects = set()

        pick_toy_nsrt = NSRT("PickToyFromTable", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _default_xy_sampler)
        nsrts.add(pick_toy_nsrt)

        # PlaceToyToBox
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        box = Variable("?box", box_type)
        parameters = [robot, toy, box]
        option_vars = [robot, toy, box]
        option = PlaceToyToBox
        preconditions = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        add_effects = {
            LiftedAtom(ToyInBox, [toy,box]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        ignore_effects = set()

        place_toy_nsrt = NSRT("PlaceToyToBox", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option, option_vars,
                            null_sampler)
        nsrts.add(place_toy_nsrt)

        # PickWiperFromBox
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, box, table]
        option_vars = [robot, wiper, box]
        option = PickWiperFromBox
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperInBox, [wiper,box]),
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        add_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperInBox, [wiper,box]),
        }
        ignore_effects = set()

        pick_wiper_box_nsrt = NSRT("PickWiperFromBox", parameters, preconditions, add_effects,
                                 delete_effects, ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(pick_wiper_box_nsrt)

        # PickWiperFromTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper]
        option = PickWiperFromTable
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperOnTable, [wiper,table]),
        }
        add_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperOnTable, [wiper,table]),
        }
        ignore_effects = set()

        pick_wiper_table_nsrt = NSRT("PickWiperFromTable", parameters, preconditions, add_effects,
                                   delete_effects, ignore_effects, option, option_vars,
                                   _default_xy_sampler)
        nsrts.add(pick_wiper_table_nsrt)

        # PlaceWiperAtTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper]
        option = PlaceWiperAtTable
        preconditions = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        add_effects = {
            LiftedAtom(WiperOnTable, [wiper,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        ignore_effects = set()

        place_wiper_table_nsrt = NSRT("PlaceWiperAtTable", parameters, preconditions, add_effects,
                                    delete_effects, ignore_effects, option, option_vars,
                                    _default_xy_sampler)
        nsrts.add(place_wiper_table_nsrt)

        # PlaceWiperToBox
        # robot = Variable("?robot", robot_type)
        # wiper = Variable("?wiper", wiper_type)
        # box = Variable("?box", box_type)
        # parameters = [robot, wiper, box]
        # option_vars = [robot, wiper, box]
        # option = PlaceWiperToBox
        # preconditions = {
        #     LiftedAtom(HoldingWiper, [robot,wiper]),
        # }
        # add_effects = {
        #     LiftedAtom(WiperInBox, [wiper,box]),
        #     LiftedAtom(HandEmpty, [robot]),
        # }
        # delete_effects = {
        #     LiftedAtom(HoldingWiper, [robot,wiper]),
        # }
        # ignore_effects = set()

        # place_wiper_box_nsrt = NSRT("PlaceWiperToBox", parameters, preconditions, add_effects,
        #                           delete_effects, ignore_effects, option, option_vars,
        #                           null_sampler)
        # nsrts.add(place_wiper_box_nsrt)

        # PushBoxOut
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box]
        option = PushBoxOut
        preconditions = {
            LiftedAtom(BoxAtCenter, [box,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(BoxAtSide, [box,table]),
        }
        delete_effects = {
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        ignore_effects = set()

        push_box_nsrt = NSRT("PushBoxOut", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(push_box_nsrt)

        # PullBoxIn
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box]
        option = PullBoxIn
        preconditions = {
            LiftedAtom(BoxAtSide, [box,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        delete_effects = {
            LiftedAtom(BoxAtSide, [box,table]),
        }
        ignore_effects = set()

        pull_box_nsrt = NSRT("PullBoxIn", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(pull_box_nsrt)

        # WipeTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        toy = Variable("?toy", toy_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = WipeTable
        preconditions = {
            LiftedAtom(BoxAtSide, [box,table]),
            LiftedAtom(NoToyAtTable, []),  # Restore - now managed manually
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        add_effects = {
            LiftedAtom(TableClean, [table]),
        }
        delete_effects = set()
        ignore_effects = set()

        wipe_table_nsrt = NSRT("WipeTable", parameters, preconditions, add_effects,
                             delete_effects, ignore_effects, option, option_vars,
                             null_sampler)
        nsrts.add(wipe_table_nsrt)

        # AchieveGoal
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [robot, table]
        option_vars = [robot, table]
        option = AchieveGoal
        preconditions = {
            LiftedAtom(TableClean, [table])
            } 
        add_effects = {
            LiftedAtom(GoalAchieved, [table]),
        }
        delete_effects = set()
        ignore_effects = set()

        achieve_goal_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                               delete_effects, ignore_effects, option, option_vars,
                               null_sampler)
        nsrts.add(achieve_goal_nsrt)

        return nsrts



class CleanTableRealRealGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the clean table real environment."""

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
        # PlaceWiperToBox = options["PlaceWiperToBox"]
        PushBoxOut = options["PushBoxOut"]
        PullBoxIn = options["PullBoxIn"]
        WipeTable = options["WipeTable"]
        AchieveGoal = options["AchieveGoal"]

        nsrts = set()

        # PickToyFromTable
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        table = Variable("?table", table_type)
        parameters = [robot, toy, table]
        option_vars = [robot, toy]
        option = PickToyFromTable
        preconditions = {
            LiftedAtom(ToyOnTable, [toy,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        delete_effects = {
            LiftedAtom(ToyOnTable, [toy,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        ignore_effects = set()

        pick_toy_nsrt = NSRT("PickToyFromTable", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           _default_xy_sampler)
        nsrts.add(pick_toy_nsrt)

        # PlaceToyToBox
        robot = Variable("?robot", robot_type)
        toy = Variable("?toy", toy_type)
        box = Variable("?box", box_type)
        parameters = [robot, toy, box]
        option_vars = [robot, toy, box]
        option = PlaceToyToBox
        preconditions = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        add_effects = {
            LiftedAtom(ToyInBox, [toy,box]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(HoldingToy, [robot,toy]),
        }
        ignore_effects = set()

        place_toy_nsrt = NSRT("PlaceToyToBox", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option, option_vars,
                            null_sampler)
        nsrts.add(place_toy_nsrt)

        # PickWiperFromBox
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, box, table]
        option_vars = [robot, wiper, box]
        option = PickWiperFromBox
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperInBox, [wiper,box]),
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        add_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperInBox, [wiper,box]),
        }
        ignore_effects = set()

        pick_wiper_box_nsrt = NSRT("PickWiperFromBox", parameters, preconditions, add_effects,
                                 delete_effects, ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(pick_wiper_box_nsrt)

        # PickWiperFromTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper]
        option = PickWiperFromTable
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperOnTable, [wiper,table]),
        }
        add_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(WiperOnTable, [wiper,table]),
        }
        ignore_effects = set()

        pick_wiper_table_nsrt = NSRT("PickWiperFromTable", parameters, preconditions, add_effects,
                                   delete_effects, ignore_effects, option, option_vars,
                                   _default_xy_sampler)
        nsrts.add(pick_wiper_table_nsrt)

        # PlaceWiperAtTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        table = Variable("?table", table_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper]
        option = PlaceWiperAtTable
        preconditions = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        add_effects = {
            LiftedAtom(WiperOnTable, [wiper,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        ignore_effects = set()

        place_wiper_table_nsrt = NSRT("PlaceWiperAtTable", parameters, preconditions, add_effects,
                                    delete_effects, ignore_effects, option, option_vars,
                                    _default_xy_sampler)
        nsrts.add(place_wiper_table_nsrt)

        # # PlaceWiperToBox
        # robot = Variable("?robot", robot_type)
        # wiper = Variable("?wiper", wiper_type)
        # box = Variable("?box", box_type)
        # parameters = [robot, wiper, box]
        # option_vars = [robot, wiper, box]
        # option = PlaceWiperToBox
        # preconditions = {
        #     LiftedAtom(HoldingWiper, [robot,wiper]),
        # }
        # add_effects = {
        #     LiftedAtom(WiperInBox, [wiper,box]),
        #     LiftedAtom(HandEmpty, [robot]),
        # }
        # delete_effects = {
        #     LiftedAtom(HoldingWiper, [robot,wiper]),
        # }
        # ignore_effects = set()

        # place_wiper_box_nsrt = NSRT("PlaceWiperToBox", parameters, preconditions, add_effects,
        #                           delete_effects, ignore_effects, option, option_vars,
        #                           null_sampler)
        # nsrts.add(place_wiper_box_nsrt)

        # PushBoxOut
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box]
        option = PushBoxOut
        preconditions = {
            LiftedAtom(BoxAtCenter, [box,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(BoxAtSide, [box,table]),
        }
        delete_effects = {
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        ignore_effects = set()

        push_box_nsrt = NSRT("PushBoxOut", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(push_box_nsrt)

        # PullBoxIn
        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        parameters = [robot, box, table]
        option_vars = [robot, box]
        option = PullBoxIn
        preconditions = {
            LiftedAtom(BoxAtSide, [box,table]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(BoxAtCenter, [box,table]),
        }
        delete_effects = {
            LiftedAtom(BoxAtSide, [box,table]),
        }
        ignore_effects = set()

        pull_box_nsrt = NSRT("PullBoxIn", parameters, preconditions, add_effects,
                           delete_effects, ignore_effects, option, option_vars,
                           null_sampler)
        nsrts.add(pull_box_nsrt)

        # WipeTable
        robot = Variable("?robot", robot_type)
        wiper = Variable("?wiper", wiper_type)
        box = Variable("?box", box_type)
        table = Variable("?table", table_type)
        toy = Variable("?toy", toy_type)
        parameters = [robot, wiper, table]
        option_vars = [robot, wiper, table]
        option = WipeTable
        preconditions = {
            LiftedAtom(BoxAtSide, [box,table]),
            LiftedAtom(NoToyAtTable, []),  # Restore - now managed manually
            LiftedAtom(HoldingWiper, [robot,wiper]),
        }
        add_effects = {
            LiftedAtom(TableClean, [table]),
        }
        delete_effects = set()
        ignore_effects = set()

        wipe_table_nsrt = NSRT("WipeTable", parameters, preconditions, add_effects,
                             delete_effects, ignore_effects, option, option_vars,
                             null_sampler)
        nsrts.add(wipe_table_nsrt)

        # AchieveGoal
        robot = Variable("?robot", robot_type)
        table = Variable("?table", table_type)

        parameters = [robot, table]
        option_vars = [robot, table]
        option = AchieveGoal
        preconditions = {
            LiftedAtom(TableClean, [table])
            } 
        add_effects = {
            LiftedAtom(GoalAchieved, [table]),
        }
        delete_effects = set()
        ignore_effects = set()

        achieve_goal_nsrt = NSRT("AchieveGoal", parameters, preconditions, add_effects,
                               delete_effects, ignore_effects, option, option_vars,
                               null_sampler)
        nsrts.add(achieve_goal_nsrt)

        return nsrts