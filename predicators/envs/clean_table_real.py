"""Table cleaning environment in real-world setting (object-centric image observations).
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
from predicators import utils


# Note: obj_to_pointcloud function not needed for mock image observations
# but imports kept for potential future point cloud integration



class TableCleanEnv(BaseEnv):
    """Table cleaning environment with object-centric image observations."""
    # Environment parameters
    table_lx: ClassVar[float] = -5.0
    table_ly: ClassVar[float] = -5.0  
    table_ux: ClassVar[float] = 5.0
    table_uy: ClassVar[float] = 5.0
    box_size: ClassVar[float] = 1.0
    close_thresh: ClassVar[float] = 10
    center_x: ClassVar[float] = 0.0  # table center x
    center_y: ClassVar[float] = 0.0  # table center y
    side_x: ClassVar[float] = 3.0    # box side position x
    side_y: ClassVar[float] = 3.0    # box side position y
    
    # Object-centric image dimensions (assuming RGB images)
    img_height: ClassVar[int] = 64
    img_width: ClassVar[int] = 64
    img_channels: ClassVar[int] = 3
    img_size: ClassVar[int] = img_height * img_width * img_channels


    # Types based on PDDL specification
    _robot_type = Type("robot", ["handempty", "goal_achieved"])
    _toy_type = Type("toy", ["pose_x", "pose_y", "on_table", "in_box"] + 
                    [f"img_{i}" for i in range(img_size)])
    _wiper_type = Type("wiper", ["pose_x", "pose_y", "on_table", "in_box"] + 
                     [f"img_{i}" for i in range(img_size)])
    _box_type = Type("box", ["pose_x", "pose_y", "at_center", "at_side"] + 
                    [f"img_{i}" for i in range(img_size)])
    _table_type = Type("table", ["pose_x", "pose_y", "is_clean"] + 
                      [f"img_{i}" for i in range(img_size)])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates based on PDDL specification
        self._ToyOnTable = Predicate("toy_on_table", [self._toy_type,self._table_type], self._ToyOnTable_holds)
        self._HandEmpty = Predicate("handempty", [self._robot_type], self._HandEmpty_holds)
        self._HoldingToy = Predicate("holdingToy", [self._robot_type,self._toy_type], self._HoldingToy_holds)
        self._ToyInBox = Predicate("toy_in_box", [self._toy_type,self._box_type], self._ToyInBox_holds)
        
        self._WiperInBox = Predicate("wiper_in_box", [self._wiper_type,self._box_type], self._WiperInBox_holds)
        self._WiperOnTable = Predicate("wiper_on_table", [self._wiper_type,self._table_type], self._WiperOnTable_holds)
        self._HoldingWiper = Predicate("holdingWiper", [self._robot_type,self._wiper_type], self._HoldingWiper_holds)
        
        self._BoxAtCenter = Predicate("box_at_center", [self._box_type,self._table_type], self._BoxAtCenter_holds)
        self._BoxAtSide = Predicate("box_at_side", [self._box_type,self._table_type], self._BoxAtSide_holds)
        
        self._NoToyAtTable = Predicate("No_toy_at_table", [], self._NoToyAtTable_holds)
        self._TableClean = Predicate("table_clean", [self._table_type], self._TableClean_holds)
        
        # Goal achievement predicate
        self._GoalAchieved = Predicate("goalAchieved", [self._robot_type], self._GoalAchieved_holds)
        
        # Static objects
        self._robot = Object("robot", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "clean-table-real"

    def simulate(self, state: State, action: Action) -> State:
        """Simulate actions based on PDDL specification.
        
        Actions: pick_toy_from_table, place_toy_to_box, pick_wiper_from_box,
                pick_wiper_from_table, place_wiper_at_table, place_wiper_to_box,
                push_box_out, pull_box_in, wipe_table
        """
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        
        # Action encoding: [action_type, x, y]
        # action_type: 0=pick_toy_table, 1=place_toy_box, 2=pick_wiper_box,
        #             3=pick_wiper_table, 4=place_wiper_table, 5=place_wiper_box,
        #             6=push_box_out, 7=pull_box_in, 8=wipe_table, 9=achieve_goal
        action_type, x, y = action.arr
        action_type = int(round(action_type))
        
        # Find relevant objects
        toys = [obj for obj in state if obj.type == self._toy_type]
        wipers = [obj for obj in state if obj.type == self._wiper_type]
        boxes = [obj for obj in state if obj.type == self._box_type]
        tables = [obj for obj in state if obj.type == self._table_type]
        
        if action_type == 0:  # pick_toy_from_table
            return self._pick_toy_from_table(state, next_state, toys, x, y)
        elif action_type == 1:  # place_toy_to_box
            return self._place_toy_to_box(state, next_state, toys, boxes)
        elif action_type == 2:  # pick_wiper_from_box
            return self._pick_wiper_from_box(state, next_state, wipers, boxes)
        elif action_type == 3:  # pick_wiper_from_table
            return self._pick_wiper_from_table(state, next_state, wipers, x, y)
        elif action_type == 4:  # place_wiper_at_table
            return self._place_wiper_at_table(state, next_state, wipers, x, y)
        elif action_type == 5:  # place_wiper_to_box
            return self._place_wiper_to_box(state, next_state, wipers, boxes)
        elif action_type == 6:  # push_box_out
            return self._push_box_out(state, next_state, boxes)
        elif action_type == 7:  # pull_box_in
            return self._pull_box_in(state, next_state, boxes)
        elif action_type == 8:  # wipe_table
            return self._wipe_table(state, next_state, wipers, toys, boxes, tables)
        elif action_type == 9:  # achieve_goal
            return self._achieve_goal(state, next_state, toys, wipers, boxes)
        else:
            return next_state  # invalid action

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_train_tasks,
            num_toys_lst=[2],  # 2 toys for training
            num_wipers_lst=[1],  # 1 wiper for training
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_test_tasks,
            num_toys_lst=[3],  # 3 toys for testing
            num_wipers_lst=[2],  # 2 wipers for testing
            rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._ToyOnTable, self._HandEmpty, self._HoldingToy, self._ToyInBox,
            self._WiperInBox, self._WiperOnTable, self._HoldingWiper,
            self._BoxAtCenter, self._BoxAtSide, self._NoToyAtTable, self._TableClean,
            self._GoalAchieved
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._GoalAchieved
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._toy_type, self._wiper_type, self._box_type, self._table_type
        }

    @property
    def derived_predicates(self) -> List[str]:
        return [
            "(:derived (No_toy_at_table)\n    (forall (?t - toy ?tb - table) (not (toy_on_table ?t ?tb))))"
        ]

   

    # Actions: [action_type, x, y]
    # action_type: discrete 0-9 for different PDDL actions
    @property
    def action_space(self) -> Box:
        return Box(
            np.array([0, self.table_lx, self.table_ly], dtype=np.float32),
            np.array([9, self.table_ux, self.table_uy], dtype=np.float32),
        )


    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError
    
    # PDDL Action implementations
    def _pick_toy_from_table(self, state: State, next_state: State, toys: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        """Pick toy from table - precond: toy_on_table and handempty"""
        if not state.get(self._robot, "handempty") > 0.5:
            return state  # hand not empty
        
        # Find any toy on table (no location sampling needed)
        target_toy = None
        for toy in toys:
            if state.get(toy, "on_table") > 0.5:
                target_toy = toy
                break
        
        if target_toy is None:
            return state  # no toy on table
        
        # Execute action effects
        next_state.set(self._robot, "handempty", 0.0)  # not handempty
        next_state.set(target_toy, "on_table", 0.0)    # not toy_on_table
        return next_state
    
    def _place_toy_to_box(self, state: State, next_state: State, toys: List[Object], boxes: List[Object]) -> State:
        """Place toy to box - precond: holdingToy"""
        # Check if holding a toy
        holding_toy = None
        for toy in toys:
            if not (state.get(toy, "on_table") > 0.5 or state.get(toy, "in_box") > 0.5):
                if not state.get(self._robot, "handempty") > 0.5:
                    holding_toy = toy
                    break
        
        if holding_toy is None:
            return state  # not holding toy
        
        # Execute effects
        next_state.set(holding_toy, "in_box", 1.0)
        next_state.set(self._robot, "handempty", 1.0)
        return next_state
    
    def _pick_wiper_from_box(self, state: State, next_state: State, wipers: List[Object], boxes: List[Object]) -> State:
        """Pick wiper from box - precond: handempty, wiper_in_box, box_at_center"""
        if not state.get(self._robot, "handempty") > 0.5:
            return state  # hand not empty
        
        # Check box at center
        box_at_center = any(state.get(box, "at_center") > 0.5 for box in boxes)
        if not box_at_center:
            return state  # box not at center
        
        # Find wiper in box
        target_wiper = None
        for wiper in wipers:
            if state.get(wiper, "in_box") > 0.5:
                target_wiper = wiper
                break
        
        if target_wiper is None:
            return state  # no wiper in box
        
        # Execute effects
        next_state.set(self._robot, "handempty", 0.0)
        next_state.set(target_wiper, "in_box", 0.0)
        return next_state
    
    def _pick_wiper_from_table(self, state: State, next_state: State, wipers: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        """Pick wiper from table - precond: handempty, wiper_on_table"""
        if not state.get(self._robot, "handempty") > 0.5:
            return state  # hand not empty
        
        # Find any wiper on table (no location sampling needed)
        target_wiper = None
        for wiper in wipers:
            if state.get(wiper, "on_table") > 0.5:
                target_wiper = wiper
                break
        
        if target_wiper is None:
            return state  # no wiper on table
        
        # Execute effects
        next_state.set(self._robot, "handempty", 0.0)
        next_state.set(target_wiper, "on_table", 0.0)
        return next_state
    
    def _place_wiper_at_table(self, state: State, next_state: State, wipers: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        """Place wiper at table - precond: holdingWiper"""
        # Check if holding wiper
        holding_wiper = None
        for wiper in wipers:
            if not (state.get(wiper, "on_table") > 0.5 or state.get(wiper, "in_box") > 0.5):
                if not state.get(self._robot, "handempty") > 0.5:
                    holding_wiper = wiper
                    break
        
        if holding_wiper is None:
            return state  # not holding wiper
        
        # Execute effects - place at default table center location
        next_state.set(holding_wiper, "on_table", 1.0)
        next_state.set(holding_wiper, "pose_x", self.center_x)
        next_state.set(holding_wiper, "pose_y", self.center_y)
        next_state.set(self._robot, "handempty", 1.0)
        return next_state
    
    def _place_wiper_to_box(self, state: State, next_state: State, wipers: List[Object], boxes: List[Object]) -> State:
        """Place wiper to box - precond: holdingWiper"""
        # Check if holding wiper
        holding_wiper = None
        for wiper in wipers:
            if not (state.get(wiper, "on_table") > 0.5 or state.get(wiper, "in_box") > 0.5):
                if not state.get(self._robot, "handempty") > 0.5:
                    holding_wiper = wiper
                    break
        
        if holding_wiper is None:
            return state  # not holding wiper
        
        # Execute effects
        next_state.set(holding_wiper, "in_box", 1.0)
        next_state.set(self._robot, "handempty", 1.0)
        return next_state
    
    def _push_box_out(self, state: State, next_state: State, boxes: List[Object]) -> State:
        """Push box out - precond: box_at_center, handempty"""
        if not state.get(self._robot, "handempty") > 0.5:
            return state  # hand not empty
        
        # Find box at center
        target_box = None
        for box in boxes:
            if state.get(box, "at_center") > 0.5:
                target_box = box
                break
        
        if target_box is None:
            return state  # no box at center
        
        # Execute effects
        next_state.set(target_box, "at_side", 1.0)
        next_state.set(target_box, "at_center", 0.0)
        next_state.set(target_box, "pose_x", self.side_x)
        next_state.set(target_box, "pose_y", self.side_y)
        return next_state
    
    def _pull_box_in(self, state: State, next_state: State, boxes: List[Object]) -> State:
        """Pull box in - precond: box_at_side, handempty"""
        if not state.get(self._robot, "handempty") > 0.5:
            return state  # hand not empty
        
        # Find box at side
        target_box = None
        for box in boxes:
            if state.get(box, "at_side") > 0.5:
                target_box = box
                break
        
        if target_box is None:
            return state  # no box at side
        
        # Execute effects
        next_state.set(target_box, "at_center", 1.0)
        next_state.set(target_box, "at_side", 0.0)
        next_state.set(target_box, "pose_x", self.center_x)
        next_state.set(target_box, "pose_y", self.center_y)
        return next_state
    
    def _wipe_table(self, state: State, next_state: State, wipers: List[Object], toys: List[Object], boxes: List[Object], tables: List[Object]) -> State:
        """Wipe table - precond: box_at_side, No_toy_at_table, holdingWiper"""
        # Check box at side
        box_at_side = any(state.get(box, "at_side") > 0.5 for box in boxes)
        if not box_at_side:
            return state  # box not at side
        
        # Check no toy at table
        toy_on_table = any(state.get(toy, "on_table") > 0.5 for toy in toys)
        if toy_on_table:
            return state  # toy still on table
        
        # Check holding wiper
        holding_wiper = None
        for wiper in wipers:
            if not (state.get(wiper, "on_table") > 0.5 or state.get(wiper, "in_box") > 0.5):
                if not state.get(self._robot, "handempty") > 0.5:
                    holding_wiper = wiper
                    break
        
        if holding_wiper is None:
            return state  # not holding wiper
        
        # Execute effects - table becomes clean
        # Only update table object state
        if tables:
            next_state.set(tables[0], "is_clean", 1.0)
        return next_state
    
    def _achieve_goal(self, state: State, next_state: State, toys: List[Object], wipers: List[Object], boxes: List[Object]) -> State:
        """Achieve goal - precond: table_clean"""
        # Check table is clean
        table_clean = False
        for obj in state:
            if obj.type == self._table_type and state.get(obj, "is_clean") > 0.5:
                table_clean = True
                break
        if not table_clean:
            return state  # table not clean
        
        # Execute effect - goal is achieved
        next_state.set(self._robot, "goal_achieved", 1.0)
        return next_state


    # Predicate holding functions
    @staticmethod
    def _ToyOnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        toy,table = objects
        return state.get(toy, "on_table") > 0.5
    
    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "handempty") > 0.5
    
    def _HoldingToy_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot,toy = objects
        # Toy is being held if it's not on table, not in box, and hand is not empty
        on_table = state.get(toy, "on_table") > 0.5
        in_box = state.get(toy, "in_box") > 0.5
        hand_empty = state.get(self._robot, "handempty") > 0.5
        return not on_table and not in_box and not hand_empty
    
    @staticmethod
    def _ToyInBox_holds(state: State, objects: Sequence[Object]) -> bool:
        toy,box = objects
        return state.get(toy, "in_box") > 0.5
    
    @staticmethod
    def _WiperInBox_holds(state: State, objects: Sequence[Object]) -> bool:
        wiper,box = objects
        return state.get(wiper, "in_box") > 0.5
    
    @staticmethod
    def _WiperOnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        wiper, table = objects
        return state.get(wiper, "on_table") > 0.5
    
    def _HoldingWiper_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot,wiper = objects
        # Wiper is being held if it's not on table, not in box, and hand is not empty
        on_table = state.get(wiper, "on_table") > 0.5
        in_box = state.get(wiper, "in_box") > 0.5
        hand_empty = state.get(self._robot, "handempty") > 0.5
        return not on_table and not in_box and not hand_empty
    
    @staticmethod
    def _BoxAtCenter_holds(state: State, objects: Sequence[Object]) -> bool:
        box,table = objects
        return state.get(box, "at_center") > 0.5
    
    @staticmethod
    def _BoxAtSide_holds(state: State, objects: Sequence[Object]) -> bool:
        box, table= objects
        return state.get(box, "at_side") > 0.5
    
    def _NoToyAtTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # Check all toys in the state
        for obj in state:
            if obj.type == self._toy_type and state.get(obj, "on_table") > 0.5:
                return False
        return True
    
    def _TableClean_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # Check if table has been cleaned (check table object state)
        for obj in state:
            if obj.type == self._table_type:
                return state.get(obj, "is_clean") > 0.5
        return False  # No table found
    
    def _GoalAchieved_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # Check if overall goal has been achieved
        # Find the robot object in the state (since self._robot might not match after serialization)
        for obj in state:
            if obj.type == self._robot_type:
                print("Robot found in state for goal check.")
                print(f"Goal achieved value: {state.get(obj, 'goal_achieved')}")
                return state.get(obj, "goal_achieved") > 0.5
        print("Robot object not found in state for goal check.")
        return False  # No robot found

    def _get_tasks(self,
                num_tasks: int,
                num_toys_lst: List[int],
                num_wipers_lst: List[int],
                rng: np.random.Generator) -> List[EnvironmentTask]:
        """Generate train / test tasks for table cleaning scenario."""
        tasks: List[EnvironmentTask] = []
        
        # Always 1 box regardless of train/test
        num_boxes_lst = [1]

        def _collides(x: float, y: float,
                    existing: List[Tuple[float, float]],
                    thresh: float) -> bool:
            return any(abs(ex_x - x) < thresh and abs(ex_y - y) < thresh
                    for ex_x, ex_y in existing)
        
        def _generate_mock_image() -> np.ndarray:
            """Generate mock object-centric image data."""
            return rng.random((self.img_height, self.img_width, self.img_channels)).flatten()

        for i in range(num_tasks):
            num_toys = num_toys_lst[0]  # Fixed number for train or test
            num_wipers = num_wipers_lst[0]  # Fixed number for train or test
            num_boxes = num_boxes_lst[0]  # Always 1 box

            # Initialize state dictionary
            state_dict: Dict[Object, Dict[str, float | np.ndarray]] = {}

            # Robot - starts with empty hands, goal not achieved
            state_dict[self._robot] = {
                "handempty": 1.0,
                "goal_achieved": 0.0
            }

            # Table - static object at origin
            table = Object("table0", self._table_type)
            mock_img = _generate_mock_image()
            state_dict[table] = {
                "pose_x": 0.0,
                "pose_y": 0.0,
                "is_clean": 0.0,
                **{f"img_{j}": mock_img[j] for j in range(self.img_size)}
            }
            
            # Box - starts at center
            box = Object("box0", self._box_type)
            mock_img = _generate_mock_image()
            state_dict[box] = {
                "pose_x": self.center_x,
                "pose_y": self.center_y,
                "at_center": 1.0,
                "at_side": 0.0,
                **{f"img_{j}": mock_img[j] for j in range(self.img_size)}
            }
            boxes = [box]

            # Toys - start on table
            toys: List[Object] = []
            toy_coords: List[Tuple[float, float]] = []
            goal: Set[GroundAtom] = set()
            
            for j in range(num_toys):
                toy = Object(f"toy{j}", self._toy_type)
                while True:
                    tx = rng.uniform(self.table_lx, self.table_ux)
                    ty = rng.uniform(self.table_ly, self.table_uy)
                    break
                    # # Avoid box center area
                    # if abs(tx - self.center_x) > 1.0 or abs(ty - self.center_y) > 1.0:
                    #     if not _collides(tx, ty, toy_coords, self.close_thresh):
                    #         break
                
                mock_img = _generate_mock_image()
                state_dict[toy] = {
                    "pose_x": tx,
                    "pose_y": ty,
                    "on_table": 1.0,
                    "in_box": 0.0,
                    **{f"img_{j}": mock_img[j] for j in range(self.img_size)}
                }
                toys.append(toy)
                toy_coords.append((tx, ty))
            # Wipers - start in box
            wipers: List[Object] = []
            for j in range(num_wipers):
                wiper = Object(f"wiper{j}", self._wiper_type)
                mock_img = _generate_mock_image()
                state_dict[wiper] = {
                    "pose_x": self.center_x,  # in box initially
                    "pose_y": self.center_y,
                    "on_table": 0.0,
                    "in_box": 1.0,
                    **{f"img_{j}": mock_img[j] for j in range(self.img_size)}
                }
                wipers.append(wiper)
            
            # Only goal is to achieve the overall goal
            goal.add(GroundAtom(self._GoalAchieved, [self._robot]))

            # Create state and task
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))

        return tasks

   


class TableCleanRealEnv(BaseEnv):
    # Environment parameters (needed for action space)
    table_lx: ClassVar[float] = -5.0
    table_ly: ClassVar[float] = -5.0
    table_ux: ClassVar[float] = 5.0
    table_uy: ClassVar[float] = 5.0
    
    dino_feature_dim: ClassVar[int] = 1024  
    
    _robot_type = Type("robot", ["goal_achieved"]+[f"dino_{i}" for i in range(dino_feature_dim)])
    _toy_type = Type("toy", [f"dino_{i}" for i in range(dino_feature_dim)])
    _wiper_type = Type("wiper", [f"dino_{i}" for i in range(dino_feature_dim)])
    _box_type = Type("box", [f"dino_{i}" for i in range(dino_feature_dim)])
    _table_type = Type("table", [f"dino_{i}" for i in range(dino_feature_dim)])


    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates based on PDDL specification
        self._ToyOnTable = Predicate("toy_on_table", [self._toy_type,self._table_type], self._ToyOnTable_holds)
        self._HandEmpty = Predicate("handempty", [self._robot_type], self._HandEmpty_holds)
        self._HoldingToy = Predicate("holdingToy", [self._robot_type,self._toy_type], self._HoldingToy_holds)
        self._ToyInBox = Predicate("toy_in_box", [self._toy_type,self._box_type], self._ToyInBox_holds)
        
        self._WiperInBox = Predicate("wiper_in_box", [self._wiper_type,self._box_type], self._WiperInBox_holds)
        self._WiperOnTable = Predicate("wiper_on_table", [self._wiper_type,self._table_type], self._WiperOnTable_holds)
        self._HoldingWiper = Predicate("holdingWiper", [self._robot_type,self._wiper_type], self._HoldingWiper_holds)
        
        self._BoxAtCenter = Predicate("box_at_center", [self._box_type,self._table_type], self._BoxAtCenter_holds)
        self._BoxAtSide = Predicate("box_at_side", [self._box_type,self._table_type], self._BoxAtSide_holds)
        
        self._NoToyAtTable = Predicate("No_toy_at_table", [], self._NoToyAtTable_holds)
        self._TableClean = Predicate("table_clean", [self._table_type], self._TableClean_holds)
        
        # Goal achievement predicate
        self._GoalAchieved = Predicate("goalAchieved", [self._robot_type], self._GoalAchieved_holds)
        
        # Static objects
        self._robot = Object("robot", self._robot_type)


    @classmethod
    def get_name(cls) -> str:
        return "clean-table-real-real"

    # ----------------------------------------------------------------------
    # PLACEHOLDER SIMULATION (no-ops except action 9 toggles goal_achieved)
    # ----------------------------------------------------------------------
    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        action_type, x, y = action.arr
        action_type = int(round(float(action_type)))

        if action_type == 9:
            # Achieve goal: set robot's goal_achieved to 1.0
            next_state.set(self._robot, "goal_achieved", 1.0)
            return next_state

        # All other actions are no-ops in this placeholder env
        return state

    # Simple train/test task generators delegating to _get_tasks
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_train_tasks,
            num_toys_lst=[2],
            num_wipers_lst=[1],
            rng=self._train_rng
        )

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_test_tasks,
            num_toys_lst=[3],
            num_wipers_lst=[2],
            rng=self._test_rng
        )

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._ToyOnTable, self._HandEmpty, self._HoldingToy, self._ToyInBox,
            self._WiperInBox, self._WiperOnTable, self._HoldingWiper,
            self._BoxAtCenter, self._BoxAtSide, self._NoToyAtTable, self._TableClean,
            self._GoalAchieved
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return { self._GoalAchieved }

    @property
    def types(self) -> Set[Type]:
        return { self._robot_type, self._toy_type, self._wiper_type, self._box_type, self._table_type }

    @property
    def derived_predicates(self) -> List[str]:
        return [
            "(:derived (No_toy_at_table)\n    (forall (?t - toy ?tb - table) (not (toy_on_table ?t ?tb))))"
        ]

    @property
    def action_space(self) -> Box:
        return Box(
            np.array([0, self.table_lx, self.table_ly], dtype=np.float32),
            np.array([9, self.table_ux, self.table_uy], dtype=np.float32),
        )

    # Placeholder renderer: return an empty figure
    def render_state_plt(
        self,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        return matplotlib.figure.Figure()

    # ----------------------------------------------------------------------
    # PLACEHOLDER ACTION HELPERS (no-ops) + achieve_goal sets the flag
    # ----------------------------------------------------------------------
    def _pick_toy_from_table(self, state: State, next_state: State, toys: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        return state

    def _place_toy_to_box(self, state: State, next_state: State, toys: List[Object], boxes: List[Object]) -> State:
        return state

    def _pick_wiper_from_box(self, state: State, next_state: State, wipers: List[Object], boxes: List[Object]) -> State:
        return state

    def _pick_wiper_from_table(self, state: State, next_state: State, wipers: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        return state

    def _place_wiper_at_table(self, state: State, next_state: State, wipers: List[Object], x: float = 0.0, y: float = 0.0) -> State:
        return state

    def _place_wiper_to_box(self, state: State, next_state: State, wipers: List[Object], boxes: List[Object]) -> State:
        return state

    def _push_box_out(self, state: State, next_state: State, boxes: List[Object]) -> State:
        return state

    def _pull_box_in(self, state: State, next_state: State, boxes: List[Object]) -> State:
        return state

    def _wipe_table(self, state: State, next_state: State, wipers: List[Object], toys: List[Object], boxes: List[Object], tables: List[Object]) -> State:
        return state

    def _achieve_goal(self, state: State, next_state: State, toys: List[Object], wipers: List[Object], boxes: List[Object]) -> State:
        next_state.set(self._robot, "goal_achieved", 1.0)
        return next_state

    # ----------------------------------------------------------------------
    # PLACEHOLDER PREDICATES (keep your simple returns), only goal works
    # ----------------------------------------------------------------------
    @staticmethod
    def _ToyOnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    def _HoldingToy_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _ToyInBox_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _WiperInBox_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _WiperOnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    def _HoldingWiper_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _BoxAtCenter_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    @staticmethod
    def _BoxAtSide_holds(state: State, objects: Sequence[Object]) -> bool:
        return False

    def _NoToyAtTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return False

    def _TableClean_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return False

    def _GoalAchieved_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # Find the robot of the correct type and read its flag
        for obj in state:
            if obj.type == self._robot_type:
                return state.get(obj, "goal_achieved") > 0.5
        return False

    # ----------------------------------------------------------------------
    # PLACEHOLDER TASK GENERATOR (objects with only DINO features)
    # ----------------------------------------------------------------------
    def _get_tasks(
        self,
        num_tasks: int,
        num_toys_lst: List[int],
        num_wipers_lst: List[int],
        rng: np.random.Generator
    ) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []

        def _rand_dino() -> np.ndarray:
            return rng.random(self.dino_feature_dim, dtype=float)

        for _ in range(num_tasks):
            state_dict: Dict[Object, Dict[str, float]] = {}

            # Robot with goal flag and DINO
            state_dict[self._robot] = {
                "goal_achieved": 0.0,
                **{f"dino_{i}": v for i, v in enumerate(_rand_dino())}
            }

            # One table
            table = Object("table0", self._table_type)
            state_dict[table] = {f"dino_{i}": v for i, v in enumerate(_rand_dino())}

            # One box
            box = Object("box0", self._box_type)
            state_dict[box] = {f"dino_{i}": v for i, v in enumerate(_rand_dino())}

            # Toys
            num_toys = num_toys_lst[0]
            for j in range(num_toys):
                toy = Object(f"toy{j}", self._toy_type)
                state_dict[toy] = {f"dino_{i}": v for i, v in enumerate(_rand_dino())}

            # Wipers
            num_wipers = num_wipers_lst[0]
            for j in range(num_wipers):
                wiper = Object(f"wiper{j}", self._wiper_type)
                state_dict[wiper] = {f"dino_{i}": v for i, v in enumerate(_rand_dino())}

            # Goal: achieve goal
            goal: Set[GroundAtom] = {GroundAtom(self._GoalAchieved, [self._robot])}

            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))

        return tasks