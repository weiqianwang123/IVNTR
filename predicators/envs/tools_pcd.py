"""Tools domain, where the robot must interact with a variety of items and
tools. Items are screws, nails, and bolts. Tools are screwdrivers, wrenches,
and hammers. Screws are fastened using screwdrivers or the robot's hand. Nails
are fastened using hammers. Bolts are fastened using wrenches.

Screwdrivers have a shape and a size. The shape must match the screw
shape, but is not captured using any given predicate. Some screw shapes
can be fastened using hands directly. The screwdriver's size must be
small enough that it is graspable by the robot, which is captured by a
predicate. Hammer sizes work the same way as screwdriver sizes. Wrench
sizes don't matter.
"""

from typing import ClassVar, List, Optional, Sequence, Set,Tuple

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
from predicators import utils


import argparse
import numpy as np
import trimesh
import open3d as o3d
from typing import Optional

def obj_to_pointcloud(obj_path: str,
                      n_points: int = 50_000,
                      save_path: Optional[str] = None
                      ) -> o3d.geometry.PointCloud:
    """Sample exactly `n_points` from an OBJ/PLY/STL surface.

    Returns
    -------
    o3d.geometry.PointCloud
        Always has `len(pcd.points) == n_points`.
    """
    # ---- Load mesh --------------------------------------------------------
    mesh = trimesh.load(obj_path, force='mesh')        # handles .obj/.ply/.stl …
    if mesh.is_empty:
        raise ValueError(f"Failed to load a mesh from {obj_path}")

    # ---- Uniform surface sampling ----------------------------------------
    points, _ = trimesh.sample.sample_surface_even(mesh, 5000)

    # Guarantee the exact count
    if points.shape[0] < n_points:                     # too few → up-sample
        extra, _ = trimesh.sample.sample_surface_even(
            mesh, n_points - points.shape[0])
        points = np.vstack([points, extra])
    elif points.shape[0] > n_points:                   # too many → trim
        points = points[:n_points]

    assert points.shape[0] == n_points, "Point-cloud size mismatch!"

    # ---- Wrap in Open3D ---------------------------------------------------
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # ---- Optional save ----------------------------------------------------
    if save_path:
        ok = o3d.io.write_point_cloud(save_path, pcd)
        if not ok:
            raise IOError(f"Open3D couldn’t write {save_path}")
        print(f"Saved {n_points} pts → {save_path}")

    return points.astype(np.float32)


class ToolsPCDEnv(BaseEnv):
    """Tools domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    table_lx: ClassVar[float] = -10.0
    table_ly: ClassVar[float] = -10.0
    table_ux: ClassVar[float] = 10.0
    table_uy: ClassVar[float] = 10.0
    contraption_size: ClassVar[float] = 2.0
    close_thresh: ClassVar[float] = 0.1
    # For a screw of a particular shape, if the shape of every graspable
    # screwdriver differs by at least this amount, then this screw is required
    # to be fastened by hand. Otherwise, it is required to be fastened by the
    # graspable screwdriver that has the smallest difference in shape.
    screw_shape_hand_thresh: ClassVar[float] = 0.25
    # Number of each type of tool is fixed
    num_screwdrivers: ClassVar[int] = 1
    num_hammers: ClassVar[int] = 2
    num_wrenches: ClassVar[int] = 1
    SHAPES: ClassVar[Tuple[float, float]] = (0.0, 1.0)
    pcd_dim: ClassVar[int] = 1024  # number of points in the point cloud
    pcd_attrs = [f"pcd_{i}" for i in range(pcd_dim*3)]


    # Types
    _robot_type = Type("robot", ["fingers"])
    _screw_type = Type("screw", 
    ["pcd","pose_x", "pose_y", "shape", "is_fastened", "is_held"])

    _screwdriver_type = Type("screwdriver", 
    ["pcd","pose_x", "pose_y", "shape", "size", "is_held"])
    _nail_type = Type("nail", ["pose_x", "pose_y", "is_fastened", "is_held"])
    _hammer_type = Type("hammer", ["pose_x", "pose_y", "size", "is_held"])
    _bolt_type = Type("bolt", ["pose_x", "pose_y", "is_fastened", "is_held"])
    _wrench_type = Type("wrench", ["pose_x", "pose_y", "size", "is_held"])
    _contraption_type = Type("contraption", ["pose_lx", "pose_ly"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._HoldingScrew = Predicate("HoldingScrew", [self._screw_type],
                                       self._Holding_holds)
        self._HoldingScrewdriver = Predicate("HoldingScrewdriver",
                                             [self._screwdriver_type],
                                             self._Holding_holds)
        self._HoldingNail = Predicate("HoldingNail", [self._nail_type],
                                      self._Holding_holds)
        self._HoldingHammer = Predicate("HoldingHammer", [self._hammer_type],
                                        self._Holding_holds)
        self._HoldingBolt = Predicate("HoldingBolt", [self._bolt_type],
                                      self._Holding_holds)
        self._HoldingWrench = Predicate("HoldingWrench", [self._wrench_type],
                                        self._Holding_holds)
        self._ScrewPlaced = Predicate(
            "ScrewPlaced", [self._screw_type, self._contraption_type],
            self._Placed_holds)
        self._NailPlaced = Predicate("NailPlaced",
                                     [self._nail_type, self._contraption_type],
                                     self._Placed_holds)
        self._BoltPlaced = Predicate("BoltPlaced",
                                     [self._bolt_type, self._contraption_type],
                                     self._Placed_holds)
        self._ScrewFastened = Predicate("ScrewFastened", [self._screw_type],
                                        self._Fastened_holds)
        self._NailFastened = Predicate("NailFastened", [self._nail_type],
                                       self._Fastened_holds)
        self._BoltFastened = Predicate("BoltFastened", [self._bolt_type],
                                       self._Fastened_holds)
        self._ScrewdriverGraspable = Predicate(
            "ScrewdriverGraspable", [self._screwdriver_type],
            self._ScrewdriverGraspable_holds)
        self._HammerGraspable = Predicate("HammerGraspable",
                                          [self._hammer_type],
                                          self._HammerGraspable_holds)
        self._ScrewMatch = Predicate(
            "ScrewMatch", [self._screw_type, self._screwdriver_type],
            self._ScrewMatch_holds)
        self._ScrewNotMatch = Predicate(
            "ScrewNotMatch", [self._screw_type, self._screwdriver_type],
            self._ScrewNotMatch_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "tools-pcd"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        x, y, is_pick, is_place,is_adjust = action.arr

        if (is_pick > 0.5) + (is_place > 0.5) + (is_adjust > 0.5) > 1:
            return next_state  # invalid combo

        
        held = self.get_held_item_or_tool(state)

        target = self._get_closest_item_or_tool(state, x, y)

        if is_adjust > 0.5:
            if held is None or held.type != self._screwdriver_type:
                return next_state  # failure: need to hold screwdriver
            screwdriver_shape = state.get(held, "shape")
            screwdriver_new_shape =  abs(screwdriver_shape - 1.0) 
            next_state.set(held, "shape", screwdriver_new_shape)
            screw_mesh_file = f"mesh/Screwdriver_{int(screwdriver_new_shape) + 1}.obj"
            pcd = obj_to_pointcloud(screw_mesh_file, n_points=self.pcd_dim)  
            next_state.set(held, "pcd", pcd)  
            # Update the point cloud of the screwdriver
            return next_state
        if is_place > 0.5:
            # Handle placing
            if held is None:
                # Failure: not holding anything
                return next_state
            if self._is_tool(held) and \
               self._get_contraption_pose_is_on(state, x, y) is not None:
                # Failure: cannot place a tool on a contraption
                return next_state
            next_state.set(held, "is_held", 0.0)
            next_state.set(held, "pose_x", x)
            next_state.set(held, "pose_y", y)
            next_state.set(self._robot, "fingers", 1.0)
            return next_state
        
        if target is None:
            # Failure: not doing a place, so something must be at this (x, y)
            return next_state
        del x, y  # no longer needed
        pose_x = state.get(target, "pose_x")
        pose_y = state.get(target, "pose_y")
        contraption = self._get_contraption_pose_is_on(state, pose_x, pose_y)
        if is_pick < 0.5:  # no pick flag => treat as fasten
            if contraption is None or not self.is_item(target):
                return next_state

            if target.type == self._screw_type:
                # Case A: attempt to fasten by hand
                if held is None:
                   return next_state  # screwdriver available → hand not ok
                # Case B: holding screwdriver – must match shape
                elif held.type == self._screwdriver_type:
                    if not self._ScrewMatch_holds(state, [target, held]):
                        return next_state
                else:
                    return next_state  # wrong tool in hand
            elif target.type == self._nail_type:
                if held is None or held.type != self._hammer_type:
                    return next_state
            elif target.type == self._bolt_type:
                if held is None or held.type != self._wrench_type:
                    return next_state

            next_state.set(target, "is_fastened", 1.0)
            return next_state
        # Handle picking
        if held is not None:
            # Failure: holding something already
            return next_state
        if self._is_screwdriver_or_hammer(target) and \
           state.get(target, "size") > 0.5:
            # Failure: screwdriver/hammer is not graspable
            return next_state
        if self.is_item(target) and contraption is not None:
            # Failure: can't pick an item when it's on a contraption
            return next_state
        # Note: we don't update the pose of an object when it is
        # picked. This is a sort of hack that provides "memory",
        # so that when placing tools down, there is always the
        # easy choice of placing it back where you got it from.
        # The oracle sampler makes use of this.
        next_state.set(target, "is_held", 1.0)
        next_state.set(self._robot, "fingers", 0.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_train_tasks,
            num_items_lst=CFG.tools_num_items_train,
            num_contraptions_lst=CFG.tools_num_contraptions_train,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(
            num_tasks=CFG.num_test_tasks,
            num_items_lst=CFG.tools_num_items_test,
            num_contraptions_lst=CFG.tools_num_contraptions_test,
            rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._HoldingScrew, self._HoldingScrewdriver,
            self._HoldingNail, self._HoldingHammer, self._HoldingBolt,
            self._HoldingWrench, self._ScrewPlaced, self._NailPlaced,
            self._BoltPlaced, self._ScrewFastened, self._NailFastened,
            self._BoltFastened, self._ScrewdriverGraspable,
            self._HammerGraspable,self._ScrewMatch,self._ScrewNotMatch
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._ScrewPlaced, self._NailPlaced, self._BoltPlaced,
            self._ScrewFastened, self._NailFastened, self._BoltFastened
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._screw_type, self._screwdriver_type,
            self._nail_type, self._hammer_type, self._bolt_type,
            self._wrench_type, self._contraption_type
        }

   

    # Actions: [x, y, is_pick, is_place, is_adjust]
    @property
    def action_space(self) -> Box:
        return Box(
            np.array([self.table_lx, self.table_ly, 0, 0, 0], dtype=np.float32),
            np.array([self.table_ux, self.table_uy, 1, 1, 1], dtype=np.float32),
        )


    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError


    def _get_tasks(self,
                num_tasks: int,
                num_items_lst: List[int],
                num_contraptions_lst: List[int],
                rng: np.random.Generator) -> List[EnvironmentTask]:
        """Generate train / test tasks with new pcd-in-dict format."""
        tasks: List[EnvironmentTask] = []

        # ----------------------------------------------------------------------- #
        #  Helper：检测 (x,y) 是否与已存在物体/装置碰撞                              #
        # ----------------------------------------------------------------------- #
        def _collides(x: float, y: float,
                    existing: List[Tuple[float, float]],
                    thresh: float) -> bool:
            return any(abs(ex_x - x) < thresh and abs(ex_y - y) < thresh
                    for ex_x, ex_y in existing)

        for i in range(num_tasks):
            num_items         = num_items_lst[i % len(num_items_lst)]
            num_contraptions  = num_contraptions_lst[i % len(num_contraptions_lst)]

            # =============== 1. 组装 state_dict ================================= #
            state_dict: Dict[Object, Dict[str, np.ndarray | float]] = {}

            # 1-a 机器人
            state_dict[self._robot] = {"fingers": 1.0}

            # ---------------- Contraptions ------------------------------------- #
            contraptions: List[Object] = []
            contraption_coords: List[Tuple[float, float]] = []
            for j in range(num_contraptions):
                contraption = Object(f"contraption{j}", self._contraption_type)
                while True:
                    lx = rng.uniform(self.table_lx,
                                    self.table_ux - self.contraption_size)
                    ly = rng.uniform(self.table_ly,
                                    self.table_uy - self.contraption_size)
                    if not _collides(lx, ly, contraption_coords, self.contraption_size):
                        break
                contraption_coords.append((lx, ly))
                contraptions.append(contraption)
                state_dict[contraption] = {"pose_lx": lx, "pose_ly": ly}

            # ---------------- Items (screw / nail / bolt) ---------------------- #
            items: List[Object] = []
            item_coords: List[Tuple[float, float]] = []
            screw_cnt = nail_cnt = bolt_cnt = 0
            goal: Set[GroundAtom] = set()

            for _ in range(num_items):
                while True:
                    ix = rng.uniform(self.table_lx, self.table_ux)
                    iy = rng.uniform(self.table_ly, self.table_uy)
                    # 与 contraption / 现有 item 不冲突
                    contraption_hit = any(
                        (lx < ix < lx + self.contraption_size) and
                        (ly < iy < ly + self.contraption_size)
                        for lx, ly in contraption_coords)
                    if not contraption_hit and \
                    not _collides(ix, iy, item_coords, self.close_thresh):
                        break

                is_fastened = is_held = 0.0
                choices = ["screw", "nail", "bolt"]
                if screw_cnt > 0:
                    choices.remove("screw")        # 题目限定至多一个螺丝
                choice = rng.choice(choices)
                goal_contraption = rng.choice(contraptions)

                if choice == "screw":
                    item = Object(f"screw{screw_cnt}", self._screw_type)
                    screw_cnt += 1
                    shape = float(rng.choice(self.SHAPES))

                    mesh_file = f"mesh/Screw_{int(shape)+1}.obj"
                    pcd = obj_to_pointcloud(mesh_file, n_points=self.pcd_dim)

                    state_dict[item] = {
                        "pcd"        : pcd,          # (N,3)
                        "pose_x"     : ix,
                        "pose_y"     : iy,
                        "shape"      : shape,
                        "is_fastened": is_fastened,
                        "is_held"    : is_held,
                    }
                    goal.add(GroundAtom(self._ScrewFastened, [item]))
                    goal.add(GroundAtom(self._ScrewPlaced, [item, goal_contraption]))

                elif choice == "nail":
                    item = Object(f"nail{nail_cnt}", self._nail_type)
                    nail_cnt += 1
                    state_dict[item] = {
                        "pose_x"     : ix,
                        "pose_y"     : iy,
                        "is_fastened": is_fastened,
                        "is_held"    : is_held,
                    }
                    goal.add(GroundAtom(self._NailFastened, [item]))
                    goal.add(GroundAtom(self._NailPlaced, [item, goal_contraption]))

                else:   # bolt
                    item = Object(f"bolt{bolt_cnt}", self._bolt_type)
                    bolt_cnt += 1
                    state_dict[item] = {
                        "pose_x"     : ix,
                        "pose_y"     : iy,
                        "is_fastened": is_fastened,
                        "is_held"    : is_held,
                    }
                    goal.add(GroundAtom(self._BoltFastened, [item]))
                    goal.add(GroundAtom(self._BoltPlaced, [item, goal_contraption]))

                items.append(item)
                item_coords.append((ix, iy))

            # ---------------- Tools -------------------------------------------- #
            tools: List[Object] = []
            tool_coords: List[Tuple[float, float]] = []

            screwdriver_sizes = [rng.uniform(0, 0.5) for _ in range(self.num_screwdrivers)]
            hammer_sizes      = [rng.uniform(0, 0.5) for _ in range(self.num_hammers)]
            hammer_sizes[rng.integers(self.num_hammers)] = rng.uniform(0.5, 1.0)  # 至少一个过大
            wrench_sizes      = [rng.uniform(0, 1.0) for _ in range(self.num_wrenches)]

            all_sizes = screwdriver_sizes + hammer_sizes + wrench_sizes
            for idx, size in enumerate(all_sizes):
                while True:
                    tx = rng.uniform(self.table_lx, self.table_ux)
                    ty = rng.uniform(self.table_ly, self.table_uy)
                    if not _collides(tx, ty, contraption_coords, self.contraption_size) and \
                    not _collides(tx, ty, item_coords + tool_coords, self.close_thresh):
                        break

                is_held = 0.0
                if idx < self.num_screwdrivers:
                    tool = Object(f"screwdriver{idx}", self._screwdriver_type)
                    shape = float(rng.choice(self.SHAPES))
                    mesh_file = f"mesh/Screwdriver_{int(shape)+1}.obj"
                    pcd = obj_to_pointcloud(mesh_file, n_points=self.pcd_dim)

                    state_dict[tool] = {
                        "pcd"     : pcd,
                        "pose_x"  : tx,
                        "pose_y"  : ty,
                        "shape"   : shape,
                        "size"    : size,
                        "is_held" : is_held,
                    }
                elif idx < self.num_screwdrivers + self.num_hammers:
                    h_idx = idx - self.num_screwdrivers
                    tool = Object(f"hammer{h_idx}", self._hammer_type)
                    state_dict[tool] = {
                        "pose_x"  : tx,
                        "pose_y"  : ty,
                        "size"    : size,
                        "is_held" : is_held,
                    }
                else:
                    w_idx = idx - self.num_screwdrivers - self.num_hammers
                    tool = Object(f"wrench{w_idx}", self._wrench_type)
                    state_dict[tool] = {
                        "pose_x"  : tx,
                        "pose_y"  : ty,
                        "size"    : size,
                        "is_held" : is_held,
                    }

                tools.append(tool)
                tool_coords.append((tx, ty))

            # =============== 2. 生成 State & Task ============================== #
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))

        return tasks


    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.5

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        # Works for any item or tool
        item_or_tool, = objects
        return state.get(item_or_tool, "is_held") > 0.5

    def _Placed_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # Works for any item
        item, contraption = objects
        pose_x = state.get(item, "pose_x")
        pose_y = state.get(item, "pose_y")
        return self.is_pose_on_contraption(state, pose_x, pose_y, contraption)

    @staticmethod
    def _Fastened_holds(state: State, objects: Sequence[Object]) -> bool:
        # Works for any item
        item, = objects
        return state.get(item, "is_fastened") > 0.5

    @staticmethod
    def _ScrewdriverGraspable_holds(state: State,
                                    objects: Sequence[Object]) -> bool:
        screwdriver, = objects
        return state.get(screwdriver, "size") < 0.5

    @staticmethod
    def _HammerGraspable_holds(state: State,
                               objects: Sequence[Object]) -> bool:
        hammer, = objects
        return state.get(hammer, "size") < 0.5
    

    def _ScrewMatch_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        screw, screwdriver = objects
        assert screw.type == self._screw_type
        assert screwdriver.type == self._screwdriver_type
        screw_shape = state.get(screw, "shape")
        screwdriver_shape = state.get(screwdriver, "shape")
        return abs(screw_shape - screwdriver_shape) < self.screw_shape_hand_thresh
    def _ScrewNotMatch_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        screw, screwdriver = objects
        assert screw.type == self._screw_type
        assert screwdriver.type == self._screwdriver_type
        screw_shape = state.get(screw, "shape")
        screwdriver_shape = state.get(screwdriver, "shape")
        return abs(screw_shape - screwdriver_shape) >= self.screw_shape_hand_thresh

    def _get_closest_item_or_tool(self, state: State, x: float,
                                  y: float) -> Optional[Object]:
        closest_obj = None
        closest_dist = float("inf")
        for obj in state:
            if obj == self._robot:
                continue
            if obj.type == self._contraption_type:
                continue
            x_dist = abs(state.get(obj, "pose_x") - x)
            y_dist = abs(state.get(obj, "pose_y") - y)
            if x_dist > self.close_thresh or y_dist > self.close_thresh:
                continue
            dist = x_dist + y_dist
            if dist < closest_dist:
                closest_dist = dist
                closest_obj = obj
        return closest_obj

    @classmethod
    def get_held_item_or_tool(cls, state: State) -> Optional[Object]:
        """Public for use by oracle options."""
        held_obj = None
        for obj in state:
            if obj.type == cls._robot_type:
                continue
            if obj.type == cls._contraption_type:
                continue
            if state.get(obj, "is_held") > 0.5:
                assert held_obj is None
                held_obj = obj
        return held_obj

    def _get_contraption_pose_is_on(self, state: State, x: float,
                                    y: float) -> Optional[Object]:
        for obj in state:
            if obj.type != self._contraption_type:
                continue
            if self.is_pose_on_contraption(state, x, y, obj):
                return obj
        return None

    @classmethod
    def is_pose_on_contraption(cls, state: State, x: float, y: float,
                               contraption: Object) -> bool:
        """Public for use by oracle options."""
        pose_lx = state.get(contraption, "pose_lx")
        pose_ly = state.get(contraption, "pose_ly")
        pose_ux = pose_lx + cls.contraption_size
        pose_uy = pose_ly + cls.contraption_size
        return pose_lx < x < pose_ux and pose_ly < y < pose_uy

    def _is_tool(self, obj: Object) -> bool:
        return obj.type in (self._screwdriver_type, self._hammer_type,
                            self._wrench_type)

    @classmethod
    def is_item(cls, obj: Object) -> bool:
        """Public for use by oracle options."""
        return obj.type in (cls._screw_type, cls._nail_type, cls._bolt_type)

    def _is_screwdriver_or_hammer(self, obj: Object) -> bool:
        return obj.type in (self._screwdriver_type, self._hammer_type)

    def _get_best_screwdriver_or_none(self, state: State,
                                      screw: Object) -> Optional[Object]:
        """Use the shape of the given screw to figure out the best graspable
        screwdriver for it, or None if no graspable screwdriver has a shape
        within the threshold self.screw_shape_hand_thresh."""
        assert screw.type == self._screw_type
        closest_screwdriver = None
        closest_diff = float("inf")
        screw_shape = state.get(screw, "shape")
        for obj in state:
            if obj.type != self._screwdriver_type:
                continue
            if state.get(obj, "size") > 0.5:
                # Ignore non-graspable screwdrivers
                continue
            screwdriver_shape = state.get(obj, "shape")
            diff = abs(screw_shape - screwdriver_shape)
            if diff < closest_diff:
                closest_diff = diff
                closest_screwdriver = obj
        if closest_diff > self.screw_shape_hand_thresh:
            return None
        return closest_screwdriver
