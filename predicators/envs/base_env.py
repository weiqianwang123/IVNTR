"""Base class for an environment."""

import os
import abc
import json
from natsort import natsorted
import dill as pkl
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pretrained_model_interface import OpenAILLM
from predicators.settings import CFG
from predicators.structs import Action, DefaultEnvironmentTask, \
    EnvironmentTask, GroundAtom, Object, Observation, Predicate, State, Task, \
    Type, Video


class BaseEnv(abc.ABC):
    """Base environment."""

    def __init__(self, use_gui: bool = True) -> None:
        self._current_observation: Observation = None  # set in reset
        self._current_task = DefaultEnvironmentTask  # set in reset
        self._set_seed(CFG.seed)
        # These are generated lazily when get_train_tasks or get_test_tasks is
        # called. This is necessary because environment attributes are often
        # initialized in __init__ in subclasses, and super().__init__ needs
        # to be called in those subclasses first, to set the env seed.
        self._train_tasks: List[EnvironmentTask] = []
        self._test_tasks: List[EnvironmentTask] = []
        # If the environment has a GUI, this determines whether to launch it.
        self._using_gui = use_gui

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this environment, used as the argument to
        `--env`."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        This function is primarily used in the default option model, and
        for implementing the default self.step(action). It is not meant to
        be part of the "final system", where the environment is the real world.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for testing / evaluation."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        raise NotImplementedError("Override me!")

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        """Get the goal predicates that we want the agent to use, which may be
        different than the ones the demonstrator uses.

        This is used when inventing VLM predicates. Unless overridden,
        these are the same as the original goal predicates.
        """
        return self.goal_predicates

    @property
    @abc.abstractmethod
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def action_space(self) -> Box:
        """Get the action space of this environment."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render a state and action into a Matplotlib figure.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        For environments which don't use Matplotlib for rendering, this
        function should be overriden to simply crash.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        raise NotImplementedError("Matplotlib rendering not implemented!")

    @property
    def using_gui(self) -> bool:
        """Whether the GUI for this environment is activated."""
        return self._using_gui

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        """Render a state and action into a list of images.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        By default, calls render_state_plt, but subclasses may override,
        e.g. if they do not use Matplotlib for rendering, and thus do not
        define a render_state_plt() function.
        """
        fig = self.render_state_plt(state, task, action, caption)
        img = utils.fig2data(fig, dpi=CFG.render_state_dpi)
        plt.close()
        return [img]

    def render_plt(self,
                   action: Optional[Action] = None,
                   caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render the current state and action into a Matplotlib figure.

        By default, calls render_state_plt, but subclasses may override.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        assert isinstance(self._current_observation, State), \
            "render_plt() only works in fully-observed environments."
        return self.render_state_plt(self._current_observation,
                                     self._current_task, action, caption)

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        """Render the current state and action into a list of images.

        By default, calls render_state, but subclasses may override.
        """
        assert isinstance(self._current_observation, State), \
            "render_state() only works in fully-observed environments."
        return self.render_state(self._current_observation, self._current_task,
                                 action, caption)

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """Return the ordered list of tasks for training."""
        if not self._train_tasks:
            tasks_fname, _ = utils.create_task_filename_str()
            if os.path.exists(tasks_fname):
                files = natsorted(Path(tasks_fname).glob("*.json"))
                assert len(files) >= CFG.num_train_tasks
                self._train_tasks = [
                    self._load_task_from_json(f)
                    for f in files[:CFG.num_train_tasks]
                ]
            else:
                self._train_tasks = self._generate_train_tasks()
        return self._train_tasks

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Return the ordered list of tasks for testing / evaluation."""
        if not self._test_tasks:
            tasks_fname, _ = utils.create_task_filename_str(train=False)
            if os.path.exists(tasks_fname):
                files = natsorted(Path(tasks_fname).glob("*.json"))
                assert len(files) >= CFG.num_test_tasks
                self._test_tasks = [
                    self._load_task_from_json(f)
                    for f in files[:CFG.num_test_tasks]
                ]
            else:
                self._test_tasks = self._generate_test_tasks()
        return self._test_tasks

    @property
    def _current_state(self) -> State:
        """Default for environments where states are observations."""
        assert isinstance(self._current_observation, State)
        return self._current_observation

    def goal_reached(self) -> bool:
        """Default implementation assumes environment tasks are tasks.

        Subclasses may override.
        """
        # NOTE: this is a convenience hack because most environments that are
        # currently implemented have goal descriptions that are simply sets of
        # ground atoms. In the future, it may be better to implement this on a
        # per-environment basis anyway, to make clear that we do not need to
        # make this assumption about goal descriptions in general.
        goal = self._current_task.goal_description
        assert isinstance(goal, set)
        assert not goal or isinstance(next(iter(goal)), GroundAtom)
        return all(goal_atom.holds(self._current_state) for goal_atom in goal)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        """Create a task from a JSON file.

        By default, we assume JSON files are in the following format:

        {
            "objects": {
                <object name>: <type name>
            }
            "init": {
                <object name>: {
                    <feature name>: <value>
                }
            }
            "goal": {
                <predicate name> : [
                    [<object name>]
                ]
            }
        }

        Instead of "goal", "language_goal" can also be used.

        Environments can override this method to handle different formats.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        # Parse objects.
        type_name_to_type = {t.name: t for t in self.types}
        object_name_to_object: Dict[str, Object] = {}
        for obj_name, type_name in json_dict["objects"].items():
            obj_type = type_name_to_type[type_name]
            obj = Object(obj_name, obj_type)
            object_name_to_object[obj_name] = obj
        assert set(object_name_to_object).issubset(set(json_dict["init"])), \
            "The init state can only include objects in `objects`."
        assert set(object_name_to_object).issuperset(set(json_dict["init"])), \
            "The init state must include every object in `objects`."
        # Parse initial state.
        init_dict: Dict[Object, Dict[str, float]] = {}
        for obj_name, obj_dict in json_dict["init"].items():
            obj = object_name_to_object[obj_name]
            init_dict[obj] = obj_dict.copy()
        init_state = utils.create_state_from_dict(init_dict)
        # Parse goal.
        if "goal" in json_dict:
            goal = self._parse_goal_from_json(json_dict["goal"],
                                              object_name_to_object)
        else:
            assert "language_goal" in json_dict
            goal = self._parse_language_goal_from_json(
                json_dict["language_goal"], object_name_to_object)
        return EnvironmentTask(init_state, goal)
    
    def _load_task_from_pkl(self, pkl_file: Path) -> EnvironmentTask:
        """Create a task from a pickle file.
        """
        assert os.path.exists(pkl_file), f"File {pkl_file} does not exist."
        with open(pkl_file, "rb") as f:
            test_tasks = pkl.load(f)
        self._test_tasks = test_tasks
        return self._test_tasks

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        """Create a prompt to prepend to a language model query for parsing
        language-based goals into goal atoms.

        Since the language model is queried with "#" as the stop token,
        and since the goal atoms are processed with _parse_goal_from_json(),
        the following format of hashtags and JSON dicts is necessary:

        # Build a tower of block 1, block 2, and block 3, with block 1 on top
        {"On": [["block1", "block2"], ["block2", "block3"]]}

        # Put block 4 on block 3 and block 2 on block 1 and block 1 on table
        {"On": [["block4", "block3"], ["block2", "block1"]],
         "OnTable": [["block1"]]}
        """
        raise NotImplementedError("This environment did not implement an "
                                  "interface for language-based goals!")

    def _parse_goal_from_json(self, spec: Dict[str, List[List[str]]],
                              id_to_obj: Dict[str, Object]) -> Set[GroundAtom]:
        """Helper for parsing goals from JSON task specifications."""
        goal_pred_names = {p.name for p in self.goal_predicates}
        assert set(spec.keys()).issubset(goal_pred_names)
        pred_to_args = {p: spec.get(p.name, []) for p in self.goal_predicates}
        goal: Set[GroundAtom] = set()
        for pred, args in pred_to_args.items():
            for id_args in args:
                obj_args = [id_to_obj[a] for a in id_args]
                goal_atom = GroundAtom(pred, obj_args)
                goal.add(goal_atom)
        return goal

    def _parse_language_goal_from_json(
            self, language_goal: str,
            id_to_obj: Dict[str, Object]) -> Set[GroundAtom]:
        """Helper for parsing language-based goals from JSON task specs."""
        object_names = set(id_to_obj)
        prompt_prefix = self._get_language_goal_prompt_prefix(object_names)
        prompt = prompt_prefix + f"\n# {language_goal}"
        llm = OpenAILLM(CFG.llm_model_name)
        responses = llm.sample_completions(prompt,
                                           None,
                                           temperature=0.0,
                                           seed=CFG.seed,
                                           stop_token="#")
        response = responses[0]
        # Currently assumes that the LLM is perfect. In the future, will need
        # to handle various errors and perhaps query the LLM for multiple
        # responses until we find one that can be parsed.
        goal_spec = json.loads(response)
        return self._parse_goal_from_json(goal_spec, id_to_obj)

    def get_task(self, train_or_test: str, task_idx: int) -> EnvironmentTask:
        """Return the train or test task at the given index."""
        if train_or_test == "train":
            tasks = self.get_train_tasks()
        elif train_or_test == "test":
            tasks = self.get_test_tasks()
        else:
            raise ValueError(f"get_task called with invalid train_or_test: "
                             f"{train_or_test}.")
        return tasks[task_idx]

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rngs."""
        self._seed = seed
        # The train/test rng should be used when generating
        # train/test tasks respectively.
        self._train_rng = np.random.default_rng(self._seed)
        self._test_rng = np.random.default_rng(self._seed +
                                               CFG.test_env_seed_offset)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        # Copy to prevent external changes to the environment's state.
        # This default implementation of reset assumes that observations are
        # states. Subclasses with different states should override.
        assert isinstance(self._current_observation, State)
        return self._current_observation.copy()

    def step(self, action: Action) -> Observation:
        """Apply the action, update the state, and return an observation.

        Note that this action is a low-level action (i.e., action.arr
        is a member of self.action_space), NOT an option.

        By default, this function just calls self.simulate. However,
        environments that maintain a more complicated internal state,
        or that don't implement simulate(), may override this method.
        """
        assert isinstance(self._current_observation, State)
        self._current_observation = self.simulate(self._current_observation,
                                                  action)
        # Copy to prevent external changes to the environment's state.
        return self._current_observation.copy()

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        """The optional environment-specific method that is used for generating
        demonstrations from a human, with a GUI.

        Returns a function that maps state and Matplotlib event to an
        action in this environment; before returning this function, it's
        recommended to log some instructions about the controls.
        """
        raise NotImplementedError("This environment did not implement an "
                                  "interface for human demonstrations!")

    def get_observation(self) -> Observation:
        """Get the current observation of this environment."""
        assert isinstance(self._current_observation, State)
        return self._current_observation.copy()

    def get_vlm_debug_atom_strs(self, train_tasks: List[Task]) -> Set[str]:
        """A 'debug grammar' set of predicates that should be sufficient for
        completing the task; useful for comparing different methods of VLM
        truth-value labelling given the same set of atom proposals to label.

        For the BaseEnv, this method simply takes the names of all
        excluded predicates and uses these (i.e., forcing the VLM to
        learn a classifier for these predicates). Subclasses can
        override to handle more specific use cases.
        """
        _, excluded_preds = utils.parse_config_excluded_predicates(self)
        all_ground_atoms_set: Set[GroundAtom] = set()
        for tt in train_tasks:
            all_ground_atoms_set |= set(
                utils.all_possible_ground_atoms(tt.init, excluded_preds))
        atom_strs = {
            atom.predicate.name + "(" +
            ", ".join([o.name for o in atom.objects]) + ")"
            for atom in sorted(all_ground_atoms_set)
        }
        return atom_strs
