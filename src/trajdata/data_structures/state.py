"""
Defines subclasses of np.array and torch.Tensor which give
property access to different state elements and allow for easy conversion
between types to help make code that works with state elements more readable
and more robust to future changes in state format (e.g. adding additional dimensions)

Currently, these subclasses are designed to be lightweight and ephemeral:
any np/torch operation on a State subclass will drop the format metadata.
TODO: we could make this more robust by making exceptions for operations which 
preserve the semantic meanings of the elements.
TODO: implement setters for all properties

If you want to add state column info to an array
"""
from abc import abstractclassmethod
from collections import defaultdict
from typing import Callable, ClassVar, Dict, List, Type, TypeVar

import numpy as np
import torch
from torch import Tensor

STATE_ELEMS_REQUIREMENTS = {
    "x": None,  # x position in world frame (m)
    "y": None,  # y position in world frame (m)
    "z": None,  # z position in world frame (m)
    "xd": None,  # x velocity in world frame (m/s)
    "yd": None,  # y velocity in world frame (m/s)
    "zd": None,  # z velocity in world frame (m/s)
    "xdd": None,  # x acceleration in world frame (m/s^2)
    "ydd": None,  # y acceleration in world frame (m/s^2)
    "zdd": None,  # z acceleration in world frame (m/s^2)
    "h": ("arctan", "s", "c"),  # heading (rad)
    "dh": None,  # heading rate (rad)
    "c": ("cos", "h"),  # cos(h)
    "s": ("sin", "h"),  # sin(h)
}

Array = TypeVar("Array", np.ndarray, torch.Tensor)


class State:
    """
    Base class implementing property access to state elements
    Needs to be subclassed for concrete underlying datatypes, e.g.
    torch.Tensor vs np.ndarray, to equip self object with
    indexing support
    """

    _format: str = ""

    # set upon subclass init
    state_dim: int = 0

    # needs to be defined in subclass
    _FUNCS: ClassVar[Dict[str, Callable]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # use subclass _format string to initialize class specific _format_dict
        cls._format_dict: Dict[str, int] = {}
        for j, attr in enumerate(cls._format.split(",")):
            cls._format_dict[attr] = j

        # intialize properties
        cls.position = cls._init_property("x,y")
        cls.position3d = cls._init_property("x,y,z")
        cls.velocity = cls._init_property("xd,yd")
        cls.acceleration = cls._init_property("xdd,ydd")
        cls.heading = cls._init_property("h")
        cls.heading_vector = cls._init_property("c,s")

        # initialize state_dim
        cls.state_dim = len(cls._format_dict)

    @abstractclassmethod
    def from_array(cls, array: Array, format: str) -> "State":
        """
        Returns State instance given Array with correct format.

        Args:
            array (Array): Array
            format (str): format string
        """
        raise NotImplementedError

    @abstractclassmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string

        Args:
            arrays (List[Array]): _description_
            format (str): _description_
        """
        raise NotImplementedError

    def as_format(self, new_format: str, create_type=True):
        """
        Returns a new StateTensor with the specified format,
        constructed using data in the current format
        """
        requested_attrs = new_format.split(",")
        components = []  # contains either indicies into self, or attrs
        index_list = None
        for j, attr in enumerate(requested_attrs):
            if attr in self._format_dict:
                if index_list is None:
                    # start a new block of indices
                    index_list = []
                    components.append(index_list)
                index_list.append(self._format_dict[attr])
            else:
                if index_list is not None:
                    # if we had been pulling indices, stop
                    index_list = None
                components.append(attr)
        # assemble
        arrays = []
        for component in components:
            if isinstance(component, list):
                arrays.append(self[..., component])
            elif isinstance(component, str):
                arrays.append(self._compute_attr(component)[..., None])
            else:
                raise ValueError

        result = self._combine(arrays)
        if create_type:
            return self.from_array(result, new_format)
        else:
            return result

    def _compute_attr(self, attr: str):
        """
        Tries to compute attr that isn't directly part of the tensor
        given the information available.

        If impossible raises ValueError
        """
        try:
            formula = STATE_ELEMS_REQUIREMENTS[attr]
            if formula is None:
                raise KeyError(f"No formula for {attr}")
            func_name, *requirements = formula
            func = self._FUNCS[func_name]
            args = [self[..., self._format_dict[req]] for req in requirements]
        except KeyError as ke:
            raise ValueError(
                f"{attr} cannot be computed from available data at the current timestep."
            )
        return func(*args)

    def get_attr(self, attr: str):
        """
        Returns slice of tensor corresponding to attr
        """
        if attr in self._format_dict:
            return self[..., self._format_dict[attr]]
        else:
            return self._compute_attr(attr)

    def set_attr(self, attr: str, val: Tensor):
        if attr in self._format_dict:
            self[..., self._format_dict[attr]] = val
        else:
            raise ValueError(f"{attr} not part of State")

    @classmethod
    def _init_property(cls, format: str) -> property:
        split_format = format.split(",")
        try:
            index_list = tuple(cls._format_dict[attr] for attr in split_format)

            def getter(self: State) -> Array:
                return self[..., index_list]

            def setter(self: State, val: Array) -> None:
                self[..., index_list] = val

        except KeyError:
            # getter is nontrivial, let as_format handle the logic
            def getter(self: State) -> Array:
                return self.as_format(format, create_type=False)

            # can't set this property since not all elements are part of format
            setter = None

        return property(
            getter,
            setter,
            doc=f"""
            Returns:
                Array: shape [..., {len(split_format)}] corresponding to {split_format}.
            """,
        )


class StateArray(State, np.ndarray):

    _FUNCS = {
        "cos": np.cos,
        "sin": np.sin,
        "arctan": np.arctan2,
    }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, type(self)):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.get("out", None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, type(self)):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * function.nout

        # call original function
        results = super().__array_ufunc__(function, method, *args, **kwargs)

        return results

    def __getitem__(self, key) -> np.ndarray:
        """
        StateArray[key] always returns an np.ndarray, as we can't
        be sure that key isn't indexing into the state elements
        without adding logic in python which adds significant overhead
        to the base numpy implementation which is in C.

        In cases where we just want to index batch dimensions, use
        StateArray.at(key). We add logic for slice indexing
        """
        return_type = np.ndarray
        if isinstance(key, (int, slice)) and self.ndim > 1:
            return_type = type(self)
        return super().__getitem__(key).view(return_type)

    def at(self, key) -> "StateArray":
        """
        Equivalent to self[key], but assumes (without checking!)
        that key selects only batch dimensions, so return type
        is the same as type(self)
        """
        return super().__getitem__(key)

    @classmethod
    def from_array(cls, array: Array, format: str):
        return array.view(NP_STATE_TYPES[format])

    @classmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string
        """
        return np.concatenate(arrays, axis=-1)


class StateTensor(State, Tensor):
    """
    Convenience class which offers property access to state elements
    Standardizes order of state dimensions
    """

    _FUNCS = {
        "cos": torch.cos,
        "sin": torch.sin,
        "arctan": torch.arctan2,
    }

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # overriding this to ensure operations yield base Tensors
        if kwargs is None:
            kwargs = {}

        new_class = Tensor
        result = super().__torch_function__(func, types, args, kwargs)

        # TODO(bivanovic): Ideally we would have Tensor.to here as well, but...
        # https://github.com/pytorch/pytorch/issues/47051
        if func in {Tensor.cpu, Tensor.cuda, Tensor.add, Tensor.add_}:
            new_class = cls

        if func == Tensor.__getitem__:
            self = args[0]
            indices = args[1]
            if isinstance(indices, int):
                if self.ndim > 1:
                    new_class = cls
            elif isinstance(indices, slice):
                if self.ndim > 1:
                    new_class = cls
                elif indices == slice(None):
                    new_class = cls
            elif isinstance(indices, tuple):
                if len(indices) < self.ndim:
                    new_class = cls
                elif len(indices) == self.ndim and indices[-1] == slice(None):
                    new_class = cls

        if isinstance(result, Tensor) and new_class != cls:
            result = result.as_subclass(new_class)

        if func == Tensor.numpy:
            result: np.ndarray
            result = result.view(NP_STATE_TYPES[cls._format])

        return result

    @classmethod
    def from_numpy(cls, state: StateArray, **kwargs):
        return torch.from_numpy(state, **kwargs).as_subclass(
            TORCH_STATE_TYPES[state._format]
        )

    @classmethod
    def from_array(cls, array: Array, format: str):
        return array.as_subclass(TORCH_STATE_TYPES[format])

    @classmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string
        """
        return torch.cat(arrays, dim=-1)


def createStateType(format: str, base: Type[State]) -> Type[State]:
    name = base.__name__ + "".join(map(str.capitalize, format.split(",")))
    cls = type(
        name,
        (base,),
        {
            "_format": format,
        },
    )
    # This is needed so that these dynamically created classes are understood
    # by pickle, which is used in multiprocessing, e.g. in a dataset
    globals()[name] = cls
    return cls


class StateTypeFactory(defaultdict):
    def __init__(self, base: Type[State]):
        self.base_type = base

    def __missing__(self, format: str) -> Type[State]:
        self[format] = createStateType(format, self.base_type)
        return self[format]


# DEFINE STATE TYPES
TORCH_STATE_TYPES: Dict[str, Type[StateTensor]] = StateTypeFactory(StateTensor)
NP_STATE_TYPES: Dict[str, Type[StateArray]] = StateTypeFactory(StateArray)
