import functools
from collections.abc import Callable

import jax.nn as jnn
import jax.numpy as jnp
from flax.linen import gelu

activations: dict[str, Callable] = {
    "elu": jnn.elu,
    "relu": jnn.relu,
    "tanh": lambda x: jnp.tanh(x),
    "gelu": gelu,
    "lrelu": functools.partial(jnn.leaky_relu, negative_slope=0.01),
    "swish": jnn.swish,
    "sin": jnp.sin,
    "none": lambda x: x,
    "const": lambda x: jnp.ones_like(x),
}
