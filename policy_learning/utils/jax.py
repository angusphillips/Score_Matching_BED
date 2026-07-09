import hashlib

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray as Key


def derive_key(seed: int, *namespace: object) -> Key:
    """Deterministically derive an independent PRNG key from a global seed.

    Unlike the common ``key, subkey = jax.random.split(key)`` iterator pattern,
    the returned key depends *only* on ``seed`` and the supplied ``namespace``
    parts -- not on how many keys were drawn before it. This makes per-object
    seeding reproducible across runs with the same ``seed`` and robust to
    reordering or to inserting/removing other random draws elsewhere (e.g.
    changing the number of training iters or callback frequencies does not
    disturb the seeds handed to other objects).

    Each object created in a script (a trainer, an estimator, a callback, ...)
    should derive its own key via a stable, unique ``namespace`` and then use
    and advance that key internally, independently of every other object.

    Args:
        seed: The global config seed.
        namespace: A tuple of stable identifiers (strings / ints) uniquely
            naming the object, e.g. ``("policy", "trainer", restart_id)``.

    Returns:
        An independent ``jax.random`` key.
    """
    h = hashlib.blake2b(digest_size=4)
    for part in namespace:
        h.update(repr(part).encode())
        h.update(b"\x00")
    fold = int.from_bytes(h.digest(), "big")
    return jax.random.fold_in(jax.random.key(seed), fold)


def x_gradient(func):
    return lambda x: jax.vjp(func, x)[1](jnp.ones_like(x))[0]


def flatten_pytree(params, leave_axes: int = 0):
    """Flatten a pytree into a single array.

    Args:
        params: A pytree of arrays.
        leave_axes: Number of leading axes to preserve (not flatten).
            If 0, flattens everything into a 1D array.
            If 1, preserves the first axis and flattens the rest,
            e.g. shape [256, 5, 2] -> [256, 10].

    Returns:
        flat: The flattened array.
        spec: A dictionary containing information needed to unflatten.
    """
    leaves, treedef = jax.tree.flatten(params)

    flat_arrays = []
    shapes = []
    sizes = []

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        shapes.append(arr.shape)
        if leave_axes == 0:
            sizes.append(arr.size)
            flat_arrays.append(arr.reshape(-1))
        else:
            # Preserve the first `leave_axes` dimensions, flatten the rest
            leading_shape = arr.shape[:leave_axes]
            # Use -1 to let JAX infer the trailing size (avoids concretization error)
            flat_arr = arr.reshape(*leading_shape, -1)
            sizes.append(flat_arr.shape[-1])
            flat_arrays.append(flat_arr)

    # Concatenate into one big vector (along the last axis if leave_axes > 0)
    if not flat_arrays:
        flat = jnp.array([])
    elif leave_axes == 0:
        flat = jnp.concatenate(flat_arrays)
    else:
        flat = jnp.concatenate(flat_arrays, axis=-1)

    # Create index ranges for each leaf
    indices = []
    start = 0
    for size in sizes:
        indices.append((start, start + size))
        start += size

    spec = {
        "treedef": treedef,
        "shapes": shapes,
        "indices": indices,
        "leave_axes": leave_axes,
    }

    return flat, spec


def unflatten_pytree(flat, spec):
    """Unflatten an array back into a pytree.

    Args:
        flat: The flattened array (created by flatten_pytree).
        spec: The spec dictionary returned by flatten_pytree.

    Returns:
        The reconstructed pytree.
    """
    treedef = spec["treedef"]
    shapes = spec["shapes"]
    indices = spec["indices"]
    leave_axes = spec.get("leave_axes", 0)

    leaves = []
    for (start, end), shape in zip(indices, shapes, strict=True):
        if leave_axes == 0:
            leaf = flat[start:end].reshape(shape)
        else:
            # Index along the last axis, then reshape
            leaf = flat[..., start:end].reshape(shape)
        leaves.append(leaf)

    return jax.tree.unflatten(treedef, leaves)
