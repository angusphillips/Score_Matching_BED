import logging

from jax.errors import JaxRuntimeError


def get_maximal_outer_batch_size(key, test_params, grad_fn_of_bs, initial_bs) -> int:
    outer_batch_size = initial_bs
    success = False
    while not success:
        grad_fn = grad_fn_of_bs(outer_batch_size)
        try:
            _, key = grad_fn(key, test_params)
            success = True
        except JaxRuntimeError as err:
            print("Outer batch size too large:", outer_batch_size)
            outer_batch_size //= 2
            print("Trying batch size: ", outer_batch_size)
            if outer_batch_size == 0:
                raise ValueError("Cannot reduce outer batch size any further") from err

    return int(outer_batch_size)


def resolve_outer_batch_size(
    key,
    params,
    *,
    policy_net,
    bed_model,
    policy_cfg: dict,
    estimate_type: str,
    data_score_fn,
    design_score_fn,
    init_batch_size: int,
    cache: dict,
    slot: str,
    log: logging.Logger | None = None,
    label: str = "",
) -> int:
    """OOM-probe the maximal safe outer batch size for one estimator config.

    Memoised in the caller-owned ``cache[slot]`` so restarts/rounds reuse one
    probe. The two conventional slots mirror the historical sharing: "general"
    (SPCE-like / non-chunked estimators) and "chunked" (``score_chunked``,
    whose memory profile is bounded by the chunk). ``estimate_type="score"``
    never needs probing — the fully vmapped estimator ignores
    ``outer_batch_size`` (callers pass 0 directly).
    """
    from policy_learning.eig_estimators import get_grad_estimate_fn

    if cache.get(slot) is None:

        def partial_grad_fn(outer_batch_size):
            return get_grad_estimate_fn(
                policy_net=policy_net,
                bed_model=bed_model,
                data_score_fn=data_score_fn,
                design_score_fn=design_score_fn,
                outer_batch_size=outer_batch_size,
                N=int(policy_cfg["train_outer_samples"]),
                M=int(policy_cfg["train_inner_samples"]),
                estimate_type=estimate_type,
            )

        if log is not None:
            log.info(
                "[oom%s] Probing safe %s outer batch size (init=%d)",
                f"/{label}" if label else "",
                estimate_type,
                init_batch_size,
            )
        cache[slot] = get_maximal_outer_batch_size(
            key, params, partial_grad_fn, init_batch_size
        )
        if log is not None:
            log.info(
                "[oom%s] %s outer batch size (no-OOM): %d",
                f"/{label}" if label else "",
                estimate_type,
                cache[slot],
            )
    return int(cache[slot])
