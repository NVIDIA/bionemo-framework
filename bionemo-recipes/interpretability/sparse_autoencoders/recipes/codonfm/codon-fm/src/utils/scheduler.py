def linear_scheduler_with_warmup_exp_decay(iteration,
                                           total_iterations,
                                           initial_lr,
                                           warmup_iterations=1000,
                                           min_lr=1e-6):

    decay_factor = (min_lr / initial_lr) ** (1 / (total_iterations - warmup_iterations))

    if iteration < warmup_iterations:
        lr = initial_lr * (iteration + 1) / warmup_iterations
    else:
        lr = initial_lr * (decay_factor ** (iteration - warmup_iterations))

    if lr < min_lr:
        lr = min_lr

    return lr

def linear_scheduler_with_warmup_lr_lambda(iteration: int, warmup_iterations: int, total_iterations: int):
    """https://github.com/huggingface/transformers/blob/0de15c988b0d27758ce360adb2627e9ea99e91b3/src/transformers/optimization.py#L102"""
    if iteration < warmup_iterations:
        return float(iteration) / float(max(1, warmup_iterations))
    return max(0.0, float(total_iterations - iteration) / float(max(1, total_iterations - warmup_iterations)))