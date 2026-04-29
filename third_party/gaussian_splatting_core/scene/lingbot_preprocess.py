import numpy as np


def random_uniform_downsample(
    xyz,
    rgb,
    conf=None,
    sample_ratio=1.0,
    max_points=0,
    seed=0,
):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz to be [N,3], got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"Expected rgb to match xyz shape, got {rgb.shape} vs {xyz.shape}")

    valid = np.isfinite(xyz).all(axis=1)
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if conf.shape[0] != xyz.shape[0]:
            raise ValueError("conf length must match number of points")
        valid &= np.isfinite(conf)
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return xyz[:0], rgb[:0], (conf[:0] if conf is not None else None)

    rng = np.random.default_rng(seed)
    target = valid_idx.size
    if sample_ratio > 0 and sample_ratio < 1.0:
        target = int(np.floor(valid_idx.size * sample_ratio))
        target = max(1, target)
    if max_points and max_points > 0:
        target = min(target, int(max_points))

    if target < valid_idx.size:
        choose = rng.choice(valid_idx, size=target, replace=False)
    else:
        choose = valid_idx

    xyz_out = xyz[choose]
    rgb_out = rgb[choose]
    conf_out = conf[choose] if conf is not None else None
    return xyz_out, rgb_out, conf_out
