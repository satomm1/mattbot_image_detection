"""Depth of a detection from the (color-aligned) depth image, uint16 millimetres."""

import numpy as np

#: Sub-windows of the box, as (x_lo, x_hi, y_lo, y_hi) fractions of its width / height.
#: People: the torso -- clear of the floor at the feet, the gaps between legs and arms, and
#: most things standing in front of them. Objects: the middle of the box.
PERSON_WINDOW = (0.30, 0.70, 0.20, 0.60)
OBJECT_WINDOW = (0.25, 0.75, 0.25, 0.75)


def bbox_percentile_depth(depth, x_min, y_min, x_max, y_max, pct=25):
    """LEGACY (the original ``get_depth``, verbatim): the ``pct`` percentile of every non-zero
    depth pixel in the box, metres; 0.0 when there is none."""
    # Get the depth values within the bounding box
    depth_values = depth[int(y_min):int(y_max), int(x_min):int(x_max)].flatten()

    # Remove zero depth values
    depth_values = depth_values[depth_values > 0]

    if len(depth_values) == 0:
        return 0.0  # 0 since no valid depth values are found

    # Get the 25th percentile depth
    estimated_depth = np.percentile(depth_values, pct) / 1000.0  # Convert to meters (/1000)

    return estimated_depth


def window_cluster_depth(depth, x_min, y_min, x_max, y_max, window=OBJECT_WINDOW, bin_m=0.05,
                         min_frac=0.15, stride=2):
    """ROBUST: depth (m) of the nearest SUBSTANTIAL surface in a central window of the box.

    Non-zero pixels of the window (every ``stride``-th row and column) go into ``bin_m`` bins;
    the nearest bin that, with its two neighbours, holds at least ``min_frac`` of the pixels is
    the surface, and its pixels' median is the depth. Speckle and thin things in front are too
    few to count; the background behind is never the nearest. When no bin qualifies, the densest
    one. 0.0 only when the window has no valid pixel.
    """
    w, h = x_max - x_min, y_max - y_min
    xa = int(x_min + window[0] * w)
    xb = max(int(x_min + window[1] * w), xa + 1)
    ya = int(y_min + window[2] * h)
    yb = max(int(y_min + window[3] * h), ya + 1)
    vals = depth[max(ya, 0):max(yb, 0):stride, max(xa, 0):max(xb, 0):stride].ravel()
    vals = vals[vals > 0].astype(np.float64) / 1000.0
    if len(vals) == 0:
        return 0.0
    bins = np.floor(vals / bin_m).astype(np.int64)
    lo = int(bins.min())
    counts = np.bincount(bins - lo)
    near = np.convolve(counts, np.ones(3, dtype=np.int64), mode="same")  # each bin with its neighbours
    ok = np.nonzero(near >= min_frac * len(vals))[0]
    # No surface holds min_frac (depth spread out, e.g. an object seen at a slant): the densest one, so a
    # detection with valid pixels is never dropped for want of a dominant surface.
    b = (int(ok[0]) if len(ok) else int(np.argmax(near))) + lo
    sel = vals[np.abs(bins - b) <= 1]
    return float(np.median(sel))
