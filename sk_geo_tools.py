import numpy as np
from scipy import linalg
import re

pmt_bolt_offset = 15
bolt_ring_radius = 29.8
bolt_distance = 2 * bolt_ring_radius * np.sin(np.pi / 24)


def get_bolt_locations_barrel(pmt_locations):
    bolt_locations = {}
    for f, pmt in pmt_locations.items():
        match = re.fullmatch(r"[0-1][0-9]{4}-00", f)
        if not match:
            continue
        phi = np.arctan2(pmt[1], pmt[0])
        bolt_locations.update({
            re.sub(r"-00$", "-" + str(i + 1).zfill(2), f): np.array([
                pmt[0] - pmt_bolt_offset * np.cos(phi) + bolt_ring_radius * np.sin(i * np.pi / 12.) * np.sin(phi),
                pmt[1] - pmt_bolt_offset * np.sin(phi) - bolt_ring_radius * np.sin(i * np.pi / 12.) * np.cos(phi),
                pmt[2] + bolt_ring_radius * np.cos(i * np.pi / 12.)])
            for i in range(0, 24)})
    return bolt_locations


def get_bolt_distances(bolt_locations):
    bolt_distances = []
    for b, l in bolt_locations.items():
        next_bolt = b[:-2] + str(int(b[-2:]) % 24 + 1).zfill(2)
        if next_bolt in bolt_locations:
            bolt_distances.append(linalg.norm(l - bolt_locations[next_bolt]))
    return bolt_distances


def get_unique_pmt_ids(feature_locations):
    return set(f[:5] for f in feature_locations.keys())


def get_bolt_ring_centres(bolt_locations):
    pmt_ids = get_unique_pmt_ids(bolt_locations)
    return {p: np.mean([bolt_locations[p + "-" + str(b).zfill(2)]
                        for b in range(1, 25) if p + "-" + str(b).zfill(2) in bolt_locations.keys()], axis=0)
            for p in pmt_ids}


def get_bolt_ring_radii(bolt_locations):
    bolt_ring_centres = get_bolt_ring_centres(bolt_locations)
    return [linalg.norm(l - bolt_ring_centres[b[:5]]) for b, l in bolt_locations.items()]


# https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
def fit_plane(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """

    assert points.shape[0] <= points.shape[0], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=0)
    x = points - ctr
    M = np.dot(x.T, x)  # Could also use np.cov(x) here.
    return ctr, linalg.svd(M)[0][:, -1]


def get_bolt_ring_planes(bolt_locations):
    pmt_ids = get_unique_pmt_ids(bolt_locations)
    planes = {}
    for p in pmt_ids:
        c, n = fit_plane(np.array([bolt_locations[p + "-" + str(b).zfill(2)]
                                   for b in range(1, 25) if p + "-" + str(b).zfill(2) in bolt_locations.keys()]))
        # flip normal if it is directed away from tank centre
        if n[0] > 0 and n[1] > 0:
            n = -n
        planes[p] = c, n
    return planes


def get_supermodule_plane(bolt_locations, min_pmt, max_pmt):
    c, n = fit_plane(np.array([l for b, l in bolt_locations.items() if min_pmt <= int(b[:5]) <= max_pmt]))
    # flip normal if it is directed away from tank centre
    if n[0] > 0 and n[1] > 0:
        n = -n
    return c, n
