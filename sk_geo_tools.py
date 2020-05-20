import numpy as np
import re

pmt_bolt_offset = 15
bolt_ring_radius = 29.8


def get_bolt_locations_barrel(pmt_locations):
    bolt_locations = {}
    for f, pmt in pmt_locations.items():
        match = re.fullmatch(r"[0-1][0-9]{4}-00", f)
        if not match:
            continue
        phi = np.arctan2(pmt[1], pmt[0])
        bolt_locations.update({
            re.sub(r"-00$", "-"+str(i+1).zfill(2), f): np.array([
                pmt[0] - pmt_bolt_offset*np.cos(phi) + bolt_ring_radius*np.sin(i*np.pi/12.)*np.sin(phi),
                pmt[1] - pmt_bolt_offset*np.sin(phi) - bolt_ring_radius*np.sin(i*np.pi/12.)*np.cos(phi),
                pmt[2] + bolt_ring_radius*np.cos(i*np.pi/12.)])
            for i in range(0, 24)})
    return bolt_locations
