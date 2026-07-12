"""DesirePath-NF: conditional Normalizing Flow over pedestrian motion in open
public space (ETH/UCY). Learns p(step | position, history, goal-bearing, crowd)
and reads out (a) a flow/desire field, (b) a surprise map, (c) the systematic
deviation of real flow from the straight-to-goal geodesic = a "desire path"
proxy. See README.md for the honest scope (no paved-path ground truth)."""
