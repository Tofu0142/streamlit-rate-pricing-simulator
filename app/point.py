from collections import namedtuple
import math

Point = namedtuple('Point', 'x y')


def new_point(current_position, points_per_turn, total_points):
    curr_turn, i = divmod(current_position, points_per_turn)
    angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
    radius = current_position / total_points
    return Point(radius * math.cos(angle), radius * math.sin(angle))
