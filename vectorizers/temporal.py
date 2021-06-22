#!/usr/bin/python3

from math import pi, sin, cos
from re import match

import numpy as np
from rdflib import Literal


_REGEX_YEAR_FRAG = "(?P<sign>-?)(?P<year>\d{4})"
_REGEX_MONTH_FRAG = "(?P<month>\d{2})"
_REGEX_DAY_FRAG = "(?P<day>\d{2})"
_REGEX_HOUR_FRAG = "(?P<hour>\d{2})"
_REGEX_MINUTE_FRAG = "(?P<minute>\d{2})"
_REGEX_SECOND_FRAG = "(?P<second>\d{2})(?:\.(?P<subsecond>\d+))?"
_REGEX_TIMEZONE_FRAG = "(?P<timezone>Z|(?:\+|-)(?:(?:0\d|1[0-3]):[0-5]\d|14:00))?"
_REGEX_DATE = "{}-{}-{}(?:{})?".format(_REGEX_YEAR_FRAG,
                                       _REGEX_MONTH_FRAG,
                                       _REGEX_DAY_FRAG,
                                       _REGEX_TIMEZONE_FRAG)
_REGEX_DATETIME = "{}-{}-{}T{}:{}:{}(?:{})?".format(_REGEX_YEAR_FRAG,
                                                    _REGEX_MONTH_FRAG,
                                                    _REGEX_DAY_FRAG,
                                                    _REGEX_HOUR_FRAG,
                                                    _REGEX_MINUTE_FRAG,
                                                    _REGEX_SECOND_FRAG,
                                                    _REGEX_TIMEZONE_FRAG)
_MINUTE_RAD = 2*pi/60
_HOUR_RAD = 2*pi/24
_DAY_RAD = 2*pi/31
_MONTH_RAD = 2*pi/12
_YEAR_DECADE_RAD = 2*pi/10


def generate_data(g, indices, datatypes):
    entity_to_class_map, entity_to_int_map, _ = indices
    is_varlength = False
    time_dim = -1

    datatypes = list(datatypes)
    data = [list() for dtype in datatypes]
    data_length = [list() for dtype in datatypes]
    data_entity_map = [list() for dtype in datatypes]

    int_to_datatype_map = dict(enumerate(datatypes))
    datatype_to_int_map = {v: k for k, v in int_to_datatype_map.items()}
    seen_datatypes = set()
    for (s, p, o), _ in g.triples((None, None, None), None):
        if type(o) is not Literal or str(o.datatype) not in datatypes:
            continue

        s = str(s)
        s_int = entity_to_int_map[s]

        o_dtype = str(o.datatype)
        value = str(o)
        try:
            if o_dtype.endswith("gYear"):
                value = match("{}{}".format(_REGEX_YEAR_FRAG,
                                            _REGEX_TIMEZONE_FRAG),
                              value)
            elif o_dtype.endswith("dateTime"):
                value = match(_REGEX_DATETIME, value)
            elif o_dtype.endswith("date"):
                value = match(_REGEX_DATE, value)
        except Exception:
            continue

        row = list()
        try:
            sign = 1. if value.group('sign') == '' else -1.
            year = value.group('year')

            # separate centuries, decades, and individual years
            separated = temporal_separate(year)

            c = int(separated.group('century'))

            decade = int(separated.group('decade'))
            dec1, dec2 = temporal_point(decade, _YEAR_DECADE_RAD)

            year = int(separated.group('year'))
            y1, y2 = temporal_point(year, _YEAR_DECADE_RAD)

            row.extend([sign, c, dec1, dec2, y1, y2])
            if o_dtype.endswith("date") or o_dtype.endswith("dateTime"):
                month = value.group('month')
                m1, m2 = temporal_point(int(month), _MONTH_RAD)

                day = value.group('day')
                d1, d2 = temporal_point(int(day), _DAY_RAD)

                row.extend([m1, m2, d1, d2])
                if o_dtype.endswith("dateTime"):
                    hour = value.group('hour')
                    h1, h2 = temporal_point(int(hour), _HOUR_RAD)

                    minutes = value.group('minute')
                    min1, min2 = temporal_point(int(minutes), _MINUTE_RAD)
                    row.extend([h1, h2, min1, min2])
        except Exception:
            continue

        o_dtype_int = datatype_to_int_map[o_dtype]
        seen_datatypes.add(o_dtype_int)

        data[o_dtype_int].append(row)
        data_length[o_dtype_int].append(len(row))
        data_entity_map[o_dtype_int].append(s_int)

    seen_datatypes = list(seen_datatypes)
    data = [data[i] for i in seen_datatypes]
    data_length = [data_length[i] for i in seen_datatypes]
    data_entity_map = [data_entity_map[i] for i in seen_datatypes]

    if len(seen_datatypes) <= 0:
        return list()

    # normalization over centuries
    c_idx = 1
    for i in range(len(data)):
        a = np.array(data[i])

        v_min = a[:, c_idx].min()
        v_max = a[:, c_idx].max()

        if v_min == v_max:
            a[:, c_idx] = 0.0
        else:
            a[:, c_idx] = (2 * (a[:, c_idx] - v_min) / (v_max - v_min)) - 1.0

        data[i] = [r for r in a]

    return list(zip([int_to_datatype_map[i] for i in seen_datatypes],
                    data, data_length, data_entity_map,
                    [is_varlength for _ in seen_datatypes],
                    [time_dim for _ in seen_datatypes]))


def temporal_point(m, rad):
    # place on circle
    return (sin(m*rad), cos(m*rad))


def temporal_separate(year):
    regex = "(?P<century>\d\d)(?P<decade>\d)(?P<year>\d)"
    return match(regex, year)
