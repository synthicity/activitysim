# ActivitySim
# See full license in LICENSE.txt.

import os

import pandas as pd
import pandas.testing as pdt

from activitysim.abm.models.util.vectorize_tour_scheduling import (
    get_previous_tour_by_tourid,
    vectorize_tour_scheduling,
)
from activitysim.core import workflow


def test_vts():
    whale = (
        workflow.Whale()
        .initialize_filesystem(
            output_dir=os.path.join(os.path.dirname(__file__), "output"),
        )
        .default_settings()
    )

    # note: need 0 duration tour on one end of day to guarantee at least one available tour
    alts = pd.DataFrame({"start": [1, 1, 2, 3], "end": [1, 4, 5, 6]})
    alts["duration"] = alts.end - alts.start
    whale.add_injectable("tdd_alts", alts)

    current_tour_person_ids = pd.Series(["b", "c"], index=["d", "e"])

    previous_tour_by_personid = pd.Series([2, 2, 1], index=["a", "b", "c"])

    prev_tour_attrs = get_previous_tour_by_tourid(
        current_tour_person_ids, previous_tour_by_personid, alts
    )

    pdt.assert_series_equal(
        prev_tour_attrs.start_previous,
        pd.Series([2, 1], index=["d", "e"], name="start_previous"),
    )

    pdt.assert_series_equal(
        prev_tour_attrs.end_previous,
        pd.Series([5, 4], index=["d", "e"], name="end_previous"),
    )

    tours = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3, 3],
            "tour_num": [1, 2, 1, 1, 2],
            "tour_type": ["x", "x", "x", "x", "x"],
        }
    )

    persons = pd.DataFrame({"income": [20, 30, 25]}, index=[1, 2, 3])

    whale.add_table("persons", persons)

    spec = pd.DataFrame({"Coefficient": [1.2]}, index=["income"])
    spec.index.name = "Expression"

    whale.settings.check_for_variability = True

    timetable = whale.get_injectable("timetable")

    tdd_choices = vectorize_tour_scheduling(
        whale,
        tours,
        persons,
        alts,
        timetable,
        tour_segments={"spec": spec},
        tour_segment_col=None,
        model_settings={},
        chunk_size=0,
        trace_label="test_vts",
    )

    # FIXME - dead reckoning regression
    # there's no real logic here - this is just what came out of the monte carlo
    # note that the result comes out ordered by the nth trips and not ordered
    # by the trip index.  shrug?
    expected = [2, 2, 2, 0, 0]
    assert (tdd_choices.values == expected).all()
