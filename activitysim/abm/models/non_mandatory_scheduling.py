# ActivitySim
# See full license in LICENSE.txt.
import logging

from activitysim.abm.models.util.tour_scheduling import run_tour_scheduling
from activitysim.core import timetable as tt
from activitysim.core import tracing, workflow
from activitysim.core.util import assign_in_place

logger = logging.getLogger(__name__)
DUMP = False


@workflow.step
def non_mandatory_tour_scheduling(
    whale: workflow.Whale, tours, persons_merged, tdd_alts, chunk_size
):
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    model_name = "non_mandatory_tour_scheduling"
    trace_label = model_name
    trace_hh_id = whale.settings.trace_hh_id
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    # - if no mandatory_tours
    if non_mandatory_tours.shape[0] == 0:
        tracing.no_results(model_name)
        return

    tour_segment_col = None

    choices = run_tour_scheduling(
        whale,
        model_name,
        non_mandatory_tours,
        persons_merged,
        tdd_alts,
        tour_segment_col,
        chunk_size,
    )

    assign_in_place(tours, choices)
    whale.add_table("tours", tours)

    # updated df for tracing
    non_mandatory_tours = tours[tours.tour_category == "non_mandatory"]

    whale.dump_df(
        DUMP,
        tt.tour_map(persons_merged, non_mandatory_tours, tdd_alts),
        trace_label,
        "tour_map",
    )

    if trace_hh_id:
        whale.trace_df(
            non_mandatory_tours,
            label=trace_label,
            slicer="person_id",
            index_label="tour_id",
            columns=None,
            warn_if_empty=True,
        )
