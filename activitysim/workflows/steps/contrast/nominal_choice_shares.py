
import logging
import pandas as pd
import altair as alt
from pypyr.context import Context
from ..progression import reset_progress_step
from ..wrapping import report_step

logger = logging.getLogger(__name__)


def parse_grouping(g):
    if isinstance(g, str):
        return g, {'shorthand': g}
    elif isinstance(g, dict):
        return g.get('field'), g
    elif g is None:
        return None, None
    else:
        raise ValueError(g)

@report_step
def compare_nominal_choice_shares(
        tablesets,
        tablename,
        nominal_col,
        row_grouping=None,
        col_grouping=None,
        count_label=None,
        share_label=None,
        share_axis_label='Share',
        title=None,
        ordinal=False,
):
    """
    Parameters
    ----------
    tablesets : Mapping
    title : str, optional
    grouping : str
    """
    if count_label is None:
        count_label = f"# of {tablename}"

    if share_label is None:
        share_label = f"share of {tablename}"

    row_g, row_g_kwd = parse_grouping(row_grouping)
    col_g, col_g_kwd = parse_grouping(col_grouping)

    d = {}
    groupings = []
    if row_g is not None:
        groupings.append(row_g)
    if col_g is not None:
        groupings.append(col_g)

    for key, tableset in tablesets.items():
        df = tableset[tablename].groupby(
            groupings + [nominal_col]
        ).size().rename(count_label).reset_index()
        if not groupings:
            df[share_label] = df[count_label] / df[count_label].sum()
        else:
            df[share_label] = df[count_label] / df.groupby(groupings)[count_label].transform('sum')
        d[key] = df

    all_d = pd.concat(d, names=['source']).reset_index()

    selection = alt.selection_multi(
        fields=[nominal_col], bind='legend',
    )

    encode = dict(
        color=alt.Color(nominal_col, type="ordinal" if ordinal else "nominal"),
        y=alt.Y('source', axis=alt.Axis(grid=False, title='')),
        x=alt.X(share_label, axis=alt.Axis(grid=False, labels=False, title=share_axis_label),
                scale=alt.Scale(domain=[0., 1.])),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[nominal_col, 'source', count_label, alt.Tooltip(f'{share_label}:Q', format='.2%')],
    )
    if row_g is not None:
        encode['row'] = alt.Row(**row_g_kwd)
    if col_g is not None:
        encode['column'] = alt.Column(**col_g_kwd)

    fig = alt.Chart(
        all_d
    ).mark_bar(
    ).encode(
        **encode,
    ).add_selection(
        selection,
    )

    if title:
        fig = fig.properties(
            title=title
        ).configure_title(
            fontSize=20,
            anchor='start',
            color='black',
        )

    if col_grouping is not None:
        fig = fig.properties(
            width=100,
        )

    return fig