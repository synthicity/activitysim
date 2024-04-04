from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel as PydanticBase
from pydantic import validator

from activitysim.core.configuration.base import PreprocessorSettings, PydanticReadable


class LogitNestSpec(PydanticBase):
    """
    Defines a nest in a nested logit model.
    """

    name: str
    """A descriptive name for this nest."""

    coefficient: str | float
    """The named parameter to be used as the logsum coefficient.

    If given as a string, this named parameter should appear in the
    logit models's `COEFFICIENTS` file.
    """

    alternatives: list[LogitNestSpec | str]
    """The alternatives within this nest.

    These can be either the names of elemental alternatives, or `LogitNestSpec`
    definitions for more nests, or a mixture of these.
    """

    @validator("coefficient")
    def prefer_float_to_str(cls, coefficient_value):
        """
        Convert string values to float directly if possible.
        """
        try:
            coefficient_value = float(coefficient_value)
        except ValueError:
            pass
        return coefficient_value


class BaseLogitComponentSettings(PydanticReadable):
    """
    Base configuration class for components that are logit models.

    These settings are common to all logit models. Component developers
    should generally prefer using a derived classes that defines a complete
    logit model such as `LogitComponentSettings`, or a compound component
    such as `LocationComponentSettings`, which melds together alternative
    sampling, logsums, and choice.
    """

    SPEC: Path
    """Utility specification filename.

    This is sometimes alternatively called the utility expressions calculator
    (UEC). It is a CSV file giving all the functions for the terms of a
    linear-in-parameters utility expression.
    """

    COEFFICIENTS: Path | None = None
    """Coefficients filename.

    This is a CSV file giving named parameters for use in the utility expression.
    If it is not provided, then it is assumed that all model coefficients are
    given explicitly in the `SPEC` as numerical values instead of named parameters.
    This is perfectly acceptable for use with ActivitySim for typical simulation
    applications, but may be problematic if used with "estimation mode".
    """

    CONSTANTS: dict[str, Any] = {}
    """Named constants usable in the utility expressions."""

    sharrow_skip: bool = False
    """Skip sharrow when evaluating this component."""


class LogitComponentSettings(BaseLogitComponentSettings):
    """
    Base configuration class for components that are individual logit models.
    """

    LOGIT_TYPE: Literal["MNL", "NL"] = "MNL"
    """Logit model mathematical form.

    * "MNL"
        Multinomial logit model.
    * "NL"
        Nested multinomial logit model.
    """

    NESTS: LogitNestSpec | None = None
    """Nesting structure for a nested logit model.

    The nesting structure is specified heirarchically from the top, so the
    value of this field should be the "root" level nest of the nested logit
    tree, which should contain references to lower level nests and/or the
    actual alternatives.

    For example, this YAML defines a simple nesting structure for four
    alternatives (DRIVE, WALK, WALK_TO_TRANSIT, DRIVE_TO_TRANSIT) with the two
    transit alternatives grouped together in a nest:

    .. code-block:: yaml

        NESTS:
          name: root
          coefficient: coef_nest_root
          alternatives:
            - DRIVE
            - WALK
            - name: TRANSIT
              coefficient: coef_nest_transit
              alternatives:
              - WALK_TO_TRANSIT
              - DRIVE_TO_TRANSIT
    """

    @validator("NESTS")
    def nests_are_for_nl(cls, nests, values):
        """
        Checks that nests are provided if (and only if) `LOGIT_TYPE` is NL.
        """
        if "LOGIT_TYPE" in values and values["LOGIT_TYPE"] == "NL":
            if nests is None:
                raise ValueError("NESTS cannot be omitted for a NL model")
        if "LOGIT_TYPE" in values and values["LOGIT_TYPE"] == "MNL":
            if nests is not None:
                raise ValueError("NESTS cannot be provided for a MNL model")
        return nests

    overflow_protection: bool = False
    """Whether to use overflow protection in the probability calculation.

    Overflow protection will shift the utility values to avoid numerical
    overflow when exponentiating the utility values to calculate probabilities.
    It will also protect against underflow if the utility values are very
    negative.  Ideally, model specifications will be constructed to avoid
    overflow, but this option is provided as a safety net.
    """


class TemplatedLogitComponentSettings(LogitComponentSettings):
    """
    Base configuration for segmented logit models with a coefficient template.
    """

    COEFFICIENT_TEMPLATE: str | None = None
    """Coefficients template filename.

    For a segmented model component, this maps the named parameters to
    segment-specific names.
    """


class LocationComponentSettings(BaseLogitComponentSettings):
    """
    Base configuration class for components that are location choice models.
    """

    SAMPLE_SPEC: Path
    """The utility spec giving expressions to use in alternative sampling."""

    SAMPLE_SIZE: int
    """This many candidate alternatives will be sampled for each choice."""

    LOGSUM_SETTINGS: Path
    """Settings for the logsum computation."""


class TourLocationComponentSettings(LocationComponentSettings, extra="forbid"):
    # Logsum-related settings
    CHOOSER_ORIG_COL_NAME: str
    ALT_DEST_COL_NAME: str
    IN_PERIOD: int | dict[str, int] | None = None
    OUT_PERIOD: int | dict[str, int] | None = None
    LOGSUM_PREPROCESSOR: str = "preprocessor"

    SEGMENTS: list[str] | None = None
    SIZE_TERM_SELECTOR: str | None = None
    annotate_tours: PreprocessorSettings | None = None

    CHOOSER_FILTER_COLUMN_NAME: str | None = None
    DEST_CHOICE_COLUMN_NAME: str | None = None
    DEST_CHOICE_LOGSUM_COLUMN_NAME: str | None = None
    """Column name for logsum calculated across all sampled destinations."""
    MODE_CHOICE_LOGSUM_COLUMN_NAME: str | None = None
    """Column name for logsum calculated across all sampled modes to selected destination."""
    DEST_CHOICE_SAMPLE_TABLE_NAME: str | None = None
    CHOOSER_TABLE_NAME: str | None = None
    CHOOSER_SEGMENT_COLUMN_NAME: str | None = None
    SEGMENT_IDS: dict[str, int] | None = None
    SHADOW_PRICE_TABLE: str | None = None
    MODELED_SIZE_TABLE: str | None = None
    annotate_persons: PreprocessorSettings | None = None
    annotate_households: PreprocessorSettings | None = None
    SIMULATE_CHOOSER_COLUMNS: list[str] | None = None
    ALT_DEST_COL_NAME: str
    LOGSUM_TOUR_PURPOSE: str | dict[str, str] | None = None
    MODEL_SELECTOR: Literal["workplace", "school", None] = None
    SAVED_SHADOW_PRICE_TABLE_NAME: str | None = None
    CHOOSER_ID_COLUMN: str = "person_id"

    ORIG_ZONE_ID: str | None = None
    """This setting appears to do nothing..."""

    ESTIMATION_SAMPLE_SIZE: int = 0
    """
    The number of alternatives to sample for estimation mode.
    If zero, then all alternatives are used.
    Truth alternative will be included in the sample.
    Larch does not yet support sampling alternatives for estimation, 
    but this setting is still helpful for estimation mode runtime.
    """


class TourModeComponentSettings(TemplatedLogitComponentSettings, extra="forbid"):
    MODE_CHOICE_LOGSUM_COLUMN_NAME: str | None = None
    use_TVPB_constants: bool = True
    COMPUTE_TRIP_MODE_CHOICE_LOGSUMS: bool = False
    tvpb_mode_path_types: dict[str, Any] | None = None
    FORCE_ESCORTEE_CHAUFFEUR_MODE_MATCH: bool = True
    annotate_tours: PreprocessorSettings | None = None
    preprocessor: PreprocessorSettings | list[PreprocessorSettings] | None = None
    nontour_preprocessor: PreprocessorSettings | list[
        PreprocessorSettings
    ] | None = None
    LOGSUM_CHOOSER_COLUMNS: list[str] = []
