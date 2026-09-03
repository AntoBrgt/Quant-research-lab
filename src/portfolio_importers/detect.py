"""Input-format detection for uploaded portfolio files.

Column/schema-based, not filename-based -- a Trade Republic export renamed
to `portfolio.csv` is still detected as a Trade Republic export, and vice
versa. Detection only reads the header row; it never inspects file names or
row content beyond the columns present.

To add a new broker: add its own `REQUIRED_COLUMNS` to its adapter module,
then add one `elif` branch here (see the module docstring in
`portfolio_importers/__init__.py` and the README's "how to add a new broker
adapter" section for the full checklist).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

import portfolio as portfolio_mod
from . import trade_republic

InputFormat = Literal["canonical", "trade_republic", "unknown"]


def detect_format(df: pd.DataFrame) -> InputFormat:
    """Determine which importer, if any, can parse this DataFrame's columns.

    Trade Republic's schema is checked first: it's the more specific/larger
    required-column set, so checking it first avoids a coincidental subset
    match against the (smaller) canonical schema. In practice the two schemas
    don't overlap at all in this codebase's current adapters (canonical has
    no `datetime`/`shares`/`type`; Trade Republic has no `ticker`/
    `average_cost`), but the ordering is kept explicit for when a future
    broker's schema is a superset of another's.
    """
    columns = set(df.columns)

    if set(trade_republic.REQUIRED_COLUMNS).issubset(columns):
        return "trade_republic"

    if set(portfolio_mod.REQUIRED_COLUMNS).issubset(columns):
        return "canonical"

    return "unknown"
