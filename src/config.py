"""Application-level constants and configuration objects."""

import logging
from dataclasses import dataclass

BASE = "https://api.openf1.org/v1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime settings used by the Streamlit app."""

    default_year: int = 2026
    default_country_filter: str = "Australia"
    cache_ttl_short: int = 300
    cache_ttl_long: int = 600


APP_CONFIG = AppConfig()
PLOT_CONFIG = {"displaylogo": False}
