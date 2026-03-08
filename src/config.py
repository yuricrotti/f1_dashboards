import logging
from dataclasses import dataclass

BASE = "https://api.openf1.org/v1"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    default_year: int = 2026
    default_country_filter: str = "Australia"
    cache_ttl_short: int = 300
    cache_ttl_long: int = 600


APP_CONFIG = AppConfig()
PLOT_CONFIG = {"displaylogo": False}
