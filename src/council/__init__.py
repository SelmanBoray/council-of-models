"""council-of-models: ask a council of LLMs, let them rank each other, get a synthesis."""

from council.config import CouncilConfig, load_config
from council.council import Council, parse_ranking
from council.models import CouncilRun
from council.store import RunStore

__version__ = "0.1.0"

__all__ = [
    "Council",
    "CouncilConfig",
    "CouncilRun",
    "RunStore",
    "__version__",
    "load_config",
    "parse_ranking",
]
