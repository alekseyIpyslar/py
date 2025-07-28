from .text_has_link import HasLinkFilter
from .check_via_bot import ViaBotFilter

# We make it so that we can just import
# from filters import HasLinkFilter
__all__ = [
    "HasLinkFilter",
    "ViaBotFilter"
]