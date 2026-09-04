"""Model layer: the CTC-headed XEUS encoder and its factory."""

from .xeus_ctc import XeusCTC, make_model

__all__ = ["XeusCTC", "make_model"]
