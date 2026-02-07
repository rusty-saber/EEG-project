"""Model architectures for 4→8 EEG channel expansion."""

from .reve_wrapper import REVEWrapper
from .expansion_module import ChannelExpansionModule, PositionalCrossAttention
from .decoder import TemporalDecoder, SimpleDecoder
from .full_model import ChannelExpansionModel, create_model

__all__ = [
    'REVEWrapper',
    'ChannelExpansionModule',
    'PositionalCrossAttention',
    'TemporalDecoder',
    'SimpleDecoder',
    'ChannelExpansionModel',
    'create_model',
]
