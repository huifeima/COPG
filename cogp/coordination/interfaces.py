"""Shared topology task interface for every learned allocation mechanism."""
from .core import HCCGTaskBuilder
from .snapshot import AuthoritativeTaskBuilder

def builder_for(method, config):
    if method not in ('HCCG', 'EdgeGNN', 'AgentDAG', 'Geometry'):
        raise ValueError('Unknown task generator: '+str(method))
    return AuthoritativeTaskBuilder(HCCGTaskBuilder(
        world_size=config.world_size, social_radius=config.sheep_social_radius,
        sheep_max_neighbors=config.sheep_max_neighbors,
        sheep_per_pressure_slot=4, max_pressure_slots_per_component=3,
        outlier_radius=75.0))
