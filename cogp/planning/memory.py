"""Legal joint reinitialization of designated history on a prediction clone."""
from __future__ import annotations
import copy
import numpy as np

FIELDS = ('_stage_active', '_stage_armed', '_stage_target', '_stage_hold',
          '_prev_high_collect_count', '_prev_high_roles', '_prev_high_targets',
          '_rear_barrier_active', '_rear_barrier_hold', '_rear_barrier_start_primitive')


def _equal(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    if isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)) and np.isnan(a) and np.isnan(b):return True
    return a == b


def reinitialize_history(controller, shepherds):
    """Run retained initializers on an isolated shell; copy only the whitelist."""
    if not all(hasattr(controller, key) for key in FIELDS):
        raise RuntimeError('Incomplete designated controller history')
    memo={id(model):model for model in (getattr(controller,'collect_model',None),getattr(controller,'drive_model',None)) if model is not None}
    before = {key: copy.deepcopy(value,memo) for key, value in vars(controller).items()
              if key not in FIELDS and key not in ('collect_model', 'drive_model')}
    previous = {key: copy.deepcopy(getattr(controller, key)) for key in FIELDS}
    shell = copy.copy(controller)
    shell.reset_staging(int(shepherds))
    shell.reset_rear_barrier(int(shepherds))
    for key in FIELDS:
        setattr(controller, key, copy.deepcopy(getattr(shell, key)))
    # Initializers only assign into the shell. No diagnostic/phase/safety reset is copied.
    for key, value in before.items():
        current = getattr(controller, key)
        if isinstance(value, (np.ndarray, bool, int, float, str, type(None))) and not _equal(value, current):
            raise RuntimeError('History intervention changed protected field: ' + key)
    changed = [key for key in FIELDS if not _equal(previous[key], getattr(controller, key))]
    return {'changed_fields': changed, 'changed': bool(changed), 'group': list(FIELDS)}
