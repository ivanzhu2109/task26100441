"""Exception types for the model handle lifecycle pool."""


class ModelPoolError(Exception):
    """Base class for all model pool errors."""


class CapacityExceeded(ModelPoolError):
    """No free resource slot; the activation was rejected without queuing."""


class Superseded(ModelPoolError):
    """The activation generation was deactivated before it became usable."""


class NotActive(ModelPoolError):
    """The model has no currently activated handle to lease."""
