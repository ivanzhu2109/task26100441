"""Exception types raised by :class:`model_pool.ModelPool`."""


class CapacityExceeded(Exception):
    """No free resource slot was available for a new load."""


class Superseded(Exception):
    """An ``activate`` wait was ended because the activation was revoked."""


class NotActive(Exception):
    """A lease was requested for a model that is not currently activated."""
