"""Kaggle integration for imggenhub."""

# Patch kagglesdk to fix User-Agent: None bug
try:
    import kagglesdk.kaggle_client
    import functools

    _original_init = kagglesdk.kaggle_client.KaggleClient.__init__

    @functools.wraps(_original_init)
    def _patched_init(self, *args, **kwargs):
        # The 6th argument (index 5) is user_agent
        if len(args) < 6 and 'user_agent' not in kwargs:
            kwargs['user_agent'] = 'kaggle-api/v1.7.0'
        return _original_init(self, *args, **kwargs)

    kagglesdk.kaggle_client.KaggleClient.__init__ = _patched_init
except ImportError:
    pass
