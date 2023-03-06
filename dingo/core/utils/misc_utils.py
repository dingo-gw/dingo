def get_version():
    try:
        from dingo import __version__
        return __version__
    except ImportError:
        return None
