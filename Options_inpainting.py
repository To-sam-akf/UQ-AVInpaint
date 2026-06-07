from base_options import BaseOptions


_CACHED_CONFIG = None


def Inpainting_Config(force_reload=False, args=None):
    global _CACHED_CONFIG
    if _CACHED_CONFIG is None or force_reload:
        _CACHED_CONFIG = BaseOptions().parse(args=args)
    return _CACHED_CONFIG

