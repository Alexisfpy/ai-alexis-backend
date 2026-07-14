import litellm

def apply_litellm_patch():
    origen_completion = litellm.completion

    def mi_completion_ultra_parcheada(*args, **kwargs):
        if 'messages' in kwargs and isinstance(kwargs['messages'], list):
            for msg in kwargs['messages']:
                if isinstance(msg, dict):
                    msg.pop('cache_breakpoint', None)
                    msg.pop('cache_control', None)
                    msg.pop('cache_control_immutable', None)
        if 'cache' in kwargs:
            del kwargs['cache']
        return origen_completion(*args, **kwargs)

    litellm.completion = mi_completion_ultra_parcheada