import streamlit.watcher.local_sources_watcher as watcher
import types

original_get_module_paths = watcher.get_module_paths

def safe_get_module_paths(module):
    # Skip any problematic module (like torch.classes)
    if module.__name__.startswith("torch.classes"):
        return []
    try:
        return original_get_module_paths(module)
    except Exception:
        return []

watcher.get_module_paths = safe_get_module_paths
