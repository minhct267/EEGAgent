"""Hold the loaded EEG so tools can slice it without the planner passing the array."""
_register_data = None

def registerData(data):
    """Store the current recording for later tool calls."""
    global _register_data
    _register_data = data

def getRegisteredData(s: int = None, e: int = None, config: dict = None):
    """Return the full array, or samples from second s to e when config supplies fs."""
    if _register_data is None:
        raise RuntimeError("The data is not registered, please call register_data to load the data first")
    
    if s is not None and e is not None and config is not None:
        fs = config['fs']
        return _register_data[:, s*fs:e*fs]
    
    return _register_data