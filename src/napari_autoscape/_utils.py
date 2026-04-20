def _find_in_nested_dict(data, key, value):
    """
    Recursively search for objects containing a specific key-value pair.
    Returns a list of all matching objects.
    """
    results = []
    
    def search(obj):
        if isinstance(obj, dict):
            if obj.get(key) == value:
                results.append(obj)
            for v in obj.values():
                search(v)
        elif isinstance(obj, list):
            for item in obj:
                search(item)
    
    search(data)
    return results  