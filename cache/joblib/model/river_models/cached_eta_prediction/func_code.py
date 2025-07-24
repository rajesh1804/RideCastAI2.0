# first line: 159
@memory.cache
def cached_eta_prediction(x_dict_serialized):
    global _eta_model
    CACHE_STATS["eta_misses"] += 1  # Called only if cache MISS
    return _eta_model.predict_one(x_dict_serialized)
