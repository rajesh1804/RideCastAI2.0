# first line: 152
@memory.cache
def cached_fare_prediction(x_dict_serialized):
    global _fare_model
    CACHE_STATS["fare_misses"] += 1  # Joblib only calls this if cache MISS
    return _fare_model.predict_one(x_dict_serialized)
