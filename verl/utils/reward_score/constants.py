LENGTH_THRESHOLDS = list(range(2048, 16384 + 1024, 1024))

MONITORH_FIELD = "accuracy_bands_monitor"
MONITORH_FIELD_FINED = "accuracy_bands_fined_monitor"

TRAINING_MONITORH_FIELD = "accuracy_bands_training"
TRAINING_MONITORH_FIELD_FINED = "accuracy_bands_fined_training"


DIFFICULTY_LEVEL = ["low", "medium", "high"]

def get_length_threshold(lower_bound: int, upper_bound: int, interval: int) -> list[int]:
    
    """
    if lower_bound == 512:
        return list(range(1024, upper_bound + interval, interval)) + [512]
    else:
        return list(range(lower_bound, upper_bound + interval, interval))
    """
    
    if lower_bound < 1024:
        obtained_list = [lower_bound] + list(range(1024, upper_bound + interval, interval))
        return obtained_list
    else:
        return list(range(lower_bound, upper_bound + interval, interval))

