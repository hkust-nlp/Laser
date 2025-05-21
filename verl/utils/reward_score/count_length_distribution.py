from collections import defaultdict
from verl.utils.reward_score.constants import DIFFICULTY_LEVEL, LENGTH_THRESHOLDS

def _calculate_length_coverage_statistics(lengths, length_thresholds, prefix):
    """
    Calculate coverage statistics for a list of lengths against various thresholds.
    
    Args:
        lengths: List of token lengths
        length_thresholds: List of threshold values to check coverage against
        prefix: Metric name prefix for the returned statistics
        
    Returns:
        Dictionary of coverage statistics
    """
    metrics = {}
    
    if not lengths:
        for threshold in length_thresholds:
            metrics[f'{prefix}/length_{threshold}'] = 0.0
        return metrics
    
    total_responses = len(lengths)
    for threshold in length_thresholds:
        # Count responses below or equal to the threshold
        covered = sum(1 for length in lengths if length <= threshold)
        coverage_ratio = covered / total_responses
        metrics[f'{prefix}/length_{threshold}'] = coverage_ratio
        
    return metrics

def count_accuracy_bands_metrics(batch, response_length, data_source: str, eval_type: str):
    """
    Calculate metrics for questions with different accuracy bands:
    - Questions with accuracy > 1/3
    - Questions with accuracy between 1/3 and 2/3
    - Questions with accuracy > 2/3
    
    Also calculates the average response length for each band.
    """
    
    index = batch.non_tensor_batch['uid']
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # Group correctness and lengths by question ID
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    total_questions = len(id2correctness)
    if total_questions == 0:
        return {}
    
    # Calculate accuracy for each question
    question_accuracies = {}
    for qid in id2correctness:
        correct_count = sum(id2correctness[qid])
        total_responses = len(id2correctness[qid])
        question_accuracies[qid] = correct_count / total_responses if total_responses > 0 else 0
    
    # Define accuracy bands
    low_threshold = 1/3
    high_threshold = 2/3
    
    # Group questions by accuracy band
    accuracy_bands = {
        DIFFICULTY_LEVEL[0]: [],    # accuracy <= 1/3
        DIFFICULTY_LEVEL[1]: [], # 1/3 < accuracy <= 2/3
        DIFFICULTY_LEVEL[2]: []    # accuracy > 2/3
    }
    
    lengths_by_band = {
        DIFFICULTY_LEVEL[0]: [],
        DIFFICULTY_LEVEL[1]: [],
        DIFFICULTY_LEVEL[2]: []
    }
    
    for qid, accuracy in question_accuracies.items():
        if accuracy <= low_threshold:
            band = DIFFICULTY_LEVEL[0]
        elif accuracy <= high_threshold:
            band = DIFFICULTY_LEVEL[1]
        else:
            band = DIFFICULTY_LEVEL[2]
        
        accuracy_bands[band].append(qid)
        lengths_by_band[band].extend(id2length[qid])
    
    # Calculate metrics
    metrics = {}
    
    # Proportions of questions in each band
    for band, qids in accuracy_bands.items():
        band_ratio = len(qids) / total_questions
        metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_{band}_accuracy'] = band_ratio
    
    # Average, min, max lengths for each band
    for band, lengths in lengths_by_band.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_{band}_accuracy'] = avg_length
            metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_{band}_accuracy'] = min(lengths)
            metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_{band}_accuracy'] = max(lengths)
        else:
            metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_{band}_accuracy'] = 0.0
            metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_{band}_accuracy'] = 0.0
            metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_{band}_accuracy'] = 0.0
    
    # Additional metrics - specific thresholds as requested
    above_one_third = len(accuracy_bands['medium']) + len(accuracy_bands['high'])
    above_two_thirds = len(accuracy_bands['high'])
    
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_above_one_third'] = above_one_third / total_questions
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_between_one_third_two_thirds'] = len(accuracy_bands['medium']) / total_questions
    metrics[f'accuracy_bands_{eval_type}/{data_source}/proportion_above_two_thirds'] = above_two_thirds / total_questions
    
    # Collect all lengths for questions above specific thresholds
    lengths_above_one_third = lengths_by_band['medium'] + lengths_by_band['high']
    lengths_above_two_thirds = lengths_by_band['high']
    
    if lengths_above_one_third:
        metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_above_one_third'] = sum(lengths_above_one_third) / len(lengths_above_one_third)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_above_one_third'] = min(lengths_above_one_third)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_above_one_third'] = max(lengths_above_one_third)
    
    if lengths_above_two_thirds:
        metrics[f'accuracy_bands_{eval_type}/{data_source}/avg_length_above_two_thirds'] = sum(lengths_above_two_thirds) / len(lengths_above_two_thirds)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/min_length_above_two_thirds'] = min(lengths_above_two_thirds)
        metrics[f'accuracy_bands_{eval_type}/{data_source}/max_length_above_two_thirds'] = max(lengths_above_two_thirds)
    
    # Calculate coverage at different length thresholds from 2048 to 16384 with step 1024
    length_thresholds = LENGTH_THRESHOLDS
    
    # Add length coverage statistics
    for band in lengths_by_band:
        coverage_metrics = _calculate_length_coverage_statistics(
            lengths_by_band[band],
            length_thresholds,
            f'accuracy_bands_{eval_type}/{data_source}/coverage_{band}_accuracy'
        )
        metrics.update(coverage_metrics)
    
    # Coverage for above thresholds groups
    above_one_third_coverage = _calculate_length_coverage_statistics(
        lengths_above_one_third,
        length_thresholds,
        f'accuracy_bands_{eval_type}/{data_source}/coverage_above_one_third'
    )
    metrics.update(above_one_third_coverage)
    
    above_two_thirds_coverage = _calculate_length_coverage_statistics(
        lengths_above_two_thirds,
        length_thresholds,
        f'accuracy_bands_{eval_type}/{data_source}/coverage_above_two_thirds'
    )
    metrics.update(above_two_thirds_coverage)
    
    return metrics

def count_accuracy_bands_fined_metrics(batch, response_length, data_source: str, eval_type: str):
    """
    Calculate more fine-grained metrics for questions based on exact number of correct answers:
    - Questions with exactly 0 correct answers
    - Questions with exactly 1 correct answer
    - ...
    - Questions with all correct answers
    
    Also calculates the average response length and coverage statistics for each group.
    """
    
    index = batch.non_tensor_batch['uid']
    correctness_tensor = batch.batch['correctness']  # (batch_size,)
    
    # Group correctness and lengths by question ID
    id2correctness = defaultdict(list)
    id2length = defaultdict(list)
    
    for i in range(len(correctness_tensor)):
        id2correctness[index[i]].append(correctness_tensor[i].item())
        id2length[index[i]].append(response_length[i].item())
    
    total_questions = len(id2correctness)
    if total_questions == 0:
        return {}
    
    # Determine number of rollouts per question
    n_rollouts = len(id2correctness[list(id2correctness.keys())[0]])
    
    # Group questions by exact number of correct answers (0 to n_rollouts)
    accuracy_counts = {i: [] for i in range(n_rollouts + 1)}  # From 0 to n_rollouts
    lengths_by_count = {i: [] for i in range(n_rollouts + 1)}
    
    for qid in id2correctness:
        correct_count = sum(id2correctness[qid])
        accuracy_counts[correct_count].append(qid)
        lengths_by_count[correct_count].extend(id2length[qid])
    
    # Calculate metrics
    metrics = {}
    
    # Proportions of questions in each count group
    for count, qids in accuracy_counts.items():
        count_ratio = len(qids) / total_questions
        metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/proportion_correct_{count}'] = count_ratio
    
    # Average, min, max lengths for each count group
    for count, lengths in lengths_by_count.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/avg_length_correct_{count}'] = avg_length
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/min_length_correct_{count}'] = min(lengths)
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/max_length_correct_{count}'] = max(lengths)
        else:
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/avg_length_correct_{count}'] = 0.0
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/min_length_correct_{count}'] = 0.0
            metrics[f'accuracy_bands_fined_{eval_type}/{data_source}/max_length_correct_{count}'] = 0.0
    
    # Calculate coverage at different length thresholds
    length_thresholds = LENGTH_THRESHOLDS
    
    # Add length coverage statistics for each count group
    for count, lengths in lengths_by_count.items():
        coverage_metrics = _calculate_length_coverage_statistics(
            lengths,
            length_thresholds,
            f'accuracy_bands_fined_{eval_type}/{data_source}/coverage_correct_{count}'
        )
        metrics.update(coverage_metrics)
    
    return metrics