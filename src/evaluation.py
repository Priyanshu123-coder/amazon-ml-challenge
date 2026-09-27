import pandas as pd

def calculate_entity_f05(y_true, y_pred):
    """
    Calculates the F0.5 score for a single Source 1 entity.
    y_true: list/set of true matching Source 2/3 IDs
    y_pred: list/set of predicted matching Source 2/3 IDs
    """
    true_set = set(y_true) if y_true else set()
    pred_set = set(y_pred) if y_pred else set()

    # Singleton logic
    if not true_set and not pred_set:
        return 1.0  # Correctly predicted no matches
    if not true_set and pred_set:
        return 0.0  # False positive on a singleton
    if true_set and not pred_set:
        return 0.0  # Missed all matches
        
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
    return f05

def macro_f05(ground_truth_df, predictions_df):
    """
    Calculates the macro-averaged F0.5 score across all Source 1 entities.
    Expects dataframes with 'source1_entity_id' and 'matched_entity_ids' columns.
    """
    # Merge on source1_entity_id to ensure we align the predictions with truth
    merged = pd.merge(
        ground_truth_df, 
        predictions_df, 
        on='source1_entity_id', 
        how='left', 
        suffixes=('_true', '_pred')
    )
    
    # Fill NaNs with empty strings for singleton handling
    merged['matched_entity_ids_true'] = merged['matched_entity_ids_true'].fillna('')
    merged['matched_entity_ids_pred'] = merged['matched_entity_ids_pred'].fillna('')

    scores = []
    for _, row in merged.iterrows():
        # Split comma-separated strings into lists, handling empty strings
        y_true_str = str(row['matched_entity_ids_true']).strip()
        y_pred_str = str(row['matched_entity_ids_pred']).strip()
        
        y_true = y_true_str.split(',') if y_true_str else []
        y_pred = y_pred_str.split(',') if y_pred_str else []
        
        scores.append(calculate_entity_f05(y_true, y_pred))

    # Macro-average
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == "__main__":
    # Quick sanity check with the example from the PDF
    y_true_example = ["S2-00047", "S3-00812"]
    y_pred_example = ["S2-00047", "S2-00193", "S3-00812"]
    score = calculate_entity_f05(y_true_example, y_pred_example)
    print(f"Example F0.5 Score (should be ~0.714): {score:.3f}")
