import pandas as pd
import re
import os
import time

def clean_text(text):
    if pd.isna(text):
        return ""
    # Lowercase and remove punctuation
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(data_dir, mode="train"):
    print(f"Loading {mode} data from {data_dir}...")
    s1 = pd.read_csv(os.path.join(data_dir, f"{mode}_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, f"{mode}_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, f"{mode}_source3.tsv"), sep="\t", dtype=str)
    
    # Pre-process columns for joining
    for df in [s1, s2, s3]:
        df['clean_name'] = df['business_name'].apply(clean_text)
        df['country'] = df['country'].fillna('').str.strip().str.upper()
        
    return s1, s2, s3

def generate_exact_matches(s1, s2, s3):
    print("Generating exact-match baseline...")
    # Blocking Strategy: Exact match on (Country + Cleaned Business Name)
    
    # 1. Build a lookup index for S2 and S3
    lookup = {}
    
    def add_to_index(df):
        for _, row in df.iterrows():
            key = (row['country'], row['clean_name'])
            if not key[1]: # skip empty names
                continue
            if key not in lookup:
                lookup[key] = []
            lookup[key].append(row['entity_id'])
            
    add_to_index(s2)
    add_to_index(s3)
    
    # 2. Find matches for S1
    matches = []
    candidates = []
    
    for _, row in s1.iterrows():
        s1_id = row['entity_id']
        key = (row['country'], row['clean_name'])
        
        matched_ids = lookup.get(key, [])
        # We must avoid duplicate IDs in the list (rules requirement)
        matched_ids = list(set(matched_ids))
        
        matches.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })
        
        # For this naive baseline, candidates = matches
        candidates.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(matched_ids)
        })
        
    return pd.DataFrame(matches), pd.DataFrame(candidates)

if __name__ == "__main__":
    start_time = time.time()
    
    # Set directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, "dataset", "train")
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load training data
    s1, s2, s3 = load_data(train_dir, mode="train")
    
    # --- Local Validation Split ---
    # We hold out 15% of S1 as our validation set
    val_s1 = s1.sample(frac=0.15, random_state=42).copy()
    print(f"Validation set created: {len(val_s1)} Source 1 entities.")
    
    # Run the naive baseline
    matches_df, candidates_df = generate_exact_matches(val_s1, s2, s3)
    
    # Save the output
    matches_df.to_csv(os.path.join(out_dir, "val_matching_results.tsv"), sep="\t", index=False)
    candidates_df.to_csv(os.path.join(out_dir, "val_candidate_pairs.tsv"), sep="\t", index=False)
    
    print(f"Outputs saved to {out_dir}")
    
    # --- Evaluate F0.5 ---
    try:
        from evaluation import macro_f05
        # Load ground truth
        gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str)
        # Filter ground truth to only our validation split
        val_gt = gt[gt['source1_entity_id'].isin(val_s1['entity_id'])]
        
        score = macro_f05(val_gt, matches_df)
        print("-" * 30)
        print(f"Baseline F0.5 Score on Validation Split: {score:.4f}")
        print("-" * 30)
        
    except ImportError:
        print("evaluation.py not found. Skipping scoring.")
        
    print(f"Finished in {time.time() - start_time:.2f} seconds.")
