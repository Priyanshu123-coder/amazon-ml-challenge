import pandas as pd
import re
import os
import time

def clean_name(series):
    # Vectorized string cleaning for massive speedup
    return series.fillna('').str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

def main():
    start = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, "dataset", "test")
    out_dir = os.path.join(base_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading TEST data...")
    s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t", dtype=str)
    
    print("Cleaning strings...")
    for df in [s1, s2, s3]:
        df['clean_name'] = clean_name(df['business_name'])
        df['clean_country'] = df['country'].fillna('').str.strip().str.upper()
        
    # Drop rows in S2/S3 where clean_name is empty to avoid massive cartesian products
    s2 = s2[s2['clean_name'] != '']
    s3 = s3[s3['clean_name'] != '']
        
    print("Finding matches (Vectorized Join)...")
    m12 = pd.merge(s1[['entity_id', 'clean_country', 'clean_name']], 
                   s2[['entity_id', 'clean_country', 'clean_name']], 
                   on=['clean_country', 'clean_name'], 
                   how='inner', suffixes=('_s1', '_s2'))
                   
    m13 = pd.merge(s1[['entity_id', 'clean_country', 'clean_name']], 
                   s3[['entity_id', 'clean_country', 'clean_name']], 
                   on=['clean_country', 'clean_name'], 
                   how='inner', suffixes=('_s1', '_s3'))
    
    print("Aggregating matches...")
    m12 = m12[['entity_id_s1', 'entity_id_s2']].rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_s2': 'matched_id'})
    m13 = m13[['entity_id_s1', 'entity_id_s3']].rename(columns={'entity_id_s1': 'source1_entity_id', 'entity_id_s3': 'matched_id'})
    
    all_matches = pd.concat([m12, m13])
    
    grouped = all_matches.groupby('source1_entity_id')['matched_id'].apply(lambda x: ','.join(sorted(list(set(x))))).reset_index()
    grouped.rename(columns={'matched_id': 'matched_entity_ids'}, inplace=True)
    
    print("Formatting final output...")
    final_df = pd.merge(s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'}), 
                        grouped, on='source1_entity_id', how='left')
                        
    final_df['matched_entity_ids'] = final_df['matched_entity_ids'].fillna('')
    
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")
    
    final_df.to_csv(matching_path, sep="\t", index=False)
    final_df.rename(columns={'matched_entity_ids': 'candidate_entity_ids'}).to_csv(candidate_path, sep="\t", index=False)
    
    print(f"MVP script done in {time.time() - start:.2f} seconds!")

if __name__ == "__main__":
    main()
