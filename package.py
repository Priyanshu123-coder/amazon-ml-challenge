import os
import zipfile
import shutil

# 1. Prepare directories
os.makedirs('submission_temp/output', exist_ok=True)
os.makedirs('submission_temp/code/business_entity_resolution/src', exist_ok=True)

# 2. Copy outputs
shutil.copy('output/matching_results.tsv', 'submission_temp/output/')
shutil.copy('output/candidate_pairs.tsv', 'submission_temp/output/')

# 3. Copy code and docs
for f in os.listdir('src'):
    shutil.copy(f'src/{f}', 'submission_temp/code/business_entity_resolution/src/')
    
shutil.copy('README.md', 'submission_temp/code/business_entity_resolution/')
shutil.copy('requirements.txt', 'submission_temp/code/business_entity_resolution/')
shutil.copy('Documentation_template.md', 'submission_temp/')

# 4. Zip it up to the correct name
zip_name = 'Priyanshu123-coder_submission.zip'
print(f"Creating {zip_name}...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk('submission_temp'):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, 'submission_temp')
            zipf.write(file_path, arcname)
            
shutil.rmtree('submission_temp')
print("Packaging complete! Ready for submission.")
