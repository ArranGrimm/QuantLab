import os
import re

def get_date_from_filename(filename):
    match = re.search(r'\[(\d{8})\]', filename)
    if match:
        return match.group(1)
    return "00000000"

def merge_docs():
    source_dir = 'markdown/docs'
    output_file = 'combined_docs.md'
    
    # Get all markdown files
    files = [f for f in os.listdir(source_dir) if f.endswith('.md')]
    
    # Sort files by date extracted from filename
    files.sort(key=get_date_from_filename)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in files:
            file_path = os.path.join(source_dir, filename)
            
            # Use filename without extension as title
            title = os.path.splitext(filename)[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                outfile.write(f"# {title}\n\n")
                outfile.write(content)
                outfile.write("\n\n---\n\n") # Separator between articles
                
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    merge_docs()
