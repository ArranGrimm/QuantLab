import re

source_file = 'markdown/main/2024年文章汇总.md'
output_file = 'article_links.txt'

try:
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to capture the URL part of markdown links [text](url)
    # Using the same logic as before to handle potential newlines
    pattern = r'\[.*?\]\s*\((.*?)\)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    links = []
    for url in matches:
        url = url.strip()
        # Filter to keep only article links (WeChat MP articles)
        # Excluding images and javascript links
        if 'mp.weixin.qq.com/s' in url:
            links.append(url)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for link in links:
            f.write(f"{link}\n")
            
    print(f"Successfully extracted {len(links)} links to {output_file}")

except FileNotFoundError:
    print(f"File not found: {source_file}")
except Exception as e:
    print(f"Error: {e}")

