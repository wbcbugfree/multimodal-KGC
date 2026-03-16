import os
import csv

def parse_triples(file_path: str) -> set[str]:
    """
    Parse a TTL file into a set of normalized triple statements.
    Lines starting with '@' (prefix declarations) and '#' (comments) are ignored.
    Each statement is split on '.' to isolate triples, and whitespace is normalized.
    """
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Filter out prefix/comment lines
    lines = []
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('@') or s.startswith('#'):
            continue
        lines.append(line)
    text = '\n'.join(lines)
    # Split on '.' to get individual triple-like statements
    statements = text.split('.')
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt:
            continue
        # Normalize whitespace to a single space
        norm = ' '.join(stmt.split())
        triples.append(norm)
    return set(triples)

# Paths to the extracted folders
label_dir = '/Users/ali/Desktop/RA/Wageningen/Github/Uploading/V10/Soil-Health/diagram2graph Dataset/Evaluating/F1_Score/out_folder'
output_dir = '/Users/ali/Desktop/RA/Wageningen/Github/Uploading/V10/Soil-Health/diagram2graph Dataset/Evaluating/F1_Score/ZeroShot_outputs'

# Map filenames to full paths
label_files = [f for f in os.listdir(label_dir) if f.endswith('.ttl')]
output_files = [f for f in os.listdir(output_dir) if f.endswith('.ttl')]
label_map = {f: os.path.join(label_dir, f) for f in label_files}
output_map = {f: os.path.join(output_dir, f) for f in output_files}

rows = []
for name, label_path in label_map.items():
    # Only compute metrics if there is a corresponding output TTL
    if name in output_map:
        output_path = output_map[name]
        label_triples  = parse_triples(label_path)
        output_triples = parse_triples(output_path)
        overlap = label_triples & output_triples
        # Precision, recall, F1-score, and Jaccard similarity
        precision = len(overlap) / len(output_triples) if output_triples else 0
        recall    = len(overlap) / len(label_triples)  if label_triples  else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        jaccard = len(overlap) / len(label_triples | output_triples) if (label_triples | output_triples) else 0
        rows.append([
            name.replace('.ttl', ''),
            len(label_triples),
            len(output_triples),
            len(overlap),
            round(precision, 4),
            round(recall,    4),
            round(f1,        4),
            round(jaccard,   4)
        ])

# Sort rows numerically by image name if possible
def sort_key(x):
    try:
        return int(x[0])
    except ValueError:
        return x[0]

rows.sort(key=sort_key)

# Write results to CSV
with open('ZeroShot_outputs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'image',
        'label_triples',
        'output_triples',
        'overlap_triples',
        'precision',
        'recall',
        'f1',
        'jaccard'
    ])
    writer.writerows(rows)
