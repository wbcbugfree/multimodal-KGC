import os
import csv

def parse_triples(file_path: str) -> tuple[set[str], set[str]]:
    """
    Parse a TTL file into two sets: one for node triples and one for edge triples.
    Lines starting with '@' (prefix declarations) and '#' (comments) are ignored.
    
    Returns:
        tuple: (node_triples, edge_triples)
    """
    node_triples = []
    edge_triples = []
    
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
        
        # Classify as node or edge based on patterns
        # Nodes: contain "/node/" and typically have "a d2g:Node" or node type declarations
        # Edges: contain "/edge/" or relationship predicates like "d2g:follows", "d2g:branches"
        if '/node/' in norm and '/edge/' not in norm:
            # Check if it's a direct relationship statement (e.g., "node/1 d2g:follows node/2")
            # These should be classified as edges even though they contain "/node/"
            if any(rel in norm for rel in ['d2g:follows', 'd2g:branches', 'd2g:source', 'd2g:target']):
                edge_triples.append(norm)
            else:
                node_triples.append(norm)
        elif '/edge/' in norm or any(rel in norm for rel in ['d2g:follows', 'd2g:branches', 'd2g:source', 'd2g:target']):
            edge_triples.append(norm)
        else:
            # Default: if unsure, treat as node triple
            node_triples.append(norm)
    
    return set(node_triples), set(edge_triples)

def calculate_metrics(label_set: set, output_set: set) -> dict:
    """
    Calculate precision, recall, F1-score, and Jaccard similarity.
    
    Returns:
        dict with keys: precision, recall, f1, jaccard, overlap, label_count, output_count
    """
    overlap = label_set & output_set
    
    precision = len(overlap) / len(output_set) if output_set else 0
    recall = len(overlap) / len(label_set) if label_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    jaccard = len(overlap) / len(label_set | output_set) if (label_set | output_set) else 0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'jaccard': round(jaccard, 4),
        'overlap': len(overlap),
        'label_count': len(label_set),
        'output_count': len(output_set)
    }

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
        
        # Parse nodes and edges separately
        label_nodes, label_edges = parse_triples(label_path)
        output_nodes, output_edges = parse_triples(output_path)
        
        # Calculate metrics for nodes
        node_metrics = calculate_metrics(label_nodes, output_nodes)
        
        # Calculate metrics for edges
        edge_metrics = calculate_metrics(label_edges, output_edges)
        
        # Calculate overall metrics (combining nodes and edges)
        all_label_triples = label_nodes | label_edges
        all_output_triples = output_nodes | output_edges
        overall_metrics = calculate_metrics(all_label_triples, all_output_triples)
        
        rows.append([
            name.replace('.ttl', ''),
            # Node metrics
            node_metrics['label_count'],
            node_metrics['output_count'],
            node_metrics['overlap'],
            node_metrics['precision'],
            node_metrics['recall'],
            node_metrics['f1'],
            node_metrics['jaccard'],
            # Edge metrics
            edge_metrics['label_count'],
            edge_metrics['output_count'],
            edge_metrics['overlap'],
            edge_metrics['precision'],
            edge_metrics['recall'],
            edge_metrics['f1'],
            edge_metrics['jaccard'],
            # Overall metrics
            overall_metrics['label_count'],
            overall_metrics['output_count'],
            overall_metrics['overlap'],
            overall_metrics['precision'],
            overall_metrics['recall'],
            overall_metrics['f1'],
            overall_metrics['jaccard']
        ])

# Sort rows numerically by image name if possible
def sort_key(x):
    try:
        return int(x[0])
    except ValueError:
        return x[0]

rows.sort(key=sort_key)

# Write results to CSV
with open('ZeroShot_outputs_node_edge_separate.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'image',
        # Node columns
        'node_label_triples',
        'node_output_triples',
        'node_overlap_triples',
        'node_precision',
        'node_recall',
        'node_f1',
        'node_jaccard',
        # Edge columns
        'edge_label_triples',
        'edge_output_triples',
        'edge_overlap_triples',
        'edge_precision',
        'edge_recall',
        'edge_f1',
        'edge_jaccard',
        # Overall columns
        'overall_label_triples',
        'overall_output_triples',
        'overall_overlap_triples',
        'overall_precision',
        'overall_recall',
        'overall_f1',
        'overall_jaccard'
    ])
    writer.writerows(rows)

print("Evaluation complete! Results saved to 'FewShot_outputs_node_edge_separate.csv'")

# Optional: Print summary statistics
if rows:
    print("\n=== Summary Statistics ===")
    node_f1_avg = sum(row[6] for row in rows) / len(rows)
    edge_f1_avg = sum(row[13] for row in rows) / len(rows)
    overall_f1_avg = sum(row[20] for row in rows) / len(rows)
    
    print(f"Average Node F1-Score: {node_f1_avg:.4f}")
    print(f"Average Edge F1-Score: {edge_f1_avg:.4f}")
    print(f"Average Overall F1-Score: {overall_f1_avg:.4f}")