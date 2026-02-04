import argparse
import yaml
from pathlib import Path
from collections import defaultdict


def load_citations(yaml_file):
    """Load citations from a YAML file and build a lookup dict by unique_context_marker."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    citations_dict = {}
    for citation in data.get('citations', []):
        marker = citation.get('unique_context_marker', '')
        citations_dict[marker] = citation

    return citations_dict


def evaluate_predictions(ground_truth_dir, prediction_dirs):
    """
    Evaluate prediction directories against ground truth.

    Returns:
        dict: {method_name: {'intent_correct': int, 'topic_correct': int, 'total': int}}
    """
    gt_dir = Path(ground_truth_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {ground_truth_dir}")

    # Get all YAML files in ground truth directory
    gt_files = sorted(gt_dir.glob("*.yaml"))
    if not gt_files:
        raise ValueError(f"No YAML files found in ground truth directory: {ground_truth_dir}")

    print(f"Found {len(gt_files)} ground truth files in {gt_dir.name}")

    results = {}

    for pred_dir_path in prediction_dirs:
        pred_dir = Path(pred_dir_path)
        if not pred_dir.exists():
            print(f"Warning: Prediction directory not found: {pred_dir_path}, skipping...")
            continue

        method_name = pred_dir.name
        intent_correct = 0
        topic_correct = 0
        total = 0

        print(f"\nEvaluating {method_name}...")

        for gt_file in gt_files:
            pred_file = pred_dir / gt_file.name

            if not pred_file.exists():
                raise FileNotFoundError(
                    f"ERROR: Prediction file missing for {gt_file.name} in {method_name}\n"
                    f"Expected: {pred_file}\n"
                    f"Ground truth and prediction files must match exactly!"
                )

            # Load citations
            gt_citations = load_citations(gt_file)
            pred_citations = load_citations(pred_file)

            # Check that all ground truth citations exist in predictions
            for marker in gt_citations.keys():
                if marker not in pred_citations:
                    raise ValueError(
                        f"ERROR: Citation {marker} from {gt_file.name} not found in prediction {method_name}\n"
                        f"All citations must be present in both ground truth and predictions!"
                    )

            # Compare citations
            for marker, gt_citation in gt_citations.items():
                pred_citation = pred_citations[marker]

                # Assert block_ids are identical
                gt_blocks = gt_citation.get('block_ids', [])
                pred_blocks = pred_citation.get('block_ids', [])

                assert gt_blocks == pred_blocks, (
                    f"ERROR: block_ids mismatch for citation {marker} in {gt_file.name}\n"
                    f"Ground truth block_ids: {gt_blocks}\n"
                    f"Prediction block_ids: {pred_blocks}\n"
                    f"block_ids must be identical!"
                )

                # Get labels
                gt_intent_labels = gt_citation.get('intent_labels', [])
                pred_intent_labels = pred_citation.get('intent_labels', [])
                gt_topic_labels = gt_citation.get('topic_labels', [])
                pred_topic_labels = pred_citation.get('topic_labels', [])

                # Assert label lengths match block_ids length
                assert len(gt_intent_labels) == len(gt_blocks), (
                    f"ERROR: GT intent_labels length mismatch in {gt_file.name}, {marker}"
                )
                assert len(pred_intent_labels) == len(gt_blocks), (
                    f"ERROR: Prediction intent_labels length mismatch in {gt_file.name}, {marker}"
                )
                assert len(gt_topic_labels) == len(gt_blocks), (
                    f"ERROR: GT topic_labels length mismatch in {gt_file.name}, {marker}"
                )
                assert len(pred_topic_labels) == len(gt_blocks), (
                    f"ERROR: Prediction topic_labels length mismatch in {gt_file.name}, {marker}"
                )

                # Compare labels element by element
                for i in range(len(gt_blocks)):
                    total += 1

                    if gt_intent_labels[i] == pred_intent_labels[i]:
                        intent_correct += 1

                    if gt_topic_labels[i] == pred_topic_labels[i]:
                        topic_correct += 1

        results[method_name] = {
            'intent_correct': intent_correct,
            'topic_correct': topic_correct,
            'total': total
        }

    return results


def print_results(results):
    """Print evaluation results in a nice table format."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Header
    print(f"{'Method':<45} {'Intent Accuracy':>15} {'Topic Accuracy':>15}")
    print("-"*80)

    # Sort by intent accuracy (descending)
    sorted_methods = sorted(results.items(),
                           key=lambda x: x[1]['intent_correct'] / x[1]['total'] if x[1]['total'] > 0 else 0,
                           reverse=True)

    for method_name, stats in sorted_methods:
        total = stats['total']
        if total == 0:
            intent_acc = 0.0
            topic_acc = 0.0
        else:
            intent_acc = (stats['intent_correct'] / total) * 100
            topic_acc = (stats['topic_correct'] / total) * 100

        print(f"{method_name:<45} {intent_acc:>14.2f}% {topic_acc:>14.2f}%")

    print("="*80)

    # Print total stats
    if sorted_methods:
        total_samples = sorted_methods[0][1]['total']
        print(f"\nTotal samples evaluated: {total_samples}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate citation classification methods against ground truth")
    parser.add_argument("--ground_truth_dir", required=True,
                       help="Directory containing ground truth YAML files")
    parser.add_argument("--prediction_dirs", required=True, nargs='+',
                       help="One or more directories containing prediction YAML files")
    args = parser.parse_args()

    try:
        results = evaluate_predictions(args.ground_truth_dir, args.prediction_dirs)
        print_results(results)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"EVALUATION FAILED")
        print(f"{'='*80}")
        print(f"{e}")
        print(f"{'='*80}\n")
        exit(1)


if __name__ == "__main__":
    main()
