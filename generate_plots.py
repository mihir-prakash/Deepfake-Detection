import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.style.use('bmh')


os.makedirs('plots', exist_ok=True)

#  Known Deepfakes Performance
methods = ['Bloom Filter\n(Hierarchical)', 'Standard\nBloom Filter', 'SHA-256', 'Perceptual\nHashing']
false_positives = [0.07, 0.10, 0.15, 0.12]
query_times = [8, 9, 15, 12]

#  Modified Deepfakes Performance
similarity_methods = ['LSH', 'SVM', 'PCA']
f1_scores = [0.92, 0.88, 0.85]
query_times_similarity = [45, 120, 95]

#  BF Size vs FPR
filter_sizes = [1000, 2000, 3000, 4000, 5000]
fpr_values = [0.15, 0.10, 0.07, 0.05, 0.03]

def plot_false_positive_rates():
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(methods, false_positives, color='skyblue')
    plt.title('False Positive Rates by Method', pad=20, fontsize=14)
    plt.ylabel('False Positive Rate', fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/false_positive_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: false_positive_rates.png")

def plot_query_times_known():
    plt.figure(figsize=(10, 6))
    bars2 = plt.bar(methods, query_times, color='lightgreen')
    plt.title('Query Time by Method', pad=20, fontsize=14)
    plt.ylabel('Query Time (ms)', fontsize=12)
    plt.xticks(rotation=45)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/query_times_known.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: query_times_known.png")

def plot_f1_scores():
    plt.figure(figsize=(10, 6))
    bars3 = plt.bar(similarity_methods, f1_scores, color='lightcoral')
    plt.title('F1 Scores for Modified Deepfake Detection', pad=20, fontsize=14)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0.8, 1.0) 
    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: f1_scores.png")

def plot_query_times_similarity():
    plt.figure(figsize=(10, 6))
    bars4 = plt.bar(similarity_methods, query_times_similarity, color='plum')
    plt.title('Query Time for Similarity Detection', pad=20, fontsize=14)
    plt.ylabel('Query Time (ms)', fontsize=12)
    for bar in bars4:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('plots/query_times_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: query_times_similarity.png")

def plot_fpr_vs_filter_size():
    plt.figure(figsize=(10, 6))
    plt.plot(filter_sizes, fpr_values, marker='o', color='navy', linewidth=2)
    plt.title('False Positive Rate vs. Bloom Filter Size', pad=20, fontsize=14)
    plt.xlabel('Bloom Filter Size', fontsize=12)
    plt.ylabel('False Positive Rate (FPR)', fontsize=12)
    plt.grid(True)
    
    for x, y in zip(filter_sizes, fpr_values):
        plt.annotate(f'{y:.2%}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.tight_layout()
    plt.savefig('plots/fpr_vs_filter_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plot: fpr_vs_filter_size.png")

def main():
    print("Generating plots...")
    plot_false_positive_rates()
    plot_query_times_known()
    plot_f1_scores()
    plot_query_times_similarity()
    plot_fpr_vs_filter_size()
    print("All plots saved in the 'plots' directory.")

if __name__ == "__main__":
    main()