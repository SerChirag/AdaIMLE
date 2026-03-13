from helpers.improved_precision_recall import compute_prec_recall
precision, recall = compute_prec_recall('/localscratch/cva19/AdaIMLE/datasets/celeba', './new-vanilla-results/celeba-convnext-ddp-dino-fix-correctsch/train/fid', num_samples=50000)
print('Precision:', precision)
print('Recall:', recall)