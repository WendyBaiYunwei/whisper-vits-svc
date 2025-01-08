import Levenshtein as lev

# Example texts
with open('imagine.txt', 'r') as f:
    ground_truth = f.read()

predicteds = []
for i, idx in enumerate(['', '_beaut_25', '_beaut_50', '_beaut_75', '_beaut_100']):
    with open(f'imagine_bad{idx}.txt', 'r') as f:
        predicteds.append(f.read())

for predicted in predicteds:
    # Calculate WER
    gt_words = ground_truth.split()
    pred_words = predicted.split()
    word_error_rate = lev.distance(" ".join(gt_words), " ".join(pred_words)) / len(gt_words)

    # Calculate CER
    char_error_rate = lev.distance(ground_truth, predicted) / len(ground_truth)

    print(f"WER: {word_error_rate:.2f}")
    print(f"CER: {char_error_rate:.2f}")
