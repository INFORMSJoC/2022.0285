import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import json
import os
from collections import Counter


def iloc_list(samples, indexes):
    new_samples = []
    for indx in indexes:
        new_samples.append(samples[indx])
    return new_samples


samples = json.load(open("../data/reddit/reddit_data.json", encoding="utf-8"))
subreddits_label = ["android","apple","nba","movies","playstation","technology","dota2"]
doc_ids = list(range(1, len(samples) + 1))
subreddits = []
new_samples = []
count_num_clause = 0
for sample in samples:
    if sample["subreddit"] in subreddits_label:
        new_samples.append(sample)
        subreddits.append(sample["subreddit"])
subreddit_dict = Counter(subreddits)
print(subreddit_dict)
print(len(new_samples))
print(count_num_clause)
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
i = 1
for not_test_index, test_index in kf.split(X=new_samples, y=subreddits, groups=subreddits):
    not_test_samples, test_samples = iloc_list(samples, not_test_index), iloc_list(samples,test_index)

    not_test_labels = iloc_list(subreddits, not_test_index)
    print(Counter(not_test_labels))
    train_samples, dev_samples, train_labels, dev_labels = train_test_split(not_test_samples, not_test_labels,
                                                                            test_size=0.125, random_state=24,
                                                                            shuffle=True, stratify=not_test_labels)

    print("Train {} Dev {} Test {}".format(len(train_samples), len(dev_samples), len(test_samples)))
    if not os.path.exists("../data/reddit/split/"):
        os.makedirs("../data/reddit/split/")
    if not os.path.exists(os.path.join("../data/reddit/split/", str(i))):
        os.makedirs(os.path.join("../data/reddit/split/", str(i)))
    with open(os.path.join("../data/reddit/split/"+str(i), "train_ids.json"), "w", encoding="utf-8") as output_file:
        json.dump(not_test_samples, output_file, indent=4)
    with open(os.path.join("../data/reddit/split/"+str(i), "dev_ids.json"), "w", encoding="utf-8") as output_file:
        json.dump(test_samples, output_file, indent=4)
    with open(os.path.join("../data/reddit/split/"+str(i), "test_ids.json"), "w", encoding="utf-8") as output_file:
        json.dump(test_samples, output_file, indent=4)
    i += 1
