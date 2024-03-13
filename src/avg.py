import numpy as np

def read_txt(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines

lines = []
for i in range(1, 6):
    file_path = f'./log/{str(i)}/2e-050.0001/LOG_bert-base-uncased_BERTLR_2.000000e-05_LR_1.000000e-04_BS_4'
    line = read_txt(file_path)
    lines.extend(line[-9:])


precisions_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}
recalls_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}
f1s_subreddit = {'all':[], 'android':[], 'apple':[], 'technology':[], 'dota2':[], 'playstation':[], 'movies':[], 'nba':[]}

for line in lines:
    line = line.split(',')
    subreddit = line[1]
    precision = float(line[2].split(':')[1])
    recall = float(line[3].split(':')[1])
    f1 = float(line[4].split(':')[1])
    if 'doc_acc' in subreddit:
        precisions_subreddit['all'].append(precision)
        recalls_subreddit['all'].append(recall)
        f1s_subreddit['all'].append(f1)
    else:
        subreddit = subreddit.strip().split(' ')[1].strip()
        precisions_subreddit[subreddit].append(precision)
        recalls_subreddit[subreddit].append(recall)
        f1s_subreddit[subreddit].append(f1)

output_lines = []
for subreddit in precisions_subreddit.keys():
    print(subreddit)
    output_lines.append('{},{:.4f},{:.4f},{:.4f}'.format(subreddit, np.mean(precisions_subreddit[subreddit]), np.mean(recalls_subreddit[subreddit]), np.mean(f1s_subreddit[subreddit])))


with open('./log/result.txt', 'w') as f:
    f.write('\n'.join(output_lines))
