# 2022.0285

[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Fusion Pre-Trained Approach for Identifying the Cause of Sarcasm Remarks

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[A Fusion Pre-Trained Approach for Identifying the Cause of Sarcasm Remarks](https://doi.org/10.1287/ijoc.2022.0285.cd) by Q. Li, D. Xu, H. Qian, L. Wang, M. Yuan and D. Zeng. 

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0285

https://doi.org/10.1287/ijoc.2022.0285.cd

Below is the BibTex for citing this snapshot of the repository.

```
@article{A Fusion Pre-Trained Approach for Identifying the Cause of Sarcasm Remarks,
  author =        {Q. Li, D. Xu, H. Qian, L. Wang, M. Yuan and D. Zeng},
  publisher =     {INFORMS Journal on Computing},
  title =         {A Fusion Pre-Trained Approach for Identifying the Cause of Sarcasm Remarks},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0285.cd},
  url =           {https://github.com/INFORMSJoC/2022.0285},
}  
```

## Description

This repository provides data for the problem and code for the method. The main folders are 'data', 'src', 'scripts', and 'results'.

'data': This folder includes reddit data and twitter data. The detailed description can be seen [README](./data/README.md).

"src": This folder includes the code for training and testing.

"scripts": This folder provides a running script.

"results": This folder provides the results.

## Building

The following packages should be installed before you run our model.

```
python >= 3.8.13
pytorch >= 1.8.0
transformers >= 4.7.0
huggingface-hub >= 0.0.8
```

## Replicating

You should download the pre-trained model [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and put the files under ./models folder.

Then the model can be trained using the [scripts](/scripts). 

```
cd scripts
bash run.sh
```

This script will execute three Python programs.

First, the data could be split into training set, validation set and testing set by five-fold.

```
python kfold_split.py
```

Then, the model can be trained by each training set. The file`train_classifier_linear.py`is the starting program for training. The file `framework.py` includes the main framework for training and evaluating the model. The model is in the file `cross_encoder.py`. The parameter `--test_prefix` can be set to different ratios of data to train the model on partial data. The parameter `--attention_head` can be used to set the number of attention heads.

```
python train_classifier_linear.py
```

 Finally, the average evaluation metric is calculated.

```
python avg.py
```

For testing, this [script](./scripts/test.sh) can be used to infer the results based on the trained model. To test the cross-subreddit and cross-platform data, the parameters `--data_path` and `--test_prefix` should be set to corresponding subreddit or platform name.

```
cd scripts
bash test.sh
```

## Results

The [results](/results)  folder show the tables and figures reported in the paper. 

[Table 3](./results/Table_3.png) shows the overall Precision, Recall and F1 of the proposed model. And [Appendix A](./results/Appendix_A.png) is the corresponding results of each subreddit.

[Table 4](./results/Table_4.png) presents the Precision, Recall and F1 on cross-subreddit and cross-platform. 

[Table 5](./results/Table_5.png) and [Appendix B](./results/Appendix_B.png)  show the evaluation metric of the variant models. 

[Figure 2](./results/Figure_2.png) shows the  F1 scores of different training data ratios.

[Table 6](./results/Table_6.png)  and [Appendix C](./results/Appendix_C.png) are results of different attention heads.

For more detailed analysis, see Section 4.4 and Appendix.  
