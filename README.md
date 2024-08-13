# UDT

This is the pytorch implementation of our paper: **Unified Denoising Training for Recommendation**.

> Haoyan Chua,  Yingpeng Du, Zhu Sun, Ziyan Wang and Jie Zhang.

## Environment
- python 3.11.5
- pytorch 2.0.1
- numpy 1.26.3 


## Commands

We provided the code for training and inference for GMF, NeuMF & CDAE:

#### GMF & NeuMF
Go to NCF folder and simply run the code below with default settings to return results shown in the paper:
```
python main.py --model GMF --dataset amazon_book --gpu 0
```
or for Neumf
```
python main.py --model NeuMF-end --dataset movielens --gpu 0
```

#### CDAE
Go to CDAE folder and simply run the code below with default settings to return results shown in the paper:
```
python main_CDAE.py --dataset yelp --gpu 0
```


To change the hyperparameter settings, `--userfact1` & `--userfact2` controls [a , b] in the paper while  `--temp1` & `--temp2` controls [a' , b'].
To save training time and start validation evaluation later, `--epoch_eval` determines the epoch to begin evaluation.

### Example with custom setting
1. Train GMF on Yelp with different hyperparameter settings:
```
python main.py --dataset yelp --model GMF --userfact1 0.05 --userfact2 0.0 --temp1 0.1 --temp2 0.5 --epoch_eval 30 --gpu=0
```

2. Train CDAE on Amazon-book with different hyperparameter settings:
```
python main_CDAE.py --dataset amazon_book --model CDAE --userfact1 1.0 --userfact2 0.5 --temp1 0.5 --temp2 1.0 --epoch_eval 30 --gpu=0
```


## Citation  
If you use our code, please kindly cite:

```

```
