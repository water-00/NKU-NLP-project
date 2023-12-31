## 单任务

单任务（sst）预训练/微调命令：

```
python3 classifier.py --option pretrain --epochs=10 --lr=1e-3 --batch_size=64 --use_gpu
python3 classifier.py --option finetune --epochs=5 --lr=1e-5 --batch_size=32 --use_gpu
```

pretrain：
- SST dev acc = 0.423
- CFIMDB dev acc = 0.841

finetune:
- SST dev acc = 0.510
- CFIMDB dev acc = 0.963

有一个很有意思的问题：pretrain模式下`param.requires_grad = False`，那为什么pretrain下每个epoch的dev acc还会有提升？

## 多任务
多任务预训练/微调命令：
```
python3 multitask_classifier.py --option pretrain --epochs=10 --lr=1e-3 --batch_size=64 --use_gpu
python3 multitask_classifier.py --option finetune --epochs=10 --lr=1e-5 --batch_size=64 --use_gpu
```
仅测试不训练则添加`--test`参数：
```
python3 multitask_classifier.py --option finetune --epochs=10 --lr=1e-5 --batch_size=64 --use_gpu --test
```
- sst train set size = 8545
- para train set size = 141507
- sts train set size = 6041

全训练集预训练结果：
- dev sentiment acc: 0.411
- dev paraphrase acc: 0.375
- dev sts corr: 0.260
- average performance: 0.349

全训练集微调结果 (10 epochs)：
- dev sentiment acc: 0.480
- dev paraphrase acc: 0.375
- dev sts corr: 0.405
- average performance: 0.420

一般情况下请使用小训练集（添加`--small`参数）：
```
python3 multitask_classifier.py --option finetune --epochs=10 --lr=1e-5 --batch_size=32 --use_gpu --small
```
添加`round-robin`参数：
```
python3 multitask_classifier.py --option finetune --epochs=10 --lr=1e-5 --batch_size=32 --use_gpu --small --rrobin
```

参数使用说明：
- `--test`：仅测试
- `--small`：使用小数据集
- `--rrobin`：使用round-robin算法分配batch_size
- `--smartr`：计算损失函数时使用smart正则化
- `--pre`：在不使用小数据集的情况下使用较大的paraphrase数据集进行预训练
- `--rlayer`：在para和sts之间引入共享层，relational layer

small-naive:
- dev sentiment acc: 0.455
- dev paraphrase acc: 0.375
- dev sts corr: 0.359
- average performance: 0.397

small-with absolute difference and cosine similarity(之后的small都带有这个): 
- dev sentiment acc: 0.450
- dev paraphrase acc: 0.375
- dev sts corr: 0.365
- average performance: 0.397

small-rrobin:
- dev sentiment acc: 0.454
- dev paraphrase acc: 0.375
- dev sts corr: 0.388
- average performance: 0.406

small-rrobin-smart
- dev sentiment acc: 0.457
- dev paraphrase acc: 0.375
- dev sts corr: 0.445
- average performance: 0.426
