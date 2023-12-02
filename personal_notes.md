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
python3 multitask_classifier.py --option finetune --epochs=3 --lr=1e-5 --batch_size=64 --use_gpu
```
- sst train set size = 8545
- para train set size = 141507
- sts train set size = 6041

预训练结果：
- dev sentiment acc : 0.411
- dev paraphrase acc : 0.375
- dev sts corr : 0.260
- average performance: 0.472

微调结果 (10 epochs)：
- dev sentiment acc : 0.465
- dev paraphrase acc : 0.435
- dev sts corr : 0.365
- average performance: 0.538