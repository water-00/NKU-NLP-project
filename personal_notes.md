预训练/微调命令：

```
python3 classifier.py --option pretrain --epochs=20 --lr=1e-3 --batch_size=64 --use_gpu
python3 classifier.py --option finetune --epochs=5 --lr=1e-5 --batch_size=32 --use_gpu
```

pretrain：
- SST dev acc = 0.423
- CFIMDB dev acc = 0.841

finetune:
- SST dev acc = 0.510
- CFIMDB dev acc = 0.963

有一个很有意思的问题：pretrain模式下`param.requires_grad = False`，那为什么pretrain下每个epoch的dev acc还会有提升？