import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

import copy


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 定义不同下游任务的分类数
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
# N_PARAPHRASE_CLASSES = 2
# N_SIMILARITY_CLASSES = 5

class RelationalLayer(nn.Module):
    def __init__(self, hidden_size):
        super(RelationalLayer, self).__init__()
        self.transformation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        # x: 来自BERT的特征表示
        return self.transformation(x)

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.relational_layer = RelationalLayer(config.hidden_size) # for rlayer
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        # 为每个任务定义一个分类头
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, N_SENTIMENT_CLASSES) # [0, 1]^5，输出属于每个类别的概率，再计算cross-entropy
        )
        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(3 * config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  
            nn.Sigmoid() # [0, 1]，输出同义的概率，再计算cross-entropy
        )
        self.similarity_classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_size + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
            nn.Linear(1, 1, bias=False)  # 添加一个没有偏置的线性层
        )

        # [0, 1]->[0, 5]，得到相似度计算loss
        self.similarity_classifier[4].weight.data.fill_(5.0)

    def forward(self, input_ids, attention_mask, task_type='sst'):
        'Takes a batch of sentences and produces embeddings for them.'


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs['pooler_output']  # 取CLS标记的输出
        sentiment_logits = self.sentiment_classifier(sequence_output)
        return sentiment_logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        sequence_output_1 = outputs_1['pooler_output']
        sequence_output_2 = outputs_2['pooler_output']
        if (args.rlayer):
            relational_output_1 = self.relational_layer(sequence_output_1)
            relational_output_2 = self.relational_layer(sequence_output_2)
            absolute_diff = torch.abs(relational_output_1 - relational_output_2)
            combined_output = torch.cat((relational_output_1, relational_output_2, absolute_diff), dim=1)
        else:
            # 计算两个嵌入向量的绝对差异，from SBERT
            absolute_diff = torch.abs(sequence_output_1 - sequence_output_2)
            combined_output = torch.cat((sequence_output_1, sequence_output_2, absolute_diff), dim=1) # dim=0是batch_size维度
            
        paraphrase_logits = self.paraphrase_classifier(combined_output)
        return paraphrase_logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        sequence_output_1 = outputs_1['pooler_output']
        sequence_output_2 = outputs_2['pooler_output']
        # 余弦相似度，from SBERT
        cos_sim = F.cosine_similarity(sequence_output_1, sequence_output_2, dim=1).unsqueeze(-1)
        if (args.rlayer):
            relational_output_1 = self.relational_layer(sequence_output_1)
            relational_output_2 = self.relational_layer(sequence_output_2)
            combined_output = torch.cat((relational_output_1, relational_output_2, cos_sim), dim=1)
        else:
            combined_output = torch.cat((sequence_output_1, sequence_output_2, cos_sim), dim=1)
        similarity_logits = self.similarity_classifier(combined_output)
        return similarity_logits



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

#对称kl散度
def symmetric_kl_divergence(output, target):
    # 将logits转换为概率分布
    softmax_output = F.softmax(output, dim=1)
    softmax_target = F.softmax(target, dim=1)

    # 计算P对Q的KL散度
    kl_div1 = F.kl_div(softmax_output.log(), softmax_target, reduction='batchmean')
    # 计算Q对P的KL散度
    kl_div2 = F.kl_div(softmax_target.log(), softmax_output, reduction='batchmean')
    
    # 对称KL散度是两者之和
    skl_div = kl_div1 + kl_div2
    return skl_div

def compute_smart_regularization_kl(model, batch, epsilon, device, predict_function, task_type):
    # for para or sst
    # Save the original parameters of the model
    orig_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Initialize the regularization term
    regularization_term = 0.0
    
    
    # Iterate over all parameters and add perturbation
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Save the original data of the parameter
            orig_data = param.data.clone()
            
            # Generate a random perturbation
            perturbation = torch.randn_like(param) * epsilon
            # Apply perturbation and ensure it's within the epsilon ball
            param.data = orig_data + torch.clamp(perturbation, -epsilon, epsilon)
    
    if (task_type == "sst"):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                        batch['attention_mask'], batch['labels'])
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)
        
        perturbed_outputs = model.predict_sentiment(b_ids, b_mask)
        
    elif (task_type == "para"):
        (b_ids1, b_mask1,
        b_ids2, b_mask2,
        b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                    batch['token_ids_2'], batch['attention_mask_2'],
                    batch['sent_ids'], batch['labels'])

        b_ids1 = b_ids1.to(device)
        b_mask1 = b_mask1.to(device)
        b_ids2 = b_ids2.to(device)
        b_mask2 = b_mask2.to(device)
        b_labels = b_labels.to(device)
        
        perturbed_outputs = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)

    # Calculate the new loss with perturbed parameters
    
    # Restore the original parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = orig_params[name]
    
    # Calculate the loss with original parameters
    if (task_type == "sst"):
        original_outputs = model.predict_sentiment(b_ids, b_mask)
    elif (task_type == "para"):
        original_outputs = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)

    perturbed_loss = symmetric_kl_divergence(perturbed_outputs, original_outputs.detach())
    original_loss = symmetric_kl_divergence(original_outputs, original_outputs.detach())
    
    # Calculate the maximum loss difference across all perturbations
    max_loss_diff = torch.max(perturbed_loss - original_loss)
    
    # The regularization term is the average of the max loss difference
    regularization_term = max_loss_diff / args.batch_size
    
    return regularization_term

def compute_smart_regularization_mse(model, batch, epsilon, device):
    # only for sts
    (b_ids1, b_mask1,
    b_ids2, b_mask2,
    b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                batch['token_ids_2'], batch['attention_mask_2'],
                batch['sent_ids'], batch['labels'])
    
    b_ids1 = b_ids1.to(device)
    b_mask1 = b_mask1.to(device)
    b_ids2 = b_ids2.to(device)
    b_mask2 = b_mask2.to(device)
    b_labels = b_labels.to(device)
    
    # Save the original parameters of the model
    orig_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Initialize the regularization term
    regularization_term = 0.0
    
    # Iterate over all parameters and add perturbation
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Save the original data of the parameter
            orig_data = param.data.clone()
            
            # Generate a random perturbation
            perturbation = torch.randn_like(param) * epsilon
            # Apply perturbation and ensure it's within the epsilon ball
            param.data = orig_data + torch.clamp(perturbation, -epsilon, epsilon)
    
    # Calculate the new loss with perturbed parameters
    perturbed_outputs = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
    perturbed_loss = F.mse_loss(perturbed_outputs.squeeze(), b_labels.float(), reduction='sum') / args.batch_size

    # Restore the original parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = orig_params[name]
    
    # Calculate the loss with original parameters
    original_outputs = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
    original_loss = F.mse_loss(original_outputs.squeeze(), b_labels.float(), reduction='sum') / args.batch_size
    
    # Calculate the maximum loss difference across all perturbations
    max_loss_diff = torch.max(perturbed_loss - original_loss)
    
    # The regularization term is the average of the max loss difference
    regularization_term = max_loss_diff / args.batch_size
    
    return regularization_term

## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args, 'para')
    para_dev_data = SentencePairDataset(para_dev_data, args, 'para')
    sts_train_data = SentencePairDataset(sts_train_data, args, 'sts')
    sts_dev_data = SentencePairDataset(sts_dev_data, args, 'sts')

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    


    if (args.rrobin):
        # for round-robin，需要根据数据集大小重新划分一下每个训练集的batch_size
        smallest_dataset_length = min(len(para_train_dataloader), len(sst_train_dataloader), len(sts_train_dataloader))
        smallest_batch_num = smallest_dataset_length / args.batch_size
        sst_batch_size = round(len(sst_train_dataloader) / smallest_dataset_length * smallest_batch_num)
        para_batch_size = round(len(para_train_dataloader) / smallest_dataset_length * smallest_batch_num)
        sts_batch_size = round(len(sts_train_dataloader) / smallest_dataset_length * smallest_batch_num)
    else:
        sst_batch_size, para_batch_size, sts_batch_size = args.batch_size, args.batch_size, args.batch_size

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=sst_batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=para_batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                      collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=sts_batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                      collate_fn=sts_dev_data.collate_fn)
    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)
    model = MultitaskBERT(config)
    model = model.to(device)
    
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_performance = 0

    
    if (args.rrobin):
        # training with round-robin
        # 在每个epoch的开始，创建迭代器
        para_iter = iter(para_train_dataloader)
        sst_iter = iter(sst_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        for epoch in range(args.epochs):
            model.train()
            if (args.pre):
                if (args.small):
                    raise Exception("args.small cann't be True when using args.pre")
                
                para_train_loss = 0
                args.para_train = "data/quora-train-pre.csv"
                _, _,para_train_data, _ = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
                para_train_data = SentencePairDataset(para_train_data, args, 'para')
                para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=para_batch_size,
                                            collate_fn=para_train_data.collate_fn)

                for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                    (b_ids1, b_mask1,
                    b_ids2, b_mask2,
                    b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                                batch['token_ids_2'], batch['attention_mask_2'],
                                batch['sent_ids'], batch['labels'])

                    b_ids1 = b_ids1.to(device)
                    b_mask1 = b_mask1.to(device)
                    b_ids2 = b_ids2.to(device)
                    b_mask2 = b_mask2.to(device)
                    b_labels = b_labels.to(device)
                    
                    optimizer.zero_grad()
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    logits = logits.type(torch.FloatTensor)
                    b_labels = b_labels.type(torch.FloatTensor)
                    loss = F.binary_cross_entropy(logits.squeeze(), b_labels.view(-1), reduction='sum') / args.batch_size
                    if args.smartr:
                        # 计算SMART正则化项,使用对称kl散度
                        smart_reg = compute_smart_regularization_kl(model, batch, epsilon=1e-5, device=device, predict_function=model.predict_paraphrase, task_type="para")
                                
                        # 将正则化项加到原始损失上
                        total_loss = loss + 5 * smart_reg
                        total_loss.backward()
                        optimizer.step()
                                
                        # 累加para任务的损失
                        para_train_loss += total_loss.item()
                    else:
                        loss.backward()
                        optimizer.step()
                                
                        para_train_loss += loss.item()

                args.para_train = "data/quora-train-pre-left.csv"
                _, _,para_train_data, _ = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
                para_train_data = SentencePairDataset(para_train_data, args, 'para')
                para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=para_batch_size,
                                            collate_fn=para_train_data.collate_fn)
            


            sst_train_loss, para_train_loss, sts_train_loss = 0, 0, 0
            total_batches = 0

            while total_batches < max(len(para_train_dataloader), len(sst_train_dataloader), len(sts_train_dataloader)):
                # 轮询训练每个任务
                for task_iter in [para_iter, sst_iter, sts_iter]:
                    try:
                        batch = next(task_iter)
                    except StopIteration:
                        # 重置迭代器
                        if task_iter == para_iter:
                            para_iter = iter(para_train_dataloader)
                        elif task_iter == sst_iter:
                            sst_iter = iter(sst_train_dataloader)
                        else:
                            sts_iter = iter(sts_train_dataloader)
                        continue  # 跳到下一个迭代器

                    # 根据任务类型执行训练步骤
                    if task_iter == para_iter:
                        (b_ids1, b_mask1,
                        b_ids2, b_mask2,
                        b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                                    batch['token_ids_2'], batch['attention_mask_2'],
                                    batch['sent_ids'], batch['labels'])

                        b_ids1 = b_ids1.to(device)
                        b_mask1 = b_mask1.to(device)
                        b_ids2 = b_ids2.to(device)
                        b_mask2 = b_mask2.to(device)
                        b_labels = b_labels.to(device)
                        
                        optimizer.zero_grad()
                        logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                        logits = logits.type(torch.FloatTensor)
                        b_labels = b_labels.type(torch.FloatTensor)
                        loss = F.binary_cross_entropy(logits.squeeze(), b_labels.view(-1), reduction='sum') / args.batch_size

                        if args.smartr:
                            # 计算SMART正则化项,使用对称kl散度
                            smart_reg = compute_smart_regularization_kl(model, batch, epsilon=1e-5, device=device, predict_function=model.predict_paraphrase, task_type="para")
                            
                            # 将正则化项加到原始损失上
                            total_loss = loss + 5 * smart_reg
                            total_loss.backward()
                            optimizer.step()
                            
                            # 输出损失信息
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: Para, Loss: {total_loss.item()}")
                            
                            # 累加para任务的损失
                            para_train_loss += total_loss.item()
                        else:
                            loss.backward()
                            optimizer.step()
                            
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: Para, Loss: {loss.item()}")
                            para_train_loss += loss.item()

                    elif task_iter == sst_iter:
                        b_ids, b_mask, b_labels = (batch['token_ids'],
                                                batch['attention_mask'], batch['labels'])
                        b_ids = b_ids.to(device)
                        b_mask = b_mask.to(device)
                        b_labels = b_labels.to(device)
                        
                        optimizer.zero_grad()
                        
                        # 前向传播，计算原始损失
                        logits = model.predict_sentiment(b_ids, b_mask)
                        loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

                        if args.smartr:
                            # 计算SMART正则化项,使用对称kl散度
                            smart_reg = compute_smart_regularization_kl(model, batch, epsilon=1e-5, device=device, predict_function=model.predict_sentiment, task_type="sst")
                            
                            # 将正则化项加到原始损失上
                            total_loss = loss + 5 * smart_reg
                            
                            # 反向传播总损失
                            total_loss.backward()
                            optimizer.step()
                            
                            # 输出损失信息
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: SST, Loss: {total_loss.item()}")
                            
                            # 累加SST任务的损失
                            sst_train_loss += total_loss.item()
                        else:
                            loss.backward()
                            optimizer.step()

                            # 输出损失信息
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: SST, Loss: {loss.item()}")
                            
                            # 累加SST任务的损失
                            sst_train_loss += loss.item()
                    else:
                        (b_ids1, b_mask1,
                        b_ids2, b_mask2,
                        b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                                    batch['token_ids_2'], batch['attention_mask_2'],
                                    batch['sent_ids'], batch['labels'])

                        b_ids1 = b_ids1.to(device)
                        b_mask1 = b_mask1.to(device)
                        b_ids2 = b_ids2.to(device)
                        b_mask2 = b_mask2.to(device)
                        b_labels = b_labels.to(device)
                        
                        optimizer.zero_grad()
                        logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                        logits = logits.type(torch.FloatTensor)
                        b_labels = b_labels.type(torch.FloatTensor)
                        loss = F.mse_loss(logits.squeeze(), b_labels.float(), reduction='sum') / args.batch_size

                        if args.smartr:
                            # 计算SMART正则化项,使用平方差损失
                            smart_reg = compute_smart_regularization_mse(model, batch, epsilon=1e-5, device=device)

                            # 将正则化项加到原始损失上
                            total_loss = loss + 5 * smart_reg

                            # 反向传播总损失
                            total_loss.backward()
                            optimizer.step()
                            
                            # 输出损失信息
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: STS, Loss: {total_loss.item()}")
                            
                            # 累加SST任务的损失
                            sts_train_loss += total_loss.item()
                        else:
                            loss.backward()
                            optimizer.step()
                            print(f"Epoch {epoch}, Batch {total_batches}, Task: STS, Loss: {loss.item()}")
                            sts_train_loss += loss.item()

                total_batches += 1
                
            # 计算平均损失
            para_train_loss = para_train_loss / len(para_train_dataloader)
            sst_train_loss = sst_train_loss / len(sst_train_dataloader)
            sts_train_loss = sts_train_loss / len(sts_train_dataloader)
               
            # 模型评估
            (paraphrase_accuracy, para_y_pred, para_sent_ids,
            sentiment_accuracy, sst_y_pred, sst_sent_ids,
            sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, 
                                                                    para_dev_dataloader, 
                                                                    sts_dev_dataloader, 
                                                                    model, device)

            average_performance = (paraphrase_accuracy + sentiment_accuracy + sts_corr)  / 3 # 平均评估指标，TODO：找找更好的评估指标
            if average_performance > best_performance:
                best_performance = average_performance
                save_model(model, optimizer, args, config, args.filepath)

            # 打印评估结果
            print(f"Epoch {epoch}: average_performance: {average_performance:.3f}")
        

    else:
        # training without round-robin        
        # Run for the specified number of epochs
        for epoch in range(args.epochs):
            model.train()
            sst_train_loss, para_train_loss, sts_train_loss = 0, 0, 0
            
            num_batches = 0
            for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['sent_ids'], batch['labels'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)
                
                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                logits = logits.type(torch.FloatTensor)
                b_labels = b_labels.type(torch.FloatTensor)
                loss = F.binary_cross_entropy(logits.squeeze(), b_labels.view(-1), reduction='sum') / args.batch_size
                if args.smartr:
                    # 计算SMART正则化项,使用对称kl散度
                    smart_reg = compute_smart_regularization_kl(model, batch, epsilon=1e-5, device=device, predict_function=model.predict_paraphrase, task_type="para")
                            
                    # 将正则化项加到原始损失上
                    total_loss = loss + 5 * smart_reg
                    total_loss.backward()
                    optimizer.step()
                            
                    # 累加para任务的损失
                    para_train_loss += total_loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                            
                    para_train_loss += loss.item()
                num_batches += 1
                
            para_train_loss = para_train_loss / (num_batches)

            num_batches = 0
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                
                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                if args.smartr:
                    # 计算SMART正则化项,使用对称kl散度
                    smart_reg = compute_smart_regularization_kl(model, batch, epsilon=1e-5, device=device, predict_function=model.predict_sentiment, task_type="sst")
                            
                    # 将正则化项加到原始损失上
                    total_loss = loss + 5 * smart_reg
                            
                    # 反向传播总损失
                    total_loss.backward()
                    optimizer.step()
                            
                    # 累加SST任务的损失
                    sst_train_loss += total_loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                            
                    # 累加SST任务的损失
                    sst_train_loss += loss.item()
                num_batches += 1

            sst_train_loss = sst_train_loss / (num_batches)
            
            num_batches = 0
            for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_sent_ids, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['sent_ids'], batch['labels'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)
                
                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                logits = logits.type(torch.FloatTensor)
                b_labels = b_labels.type(torch.FloatTensor)
                loss = F.mse_loss(logits.squeeze(), b_labels.float(), reduction='sum') / args.batch_size
                if args.smartr:
                    # 计算SMART正则化项,使用平方差损失
                    smart_reg = compute_smart_regularization_mse(model, total_batches, epsilon=1e-5, device=device)

                    # 将正则化项加到原始损失上
                    total_loss = loss + 5 * smart_reg

                    # 反向传播总损失
                    total_loss.backward()
                    optimizer.step()
                            
                    # 累加STS任务的损失
                    sts_train_loss += total_loss.item()
                else:
                    loss.backward()
                    optimizer.step()
                    sts_train_loss += loss.item()
                num_batches += 1
                
            sts_train_loss = sts_train_loss / (num_batches)
            
            # 模型评估
            # sts_corr: 皮尔逊相关系数，[-1, 1]
            (paraphrase_accuracy, para_y_pred, para_sent_ids,
            sentiment_accuracy, sst_y_pred, sst_sent_ids,
            sts_corr, sts_y_pred, sts_sent_ids) = model_eval_multitask(sst_dev_dataloader, 
                                                                    para_dev_dataloader, 
                                                                    sts_dev_dataloader, 
                                                                    model, device)

            average_performance = (paraphrase_accuracy + sentiment_accuracy + sts_corr)  / 3 # 平均评估指标，TODO：找找更好的评估指标
            if average_performance > best_performance:
                best_performance = average_performance
                save_model(model, optimizer, args, config, args.filepath)

            # 打印评估结果
            print(f"Epoch {epoch}: average_performance: {average_performance:.3f}")
        



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    parser.add_argument("--test", action='store_true') # 仅测试
    parser.add_argument("--small", action='store_true') # 使用小规模数据集
    parser.add_argument("--rrobin", action='store_true') # 训练时对数据集使用round-robin方法
    parser.add_argument("--smartr", action='store_true') # 使用smart正则化
    parser.add_argument("--pre", action='store_true') # 使用para的数据先进行warm-up,仅在完整数据集下生效
    parser.add_argument("--rlayer", action='store_true') # 添加relational layer

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # fix the seed for reproducibility
    if not args.test:
        if args.small:
            args.sst_train = "data/ids-sst-train-small.csv"
            args.para_train = "data/quora-train-small.csv"
            args.sts_train = "data/sts-train-small.csv"
            args.filepath = f'{args.option}-{args.epochs}-{args.lr}-small-multitask.pt' # save path
        else:
            args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path

        train_multitask(args)
    test_model(args)
