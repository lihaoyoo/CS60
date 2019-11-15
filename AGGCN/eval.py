"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
import json

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab


parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
#print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
#print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

#print(eval_examples[0])
helper.print_config(opt)
label2id = constant.LABEL_TO_ID
label2objid = constant.OBJ_NER_TO_ID
label2subid = constant.SUBJ_NER_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])
objid2label = dict([(v,k) for k,v in label2objid.items()])
subid2label = dict([(v,k) for k,v in label2subid.items()])

eval_examples = []

for l in range(len(batch)):
    eval_examples += batch.data[l]
    
pred_nos = []
predictions = []
all_probs = []
batch_iter = tqdm(batch)
for i, b in enumerate(batch_iter):
    preds, probs, _ = trainer.predict(b)
    pred_nos += preds
    all_probs += probs

    
predictions = [id2label[p] for p in pred_nos]

            
# #for gold, pred, score, p in zip(batch.gold(), predictions, all_probs, pred_nos):
result_final = []
result_output = []
eachsentence_dict = {}
entity_dict_noindex = {}
entity_list = []
entity_dict = {}
relation_list = []
relation_dict = {}
sentence_list = []
sentence_dict = {}
last_sentence = ''

c = 0
entity_count = 0
with open( "scores.txt", "w") as f:
    for ex, pred, prob, gold, pred_no in zip(eval_examples, predictions, all_probs, batch.gold(), pred_nos):
        count = 0
        entity1 = ''
        entity2 = ''   
        
        sub_type = subid2label[ex[7][0]]
        obj_type = objid2label[ex[8][0]]
        
        for i in range(len(ex[5])):
            if ex[5][i] == 0:
                entity1 += ex[9][i] + ' '
        
        for i in range(len(ex[6])):
            if ex[6][i] == 0:
                entity2 += ex[9][i] + ' '
        
        entity1 = entity1.strip()
        entity2 = entity2.strip()               
        
        if last_sentence != ex[9]:
            if c != 0:
                #entity_list.append(entity_dict)
                entity_list.append(entity_dict_noindex)
                result_output.append(entity_list)
                result_output.append(relation_list)
                sentence_list.append(sentence_dict)
                result_output.append(sentence_list)
                result_final.append(result_output)
                #print(result_output)
            
            last_sentence = ex[9]
            entity_list = []
            entity_dict = {}
            entity_dict_noindex = {}
            relation_list = []
            relation_dict = {}
            sentence_list = []
            sentence_dict = {}
            result_output = []  
            c += 1
            
        if sub_type in entity_dict_noindex:
            if entity1 not in entity_dict_noindex[sub_type]:
                entity_dict_noindex[sub_type].append(entity1)
        else:
            entity_dict_noindex[sub_type] = [entity1]       
            
        if obj_type in entity_dict_noindex:
            if entity2 not in entity_dict_noindex[obj_type]:
                entity_dict_noindex[obj_type].append(entity2)
        else:
            entity_dict_noindex[obj_type] = [entity2]
        
        if pred != 'no_relation':
            relation_dict['entity1'] = entity1
            relation_dict['entity2'] = entity2
            relation_dict['relation'] = pred
            relation_dict['score'] = prob[pred_no]
            relation_list.append(relation_dict)
            relation_dict = {}
        
        sentence_dict['sentence'] = (' '.join(ex[9])).strip()
        
        f.write("%s\t%s\t%0.3f\n" % (gold, pred, prob[pred_no]))
       
        
        entity_count += 1
    #entity_list.append(entity_dict)
    entity_list.append(entity_dict_noindex)
    result_output.append(entity_list)
    result_output.append(relation_list)
    sentence_list.append(sentence_dict)
    result_output.append(sentence_list)
    #print(result_output)
    result_final.append(result_output)
       

with open("predictions.json", "w") as f:
    json.dump(result_final, f)

# p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
#print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

#print("Evaluation ended.")

