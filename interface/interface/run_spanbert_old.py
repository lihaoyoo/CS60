# coding: utf-8
from django.shortcuts import render
from django.views.decorators import csrf

import uuid
import json
import os
import corenlp
import sys
sys.path.append(r'/home/jupyter/Spanbert/code')
import run_tacred_1
import argparse
import json
import pygraphviz as pgv

def spanbert(request):
    os.environ["CORENLP_HOME"] = r'/home/jupyter/Spanbert/stanford-corenlp-full-2018-10-05'
    ctx ={}
    if request.POST:
        text = request.POST['paragraph']
        res = extract_entity_sentence(text)
        result = transfer_to_multi_relation(res)
        data_path = "/home/jupyter/Spanbert/tacred"
        with open('/home/jupyter/Spanbert/tacred/preprocessing_data.json', 'w', encoding='utf-8') as f:
            json.dump(result, f)
        run_tacred_1.main(data_path)
        with open('/home/jupyter/Spanbert/output/predictions.json', 'r') as f:
             data = json.load(f)
        
        result_final = []
        for d in data:
            output = {}
            output['entities'] = d[0]
            output['relations'] = d[1]
            output['sentence'] = d[2]
            #print(output)
            result_final.append(output)
            print(json.dumps(output, sort_keys=True, indent=2))
        ctx['rlt'] = result_final
        visualization()
    return render(request, "post.html", ctx)

def extract_entity_sentence(text):

	new_dict = []
	with corenlp.CoreNLPClient(annotators=['tokenize','ssplit','pos','ner'], timeout=300000, memory='16G') as client:
		print("1212")
		ann = client.annotate(text)

		for sen in ann.sentence:
			print("1313")
			sen_dict = {}
			ners = []
			tokens = []
			pos = []
			dels = []
			for i in sen.token:
			    tokens.append(i.word)
			    ners.append(i.ner)
			    pos.append(i.pos)
			sen_dict = {"token":tokens, "stanford_ner":ners, "stanford_pos":pos}
			new_dict.append(sen_dict)

		print(new_dict)
		return new_dict

    
def transfer_to_multi_relation(new_dict):
    result = []
    for i in new_dict:
    
            location_start = 0
            location_end = 0
            entity_count = 0
            subj_start = []
            subj_end = []
            subj_type = []
            subjType = ''
            
            for j in i['stanford_ner']:
                if subjType != j:
                    if j != 'O':
                        if subjType != 'O' and subjType != '':
                            subj_end.append(location_end)
                        subj_start.append(location_start)
                        subj_type.append(j)
                        subjType = j
                        location_end = location_start
                        entity_count += 1
                    else:
                        subjType = j
                        if location_start != 0:
                            subj_end.append(location_end)
                else:
                    location_end += 1
                location_start += 1
            if len(subj_end) != len(subj_start):
                subj_end.append(len(i['token']) - 1)
                
            if entity_count >= 2:
                sub_count = 0
                while sub_count < len(subj_start):
                    obj_count = 0
                    while obj_count < len(subj_start):
                        if subj_type[sub_count] == 'PERSON' or  subj_type[sub_count] == 'ORGANIZATION':
                            if sub_count != obj_count:
                                dict_so = {}
                                dict_so['id'] = str(uuid.uuid1())
                                dict_so['relation'] = 'no_relation'
                                dict_so['token'] = i['token']
                                dict_so['subj_start'] = subj_start[sub_count]
                                dict_so['subj_end'] = subj_end[sub_count]
                                dict_so['obj_start'] = subj_start[obj_count]
                                dict_so['obj_end'] = subj_end[obj_count]
                                dict_so['subj_type'] = subj_type[sub_count]
                                dict_so['obj_type'] = subj_type[obj_count]
                                dict_so['stanford_ner'] = i['stanford_ner']
                                result.append(dict_so)
                            obj_count += 1
                        else:
                            break
                    sub_count += 1
    return result


def visualization():
    with open('/home/jupyter/Spanbert/output/predictions.json') as f:
      data = json.load(f)
    #Extract relations and entities 
    d=[] 
    for a in data:
        for b in a:
            for c in b:
                for key in c.keys():
                    if key =="relation":
                        d.append(dict(c))  
    #Ceate nodes & edges and set attributes
    G=pgv.AGraph(strict=False,directed=True)
    for i in d:
        G.add_node(i["entity1"], type= "entity",color="red")
        G.add_node(i["entity2"], type= "entity",color="red")
    for i in d:
        G.add_edge(i["entity1"],i["entity2"],label=i["relation"])  
    #Set layout and draw graph
    G.layout(prog='dot') 
    G.draw('/home/jupyter/interface/static/network_graph.png')
