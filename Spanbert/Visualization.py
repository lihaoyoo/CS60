# coding: utf-8

import json
import pygraphviz as pgv
#load output json file
with open('/home/jupyter/Spanbert_new/output/predictions_1.json') as f:
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
    G.add_edge(i["entity2"],i["entity1"],label=i["relation"])  
#Set layout and draw graph
G.layout(prog='dot') 
G.draw('121.png') 

