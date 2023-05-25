#!/usr/bin/python3 -W ignore::DeprecationWarning
# -*- coding:utf8 -*-

import codecs
import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

class C:
    def __init__(self, filename):
        self.filename = filename
        self.doc = ""
        self.rand_context=""
        self.named_ent=""
        self.named_ent_type=""
        self.split=[]
    def convert(self):
        BLOCKSIZE = 1048576
        with codecs.open(self.filename, "r", "utf-8") as sourceFile:
            contents = sourceFile.read(BLOCKSIZE)
            contents = contents.encode("ascii", "ignore")
            contents = contents.decode()
            self.doc = contents
        self.doc = self.doc.replace("\n","")
        self.doc = self.doc.replace("\t","")
        self.doc = re.sub(r"\(([^()]*)\)","",self.doc)
        self.doc = re.sub(r"\=([^()]*)\=","",self.doc)
        self.split = self.doc.split(".")
    def preprocess(self):
        self.split = [sent.strip()+". " for sent in self.split]
        return 0
    def number_tokens(self,s):
        return len(nltk.word_tokenize(s)) 
    def get_random_context(self):
        self.preprocess()
        n = len(self.split)
        r = random.randrange(0,n-5)
        res = self.split[r]
        k=1
        while self.number_tokens(res)<100:
            try:
                res+= self.split[r+k]
            except IndexError:
                return res
            k+=1
        self.rand_context = res
        return res
    def get_named_entity(self):
        self.named_ent=""
        selection = nlp(self.rand_context)
        #print("entlist:",[(X.text, X.label_) for X in selection.ents])

        entlist=[X.text for X in selection.ents]
        labels =[X.label_ for X in selection.ents]

        primary_ne_types= ["PERSON","LOC","DATE","ORDINAL","GPE","other"]
        primary_ne_weighting= [1,1,1,.5,.3,.5]

        while self.named_ent=="":
            rand_type = random.choices(primary_ne_types,primary_ne_weighting)[0]
            indices = [i for i, x in enumerate(labels) if x == rand_type or rand_type=="other"]
            #print(rand_type,indices)
            if indices!=[]:
                self.named_ent = entlist[indices[random.randrange(0,len(indices))]]
                self.type = rand_type
        
        return self.named_ent
    def find_named_entity(self):
        a=re.search(rf'\b{self.named_ent}\b', self.rand_context)
        if type(a)==None:
            return False
        return a.start()

