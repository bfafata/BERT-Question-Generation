from typing import Any, List
import spacy
from spacy.lang.en import English
import copy
import random

WH_WORD = "[HL] answer phrase[/HL]"

class QGModel:
    def __init__(self, source) -> None:
        self.source = source
        self.sentences = None
        self.doc = None
        
        self.proc = spacy.load("en_core_web_sm")
        #self.lang = English()
        
        self.answer_pos = ["VERB", "NOUN", "PROPN", "PRON", "NUM"]
        self.clause_heads = ["NOUN", ""]
        
        
        self.build()
        
    def build(self):
        with open(self.source, "r") as f:
            lines = f.readlines() # will consider other forms of punctuation later
            doc = ""
            for line in lines:
                if not ("==" in line):
                    doc += line
            
            self.doc = self.proc(doc)
            self.sentences = [sent.text.strip() for sent in self.doc.sents]
            #self.prs = self.lang(doc)
     
      
    def get_context(self, sent_id, bandwidth=0):
        context = ""
        for j in range(max(0, sent_id - bandwidth),
                       min(len(self.sentences), sent_id + bandwidth + 1)):
            context += self.sentences[j]
            
        return context
        
    def block_phrase(self, subject):
        iK = self.proc(subject)
        
        sents = []
        for i, focus in enumerate(iK):
            if (focus.pos_ in self.answer_pos):
                if (focus.dep_ != "ROOT"):
                    sent = [token.text if token not in focus.subtree else "/" + token.text for token in iK]
                    sent[i] = WH_WORD
                    for _ in range(sent.count("_NUT_")):
                        sent.remove("_NUT_")
                    sents.append((i, sent))
                else:
                    sent = [token.text for token in iK]
                    sent[i] = WH_WORD
                    sents.append((i, sent))
                
        return sents
            
    def generateQuestion(self, k=1, bandwidth=0):
        sent_id = random.randrange(0, len(self.sentences))
        subject = self.sentences[sent_id]
        
        iK = self.proc(subject)
        print([(token.text, token.lemma_, token.dep_, token.head) for token in iK], end='\n\n')
        
        for sent in self.block_phrase(subject):
            print(sent)

qg = QGModel("documents\chinese_dynasties\Han_dynasty.txt")

print(qg.generateQuestion(bandwidth=0))