import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qgcontext import C
from transformers import logging
logging.set_verbosity_error()

def get_config():
    parser = argparse.ArgumentParser()

    """init options"""
    parser.add_argument("filename", type=str,nargs='?',default="documents\languages\English_language.txt", help="(default: bert-base-uncased)")
    parser.add_argument("n", type=int, default=10,nargs='?', help="Number of questions (default: 10)")

    parser.add_argument("--model_path", type=str, default="model_weights_5_3.970575523446314.pth", help="(default: model_weights_4_72.512201225647.pth)")

    """prediction options"""
    args = parser.parse_args()
    return args



def inference(args,context):
    # set tokenizer and model
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)


    # add [HL], [/HL] token
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[HL]", "[/HL]"]})
    # add token number
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token

    # load model
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # evaluation
    model.eval()

    context_tokenized = tokenizer.encode(context, add_special_tokens=False)
    pred_str_list = [] 
    model.to(device)
    for _ in range(args.max_question_token_len): 
        pred_str_ids = tokenizer.convert_tokens_to_ids(pred_str_list + [mask_token])
        predict_token = context_tokenized + pred_str_ids
        if len(predict_token) >= args.max_len:
            break
        predict_token = torch.tensor([predict_token]).to(device)
        predictions = model(predict_token)
        predicted_index = torch.argmax(predictions[0][0][-1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        if "[SEP]" in predicted_token:
            break
        pred_str_list.append(predicted_token[0])
    token_ids = tokenizer.convert_tokens_to_ids(pred_str_list)
    question = tokenizer.decode(token_ids)
    #print(f"Context:{context} \n Question: {question}")
    return question

def checks(s):
    se = set(s.split(" "))
    mask = "[MASK]" in se
    pad = "[PAD]" in se
    return s[::-1][0]=="?" and (not mask) and (not pad)

def get_questions(n):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    qlist=[]
    while len(qlist)!=n:
        qg = C(args.filename)
        qg.convert()
        randcontext = qg.get_random_context()
        randent = qg.get_named_entity()
        answer_start = qg.find_named_entity()
        if not answer_start:
            break
        input_text = f"{cls_token} {randcontext[:answer_start]}[HL] {randcontext[answer_start:answer_start+len(randent)]} [/HL]{randcontext[answer_start+len(randent):]}{sep_token}"
        q=inference(args,input_text)
        #checks
        if checks(q) and (q not in qlist):
            qlist.append(q)
    return qlist


if __name__ == "__main__":
    args = get_config()
    questions = get_questions(args.n)
    for q in questions:
        print(q)
