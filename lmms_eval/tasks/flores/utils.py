import os
import json
import re
import logging
import evaluate

lmms_logger = logging.getLogger('lmms-eval')

code_to_lang = {
    'eng_Latn': 'English',
    'hin_Deva': 'Hindi',
    '': 'English',
}

BLEU = evaluate.load('bleu')
CHRF = evaluate.load('chrf')

def construct_prompt(doc, model_specific_prompt_kwargs=None):
    if model_specific_prompt_kwargs is None:
        model_specific_prompt_kwargs = {}
    src_lang = code_to_lang[model_specific_prompt_kwargs.get('src-lang', '')]
    tgt_lang = code_to_lang[model_specific_prompt_kwargs.get('tgt-lang', '')]

    preprompt = f'Translate the following sentence from {src_lang} to {tgt_lang}: '
    postprompt = f'\nTranslation:'

    sentence = doc[f"sentence_{model_specific_prompt_kwargs.get('src-lang', 'eng_Latn')}"]

    return f"{preprompt}{sentence}{postprompt}"

def bleu(label, pred):
    try:
        return BLEU.compute(references=[label], predictions=[pred])['bleu'] * 100
    except:
        return 0.0

def chrf(label, pred):
    return CHRF.compute(references=[label], predictions=[pred])['score']

def flores_doc_to_text(doc, model_specific_prompt_kwargs=None):
    return construct_prompt(doc, model_specific_prompt_kwargs)


