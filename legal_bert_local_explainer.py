from model_wrappers import bert_huggingface_model_wrapper
from transformers import AutoModelForSequenceClassification,AutoTokenizer,BertConfig, TFAutoModel, AutoConfig

import explainer
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from models import *
from visualizer import LocalExplanationReportVisualizer


def clean_text(text):
    text = re.sub("@\S+", " ", text) # Remove Mentions
    text = re.sub("https*\S+", " ", text) # Remove URL
    text = re.sub("#\S+", " ", text) # Remove Hastags
    text = re.sub('&lt;/?[a-z]+&gt;', '', text) # Remove special Charaters
    text = re.sub('#39', ' ', text) # Remove special Charaters
    text = re.sub('<.*?>', '', text) # Remove html
    text = re.sub(' +', ' ', text) # Merge multiple blank spaces
    text = text.replace("<br>", "")
    text = text.replace("</br>", "")
    return text

def get_model_from_dir(path, tokenizer_string="bert-base-uncased"):
    config = BertConfig.from_pretrained(path, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_string)

    return model, tokenizer

if __name__ == "__main__":
    """USE_CASE_NAME = "imdb_huggingface_bert"
    MODEL_PATH = "textattack/bert-base-uncased-imdb"
    TOKENIZER_STRING = "bert-base-uncased"
    DATASET_NAME = "imdb"
    N = 3

    model, tokenizer = get_model_from_dir(MODEL_PATH, TOKENIZER_STRING)"""


    path = "echr_corpus/ECHR_Corpus.json"
    model_name = "nlpaueb/bert-base-uncased-echr"

    df_train, df_test = build_dataset(path)
    model, tokenizer = build_model(model_name)

    #print(model.get_layer("tf_bert_model").output)
    #train_model(model, tokenizer, df_train)
    #test_model(model, tokenizer, df_test)

    """raw_datasets = load_dataset(DATASET_NAME)

    test_texts = raw_datasets["test"]["text"]
    test_labels = raw_datasets["test"]["label"]

    texts = test_texts[:N]
    true_labels = test_labels[:N]"""


    texts = df_train.head(1)[["TEXT"]].values.tolist()[0][0]
    contexts = df_train.head(1)[["CONTEXT"]].values.tolist()[0][0]
    true_labels = df_train.head(1)["LABEL"].values.tolist()

    inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=500)
    inputs = [np.asarray(inputs.input_ids, dtype="int32"), np.asarray(inputs.attention_mask, dtype="int32")]


    submodel = Model(model.inputs, model.get_layer("tf_bert_model").output) 
    #print(model.get_layer("tf_bert_model"))
    print(len(submodel.predict(inputs).hidden_states))


    print(true_labels)
    input()

    bert_model_wrapper = bert_huggingface_model_wrapper.BertModelWrapper(model, tokenizer, clean_function=clean_text, batch_size=16)

    coi = -1

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(bert_model_wrapper, model_name=model_name)

    local_explanations = exp.fit_transform(input_texts=texts,
                                          classes_of_interest=[coi] * len(texts),
                                          expected_labels=true_labels,
                                          flag_pos=False,
                                          flag_sen=False,
                                          flag_mlwe=True,
                                          flag_combinations=True,
                                          flag_rnd=False,
                                          flag_offline_mode=True)

    for explanation in local_explanations:
        visualizer = LocalExplanationReportVisualizer()
        visualizer.fit(explanation)
        visualizer.visualize_report_as_html()

        print("PRESS KEY+ENTER TO CONTINUE")
        input()
    print("End Main.")