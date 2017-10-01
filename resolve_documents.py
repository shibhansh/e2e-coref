#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np
import pickle

import cgi
import BaseHTTPServer
import ssl

import tensorflow as tf
import coref_model as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
  # This is where tokenization is taking place
  # need to edit here to coref resolution of multiple documents in a single file


  # assumes sentences are seperated by ' <eos> ' in the story
  raw_sentences = text.strip().split(' <eos> ')
  # splitting as splitted in the AMR, this makes sure that the indexing for each work in coref resolution and 
  # AMR sentences is same
  sentences = [s.split() for s in raw_sentences if len(s) != 0]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  a, b, c, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = \
                                        session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  # print antecedents
  # print antecedent_scores, len(antecedent_scores), len(antecedent_scores[0])

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  # print predicted_antecedents, len(predicted_antecedents)


  example["predicted_clusters"], mention_to_predicted = \
                                model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)

  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()

  return example

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)

    file_name = sys.argv[2]

    with open(file_name,'r') as f:
      temp = f.readlines()
      documents = [doc.strip() for doc in temp if len(doc.strip()) != 0]

    resovled_documents = []
    for doc in documents:
      text = doc.strip()
      output = make_predictions(text, model)
      resovled_documents.append([output["predicted_clusters"],\
                                util.flatten(output["sentences"]),\
                                output["head_scores"]])

    with open('predicted_resolutions.txt','w') as f:
      pickle.dump(resovled_documents,f)
