# MbaQA
## Introduction
This is a modified version of facebook [DrQA](https://github.com/facebookresearch/DrQA) module which supports Chinese Language. This module is currently able to answer factiod questions in the financial field.

## DrQA Introduction
> DrQA is a system for reading comprehension applied to open-domain question answering. In particular, DrQA is targeted at the task of "machine reading at scale" (MRS). In this setting, we are searching for an answer to a question in a potentially very large corpus of unstructured documents (that may not be redundant). Thus the system has to combine the challenges of document retrieval (finding the relevant documents) with that of machine comprehension of text (identifying the answers from those documents).

>Our experiments with DrQA focus on answering factoid questions while using Wikipedia as the unique knowledge source for documents. Wikipedia is a well-suited source of large-scale, rich, detailed information. In order to answer any question, one must first retrieve the few potentially relevant articles among more than 5 million, and then scan them carefully to identify the answer.

>Note that DrQA treats Wikipedia as a generic collection of articles and does not rely on its internal graph structure. As a result, DrQA can be straightforwardly applied to any collection of documents, as described in the retriever README.

## Installation

* install python (3.5 or higer)
* install pyltp following the instructions [here](http://pyltp.readthedocs.io/zh_CN/develop/api.html), download model data and edit the code in ```MbaQA/tokenizers/__init__.py, line 11: ltp_datapath```
* install [pytorch](https://pytorch.org/)

## Structures
```
MbaQA
├── data 
    ├── db
    │   └── mba_2000.db
    ├── reader
    │   └── single.mdl
    └── retriever
        └── model 
            └── mba_2000-tfidf-ngram=2-hash=16777216-tokenizer=ltp-numdocs=2002.npz
```

## Notes
This module takes the MBA Wiki articles(78459) as the unique knowledge source. The articles originally came from [Datayes Corporation](https://www.datayes.com/). According to company regulations, we built a demonstration system based on a subset of 2000 articles.

