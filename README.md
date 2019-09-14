# mlt

Multilingual Neural Machine Translation using Transformers with Conditional Normalization.

Code and Demos for the following use-cases

#### One to Many

> https://colab.research.google.com/github/suyash/mlt/blob/master/one_to_many_demo.ipynb

English to French, German, Spanish and Italian translation.

This uses split vocabularies of size 8192 each.

#### Many to One

> https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_one_demo.ipynb

French, German, Spanish and Italian to English translation.

This uses split vocabularies of size 8192 each.

#### Many to Many

> https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_many_demo.ipynb

English, French and German on both ends.

This uses a shared vocabulary of size 40,960.

#### Many to Many Fine-Tuned

> https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_many_fine_tune_demo.ipynb

Portuguese added from ted_hrlr_translate to the Many to Many model.
