# mlt

Multilingual Neural Machine Translation using Transformers with Conditional Normalization.

Transformers with One-to-Many, Many-to-One, Many-to-Many translation, with another demo for fine-tuning, by replacing regular layer norm in both encoder and decoder with Conditional Norm.

Conditional Norm is a technique from Style Transfer, used to train a single network with multiple styles, as well as mix styles, for example 

<img src="https://i.imgur.com/U0L1x8u.png" width=400 />

Broader set of demos at https://github.com/suyash/stylizer/blob/master/image_stylization_demo.ipynb

The goal here is similar, make the rest of the network learn a common representation, while making the normalization parameters learn language specific semantics.

The One-to-Many and Many-to-One models are trained for English to French, German, Italian and Spanish Translation and Vice-Versa.

The Many to Many model is trained on English-French, French-English, English-German and German-English.

The image stylization paper specifies how a N-style network can pick up an N+1th style through fine-tuning an existing model. Similarly, I fine-tune my Many-to-Many model to pick up Portuguese.

I also provide demos for Few-Shot and Zero-Shot translation.

Have provided `SavedModel`s, as well as standalone notebooks that run in Google Colaboratory.

Code and Demos for the following use-cases

#### One to Many

> Demo: https://colab.research.google.com/github/suyash/mlt/blob/master/one_to_many_demo.ipynb
>
> Code: https://github.com/suyash/mlt/blob/master/mlt/one_to_many.py
>
> Model: https://dl.dropboxusercontent.com/s/bssvce8t00zjwxj/one_to_many.zip?dl=0

English to French, German, Spanish and Italian translation.

This uses split vocabularies of size 8192 each.

#### Many to One

> Demo: https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_one_demo.ipynb
>
> Code: https://github.com/suyash/mlt/blob/master/mlt/many_to_one.py
>
> Model: https://dl.dropboxusercontent.com/s/xmroc5u6k17t5ds/many_to_one.zip?dl=0

French, German, Spanish and Italian to English translation.

This uses split vocabularies of size 8192 each.

#### Many to Many

> Demo: https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_many_demo.ipynb
>
> Code: https://github.com/suyash/mlt/blob/master/mlt/many_to_many.py
>
> Model: https://dl.dropboxusercontent.com/s/kzxryp6snnx5dw6/many_to_many.zip?dl=0

English, French and German on both ends.

This uses a shared vocabulary of size 40,960.

#### Many to Many Fine-Tuned

> Demo: https://colab.research.google.com/github/suyash/mlt/blob/master/many_to_many_fine_tune_demo.ipynb
>
> Code: https://github.com/suyash/mlt/blob/master/mlt//many_to_many_fine_tune.py
>
> Model: https://dl.dropboxusercontent.com/s/3ixjojbpho2emd8/many_to_many_fine_tune.zip?dl=0

Portuguese added from ted_hrlr_translate to the Many to Many model.
