|ptb|_ |wiki2|_ |wiki103|_

.. |ptb| image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190409408/language-modelling-on-penn-treebank-word
.. _ptb: https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word?p=190409408

.. |wiki2| image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190409408/language-modelling-on-wikitext-2
.. _wiki2: https://paperswithcode.com/sota/language-modelling-on-wikitext-2?p=190409408

.. |wiki103| image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190409408/language-modelling-on-wikitext-103
.. _wiki103: https://paperswithcode.com/sota/language-modelling-on-wikitext-103?p=190409408


Language Models with Transformers
-----------------------------------

Installation
~~~~~~~~~~~~~~~~

.. code::

    pip install --pre --upgrade mxnet
    pip install gluonnlp

Results
~~~~~~~~~~~~~~~~

The datasets used for training the models are wikitext-2 and wikitext-103 respectively.

The key features used to reproduce the results on wikitext-2 based on the corresponding pre-trained models are listed in the following tables.

.. editing URL for the following table: https://bit.ly/2GAWwkD

+-------------+----------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Model       | bert_lm_12_768_12_300_1150_wikitext2                                                                                                   | bert_lm_24_1024_16_300_1150_wikitext2                                                                                                   |
+=============+========================================================================================================================================+=========================================================================================================================================+
| Val PPL     | 38.43                                                                                                                                  | 37.79                                                                                                                                   |
+-------------+----------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Test PPL    | 34.64                                                                                                                                  | 34.11                                                                                                                                   |
+-------------+----------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Command     | [1]                                                                                                                                    | [2]                                                                                                                                     |
+-------------+----------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Result logs | `log <https://github.com/dmlc/web-data/tree/master/gluonnlp/logs/language_model/bert_lm_12_768_12_300_1150_wikitext2.log>`__           | `log <https://github.com/dmlc/web-data/tree/master/gluonnlp/logs/language_model/bert_lm_24_1024_16_300_1150_wikitext2.log>`__           |
+-------------+----------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+

[1] bert_lm_12_768_12_300_1150_wikitext2 (Val PPL 38.43 Test PPL 34.64)

.. code-block:: console

   $ cd scripts/language_model
   $ python transformer_language_model.py --data wikitext2 --model bert_lm_12_768_12_300_1150 --val_batch_size 8 --test_batch_size 8 --bptt 128 --seed 1882 --batch_size 16 --gpus 0

[2] bert_lm_24_1024_16_300_1150_wikitext2 (Val PPL 37.79 Test PPL 34.11)

.. code-block:: console

   $ cd scripts/language_model
   $ python transformer_language_model.py --data wikitext2 --model bert_lm_24_1024_16_300_1150 --val_batch_size 8 --test_batch_size 8 --bptt 128 --seed 1882 --batch_size 16 --gpus 0

The key features used to reproduce the results on wikitext-103 based on the corresponding pre-trained models are listed in the following tables.

.. editing URL for the following table: https://bit.ly/2Du8061

+-------------+------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Model       | bert_lm_12_768_12_400_2500_wikitext103                                                                                                   | bert_lm_24_1024_16_400_2500_wikitext103                                                                                                   |
+=============+==========================================================================================================================================+===========================================================================================================================================+
| Val PPL     | 40.70                                                                                                                                    | 20.33                                                                                                                                     |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Test PPL    | 39.85                                                                                                                                    | 20.54                                                                                                                                     |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Command     | [1]                                                                                                                                      | [2]                                                                                                                                       |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Result logs | `log <https://github.com/dmlc/web-data/tree/master/gluonnlp/logs/language_model/bert_lm_12_768_12_400_2500_wikitext103.log>`__           | `log <https://github.com/dmlc/web-data/tree/master/gluonnlp/logs/language_model/bert_lm_24_1024_16_400_2500_wikitext103.log>`__           |
+-------------+------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+

[1] bert_lm_12_768_12_400_2500_wikitext103 (Val PPL 40.70  Test PPL 39.85)

.. code-block:: console

   $ cd scripts/language_model
   $ python transformer_language_model.py --data wikitext103 --model bert_lm_12_768_12_400_2500 --val_batch_size 8 --test_batch_size 8 --bptt 64 --seed 1111 --batch_size 20 --gpus 0

[2] bert_lm_24_1024_16_400_2500_wikitext103 (Val PPL 20.33 Test PPL 20.54)

.. code-block:: console

   $ cd scripts/language_model
   $ python transformer_language_model.py --data wikitext103 --model bert_lm_24_1024_16_400_2500 --val_batch_size 8 --test_batch_size 8 --bptt 64 --seed 1111 --batch_size 12 --gpus 0

Note that the corresponding multi-gpu evaluations are also supported. The pre-trained model `bert_lm_24_1024_16_400_2500_wikitext103` would be updated soon.
