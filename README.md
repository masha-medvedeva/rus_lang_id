# Large-Scale Language Identification for Closely Related Languages

This repostory consists of data and code used for master thesis 'Large-Scale Language Identification for Closely Related Languages'.

Text files are the datasets used for training, developing and testing, including out-of-domain testing (VK)

create_dataset.py is the script that we used to extract sentences from the data available at http://web-corpora.net/wsgi3/minorlangs/download

The other scripts are the 3 models that have achived the best results during our research. Please see the thesis for more details.

|      |          1-step |          2-step |    MLP |
|------|----------------:|----------------:|-------:|
| test |          0.9423 |          0.9415 | 0.8597 |
| VK   |          0.8187 |          0.8251 | 0.7618 |
