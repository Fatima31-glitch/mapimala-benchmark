# MAPIMALA extended benchmark - summary

## Main results (Table 3)

| model | accuracy | 95% CI | P | R | F1 | F1 95% CI | seeds |
|---|---|---|---|---|---|---|---|
| efficientnetv2b0_ft_noaug | 0.9756 | 0.965–0.983 | 0.9762 | 0.9758 | 0.9757 | 0.966–0.984 | 1 |
| mobilenetv3l_ft | 0.9734 ± 0.0131 | 0.967–0.984 | 0.9785 | 0.9777 | 0.9733 ± 0.0136 | 0.968–0.985 | 3 |
| mobilenetv3l_ft_noaug | 0.9731 | 0.962–0.981 | 0.9740 | 0.9733 | 0.9728 | 0.963–0.981 | 1 |
| densenet121_ft | 0.9630 ± 0.0075 | 0.954–0.975 | 0.9688 | 0.9667 | 0.9628 ± 0.0083 | 0.955–0.975 | 3 |
| efficientnetv2b0_ft | 0.9619 ± 0.0078 | 0.947–0.969 | 0.9634 | 0.9618 | 0.9626 ± 0.0076 | 0.948–0.970 | 3 |
| efficientnetv2b0_frozen | 0.9588 | 0.946–0.969 | 0.9619 | 0.9600 | 0.9600 | 0.948–0.970 | 1 |
| mobilenetv3l_frozen | 0.9580 | 0.945–0.968 | 0.9600 | 0.9586 | 0.9585 | 0.947–0.969 | 1 |
| resnet50_frozen | 0.9420 | 0.927–0.954 | 0.9447 | 0.9434 | 0.9423 | 0.927–0.955 | 1 |
| densenet121_frozen | 0.9395 | 0.924–0.952 | 0.9438 | 0.9411 | 0.9410 | 0.927–0.953 | 1 |
| vgg16_frozen | 0.9202 | 0.903–0.934 | 0.9228 | 0.9221 | 0.9202 | 0.904–0.934 | 1 |
| cnn_scratch | 0.6538 | 0.626–0.680 | 0.6700 | 0.6594 | 0.6542 | 0.624–0.677 | 1 |

## Cost (Table 7)

| model | params | train (min) | latency (ms) | size (MB) | acc (%) |
|---|---|---|---|---|---|
| cnn_scratch | 6,673,896 / 6,673,896 | 219.4 | 113.1 | 80.14 | 65.38 |
| densenet121_frozen | 7,310,184 / 272,680 | 0.5 | 313.9 | 30.76 | 93.95 |
| densenet121_ft | 7,310,184 / 3,127,592 | 12.7 | 95.4 | 55.83 | 96.30 |
| efficientnetv2b0_frozen | 6,257,528 / 338,216 | 0.4 | 131.3 | 25.99 | 95.88 |
| efficientnetv2b0_ft_noaug | 6,257,528 / 3,406,616 | 4.3 | 86.0 | 53.27 | 97.56 |
| efficientnetv2b0_ft | 6,257,528 / 3,406,616 | 5.6 | 80.0 | 53.29 | 96.19 |
| mobilenetv3l_frozen | 3,252,648 / 256,296 | 0.3 | 102.5 | 13.7 | 95.80 |
| mobilenetv3l_ft_noaug | 3,252,648 / 2,724,408 | 2.9 | 81.7 | 35.52 | 97.31 |
| mobilenetv3l_ft | 3,252,648 / 2,724,408 | 4.6 | 78.3 | 35.54 | 97.34 |
| resnet50_frozen | 24,122,536 / 534,824 | 0.3 | 352.8 | 97.15 | 94.20 |
| vgg16_frozen | 14,856,296 / 141,608 | 0.4 | 673.5 | 59.52 | 92.02 |

Best model: **mobilenetv3l_ft_seed44** (97.34 %)

## Ten most frequent confusions of the best model

- Henna -> Nooni: 3
- Nithyapushpa -> Castor: 2
- Nagadali -> Ganike: 2
- Tulasi -> Jasmine: 1
- Tulasi -> Ashwagandha: 1
- Rose -> Pappaya: 1
- Rose -> Hibiscus: 1
- Rose -> Castor: 1
- Raktachandini -> Ashoka: 1
- Pepper -> Ashoka: 1