# Sign Language Processing
This repository contains models, trainers and utils for Natural Language Processing (NLP) applied to Sign Languages.
Jusqu'à présent, ce répertoire ne contient que la tâche de Segmentation de la langue des signes.

# Installer les dépendances

In order to use this code, you will need a version of Python (> 3.10) as well as the dependencies (PyTorch, etc.).
You can install them using the command: `pip install -r requirements.txt`

# Répliquer les résultats des expériences

To replicate the results of the experiments, simply update or copy a configuration file in the `./configs` folder.
Make sure that the dataset paths and the output directory exist.
Then, you can run the following command: `python ./scripts/segmentation/train.py --config-path ./configs/<your_config>.yaml`

You can also test a model by using the command:  `python ./scripts/segmentation/test.py --config-path ./configs/<your_config>.yaml`

Web-datasets used in our experiments (LSFB, DGS, PHOENIX) (training, testing): https://drive.google.com/drive/folders/1I1OQf_BQ_22IrYXo9F3wX3DDp-fn7z6s?usp=sharing


