# MLOps – Course

Here is the link of the HuggingFace repos with the trained model pushed : https://huggingface.co/Azese/distilbert-imdb-sentiment-analysis

Here is the link as proof of executing the end to end MLProject on google colab, as on my computer i have sever issues and limitation : https://colab.research.google.com/drive/1aGrkLiKKccT1QHWBeuiW95lsCheXDQ5S?usp=sharing


This repository contains the slides, labs, and project scaffold for a  MLOps training:

## Structure
```
mlops_2day_course/
├── slides/
│   └── MLOps_SupdeVinci.pptx
├── labs/
│   ├── Lab1/
│   └── Lab2_SA_HuggingFace/
├── project/
│   └── ml_microservice/
├── docs/
│   └── syllabus.md

```

## Quickstart
```bash
# create & activate env
python3 -m venv .venv && source .venv/bin/activate
pip install -r project/ml_microservice/requirements.txt

# run microservice locally
make -C project/ml_microservice run

# run tests
make -C project/ml_microservice test
```
