# Emotion Classification on Textual Data from MELD

### Project Setup
- All our Jupyter Notebooks were created and run in Google Colab.
- You can use our Colab links provided below (recommended), or manually import the notebooks in this repository into Colab.
- You are highly recommended to use Colab as we do not explicitly pip install any Python libraries already included in Colab.

### Data Downloads
- Our notebooks already perform the downloading of data available in this repository automatically.
- However, in the unlikely event that the data is unavailable, you may download the contents of the "data" folder available in this repository.

| Directory | Contents |
|:--------|:-----------|
| model output | Output of analysing attention scores from running "BERT visualisation of attention" section in "Deep Learning (Dialogues).ipynb". In (.txt) format. |
| processed | Contains data records that have undergone preprocessing and outputed into JSON format. (We have variations with and without punctuation) <br> All our models use this data. |
| raw | Raw textual data provided in CSV format |

## Jupyter Notebooks and Scripts
- All codes have been organised into sections with headers for ease of navigation within each notebook.
- Please use the "Table Of Contents" in Colab to jump to sections of interest.
<br>![image](https://github.com/r0b0pp1/CS5228_emotion_in_conversation/assets/22906940/a3c12516-20d2-41de-9910-4398cf6005e3)


### Data Pre-Processing
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oRXfVLD7EH-108klTXtGIkDnaHWz9SOM?usp=sharing) | Data Pre-processing.ipynb | Data Cleaning Steps <br> Rebalancing of Dataset and related diagrams |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_m9kImUDGg3dqJX111QeX27ba2LPaJwO?usp=sharing) | Data Visualisation Analysis.ipynb | Utterance Length Analysis <br> Word Clouds <br> Linear Separability of Data with TSNE  |

### Classical Models
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S8Y_Y87lh7K8TGZF4erm17bo2Zx-bdqm?usp=sharing) | Decision Trees.ipynb | Experiments on emotion classification with Decision Trees and Random Forests.  |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kSzEwnC3tnWLN7ruEzzEQPuWf_46GeHe?usp=sharing) | Logistic Regression.ipynb | Experiments on emotion classification with Logistic Regression. |

### Deep Learning Models
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fuY7fTGydp4WmSkN_s85WvtGd-BWuRVp?usp=sharing) | Deep Learning (Dialogues).ipynb | Experiments on emotion classification at utterance level (upper-bound/wo context/w context) and at dialogue level (CRF/RNN).  <br> For upper-bound, please run trainer.py (detailed steps described in "Deep Learning (Dialogues).ipynb") |
| N/A | trainer.py | Run this file for hyperparameter-tuning, training and evaluating the upper-bound. As it is time-consuming to run, we output the hyperparameter-tuning results and test results in the "logs" folder. |

### Citation
The MELD dataset is obtained from the following research:

[1] Chen, S.Y., Hsu, C.C., Kuo, C.C. and Ku, L.W. EmotionLines: An Emotion Corpus of Multi-Party
Conversations. arXiv preprint arXiv:1802.08379 (2018)., https://affective-meld.github.io/

[2] S. Poria, D. Hazarika, N. Majumder, G. Naik, R. Mihalcea, E. Cambria. MELD: A Multi-
modal Multi-Party Dataset for Emotion Recognition in Conversations. (2018), https://affective-meld.github.io/
