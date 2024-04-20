# Emotion Classification on Textual Data from MELD

### Project Setup
- All our jupyter notebooks were created and run in Google Colab.
- You can use our Colab links provided below (recommended), or manually import the notebooks in this repository into Colab.
- You are highly recommended to use Colab as we do not explicitly pip install any python libraries already included in Colab.

### Data Downloads
- Our notebook already perform the downloading of data available in this repository automatically.
- However, in the unlikely event that the data is unavailable, you may download the contents of the "data" folder available in this repository.

| Directory | Contents |
|:--------|:-----------|
| model output | Output data produced by trainer.py. In (.txt) format. |
| processed | Contains data records that have undergone preprocessing, output into JSON format. (We have variations with and without punctuation) <br> All our models use this data. |
| raw | Raw textual data provided in CSV format |

## Jupyter Notebooks and Scripts
- All codes have been organised into sections with headers for ease of navigation within each notebook.
- Please use the "Table Of Contents" in Colab to jump to sections of interest.
<br>![image](https://github.com/r0b0pp1/CS5228_emotion_in_conversation/assets/22906940/a3c12516-20d2-41de-9910-4398cf6005e3)


### Data Pre-Processing
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oRXfVLD7EH-108klTXtGIkDnaHWz9SOM?usp=sharing) | data_pre-processing.ipynb | Data Cleaning Steps <br> Rebalancing of Dataset and related diagrams |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_m9kImUDGg3dqJX111QeX27ba2LPaJwO?usp=sharing) | data_visualisation_analysis.ipynb | Utterance Length Analysis <br> Word Clouds <br> Linear Separability of Data with TSNE  |

### Classical Models
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S8Y_Y87lh7K8TGZF4erm17bo2Zx-bdqm?usp=sharing) | decision_trees.ipynb | Experiments on emotion classification with Decision Trees and Random Forests.  |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1rhXR8DFGUbh8oqqQUC1UBN0dkOmUApUE/view?usp=sharing) | logistic_regression.ipynb | Experiments on emotion classification with Logistic Regression. |

### Deep Learning Models
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Script | Contents |
|---------------|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1aGr6XFTTV0iChHZbDDiW6kRB71keMU5a/view?usp=sharing) | deep_learn_dialogues.ipynb | Experiments on emotion classification at the dialogue level with CRF and RNN.  <br> For Upper-bound, please run trainer.py (detailed steps described in "deep_learn_dialogues.ipynb") |
| N/A | trainer.py | Run this file for the trainning and results of the Upper Bound (binary classification). |
