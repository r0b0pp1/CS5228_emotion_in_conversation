# Emotion Classification on Textual Data from MELD

### Project Setup
- All our jupyter notebooks were created and run in Google Colab.
- You can use our Colab links provided below (recommended), or manually import the notebooks in this repository into Colab.
- You are highly recommended to use Colab as we do not explicitly pip install any python libraries already included in Colab.
- Please access our Project Workspace here: [Google Drive](https://drive.google.com/drive/folders/12nyL0c1F0xrYTEIx4TzRlNKcxRRwpyGk?usp=drive_link)

### Data Downloads
- Our scripts already perform the downloading of data available in this repository automatically.
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
| Link to Colab | Script | Contents |
|-----|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14eWUL5Gm4H174d11yCNpVvLjZqoPMBqo?usp=sharing) | data_pre-processing.ipynb | Data Cleaning Steps <br> Rebalancing of Dataset and related diagrams |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U_Fk6Wlg_LXSfZrtPVXDYG5tUCn_pTkI?usp=sharing) | data_visualisation_analysis.ipynb | Utterance Length Analysis <br> Word Clouds <br> Linear Separability of Data with TSNE  |

### Classical Models
| Link to Colab | Script | Contents |
|-----|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W0lXmPfl_3I41pnIOJ1OttLyOpvdokVJ?usp=sharing) | decision_trees.ipynb | Experiments on emotion classification with Decision Trees and Random Forests.  |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() | logistic_regression.ipynb | Experiments on emotion classification with Logistic Regression. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1e8Q0trb1bVDK5XQPuHo0pvBQu3dzrYW9/view?usp=sharing) | logistic_regression_sentiment.ipynb | Experiments on sentiment classification with Logistic Regression. |

### Deep Learning Models
| Link to Colab | Script | Contents |
|-----|:--------|:-----------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/151wnoLEw3v8cQj7mNbWQhS1P8i_g0FPY/view?usp=sharing) | deep_learn_dialogues.ipynb | Experiments on emotion classification at the dialogue level with CRF and RNN.  |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1v9k3hxvFgD9Za72pnwjz9guxRR8ypGOa/view?usp=sharing) | deep_learn_dialogue_sentiment.ipynb | Experiments on sentiment classification.  |
| N/A | trainer.py | Python script for running model training in the background. |
