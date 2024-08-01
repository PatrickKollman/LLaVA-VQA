# LLaVA-VQA
Benchmarking and iterating LLaVA performance on the VQA and VizWiz-VQA datasets.

[Data visualization](#data-visualization), [training](#training), and [evaluation](#evaluation) rely on ```src``` and ```LLaVA``` submodule dependencies.

See [VizWiz VQA Challenge 2024 results](#vizwiz-vqa-challenge-2024) for competition results.

# Repo Structure
### Data Visualization
- ```notebooks/vqa_visualization.ipynb```
- ```notebooks/vizwiz_visualization.ipynb```

### Training
- ```notebooks/llava_vizwiz_train.ipynb```

### Evaluation
- ```notebooks/llava_vizwiz_eval.ipynb```
- ```notebooks/llava_vqa_eval.ipynb```

# VizWiz VQA Challenge 2024
https://vizwiz.org/tasks-and-datasets/vqa/

### Overview
"We propose an artificial intelligence challenge to design algorithms that answer visual questions asked by people who are blind. For this purpose, we introduce the visual question answering (VQA) dataset coming from this population, which we call VizWiz-VQA.  It originates from a natural visual question answering setting where blind people each took an image and recorded a spoken question about it, together with 10 crowdsourced answers per visual question. Our proposed challenge addresses the following two tasks for this dataset: predict the answer to a visual question and (2) predict whether a visual question cannot be answered. Ultimately, we hope this work will educate more people about the technological needs of blind people while providing an exciting new opportunity for researchers to develop assistive technologies that eliminate accessibility barriers for blind people."

### Eval Leaderboard:
https://eval.ai/web/challenges/challenge-page/2185/leaderboard

### _**Team Kollman**_ Results:

| Question Type | LLaVA Baseline (llava-v1.5-7b) | Fine Tuning Results |
|:-------------:| :-----------------------------:|:------------------:|
| "yes/no" | 0.0   | 73.81 |
| "number" | 0.0   | 42.20 |
| "other" | 0.17 | 54.13 |
| "unanswerable" | 0.08 | 97.39 |
| "overall" | 0.14 | 66.41 |



