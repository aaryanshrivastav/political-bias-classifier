# Political Bias Classifier

A machine learning pipeline that detects **political bias** (Left, Right, Center) in text or YouTube videos.
The model is powered by a **fine-tuned Transformer** hosted on Hugging Face, and includes support for **multilingual input**, **automatic transcription**, and **translation** using **OpenAI Whisper** and **Google Gemini**.

##  Table of Contents

- [ Overview](#overview)
- [ Features](#features)
- [ Project Structure](#project-structure)
- [ Getting Started](#getting-started)
  - [ Prerequisites](#prerequisites)
  - [ Installation](#installation)
  - [ Usage](#usage)
- [ Contributors](#contributors)
- [ License](#license)

##  Overview

**Political Bias Classifier** provides an end-to-end solution for detecting ideological bias in media.
It supports:
* Raw text classification
* YouTube video transcription
* Automatic translation to English
* Bias detection using a fine-tuned Hugging Face model: **[`Arstacity/political-bias-classifier`](https://huggingface.co/Arstacity/political-bias-classifier)**

### Bias Categories:
* **Left**
* **Right**
* **Center**


##  Features

* **Automatic YouTube Transcription** using OpenAI Whisper
* **Language Detection & Translation** via Google Gemini
* **Fine-tuned Transformer Model** for political bias
* **Chunked Long-Text Processing** for consistent predictions
* **Clean, Modular Architecture** (Training, Classification, Transcription)


##  Project Structure

```sh
└── political-bias-classifier/
    ├── Generation
    │   ├── channels.py
    │   └── data_pull.py
    ├── LICENSE
    ├── Model.py
    ├── README.md
    ├── TranscriptGenerator.py
    ├── TranscriptTranslator.py
    ├── data_prep.py
    ├── main.py
    ├── requirements.txt
    └── train_transfromer.py
```

##  Getting Started

###  Prerequisites

Before getting started with political-bias-classifier, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

1. Clone the political-bias-classifier repository:
```sh
❯ git clone https://github.com/aaryanshrivastav/political-bias-classifier
```

2. Navigate to the project directory:
```sh
❯ cd political-bias-classifier
```

3. Install the project dependencies:
```sh
❯ pip install -r requirements.txt
```




###  Usage
Run political-bias-classifier using the following command:
**Using `pip`** 

```sh
❯ python main.py
```

##  Contributors
<table>
    <tr align="center" style="font-weight:bold">
        <td>
        Amlan Pal
        <p align="center">
            <img src="https://avatars.githubusercontent.com/Amlan2005ED" width="150" height="150" alt="Aryan Deshpande">
        </p>
            <p align="center">
                <a href="https://github.com/Amlan2005ED">
                    <img src="http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height="36" alt="GitHub"/>
                </a>
            </p>
        </td>
        <td>
        Aaryan Shrivastav
        <p align="center">
            <img src="https://avatars.githubusercontent.com/aaryanshrivastav" width="150" height="150" alt="Aaryan Shrivastav">
        </p>
            <p align="center">
                <a href="https://github.com/aaryanshrivastav">
                    <img src="http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height="36" alt="GitHub"/>
                </a>
            </p>
        </td>
     <td>
        Aditya Saha
        <p align="center">
            <img src="https://avatars.githubusercontent.com/AdityaSaha19" width="150" height="150" alt="Rishab Nagwani">
        </p>
            <p align="center">
                <a href="https://github.com/AdityaSaha19">
                    <img src="http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height="36" alt="GitHub"/>
                </a>
            </p>
        </td>
    </tr>
</table>


##  License

This project is protected under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.
