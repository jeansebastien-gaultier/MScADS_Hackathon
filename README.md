# MScADS_Hackathon
### Team
**Jean-Sebastien Gaultier** - Current Master in Applied Data Science Student at the University of Chicago

## Challenge
This Hackathon was organised by the Master of Science in Applied Data Science Program from The University of Chicago from June 27th to July 2nd. The goal being to: "Develop a Gen AI application using foundational models (LLM & MLLM) to enhance the analysis of unstructured (text & image), semi-structured (i.e. CSV), and structured data (i.e. tables), such as research papers, financial and medical mixed data documents. This will help organizations improve the quality of their analysis, boost productivity, and streamline workflows." 


## Problem Statement
### For Hospitals
- **Goal**: Leverage a Large Language Model (LLM) to reduce the average wait time in Emergency Rooms (ER) by assisting doctors in quickly assessing patient conditions.
- **Background**: According to an article published on October 5th, 2023, by Employee Benefit News, the average wait time for Americans in the ER exceeds 2 hours and 25 minutes. Long wait times in the ER can lead to increased patient discomfort, delayed treatments, and overall strain on healthcare resources.
- **Challenge**: Develop a solution that utilizes an LLM to assist doctors by linking the current patient's symptoms with similar past cases. This could enable faster and more accurate preliminary assessments, ultimately reducing the overall time patients spend in the ER.
- **Advantages**:
  - Reduce wait time in the ER
  - Create a centralised database to easily access historical patient data (i.e. How many diabetic patient's were met this year?)

## Project Purpose
### Data
- **mtsamples_with_rand_names.csv**: contains semi-structured and unstructured medical documents, which include de-identified information and mock patient names.
### Description
- 8 columns - all string
  - **Unnamed**: index
  - **description**: Quick resume of patient's situation
  - **medical_specialty**: Field of medicine
  - **sample_name**: identifier for each record
  - **transcription**: Meeting description with doctor (doctor's notes), includes various elements such as MEDICATION, ASSESSMENT, ALLERGIES, PHYSICAL EXAMINATION (different for each row)
  - **keywords**
  - **first_name**
  - **last_name**
- 4999 rows
- Missing Values
  - 33 in transcription
  - 1068 in keywords
- Bias: Surgery Accounts for around 20% of the medical specialty in the dataset
- **Unit of Analysis**: A unique patient examination

 ## Methodology
- Model Engineering
  - There were different ideas in terms of chunking (some parts of the CSV were very short, but the transcription is longer)

- Tools:
  - Document Loading: "CSVLoader" - Load the csv file into a list of Document files (text + metadata)
  - Document Chunking: custom function using "RecursiveCharacterTextSplitter" in an attempt at keeping the biggest chunks together
  - Embeddings: There are various models that can be used.
    - MistralAI: crashed midway so I had to change
    - CohereAI: stopped me at 1000 API calls
    - HuggingFace
    - Ollama
  - Vectorstore: "Chroma" it is an open-source database where we can store our embeddings

 ## Results
- After attempting to run various different LLM and embedding, we have decided to stay with the CohereAI and HuggingFace BGE embedding for the final deployment.

 ## Conclusion
 - THE FINAL CODE IS IN THE FINAL FOLDER. The folder includes the code for the RAG model and the streamlit app. I am content with the work I have produced, I believe there is a lot that can be optimized and upgraded in the future. I really appreciated the opportunity to work on this Hackathon (even if I was by myself), because I believe it allowed me to do a lot of research on a subject that is also part of my Capstone Project. I believe there are a lot of things that I would want to add to my work so far, I started reading about RAPTOR (for high-level questions and I thought was interesting).
 - 

 # Running the UI
 The UI URL would not work (I do not understand why I cannot deploy on Streamlit). Please run the following command in your terminal to see the model run (this is with smaller amounts of data so that it does not take too long for you to have it working if you would like to test it out).
 
 ```shell
streamlit run Final\ Code/streamlit_app.py
```

