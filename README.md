# Vector-search-on-LLM-using-MongoDB-Atlas

# Overview
This experiment involves evaluating a LLM using vector search  on ragas-wikiqa dataset. The objective is to assess the model's performance in answering questions based on Wikipedia passages. In this experiment, we have used MongoDB Atlas to store and retrieve vector embeddings.
# Files
•	llm_gpt_3.5_turbo.ipynb: Jupyter Notebook containing the experiment's implementation, including data processing, evaluation, and analysis using gpt-3.5-turbo model.<br/>
• llm_gpt_4o_mini.ipynb: Jupyter notebook containing the experiment's implemenatation using gtp-4o-mini model.

# Requirements
To run the notebook, install the following necessary dependencies:<br/>
•	Python 3.12 or later<br/>
•	Ragas<br/>
•	Hugging Face Transformers<br/>
•	Datasets<br/>
•	Pandas<br/>
•	Jupyter Notebook<br/>
•	Ragas<br/>
•	Hugging Face Transformers<br/>
•	Datasets<br/>
•	Pandas<br/>
•	Jupyter Notebook<br/>
•	Tqdm<br/>
•	Matplotlib<br/>
•	Seaborn<br/>
# Running the Experiment
1.	Load the Dataset: Load the dataset from Hugging Face.
2.	Preprocess the Data: Perform any necessary preprocessing on the dataset.
3.	Evaluate with RAGAS: Use the RAGAS framework to evaluate the RAG system. This involves setting up the evaluation metrics and running the evaluation on the dataset.
4.	Visualize Results: Use matplotlib and seaborn to visualize the evaluation results.
Evaluation Metrics:
The experiment uses the Ragas framework to evaluate the model based on:<br/>
•	Answer Relevance: Measures how relevant the generated answer is to the query.<br/>
•	Context Relevance: Evaluates the connection between the retrieved context and the query.<br/>
•	Faithfulness: Assesses the accuracy of the generated answer concerning the context.<br/>

# Conclusion
This experiment demonstrates the use of the RAGAS framework to evaluate a RAG system on the WikiQA dataset. The results provide insights into the performance of the system and highlight areas for improvement.
# Acknowledgments
•	The RAGAS framework by Exploding Gradients.<br/>
•	The WikiQA dataset for providing a benchmark for question-answering systems.<br/>
•	OpenAI for providing the language models used in this experiment.<br/>
•	MongoDB Atlas for providing database to store and retrieve data.<br/>






