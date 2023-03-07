# Distant reading approach to scholarly articles

The purpose of this projects is to conduct data analysis on datasets containing metadata information on a text corpus of scholarly articles, as an end-of-course project for the "Digital Text in Humanities" course of the Master Degree in Digital Humanities and Digital Knowledge of the University of Bologna, held by professor Tiziana Mancinelli.

The project work is focused on the distant reading approach to literary studies. In particular, I explored the datasets containing rich metadata about corpora of scholarly articles on one specific topic: sexual orientation, homosexuality. I have retrieved three datasets belonging to three most distinct disciplines studying the chosen phenomenon: social science, religion studies and health studies to allow comparative analysis.

In particular, I studied which subtopics are related to the topic of sexual orientation, in which contexts sexual orientation has been studied, which issues have been addressed by the chosen disciplines.

## Repository content
- <a href='https://github.com/AnastasiyaSopyryaeva/DTH/tree/main/scripts'>Scripts with data analysis process</a>
- Detailed project description is reported via GitHub pages on the <a href='https://anastasiyasopyryaeva.github.io/DTH/'>following website</a>

## Resources and tecchnologies used for the project

### Datasets
The datasets were downloaded from <a href="https://www.jstor.org/action/showAdvancedSearch">JSTOR digital library</a> according to the specified settings as jsonl files. 
The datasets have been selected for their size and complexity from JSTOR digital library and are used exclusively for educational purpose.

### Tools
As a tool for data manipulation, visualisation and analysis I used Python with various libraries, mainly:

- Pandas, in order to read .csv files and convert them to Python-friendly dataframes;
- NLTK for natural language processing;
- matplotlib for drawing plots and graphs;
- gensim for LDA topic modelling.

### Measures
To answer research questions, I used various natural language processing techniques on the text corpus to prepare the data, calculated descriptive statistics on the dataset and, finally, applied LDA topic modeling for subtopics detection. The detailed description of methods can be found in Data Analysis part of the report.
