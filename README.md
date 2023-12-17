# Logistic Regression in the Diagnosis of Parkinson's Disease / Parkinson Hastalığı Tanısında Lojistik Regresyon (tr)
In this notebook we will implement logistic regression to diagnose parkinson disease, the dataset is obtained from Oxford Parkinson's Disease Detection Dataset (https://archive.ics.uci.edu/dataset/174/parkinsons)

## Dataset
### Information
- This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD. 

- The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.For further information or to pass on comments, please contact Max Little (littlem '@' robots.ox.ac.uk).

- Here is a brief of the dataset
![Dataset](https://github.com/faridnec/parkinson-regression/blob/main/img/dataset.png?raw=true)

- Reference:

  Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering
