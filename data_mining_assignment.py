import pandas as pd

"""Making Datasets from three CSVs"""

ds1 = pd.read_csv("titanic/test.csv")

ds2 = pd.read_csv("titanic/train.csv")

ds3gender = pd.read_csv("titanic/gender_submission.csv")

"""Merging two datasets(Integrating)"""

ds1["Survived"] = ds3gender["Survived"]

"""Concatenating two datasets"""

ds1 = ds1[ds2.columns]
new_dataset = pd.concat([ds2, ds1], ignore_index=True)

(new_dataset.isnull().sum())

"""Droping tupples with empty 'Embarked' value (because they are few)"""

new_dataset.dropna(subset=['Embarked'], inplace=True)

"""Filling missing values of age with the median of age column because data is unbalanced"""

new_dataset['Age'].fillna(new_dataset['Age'].median(), inplace=True)

"""Droping tupples with empty 'Fare' value (because they are few)"""

new_dataset.dropna(subset=['Fare'], inplace=True)

"""Adding a column which tells that the person owns a cabin or not"""

new_dataset['HasCabin'] = new_dataset['Cabin'].notnull().astype(int)

"""droping cabin column"""

new_dataset.drop(columns = 'Cabin', inplace=True)

"""Extracting Last names of the persons"""

new_dataset['Last_Name'] = new_dataset['Name'].str.split(',').str[0]

"""Extracting titles of the all the persons like Mr., Miss., Mrs."""

new_dataset['Title'] = new_dataset['Name'].str.split('.').str[0].str.split(',').str[1].str.strip()

"""Extracting first names of Men and Single Ladies"""

names_of_singles = new_dataset[(new_dataset['Title'] != 'Mrs') | (new_dataset['Title'] != 'Mme')]['Name']
first_name = names_of_singles.str.split(".").str[1].str.strip()
new_dataset.loc[(new_dataset['Title'] != 'Mrs') | (new_dataset['Title'] != 'Mme'), 'First_Name'] = first_name

"""Extracting first names of Married Ladies"""

names_of_ML = new_dataset[(new_dataset['Title'] == 'Mrs') | (new_dataset['Title'] == 'Mme')]['Name']
first_name_of_ML = names_of_ML.str.split("(").str[1].str.rstrip(")").str.strip()
new_dataset.loc[(new_dataset['Title'] == 'Mrs') | (new_dataset['Title'] == 'Mme'), 'First_Name'] = first_name_of_ML

"""droping the name column because useful info has been extracted"""

new_dataset.drop(columns=['Name'], inplace=True)

"""Removing dots from ticket numbers"""

new_dataset['Ticket'] = new_dataset['Ticket'].str.replace('.','')

"""Filling 'Fare' column with median of it column because generally no one travels without paying fare"""

new_dataset['Fare'].replace(0, new_dataset['Fare'].median(), inplace=True)

"""Ticket Binning"""

new_dataset['Binned_Fare'] = pd.qcut(new_dataset['Fare'], 4, labels=['low', 'medium', 'high', 'very high'])

"""Age Binning"""

bins = [0, 2, 12, 19, 35, 50, 60, float('inf')]
labels = ['Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
new_dataset['Binned_Age'] = pd.cut(new_dataset['Age'], bins=bins, labels=labels)

"""Rearranging the columns"""

new_dataset = new_dataset[['PassengerId','Survived','Title','First_Name', 'Last_Name', 'Sex', 'Age','Binned_Age','SibSp','Parch','Ticket','Pclass','Embarked','HasCabin','Fare','Binned_Fare']]

"""Saving final result into CSV"""

new_dataset.to_csv('output.csv', index=False)
