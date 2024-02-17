"""
The Python script is designed to demonstrate how to implement and evaluate decision tree classifiers 
using a dataset of animals and their characteristics. It loads the dataset, prepares the data by merging 
and selecting features, then trains decision tree models with different configurations. 

The script evaluates each model's performance using metrics like accuracy and generates reports to 
understand the models better. It's structured to provide beginners with insight into machine learning 
workflows, including data preparation, model training, evaluation, and interpretation of results.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
import sklearn.exceptions


df_zoo = pd.read_csv("zoo.csv")
df_class = pd.read_csv("class.csv")

df_all = df_zoo.merge(df_class,how='left',left_on='class_type',right_on='Class_Number')
# df_all.head()

g = df_all.groupby(by='Class_Type')['animal_name'].count()
g / g.sum() * 100


sns.countplot(df_all['Class_Type'],label="Count",
             order = df_all['Class_Type'].value_counts().index) #sort bars
plt.show()

df_all = df_zoo.merge(df_class,how='left',left_on='class_type',right_on='Class_Number')

gr = df_all
df_all.head()

columns = ['Class_Number','Number_Of_Animal_Species_In_Class','class_type','animal_name','Animal_Names']
gr.drop(columns, inplace=True, axis=1)

gr = gr.groupby(by='Class_Type').mean()
plt.subplots(figsize=(10,4))
sns.heatmap(gr, cmap="YlGnBu")

# Model 1
# Let's see how well a decision tree works if we use all of the features available to us and training with 20% of the data.¶
feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic']
X = df_zoo[feature_names]
y = df_zoo['class_type'] #there are multiple classes in this column

#split the dataframe into train and test groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

#specify the model to train with
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) #ignores warning that tells us dividing by zero equals zero
#let's see how well it worked
pred = clf.predict(X_test)

accuracy_model1 = clf.score(X_test, y_test)
print('Accuracy of classifier on test set: {:.2f}'.format(accuracy_model1))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))

df_all = df_zoo.merge(df_class,how='left',left_on='class_type',right_on='Class_Number')

#this is the order of the labels in the confusion matrix above
df_all[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type')

# What features were the most important in this model?
imp = pd.DataFrame(clf.feature_importances_)
ft = pd.DataFrame(feature_names)
ft_imp = pd.concat([ft,imp],axis=1).reindex(ft.index)
ft_imp.columns = ['Feature', 'Importance']
ft_imp.sort_values(by='Importance',ascending=False)

# What if we reduced the training set size to 10%?

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1, test_size=.9) 
clf2 = DecisionTreeClassifier().fit(X_train, y_train)
pred = clf2.predict(X_test)
accuracy_model2 = clf2.score(X_test, y_test)
print('Accuracy of classifier on test set: {:.2f}'.format(accuracy_model2))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))

imp2 = pd.DataFrame(clf2.feature_importances_)
ft_imp2 = pd.concat([ft,imp2],axis=1).reindex(ft.index)
ft_imp2.columns = ['Feature', 'Importance']
ft_imp2.sort_values(by='Importance',ascending=False)

# Model 3
# Let's go back to 20% in the training group and focus on visible features of the animals.
visible_feature_names = ['hair','feathers','toothed','fins','legs','tail']

X = df_zoo[visible_feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)
clf3= DecisionTreeClassifier().fit(X_train, y_train)

pred = clf3.predict(X_test)
accuracy_model3 = clf3.score(X_test, y_test)
print('Accuracy of classifier on test set: {:.2f}'.format(accuracy_model3))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))

imp3= pd.DataFrame(clf3.feature_importances_)
ft = pd.DataFrame(visible_feature_names)
ft_imp3 = pd.concat([ft,imp3],axis=1).reindex(ft.index)
ft_imp3.columns = ['Feature', 'Importance']
ft_imp3.sort_values(by='Importance',ascending=False)

# Model 4
# If the dataset were larger, reducing the depth size of the tree would be useful to minimize memory required to perform the analysis. Below I've limited it to two still using the same train/test groups and visible features group as above.
clf4= DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

pred = clf4.predict(X_test)
accuracy_model4 = clf4.score(X_test, y_test)
print('Accuracy of classifier on test set: {:.2f}'.format(accuracy_model4))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))

imp4= pd.DataFrame(clf4.feature_importances_)
ft_imp4 = pd.concat([ft,imp3],axis=1).reindex(ft.index)
ft_imp4.columns = ['Feature', 'Importance']
ft_imp4.sort_values(by='Importance',ascending=False)

columns = ['Model','Test %', 'Accuracy','Precision','Recall','F1','Train N']
df_ = pd.DataFrame(columns=columns)

df_.loc[len(df_)] = ["Model 1",20,.78,.80,.78,.77,81]
df_.loc[len(df_)] = ["Model 2",10,.68,.62,.68,.64,91]
df_.loc[len(df_)] = ["Model 3",20,.91,.93,.91,.91,81]
df_.loc[len(df_)] = ["Model 4",20,.57,.63,.57,.58,81]
ax=df_[['Accuracy','Precision','Recall','F1']].plot(kind='bar',cmap="YlGnBu", figsize=(10,6))
ax.set_xticklabels(df_.Model)
