import pandas as pd

# rows = obervaciones
# columns = characteristics
dataset = pd.read_csv(r'C:\Users\Ivanc\Documentos\UPY\machine\Social_Network_Ads.csv', sep = ",")
print(dataset)


# removes this column because it is not usefull
dataset = dataset.drop(columns = "User ID")

# print(dataset = dataset.add(columns = "Male"))
# print(dataset = dataset.add(columns = "Female"))
# dataset = dataset.insert(4, "Male", " ")
dataset["Male"] = (dataset["Gender"] == "Male").astype(int)
dataset["Female"] = (dataset["Gender"] == "Female").astype(int)

dataset = dataset.drop(columns = "Gender")
columns_order = list(dataset.columns.difference(["Male", "Female", "Purchased"])) + ["Male", "Female"] + ["Purchased"]
dataset = dataset[columns_order]
print("____________________________________________________")
#print(dataset.columns)
print(dataset.head(5))



from sklearn.linear_model import Perceptron
# y = label
# x = features
x_train = dataset.iloc[:319 , 0:4] 
y_train = dataset.iloc[:319 , 4]


x_test = dataset.iloc[320: , 0:4]
y_test = dataset.iloc[320: , 4]

# print(x_test.head(1))
# print(x_train.head(1))
# print(y_test.head(1))
# print(y_train.head(1))

# X, y = (return_X_y=True)


clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(x_train, y_train)
Perceptron()
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))

