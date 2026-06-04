from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Trener en LDA modell
def train_lda(X_train, y_train):
    lda = LinearDiscriminantAnalysis() # Lager modellen
    lda.fit(X_train, y_train)           #trener modell med treningsdata
    return lda

#Trender en MLP
def train_mlp(X_train, y_train):
    mlp = MLPClassifier(max_iter=5000, random_state=42) #500 iterasjoner, for å sikre samme resultat 42
    mlp.fit(X_train, y_train) #Trener med treningsdata
    return mlp

#Tester en valgfri modell og regner ut resultater
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test) #Modellen predikerer svarene

    accuracy = accuracy_score(y_test, y_pred) #Regner ut accuracy
    matrix = confusion_matrix(y_test, y_pred) #Lager confusion matrix

    return accuracy, matrix #Sender begge verdier tilbake
