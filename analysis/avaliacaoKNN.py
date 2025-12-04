# avaliacao com knn e validacao cruzada

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def avaliarKNN(features, labels, k=5, nfolds=5):
    # avalia as features com knn e validacao cruzada
    # retorna dicionario com acuracia e f1 de cada estagio
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    resultados = {}
    
    for estagio, X in features.items():
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # validacao cruzada
        acuracias = cross_val_score(knn, X, y, cv=nfolds, scoring='accuracy')
        
        # f1 score
        knn.fit(X, y)
        predicao = knn.predict(X)
        f1 = f1_score(y, predicao, average='weighted')
        
        resultados[estagio] = {
            'accuracy_mean': acuracias.mean() * 100,
            'accuracy_std': acuracias.std() * 100,
            'f1_score': f1 * 100,
            'dim': X.shape[1]
        }
    
    return resultados

