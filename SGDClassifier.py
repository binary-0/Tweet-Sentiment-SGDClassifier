import numpy as np
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import clone

class StandardSGD:
    def __init__(self, X_train, X_test, y_train, y_test, config):
        self.config = config
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier = SGDClassifier(penalty=config['penalty'], max_iter=config['max_iter'], loss=config['loss'], alpha=config['alpha'], learning_rate=config['learning_rate'], verbose=1, n_jobs=config['n_jobs'], random_state=42)

    def cross_val_train(self):
        """
        Performs K-Fold Cross-Validation to train the model. This is essential for assessing the 
        model's performance and robustness across different subsets of the data.

        Process:
        1. For each fold, clone the original model and train it using the training subset.
        2. Validate the trained model on the validation subset and append the validation accuracy to 'dev_accuracies'.

        Outputs:
        Prints the mean and standard deviation of K-Fold validation accuracies.
        """
        skfolds = StratifiedKFold(n_splits=self.config['kfolds'], shuffle=True, random_state=42,)
        dev_accuracies = []
        print("--------------------------------CROSS_VAL_TRAIN_START------------------------------\n", end='')
        ############################################## EDIT ################################################
        
        for train_index, test_index in skfolds.split(self.X_train, self.y_train):
            cloner = clone(self.classifier)
            
            X_train_folds, y_train_folds, X_test_fold, y_test_fold = self.X_train[train_index], self.y_train[train_index], self.X_train[test_index], self.y_train[test_index]

            cloner.fit(X_train_folds, y_train_folds)
            predY = cloner.predict(X_test_fold)
            foldAcc = accuracy_score(y_test_fold, predY)
            dev_accuracies.append(foldAcc)
            
        ####################################################################################################
        print("---------------------------------CROSS_VAL_TRAIN_END---------------------------\n\n\n", end='')

        print("---------------------------------CROSS_VAL_TRAIN_RESULT------------------------\n", end='')
        print(f"K-Fold Validation Mean accuracy: {np.mean(dev_accuracies)}")
        print(f"K-Fold Validation Standard deviation: {np.std(dev_accuracies)}")
        print("-------------------------------------------------------------------------------\n\n\n", end='')
    
    def train(self):
        """
        Train the model on whole train set.
        """
        print("------------------------------------TRAIN_START-------------------------------------\n", end='')
        ############################################## EDIT ################################################
        self.classifier.fit(self.X_train, self.y_train)
        ####################################################################################################
        print("-------------------------------------TRAIN_END-------------------------------------\n\n\n", end='')

    def test(self):
        """
        Test the trained model on the test set.
        """
        print("-------------------------------------TEST_START------------------------------------\n", end='')
        ############################################## EDIT ################################################
        y_test_pred = self.classifier.predict(self.X_test)
        ####################################################################################################
        print(f'Test Accuracy for SGD Classifier is {accuracy_score(self.y_test, y_test_pred):.4f}')
        print(f'Test F1 score for SGD Classifier is {f1_score(self.y_test, y_test_pred, average="macro"):.4f}')  
        print("-------------------------------------TEST_END--------------------------------------\n\n\n", end='')

    def run(self):
        self.cross_val_train()
        self.train()
        self.test()
