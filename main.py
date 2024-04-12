import time
from datamodule import SNSDataset
from SGDClassifier import StandardSGD

def main():
    dataset = SNSDataset()
    """
    Find hyperparameters to help your model perform at its best!
    """
    ############################################## EDIT ################################################
    model_config = {
        'penalty': 'l2',
        'max_iter': 100,
        'alpha': 0.00001,
        'learning_rate': 'optimal',
        'loss': 'hinge',
        'n_jobs': -1,
        'kfolds': 3
    }
    ####################################################################################################
    model = StandardSGD(dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test, config=model_config)
    model_start_time = time.time()
    model.run()
    print("Model Execution time : {:.4f} sec".format(time.time() - model_start_time))

if __name__ == '__main__':
    main()