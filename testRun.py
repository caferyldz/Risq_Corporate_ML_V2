from operations.separation import Separation
from operations.algorithms.gnb import GNB
from operations.algorithms.svm import SVM
from operations.algorithms.adaboost import AdaBoost
from operations.algorithms.knn import KNN
from operations.algorithms.randomForest import RandomForest
from operations.algorithms.decisionTree import DecisionTree
from operations.algorithms.mlp import MultiLayerPerceptron
#from operations.algorithms.ffnnwhl import FFNNWHL
#from operations.algorithms.ffnnwhl_new import FFNNWHL
from operations.algorithms.cnn_5_2_new import CNN
from testt import Test
from operations.obtainData import ObtainData
from dataPreparation.featureSelection.principalComponentAnalysis import PrincipalComponentAnalysis
from dataPreparation.featureSelection.varianceThreshold import VarianceThresholdFS
from dataPreparation.featureSelection.autoEncoder import AutoEncoder
from dataPreparation.featureSelection.linearDiscriminantAnalysis import LinearDiscriminantAnalysisFS
import datetime
from dataPreparation.unbalance.randomOversampling import RandomOversampling

try:
    start = datetime.datetime.now()
    test1 = Test("5 almost empty columns")
    test1.datasetName = "selected rows final decisions only filled with mean and mode"
    test1.comment = "Random Forest algorithm applied to 5 almost empty columns on selected rows final decisions only filled with mean and mode."
    test1.algorithmType = 'Machine Learning'
    test1.algorithmName = 'Random Forest'
    test1.start()

    obtainData = ObtainData()
    # query for getting the column names from database
    obtainData.setParameters("select kolon_adi from attribute_information where orijinal = 'EVET' "
                             "and kolon_adi not in (select kolon_adi from attribute_information where orijinal = 'HAYIR' "
                             "and kolon_adi is not null) and kategorik != 'ATILACAK' and kategorik != 'BILINMIYOR' "
                             "union select kolon_adi_uretilmis from attribute_information where orijinal = 'HAYIR'")
    test1.addOperation(obtainData)
    obtainData.run()
    rawData = obtainData.getResult()
    obtainData.completeOperation()
    del obtainData  # delete object for memory management

    sprtn = Separation()
    sprtn.setParameters(rawData)
    del rawData
    test1.addOperation(sprtn)
    sprtn.run()
    features, labels = sprtn.getResult()
    sprtn.completeOperation()
    del sprtn


    # pca = PrincipalComponentAnalysis()
    # pca.setParameters(features,labels,250)
    # del features
    # test1.addOperation(pca)
    # pca.run()
    # selectedFeatures = pca.getResult()
    # pca.completeOperation()
    # del pca

    # vt = VarianceThresholdFS()
    # vt.setParameters(features,labels,0.005)
    # del features
    # test1.addOperation(vt)
    # vt.run()
    # selectedFeatures = vt.getResult()
    # vt.completeOperation()
    # del vt

    # autoen = AutoEncoder()
    # autoen.setParameters(features,labels,300)
    # del features
    # test1.addOperation(autoen)
    # autoen.run()
    # selectedFeatures = autoen.getResult()
    # autoen.completeOperation()
    # del autoen

    # ab = AdaBoost()
    # ab.setParameters([features, labels])
    # test1.addOperation(ab)
    # ab.run()
    # predictions = ab.getResult()  # predicted values
    # ab.completeOperation()
    # test1.comment = "Adaboost algorithm applied to 5 almost empty columns on selected data rows."

    # dt = DecisionTree()
    # dt.setParameters([features, labels])
    # test1.addOperation(dt)
    # dt.run()
    # predictions = dt.getResult()  # predicted values
    # dt.completeOperation()
    # test1.comment = "Decision Tree algorithm applied to selected data rows."

    # gnb = GNB()
    # gnb.setParameters([features, labels])
    # test1.addOperation(gnb)
    # gnb.run()
    # predictions = gnb.getResult()
    # gnb.completeOperation()
    # test1.comment = "Gaussian Naive Bayes algorithm algorithm applied to selected data rows with oversampling."

    # kn = KNN()
    # kn.setParameters([features, labels])
    # test1.addOperation(kn)
    # kn.run()
    # predictions = kn.getResult()  # predicted values
    # kn.completeOperation()
    # test1.comment = "KNN algorithm algorithm applied to selected data rows."

    # mlp = MultiLayerPerceptron()
    # mlp.setParameters([features, labels])
    # test1.addOperation(mlp)
    # mlp.run()
    # predictions = mlp.getResult()  # predicted values
    # mlp.completeOperation()
    # test1.comment = "Multi-layer Perceptron algorithm algorithm applied to selected data rows."

    rf = RandomForest()
    rf.setParameters([features, labels])
    test1.addOperation(rf)
    rf.run()
    predictions = rf.getResult()  # predicted values
    rf.completeOperation()
    test1.comment = "Random Forest algorithm applied to 5 almost empty columns on selected rows final decisions only filled with mean and mode."

    # svm = SVM()
    # svm.setParameters([features, labels])
    # test1.addOperation(svm)
    # svm.run()
    # predictions = svm.getResult()
    # svm.completeOperation()
    # test1.comment = "Support Vector Machine algorithm applied to selected data rows final decisions only."

    # ffnnwhl = FFNNWHL()
    # ffnnwhl.setParameters([features,labels])
    # test1.addOperation(ffnnwhl)
    # ffnnwhl.run()
    # ffnnwhl.completeOperation()
    # test1.comment = "FFNNWHL algorithm applied to data sample of 10 thousand rows for all attributes."

    # cnn = CNN()
    # cnn.setParameters([features,labels])
    # test1.addOperation(cnn)
    # cnn.run()
    # cnn.completeOperation()
    # test1.comment = "CNN 5_2 with SVM algorithm applied to balanced data 20 filled with mean and mode with kernel_regularizer 0.01 and variance threshold 0.005 wiht removing columns having more than 80% missing values."



    end = datetime.datetime.now()
    test1.duration = str(end-start)
    test1.complete()
    print("Total time spent: ", end-start)

except Exception as e:
    print(e)
    if('test1' in locals()):        # if Test instance exists
        test1.db.update_execution_history_at_failure((test1.db_id))
        if(Test.logOfExecution is not None):
            Test.logOfExecution.exception("Execution failed!")