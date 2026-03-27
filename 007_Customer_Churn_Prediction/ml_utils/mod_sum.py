from sklearn.metrics import roc_auc_score,roc_curve,classification_report
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import pandas as pd



def train_sum_l(mlist,X_train,y_train):
  for i in mlist:
   print("summary For Train Data of :", i)
   print("score for {}, is :{}".format (i,round(i.score(X_train,y_train),2)))
   predicted_labels_model = i.predict(X_train)
   print(classification_report(y_train, predicted_labels_model))
   tn, fp, fn, tp = confusion_matrix(y_train, predicted_labels_model).ravel()
   print("True Negative:",tn)
   print("False Negative:",fn)
   print("False Positive:",fp)
   print ("True Positive:",tp)
   ConfusionMatrixDisplay.from_estimator(i, X_train, y_train,cmap='YlGn');
   plt.title("Confusion Matrix for -{}". format(i),fontsize=10);
   plt.grid(False)
   plt.show()
   from sklearn.metrics import precision_recall_fscore_support as score
   precision,recall,fscore,support=score(y_train, predicted_labels_model)
   print("For class 0 Precision:{}; class 1 Precision:{}".format(round(precision[0],2),round(precision[1],2)))
   print("For class 0 recall:{}; class 1 recall:{}".format(round(recall[0],2),round(recall[1],2)))
   print("For class 0 fscore:{}; class 1 fscore:{}".format(round(fscore[0],2),round(fscore[1],2)))
   probs=i.predict_proba(X_train)
   probs=probs[:,1]
   auc=roc_auc_score(y_train, probs)
   print("AUC For {} is :{}".format(i,round(auc,2)))
   train_fpr, train_tpr, train_threshold = roc_curve(y_train, probs)
   plt.plot([0,1],[0,1],linestyle="--")
   plt.title("RoC curve for -{}". format(i),fontsize=10);
   plt.plot(train_fpr, train_tpr,c="green");
   plt.show()
   print("-----------------------------------------------------------------------------------------------------------------------------")






def test_sum_l(mlist,X_test,y_test):
  for i in mlist:
    print("summary For Test Data of ", i)
    print("score for {}, is :{}".format (i,round(i.score(X_test,y_test),2)));
    predicted_labels_model = i.predict(X_test)
    print(classification_report(y_test, predicted_labels_model))
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels_model).ravel()
    print("True Negative:",tn)
    print("False Negative:",fn)
    print("False Positive:",fp)
    print ("True Positive:",tp)
    ConfusionMatrixDisplay.from_estimator(i, X_test, y_test,cmap='PuRd');
    plt.title("Confusion Matrix for -{}". format(i),fontsize=10);
    plt.grid(False)
    plt.show()
    from sklearn.metrics import precision_recall_fscore_support as score
    precision,recall,fscore,support=score(y_test, predicted_labels_model)
    print("For class 0 Precision:{}; class 1 Precision:{}".format(round(precision[0],2),round(precision[1],2)))
    print("For class 0 recall:{}; class 1 recall:{}".format(round(recall[0],2),round(recall[1],2)))
    print("For class 0 fscore:{}; class 1 fscore:{}".format(round(fscore[0],2),round(fscore[1],2)))
    probs=i.predict_proba(X_test)
    probs=probs[:,1]
    auc=roc_auc_score(y_test, probs)
    print("AUC For {} is :{}".format(i,round(auc,2)))
    test_fpr, test_tpr, test_threshold = roc_curve(y_test, probs)
    plt.plot([0,1],[0,1],linestyle="--")
    plt.title("RoC curve for -{}". format(i),fontsize=10);
    plt.plot(test_fpr, test_tpr,c="red");
    plt.show()
    print("-----------------------------------------------------------------------------------------------------------------------------")





def train_sum_m(model,X_train,y_train):
   model_name = type(model).__name__
   print("summary For Train Data of ", model_name)
   print("score for {}, is :{}".format (model_name,round(model.score(X_train,y_train),2)));
   predicted_labels_model = model.predict(X_train)
   print(classification_report(y_train, predicted_labels_model))
   tn, fp, fn, tp = confusion_matrix(y_train, predicted_labels_model).ravel()
   print("True Negative:",tn)
   print("False Negative:",fn)
   print("False Positive:",fp)
   print ("True Positive:",tp)
   ConfusionMatrixDisplay.from_estimator(model, X_train, y_train, cmap='YlGn')
   plt.title("Confusion Matrix for Train Data-{}". format(model_name),fontsize=10)
   plt.grid(False)
   plt.show()
   from sklearn.metrics import precision_recall_fscore_support as score
   precision,recall,fscore,support=score(y_train, predicted_labels_model)
   print("For class 0 Precision:{}; class 1 Precision:{}".format(round(precision[0],2),round(precision[1],2)))
   print("For class 0 recall:{}; class 1 recall:{}".format(round(recall[0],2),round(recall[1],2)))
   print("For class 0 fscore:{}; class 1 fscore:{}".format(round(fscore[0],2),round(fscore[1],2)))
   probs=model.predict_proba(X_train)
   probs=probs[:,1]
   auc=roc_auc_score(y_train, probs)
   print("AUC For {} is :{}".format(model_name,round(auc,2)))
   train_fpr, train_tpr, train_threshold = roc_curve(y_train, probs)
   plt.plot([0,1],[0,1],linestyle="--")
   plt.title("RoC curve for Train Data -{}". format(model_name),fontsize=10);
   plt.plot(train_fpr, train_tpr,c="green");
   plt.show()
   print("-----------------------------------------------------------------------------------------------------------------------------")



def test_sum_m(model,X_test,y_test):
   model_name = type(model).__name__
   print("summary For Test Data of ", model_name)
   print("score for {}, is :{}".format (model_name,round(model.score(X_test,y_test),2)));
   predicted_labels_model = model.predict(X_test)
   print(classification_report(y_test, predicted_labels_model))
   tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels_model).ravel()
   print("True Negative:",tn)
   print("False Negative:",fn)
   print("False Positive:",fp)
   print ("True Positive:",tp)
   ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,cmap='PuRd');
   plt.title("Confusion Matrix for Test Data -{}". format(model_name),fontsize=10);
   plt.grid(False)
   plt.show()
   from sklearn.metrics import precision_recall_fscore_support as score
   precision,recall,fscore,support=score(y_test, predicted_labels_model)
   print("For class 0 Precision:{}; class 1 Precision:{}".format(round(precision[0],2),round(precision[1],2)))
   print("For class 0 recall:{}; class 1 recall:{}".format(round(recall[0],2),round(recall[1],2)))
   print("For class 0 fscore:{}; class 1 fscore:{}".format(round(fscore[0],2),round(fscore[1],2)))
   probs=model.predict_proba(X_test)
   probs=probs[:,1]
   auc=roc_auc_score(y_test, probs)
   print("AUC For {} is :{}".format(model_name,round(auc,2)))
   test_fpr, test_tpr, test_threshold = roc_curve(y_test, probs)
   plt.plot([0,1],[0,1],linestyle="--")
   plt.title("RoC curve for Test Data -{}". format(model_name),fontsize=10);
   plt.plot(test_fpr, test_tpr,c="red");
   plt.show()
   print("-----------------------------------------------------------------------------------------------------------------------------")



def feature_importance (model):
  cw=model.coef_
  cn=model.feature_names_in_
  class_weight=[]
  class_name=[]
  for i in range(0,9):
    class_weight.append(round(cw[0][i],2))
    class_name.append(cn[i])
  data={'class_name':class_name,'class_weight':class_weight}
  df_classinfo=pd.DataFrame(data)
  return df_classinfo.sort_values(by='class_weight',ascending=False)


