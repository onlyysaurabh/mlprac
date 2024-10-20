import numpy
from sklearn import metrics
import matplotlib.pyplot as plt

n = 1000
actual = numpy.random.binomial(1, 0.9, size=n)
predicted = numpy.random.binomial(1, 0.9, size=n)

accuracy = metrics.accuracy_score(actual, predicted)
precision = metrics.precision_score(actual, predicted)
recall = metrics.recall_score(actual, predicted)
specificity = metrics.recall_score(actual, predicted, pos_label=0)
fScore = metrics.f1_score(actual, predicted)
confMat = metrics.confusion_matrix(actual, predicted)

print(f"Confusion Matrix: \n {confMat}")
print(f"Accuracy: {round(accuracy, 3)} \tPrecision: {round(precision, 3)}\nRecall: {round(recall, 3)} \t\tSpecificity: {round(specificity, 3)}\nF Score: {round(fScore, 3)}")

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=[0, 1])
cm_display.plot(cmap=plt.cm.bone_r)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Data') 
plt.show()