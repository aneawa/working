import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

x = np.random.rand(1000)
y = np.random.randn(1000) > 0.01 - x
fpr, tpr, thresholds = roc_curve(y, x)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='r', lw=2,
         label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='b', linestyle='--')
plt.title('ROC curve sample')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()