import seaborn as sns
import matplotlib.pyplot as plt
def plot_CM(cm):
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()