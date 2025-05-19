import matplotlib.pyplot as plt
 
# Plot the validation and training data separately
def plot_acc_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();