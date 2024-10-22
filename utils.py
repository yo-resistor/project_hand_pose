import torch
import matplotlib.pyplot as plt

# define the style of plots
plt.style.use('ggplot')

# function to save the trained model in local environment
def save_model(epochs, model, optimizer, criterion):
    """
    ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/model.pht')
    
# function to save the loss and accuracy plots in local environment
def save_plot(train_acc, valid_acc, train_loss, valid_loss):
    """
    ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    # accuracy plot
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color='green', linestyle='-', label='Train accuracy')
    plt.plot(valid_acc, color='blue', linestyle='-', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plot
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='green', linestyle='-', label='Train loss')
    plt.plot(valid_loss, color='blue', linestyle='-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')