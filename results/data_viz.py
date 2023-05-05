import pandas as pd
import matplotlib.pyplot as plt

def plot_all():
    data = pd.read_csv('loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'loss_gan', 'loss_gan1', 'loss_gan2'])
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, data.loc[:, 'd_loss_fake'], label='d_loss_fake')
    plt.plot(epochs, data.loc[:, 'd_loss_real'], label='d_loss_real')
    plt.plot(epochs, data.loc[:, 'loss_gan'], label='loss_gan')
    plt.plot(epochs, data.loc[:, 'loss_gan1'], label='loss_gan1')
    plt.plot(epochs, data.loc[:, 'loss_gan2'], label='loss_gan2')
    plt.title('Training and Loss', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('graphs/all.png')

def plot_d_fake():
    data = pd.read_csv('loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'loss_gan', 'loss_gan1', 'loss_gan2'])
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(15,15))
    plt.plot(epochs, data.loc[:, 'd_loss_fake'], label='d_loss_fake')
    plt.title('Discriminator Loss on Fake Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('graphs/d_loss_fake.png')

def plot_d_real():
    data = pd.read_csv('loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'loss_gan', 'loss_gan1', 'loss_gan2'])
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(15,15))
    plt.plot(epochs, data.loc[:, 'd_loss_real'], label='d_loss_real')
    plt.title('Discriminator Loss on Real Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('graphs/d_loss_real.png')

def plot_gen():
    data = pd.read_csv('loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'loss_gan', 'loss_gan1', 'loss_gan2'])
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, data.loc[:, 'loss_gan'], label='loss_gan')
    plt.plot(epochs, data.loc[:, 'loss_gan1'], label='loss_gan1')
    plt.plot(epochs, data.loc[:, 'loss_gan2'], label='loss_gan2')
    plt.title('Generator Loss', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('graphs/generator.png')

if __name__ == '__main__':
    plot_all()
    plot_d_fake()
    plot_d_real()
    plot_gen()