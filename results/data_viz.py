import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    data = pd.read_csv('results/loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'Perceptual', 'Content', 'Adversarial'])
    for col in data:
        data[col] =data[col].astype(str).str.replace("[","").str.replace("]","")
    return data

def limit_decimals(x):
        return "{:.10f}".format(float(x))

def plot_all(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, np.asarray(data.loc[:, 'd_loss_fake'], float), label='d_loss_fake', linewidth=7)
    plt.plot(epochs, np.asarray(data.loc[:, 'd_loss_real'], float), label='d_loss_real', linewidth=7)
    plt.plot(epochs, np.asarray(data.loc[:, 'Perceptual'], float), label='Perceptual', linewidth=7)
    plt.plot(epochs, np.asarray(data.loc[:, 'Content'], float), label='Content', linewidth=7)
    plt.plot(epochs, np.asarray(data.loc[:, 'Adversarial'], float), label='Adversarial', linewidth=7)
    plt.title('Training and Loss', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/all.png')

def plot_d_fake(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, np.asarray(data.loc[:, 'd_loss_fake'], float), label='d_loss_fake',linewidth=10)
    plt.title('Discriminator Loss on Fake Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/d_loss_fake.png')

def plot_d_real(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, np.asarray(data.loc[:, 'd_loss_real'], float), label='d_loss_real',linewidth=10)
    plt.title('Discriminator Loss on Real Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/d_loss_real.png')

def plot_gan(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    BCE = np.asarray(data.loc[:, 'Adversarial'], float)
    BCE = BCE * 10e-3
    plt.plot(epochs, np.asarray(data.loc[:, 'Perceptual'], float), label='Perceptual',linewidth=7)
    plt.plot(epochs, np.asarray(data.loc[:, 'Content'], float), label='Content',linewidth=7)
    plt.plot(epochs, BCE, label='Adversarial',linewidth=7)
    plt.title('GAN Loss', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/gan.png')

if __name__ == '__main__':
    data = get_data()
    plot_all(data)
    plot_d_fake(data)
    plot_d_real(data)
    plot_gan(data)