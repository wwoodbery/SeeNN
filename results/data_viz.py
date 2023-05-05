import pandas as pd
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('results/loss.txt', names=['Epochs', 'd_loss_fake', 'd_loss_real', 'loss_gan', 'loss_gan1', 'loss_gan2'])
    df = data
    for col in df:
        df[col] =df[col].astype(str).str.replace("[","").str.replace("]","")
    return data

def plot_all(data):
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
    plt.savefig('results/graphs/all.png')
    print(data)

def plot_d_fake(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, data.loc[:, 'd_loss_fake'], label='d_loss_fake')
    plt.title('Discriminator Loss on Fake Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/d_loss_fake.png')

def plot_d_real(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, data.loc[:, 'd_loss_real'], label='d_loss_real')
    plt.title('Discriminator Loss on Real Images', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/d_loss_real.png')

def plot_gen(data):
    epochs = data.loc[:, 'Epochs']
    plt.figure(figsize=(20,15))
    plt.plot(epochs, data.loc[:, 'loss_gan'], label='loss_gan')
    plt.plot(epochs, data.loc[:, 'loss_gan1'], label='loss_gan1')
    plt.plot(epochs, data.loc[:, 'loss_gan2'], label='loss_gan2')
    plt.title('Generator Loss', size=40)
    plt.xlabel('Epochs', size=30)
    plt.ylabel('Loss', size=30)
    plt.legend(fontsize=20)
    plt.savefig('results/graphs/generator.png')

if __name__ == '__main__':
    data = get_data()
    plot_all(data)
    plot_d_fake(data)
    plot_d_real(data)
    plot_gen(data)