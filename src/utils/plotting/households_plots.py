import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_timestep_logs(data):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axs = axs.flatten()

    titles = ['Total Imported Energy', 'Total Exported Energy', 'Total SOC', 'Total Produced Energy']
    colors = ['blue', 'green', 'red', 'purple']

    for i, column in enumerate(data.columns):
        axs[i].bar(data.index, data[column], color=colors[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Timestep')
        axs[i].set_ylabel(column)

    plt.tight_layout()
    plt.show()


def plot_per_household_logs(data):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 12))
    axs = axs.flatten()

    columns = data.columns
    titles = [col.replace('_', ' ').title() for col in columns]

    colors = sns.color_palette("viridis", n_colors=len(columns))

    for i, (column, color) in enumerate(zip(columns, colors)):
        sns.barplot(data=data, x=data.index, y=column, ax=axs[i], color=color)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Household Index')
        axs[i].set_ylabel(column)

    plt.tight_layout()
    plt.show()
