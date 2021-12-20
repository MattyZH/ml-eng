import click
from plotting import *
from reading_files import *


@click.command()
@click.option('--max_k', type=click.INT, default=30, help='max k for kNN')
def plot_everything(max_k):
    X, y = read_cancer_dataset(snakemake.input.cancer)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    plot_precision_recall(X_train, y_train, X_test, y_test, max_k=max_k, directory=snakemake.output.cancer)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=max_k, directory=snakemake.output.cancer)
"""
    X, y = read_spam_dataset(snakemake.input.spam)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    plot_precision_recall(X_train, y_train, X_test, y_test, max_k=max_k, directory=snakemake.output.spam)
    plot_roc_curve(X_train, y_train, X_test, y_test, max_k=max_k, directory=snakemake.output.spam)
"""

if __name__ == '__main__':
    plot_everything()
