"""Stable module entry point so preprocessing objects can be reloaded."""
import argparse
from pathlib import Path
from pipeline import BASE, run

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=BASE/'data/MWCCS_full_5844.csv')
    parser.add_argument('--output',type=Path,default=BASE/'results')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
