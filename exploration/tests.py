import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from exploration.data_read import read_recoded_track

if __name__ == '__main__':

    df=read_recoded_track('data/input-recoded/10peaks/route1/1_w.csv')
    print(df.describe())

