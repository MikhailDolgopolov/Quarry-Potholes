import matplotlib.pyplot as plt

from exploration.data_read import read_track

if __name__ == '__main__':
    # target, ws = 'class', 10
    # df_full = load_prepared(f'data/{target}{ws}', keep_latlon=True, sample_frac=1)
    # print(len(df_full))

    df = read_track('data/routes/route5/2_w.csv')
    # print(df.describe())
    #
    plt.figure(figsize=(10, 6))
    plt.plot([0] * len(df),  '--', color='gray',)
    plt.plot(df['acc_X'], label='X', alpha=0.8)
    plt.plot(df['acc_Y'], label='Y', alpha=0.8)
    plt.plot(df['acc_Z']*10, label='10х Acceleration Z', alpha=0.8)


    # Add labels and title
    plt.xlabel('1Hz readings')
    plt.ylabel('Acceleration (m/s2)')
    plt.title('Accelerometers')
    plt.legend()
    plt.show()



