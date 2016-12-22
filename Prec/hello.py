import numpy as np
import pandas as pd


def delta(N, d_H, eta):
    return np.sqrt(1.0 / N * (d_H * (np.log(2 * N / d_H) + 1) - np.log(eta / 4)))


def delta2(N, d_h, eta):
    a = np.log(2 * N / d_h) + 1
    b = d_h * a
    c = b - np.log(eta / 4)
    return np.sqrt(1.0 / N * c)


def dump(f_input, f_output):
    with open(f_input) as f:
        exec('from numpy import array')
        la = []
        for line in f.readlines():
            cmds = line.split('\t')
            for cmd in cmds:
                # print(cmd.strip())
                exec(cmd.strip())
                # print(eval(cmd.strip()))

            a = eval(
                '(prec, q, tau, opt_model[1], opt_model[2], opt_model[3], p2[1][1], p2[1][0]+p2[1][1], p1[1][0]+p1[1][1])')
            pd.DataFrame()
            la.append(a)
            # print(cmds)

    df = pd.DataFrame(la, columns=['prec', 'q', 'tau', 'theta', 'tp_valid', 'cover_valid', 'tp_test', 'cover_test',
                                   'total'])
    df['total'] = 5000
    df['prec_valid'] = df['tp_valid'] / df['cover_valid']
    df['recal_valid'] = df['tp_valid'] / df['total']
    df['prec_test'] = df['tp_test'] / df['cover_test']
    df['recal_test'] = df['tp_test'] / df['total']
    print(df)
    df.to_csv(f_output)
    pass


if __name__ == '__main__':
    # fin_list = os.listdir('.')
    #
    # for f in fin_list:
    #     fin = os.path.splitext(f)
    #     if fin[1] == '.txt':
    #         fout = fin[0] + '.csv'
    #         dump(f, fout)

    # fin_list = ['journal_opt.txt', 'year_opt_delta.txt', 'producer_opt.txt', 'genres_opt.txt',
    #             'restaurant_opt_delta.txt']
    # fout_list_ = ['journal1.csv', 'year1.csv', 'producer1.csv', 'genres1.csv', 'restaurant1.csv']
    # for (fin, fout) in zip(fin_list, fout_list_):
    #     dump(fin, fout)
    print(delta(45000, 2, 0.9))
    # print(delta2(45000, 2, 0.1))
    pass
