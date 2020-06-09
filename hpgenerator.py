# Creates json file on basis of hyper parameters 
import json


def dense_tweak(top_dense, n_dense):
    return [int(top_dense / pow(2, i)) for i in range(n_dense)]


input_shape = (32, 32, 3)

no_of_conv = [i for i in range(1, 5)]
no_of_dense_layer = [i for i in range(2, 6)]
top_dense_layer = [256, 512, 1024]
pool_krn = 2
lr = 0.001
bs = 128

a = []
for n_convo in no_of_conv:
    for n_dense in no_of_dense_layer:
        for k in top_dense_layer:
            convo_krn = [11, 9, 7, 5, 3]
            n_filter = [32 * pow(2, i) for i in range(n_convo)]
            neu_fstd = dense_tweak(k, n_dense)
            hyper_params = {'n_convo': n_convo,
                            'n_dense': n_dense,
                            'convo_krn': convo_krn,
                            'n_filter': n_filter,
                            'pool_krn': pool_krn,
                            'top_dense_layer': k,
                            'neu_fstd': neu_fstd,
                            'lr': lr,
                            'bs': bs}
            a.append(hyper_params)


b = {'hyper_params': a}
print(b)

with open('model_hparam.json', 'w') as f:
    json.dump(b, f, indent=2)

# c = {'result': []}
# with open('results.json', 'w') as g:
#     json.dump(c, g, indent=2)
   
