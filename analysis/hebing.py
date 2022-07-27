with open('plot_data', 'r') as f:
    data = eval(f.read())
with open('plot_data_lr_ctn', 'r') as f:
    data2 = eval(f.read())


data.update(data2)
with open('plot_data_lr','w') as f:
    f.write(str(data))