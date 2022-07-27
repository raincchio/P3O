plot_data = {}
path = '/home/chenxing/Downloads/ss/sense_ana'
for env in ["Enduro", 'BeamRider', "Breakout"]:
    lr_data = {}
    for i in range(1,11):
        lr = str(i/100.0)
        res = 0
        file_cont=0
        for j in range(4):
            with open(path+'/'+env+'/'+str(lr)+'_'+str(j)+'/0.0.monitor.csv', 'r') as f:
                data = f.read()

            csv_data = data.split('\n')[-21:-1]
            if len(csv_data)!=20:
                print(path+'/'+env+'/'+str(lr)+'_'+str(j)+'/0.0.monitor.csv')
                file_cont+=1
                continue
            rtt = 0
            count = 0
            for cdata in csv_data:
                tmp = cdata.split(',')
                if len(tmp)!=3:
                    count+=1
                    print(path+'/'+env+'/'+str(lr)+'_'+str(j)+'/0.0.monitor.csv')
                    print(cdata)
                    continue
                rtt += float(tmp[0])

            res += rtt/(20.0 -count)
        lr_data[lr] = res/(4.0-file_cont)
    plot_data[env]= lr_data

with open('plot_data','w') as f:
    f.write(str(plot_data))