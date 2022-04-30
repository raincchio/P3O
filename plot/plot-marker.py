import matplotlib.pylab as plt

markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
descriptions = ['point', 'pixel', 'circle', 'triangle_down', 'triangle_up','triangle_left',
                'triangle_right', 'tri_down', 'tri_up', 'tri_left', 'tri_right', 'octagon',
                'square', 'pentagon', 'plus (filled)','star', 'hexagon1', 'hexagon2', 'plus',
                'x', 'x (filled)','diamond', 'thin_diamond', 'vline', 'hline']



x=[]
y=[]

for i in range(5):
    for j in range(5):
        x.append(i)
        y.append(j)

plt.figure(figsize=(8, 8))

for i,j,m,l in zip(x,y,markers,descriptions):
    plt.scatter(i,j,marker=m)
    plt.text(i-0.15,j+0.15,s=m+' : '+l)

plt.axis([-0.1,4.8,-0.1,4.5])

plt.axis('off')
plt.tight_layout()
plt.show()