# 快速傅里叶变换（FFT）

离散傅里叶变换(discrete Fourier transform) 傅里叶分析方法是信号分析的最基本方法，傅里叶变换是傅里叶分析的核心，通过它把信号从时间域变换到频率域，进而研究信号的频谱结构和变化规律。但是它的致命缺点是：计算量太大，时间复杂度太高，当采样点数太高的时候，计算缓慢，
# 采样频率以及采样定理

采样频率：采样频率，也称为采样速度或者采样率，定义了每秒从连续信号中提取并组成离散信号的采样个数，它用赫兹（Hz）来表示。采样频率的倒数是采样周期或者叫作采样时间，它是采样之间的时间间隔。通俗的讲采样频率是指计算机每秒钟采集多少个信号样本。

采样定理：所谓采样定理 ，又称香农采样定理，奈奎斯特采样定理，是信息论，特别是通讯与信号处理学科中的一个重要基本结论。采样定理指出，如果信号是带限的，并且采样频率高于信号带宽的两倍，那么，原来的连续信号可以从采样样本中完全重建出来。

定理的具体表述为：在进行模拟/数字信号的转换过程中，当采样频率fs大于信号中最高频率fmax的2倍时,即

fs>2*fmax

采样之后的数字信号完整地保留了原始信号中的信息，一般实际应用中保证采样频率为信号最高频率的2.56～4倍；采样定理又称奈奎斯特定理。

# 使用scipy包实现快速傅里叶变换
1、产生原始信号——原始信号是三个正弦波的叠加
```python
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
 
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

 
#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,1400)      
 
#设置需要采样的信号，频率分量有200，400和600
y=7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+3*np.sin(2*np.pi*600*x)
```
这里原始信号的三个正弦波的频率分别为，200Hz、400Hz、600Hz,最大频率为600赫兹。根据采样定理，fs至少是600赫兹的2倍，这里选择1400赫兹，即在一秒内选择1400个点。
原始的函数图像如下：
```python
plt.figure()
plt.plot(x,y)   
plt.title('原始波形')
 
plt.figure()
plt.plot(x[0:50],y[0:50])   
plt.title('原始部分波形（前50组样本）')
plt.show()
```
![image1](https://https://github.com/666tugege/ai108b/blob/master/Final-term/08-%E7%A7%91%E5%AD%B8%E8%A8%88%E7%AE%97_Scientific%20Computing/result.png?raw=true)

由图可见，由于采样点太过密集，看起来不好看，我们只显示前面的50组数据，如下：

![image1](https://github.com/666tugege/ai108b/blob/master/Final-term/08-%E7%A7%91%E5%AD%B8%E8%A8%88%E7%AE%97_Scientific%20Computing/%E6%B3%A2%E5%BD%A2.png?raw=true)

# 参考资料
http://www.cs.princeton.edu/introcs/97data/FFT.java.html
https://www.zhihu.com/topic/19600515/hot
