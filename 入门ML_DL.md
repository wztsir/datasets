# 入门ML/DL

## python方法

### strip

去除两边空格

```
line = line.strip()
```



### for i in range

```python
a = [print('%d*%d=%d' % (i, j, i * j), end=' ') if j < i + 1 else print() for i in range(1, 10) for j in
     range(1, i + 2)]
```

1. range的范围 [1,10)
2. for循环：无论有几层for循环，先循环外面的，后循环里面的。循环次数就是a数组的大小
3. if-else判断：if 放bool判断，满足，直接将i,j放行前面的数据，不满足，可以填表达式，如 print()。也可以填数据，比如 -1，则a数组这个位置值为-1



### join

将列表打直，并用确定符号分割，这里用空格分隔

```python
reviews.append(" ".join(review_text))
```



### tensor

使用numpy()将Tensor转换成NumPy数组:

```python
b = a.numpy()
```

使用from_numpy()将NumPy数组转换成Tensor:

```python
b = torch.from_numpy(a)
```

torch.tensor()将NumPy数组转换成Tensor:(存在拷贝操作，内存不再共享，运行耗时)

```python
torch.tensor(a)
```



### 张量乘法

numpy与torch拥有丰富的广播与维度扩展机制

一般第0维，为自动广播，剩余的2维，做矩阵运算



## 服务器环境搭建

### bug1

```
no cacheXXXXXXXXX
```

大致意思是空间不足

1、查看机器的容器大小

```
ssh gpu027
```

仍有100多G

2、发觉可能是python环境出错

查看默认安装包的地址

```
python -m site
```

删除这个python

```
rm -rf 文件夹/文件
```



更改为当前创建的虚拟的环境

ctrl+shift+p 更改环境配置

继续右键，运行当前的python文件

which python检查是否更改成功





### bug2

```python
pip install -r freeze.txt
```

-r read文件

过程是先下载安装包，再进行安装，安装的时候会缺少提示，正在安装，不是bug

### bug3

![image-20230313221608766](.\imgs\image-20230313221608766.png)

错误关注点，应该如下:

```python
FileNotFoundError: [Errno 2] No such file or directory: '~/.cache/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8/vocab.json'
```

错误关注于中间

解释：找路径，由于有歧义，代码直接当作绝对路径查找，而不是相对路径

~可以代表根目录

![](.\imgs\2023-03-13 223029.png)

解决，直接写死，写成绝对路径

```python
 parser.add_argument('--cache_dir', type=str, default="/mnt/cephfs/home/yangjiahao/detect_gpt/detect-gpt/~/.cache")
```



### bug4

```python
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB (GPU 0; 11.91 GiB total capacity; 1.66 GiB already allocated; 15.94 MiB free; 1.71 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

解释与猜测：内存空间不够，这块板子资源不够

```python
ssh gpu027
```

换一款gpu

### bug5

![](D:\note\imgs\2023-03-14 125040.png)

解释

​    该CSV文件的编码格式是 带有UTF-8-BOM，它与我们常用的UTF-8编码格式不同；区别就是在有没有BOM。即文件的开头有没有 UFEFF。这样就会造成生成数组的第一个元素，无法进行判断匹配。

解决

```pyyhon
            if line.startswith('﻿'):
                line = line[1:]
```





### bug6GPU显存不够

```python
return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 11.91 GiB total capacity; 2.30 GiB already allocated; 12.00 MiB free; 2.36 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

模型不够智能，不会自己选择状态最好的板子运行

```python
  # 设置要跑的gpu
    train_gpu=[7]
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))
```

 os.environ["CUDA_VISIBLE_DEVICES"] 应该等于字符串

### 技巧1-vscode

利用vscode搭建环境

可以直接在底部栏看见python的环境







### 技巧2-相对路径

基础中的基础，在Linux必须学会使用，而且必须要知道一些重要的常识，如果晓得，就不会犯bug3一样的错误

**"./"：代表目前所在的目录。**./可以省去不写，但是即有可能出错，如bug3。因为在linux中~可以代表根目录

**" . ./"代表上一层目录。**

**"/"：代表根目录。**



**cd ~ 和cd $HOME
是跳转到当前用户的家目录**
root用户，cd ~ 相当于 cd /root
普通用户，cd ~ 相当于cd /home/当前用户名



可以直白的用文件树理解，当前文件所处的位置，

<img src=".\imgs\210084F4378511B41E1C435F3C2804A1.png" style="zoom: 33%;" />



终极技巧，当相对路径一直都存在问题时，只能使用绝对路径

### 显存与GPU利用率



```python
nvidia-smi
```

查看具体的GPU使用情况

![](.\imgs\2023-03-15 191706.png)

第二列是内存使用率，第三列是GPU使用率



### python运行环境搭建

pip install如果不指定版本，将会自动下载最新版的包（不得不佩服pycharm的自动导包的舒适）

ctrl+c立刻终止，如果发现下载包版本同其他包版本没对应上

cv2

```pyhon
pip install opencv-contrib-python
```



## 基础的Linux命令行

查看当前的文件路径

```
pwd
```

列出当前文件下的所有文件

```
ll
```





## 论文

### 姓名

Last Name = Family Name = 姓;

First Name = Given Name = 名

first name=zitong

last name=wang



### Detect GPT



| xums      | squad           | writtingprompt         |
| :-------- | --------------- | ---------------------- |
| fack news | machine-written | prompt stories创意故事 |



man 

| WMT16                     | PubMedQA          |      |
| ------------------------- | ----------------- | ---- |
| English and German splits | long-form answers |      |







#### 数据预处理

generate_data函数中xsum生成的数据

line627生成list数组数据

```
'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\n"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said.\n"That may not be true but it is perhaps my perspective over the last few days.\n"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\nThe Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\n"I was quite taken aback by the amount of damage that has been done," he said.\n"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."\nHe said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.'


```

数据去除空格，去除换行

```
'Of these, almost 4,000 children could not be traced by the authorities. The National Children\'s Bureau said some may be at "serious risk" of abuse and exploitation, including forced marriage, FGM and radicalisation. The Department for Education said it had issued "new guidance" to schools. Ofsted has previously raised concerns that some missing children could be hidden away in unregistered, illegal schools. The figures, obtained by the BBC\'s Victoria Derbyshire programme, show that 33,262 school-aged children were recorded as missing from education in the academic year ending in July 2015. They were collated from a Freedom of Information request to 90 local education authorities in England and Wales. Children were recorded as "missing from education" if they were of compulsory school age, and the authorities were unable to trace them - typically for four weeks or more, or two to three days in the case of vulnerable children. More than 10% of these children - 3,897 - could not be traced by local authorities. Manchester recorded the highest figure - 1,243 children were missing from education, including 810 children whose whereabouts were unknown in July 2015. In Bradford 985 school-aged children were missing - the authority was unable to trace 321 of them after "extensive enquires". In some cases, children were recorded as missing because they had moved out of the area, or gone abroad, and their parent or guardian had failed to tell the school. However, in most cases where a child had been traced, local authorities could not give a reason why they had disappeared. "When I was 15, my dad thrust a picture of my cousin towards me and said, \'This is who you\'re going to marry\'. "I didn\'t know what to say, I was scared. The only thing I could think to do was run away from home, but my brother found me." Zainab - not her real name - says that from then on, she was, in effect, a prisoner in her own home. "I was pulled out of school, I wasn\'t able to finish my GCSEs. The school did send two letters home to my dad. But he just chose to completely ignore them. "And then we moved house, and the school didn\'t know. I was completely off the radar." After seven months of being locked inside, Zainab managed to call a charity from her brother\'s phone. The National Children\'s Bureau believes there are a number of "very serious risks" with children going missing. Enver Solomon from the charity said: "Some councils do a fantastic job, but unfortunately some councils don\'t do a good enough job by any stretch of the imagination. "There shouldn\'t be one child in the country who isn\'t in school and can\'t be tracked, because of the potential risks. \'We know [of some] horrendous cases, of sexual exploitation. We also know about the correlation between missing children and the possibility that they may be involved in forced marriage, and of course, issues relating to young people\'s involvement in extremist activity." The charity - as well as other child protection agencies - said the figures were likely to underestimate the scale of the problem. Children can easily disappear from education without being reported, it said, because families may tell a plausible story to the school - like they are home-schooling or going abroad. In response to the figures, a spokesman for the Department for Education said: \'We have issued new guidance to local authorities and schools making clear that they have a duty to establish the identities of children who are not registered at a school or receiving a suitable education. "Where children are being put at risk, local authorities and the police have clear powers to take action." The Victoria Derbyshire programme is broadcast on weekdays between 09:00 and 11:00 on BBC Two and the BBC News Channel.'
```

将文本词嵌入



文本对比line600

```
origin；
'Maj Richard Scott, 40, is accused of driving at speeds of up to 95mph (153km/h) in bad weather before the smash on a B-road in Wiltshire. Gareth Hicks, 24, suffered fatal injuries when the van he was asleep in was hit by Mr Scott\'s Audi A6. Maj Scott denies a charge of causing death by careless driving. Prosecutor Charles Gabb alleged the defendant, from Green Lane in Shepperton, Surrey, had crossed the carriageway of the 60mph-limit B390 in Shrewton near Amesbury. The weather was "awful" and there was strong wind and rain, he told jurors. He said Mr Scott\'s car was described as "twitching" and "may have been aquaplaning" before striking the first vehicle; a BMW driven by Craig Reed. Mr Scott\'s Audi then returned to his side of the road but crossed the carriageway again before colliding head-on with a Ford Transit van in which Mr Hicks was a passenger, the court was told. "There is no doubt that when the Audi smashed into the panel van he was on completely the wrong side of the road," Mr Gabb said. Mr Hicks, from Bath in Somerset, was asleep in the van being driven to a construction site in Salisbury by fellow DR Groundworks colleague, Patrick Gilleece. The jury was told the Maj Scott suffered "substantial injuries" and could not recall the crash, which happened shortly after 07:00 GMT on 6 October, 2014. He does not accept the charge and suggests it was in fact Mr Reed who had crossed the carriageway, causing the collision, Mr Gabb told the court. The trial continues.'
生成：
'Maj Richard Scott, 40, is accused of driving at speeds of up to 95mph (153km/h) in bad weather before the smash-and-run happened. The pair had a heated arguments outside the local council office where he allegedly caused the accident and that he died after trying to run off the road.\n\nPolice have charged the pair with driving at 120mph (160km an hour) before the crash which occurred at 3am on Tuesday night.\n\nThe driver was not injured, but was not arrested and is in custody.\n\nMr Scott was later found with a string of injuries including a broken arm, neck, hand and leg. He was taken to the Royal Melbourne Hospital in stable condition. He is listed in serious condition in hospital but is listed in police custody and is without post-traumatic stress disorder. He is due to appeal his conviction.\n\nThe pair will remain jailed until sentencing on March 24.'
```



### 图书馆占座检测











## DL/ML



读书是螺旋上升的，有时候看不懂，就再多看几遍，好的书籍与课本值得，先把那些基础概念牢记心中。

不要害怕做无用功！！！先上车，后补票，自己要有信心，那些领域内不懂的知识，风卷残云的摄入就好，不必担心自己太菜，菜鸡才是正常的，自己的定位就是小白



多GPU预测、各种各样的方法，知识无穷无尽，似要把人淹没

### 通道数

在卷积层的计算中，假设输入是H x W x C, C是输入的深度(即通道数)，那么卷积核(滤波器)的通道数需要和输入的通道数相同，所以也为C，假设卷积核的大小为K x K，一个卷积核就为K x K x C，计算时卷积核的对应通道应用于输入的对应通道，这样一个卷积核应用于输入就得到输出的一个通道。假设有P个K x K x C的卷积核，这样每个卷积核应用于输入都会得到一个通道，所以输出有P个通道。

p的设置根据网络设计来

综上，一个卷积层的输出通道数，和输入通道数没有直接关系，有直接关系的是卷积核大小



### 卷积层与全连接层

卷积层处理干净，大量的数据，进行特征处理

全连接层做出专注于分类



### Accuracy

准确率，指的是正确预测的样本数占总预测样本数的比值，它不考虑预测的样本是正例还是负例，反映的是模型算法整体性能，



### TP/FP/FN/TN

true positive真正类，

对于二分类问题，预测出的两类状态与实际结果两类状态两两交叉，最后就存在四种情况

<img src=".\imgs\2023-03-14 223018.png" style="zoom:50%;" />

精确率(precision)：预测为正的有多少是正，TP/TP+FP

召回率(recall)：实际正类中有有多少被预测正确，TP/TP+FN

两者就是参考的范围不一样，一个是范围是预测为正，一个是实际正类



### PR曲线

选择合适的阈值

纵轴精确率(Precision) 横轴召回率(Recall),对分类器预测正例的概率倒序排序，从最大概率到最小概率移动正例/反例的阈值(大于阈值的认为是正例,小于阈值的认为是负例),在每个阈值处的标记精确率和召回率,通过这种方式画出一条曲线

显然的，当逐渐减小T阈值时，可以形象的理解为上图蓝色方框下边界下移，FN减少，FP增大

有下图PR曲线

<img src=".\imgs\v2-f747468ad9a484455ac58a5d9f901049_r.png" style="zoom:50%;" />



### F1-score

是统计学中用来衡量二分类模型精确度的一种指标，它被定义为精确率和召回率的调和平均数，它的最大值是1，最小值是0



### ROC曲线

全称:[受试者工作特征曲线](https://www.zhihu.com/search?q=受试者工作特征曲线&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"435204013"})(the Receiver Operating Characteristic),诞生于军事领域,在医疗领域应用甚广

和PR曲线思想一样,只不过横轴和纵轴的指标不一样

真阳率与假阳率考虑问题的范围都是实际样本的分类情况

纵轴:[真阳性率](https://www.zhihu.com/search?q=真阳性率&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"435204013"})(真正例率,简称TPR) = TP/(TP+FN)=40/60  即:正类中分类器判定为**正类**的数量/正类数量

横轴:[假阳性率](https://www.zhihu.com/search?q=假阳性率&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"435204013"})(假正例率,简称FPR) = FP/(FP+TN)=30/40  即:负类中分类器判定为**正类**的数量/负类数量



<img src=".\imgs\v2-da63af43491595148663be34cce030cc_r.jpg" style="zoom:50%;" />



### IOU

Intersection over Union

intersection(交叉)

交并比，指的是ground truth bbox与predict bbox的交集面积占两者并集面积的一个比率，IoU值越大说明预测检测框的模型算法性能越好，通常在目标检测任务里将IoU>=0.7的区域设定为正例（目标），而将IoU<=0.3的区域设定为负例（背景）

**AP（Average Percision）：**

下文描述中，某一类，即将目标类表示为正类，其他类表示为负类

AP为平均精度，指的是所有图片内的具体某一类的PR曲线下的面积，其计算方式有两种，第一种算法：首先设定一组recall阈值[0, 0.1, 0.2, …, 1]，然后对每个recall阈值从小到大取值，同时计算当取大于该recall阈值时top-n所对应的最大precision。这样，我们就计算出了11个precision，AP即为这11个precision的平均值，这种方法英文叫做11-point interpolated average precision；

第二种算法：该方法类似，新的计算方法假设这N个样本中有M个正例，那么我们会得到M个recall值（1/M, 2/M, …, M/M）,对于每个recall值r，该recall阈值时top-n所对应的最大precision，然后对这M个precision值取平均即得到最后的AP值。

**mAP（Mean Average Percision）：**mAP为均值平均精度，指的是所有图片内的所有类别的AP的平均值，目前，在目标检测类里用的最多的是mAP，一般所宣称的性能是在IoU为0.5时mAP的值。常见的目标检测评估指标输出样式如下：







### 优化器

| 优化器   | 数学     | 优点         | 缺点    | 思路                                                         |
| :------- | :------- | :----------- | :------ | ------------------------------------------------------------ |
| AdaGrad  | 二阶动量 | 适合稀疏数据 | 容易到0 | 经常更新的参数学习率低一点，很少更新的参数学习率高一点，考虑历史所有的参数数据 |
| AdaDelta | 二阶动量 |              |         | 考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度 |
| Adam     |          |              |         |                                                              |





### 口袋里的人工智能机器视觉





## 目标检测-安全帽检测

导入数据，运行代码，有问题再说，如何搭建一个平台展示模型



1、如何多任务检测，检测安全帽与反光衣
