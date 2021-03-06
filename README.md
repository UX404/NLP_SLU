### 创建环境

注意需要安装jieba库

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1
    pip install jieba

### 运行

在根目录下运行

我们的模型使用**原本的训练集**训练，无额外数据

    # Training
    python scripts/slu_baseline.py
    
    # Inference, the prediction results will be saved in pred.json
    python scripts/test.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
    
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU

+ `utils/vocab.py`:构建编码输入输出的词表

+ `utils/word2vec.py`:读取词向量

+ `utils/example.py`:读取数据，增加了我们的方法额外需要的属性

+ `utils/batch.py`:将数据以批为单位转化为输入，增加了我们的方法额外需要的属性

+ `model/slu_baseline_tagging.py`:baseline模型以及在此基础上的改进

+ `scripts/slu_baseline.py`:主程序脚本，将会运行我们的新模型

+ `scripts/test.py`:测试脚本，预测结果将保存在`pred.json`中

+ `seg_idx_list.json`:分词词性列表，我们方法需要读取的一些常数

+ `data/train_expand.json`:测试数据增强部分功能时使用的数据
