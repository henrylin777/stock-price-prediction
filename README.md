# auto-stock-trader
Automatic stock trader using RNN

利用 RNN 分析股票資料，並預測未來股價趨勢，再根據RNN所預測的未來趨勢與目前持有的股票來決定進行哪一種操作。


### Usage 

```
python trader.py --training <training-data> --testing <testing-data> --output <output>
```
<br>

### Requirement
```
pip install -r requirements.txt
```
<br>

### Method

RNN會預測未來兩天的股價趨勢資料，一共有五種可能:
1. 上漲再上漲
2. 上漲再下跌
3. 下跌再上漲
4. 下跌再下跌
5. 其他

再根據RNN的預測結果與目前的股票張數來決定執行何種操作，規則如下:

**Case 1: 上漲再上漲**

如果目前持有股票，則不動作
如果目前沒持股，則買進
如果目前為賣空，則不動作

**Case 2: 上漲再下跌**

如果目前持有股票，則賣出
如果目前沒持股，則不動作
如果目前為賣空，則不動作

**Case 3: 下跌再上漲**

如果目前持有股票，則不動作
如果目前沒持股，則賣空
如果目前為賣空，則買入

**Case 4: 下跌再下跌**

如果目前持有股票，則不動作
如果目前沒持股，則賣空
如果目前為賣空，則不動作

**Case 5: 其他**

不執行任何動作

<br>

### Data augment

由於Training data只有1千4百多筆，所以打算手動增加資料來增強模型的訓練效果。除了原本的 `open, close, high, low` 之外，此外還增加了MA5, MA10, MA20, MA60的資料到input data中。

此外，因為RNN要預測的是**趨勢**，所以嘗試將跟趨勢有關的資料加進input data。我將今日的開盤價與 MA5, MA10, MA20, MA60與過去5天, 10天, 20天, 60天的最大值與最小值的差價加入input data。最後原本只有4維的input data擴充到100維。

<br>

### Data preprocessing

在資料前處理的部分有加入 Normalization，除了加快訓練速度以外，還可以減少離群值所帶來的影響，改善模型效能。
