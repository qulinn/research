# cospa: 極端なパッチ画像のデータセット

## 💡 Dataset Info

- データセット情報：`/data/Users/izumi/cospa/dataset-all/train/extreme-patches-255/train-dataset`
- 手動で作ったパッチ：positive: 506枚・negative: 836枚
- 訓練で使ったパッチ：positive: 500, negative: 500, unlabeled: 1000
- モデル（パス）：`/data/Users/izumi/cospa/result/extreme-patches-255/epoch900/model`
- 結果（パス）`：/data/Users/izumi/cospa/result/extreme-patches-255/epoch900/seg`

## 💡 訓練方法

- prior=0.2, eta=0.1でepochsを上げる
- lossが横ばいになってきたepochsでprior/etaの値を変える
- stride: メモリ1枚を使うときにOOMにならない値


# 実験結果

# Test Data 2008

訓練画像と近い雲画像

### p1e1

<img src="seg/2008-testdata/p1e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p1e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


### p1e3

予測がうまくいかなかった

<img src="seg/2008-testdata/p1e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p1e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


### p1e5

<img src="seg/2008-testdata/p1e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p1e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### default  (p2e1)

<img src="seg/2008-testdata/default/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/default/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p2e3

<img src="seg/2008-testdata/p2e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p2e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p2e5

<img src="seg/2008-testdata/p2e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p2e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e1

<img src="seg/2008-testdata/p3e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p3e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e3

<img src="seg/2008-testdata/p3e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p3e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e5

<img src="seg/2008-testdata/p3e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p3e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e1

<img src="seg/2008-testdata/p4e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p4e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e3

<img src="seg/2008-testdata/p4e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p4e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e5

<img src="seg/2008-testdata/p4e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p4e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e6

<img src="seg/2008-testdata/p4e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p4e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e7

<img src="seg/2008-testdata/p4e7/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p4e7/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e1

<img src="seg/2008-testdata/p5e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p5e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e3

<img src="seg/2008-testdata/p5e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p5e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e5

<img src="seg/2008-testdata/p5e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p5e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e6

<img src="seg/2008-testdata/p5e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p5e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e7
<img src="seg/2008-testdata/p5e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"><img src="seg/2008-testdata/p5e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


# Test Data 2006

### p1e1

<img src="seg/2006-testdata/p1e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p1e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p1e3

<img src="seg/2006-testdata/p1e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p1e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p1e5

<img src="seg/2006-testdata/p1e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p1e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### default  (p2e1)

<img src="seg/2006-testdata/default/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/default/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p2e3

<img src="seg/2006-testdata/p2e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p2e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p2e5

<img src="seg/2006-testdata/p2e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p2e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p3e1

<img src="seg/2006-testdata/p3e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p3e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p3e3

<img src="seg/2006-testdata/p3e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p3e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p3e5

<img src="seg/2006-testdata/p3e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p3e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e1

<img src="seg/2006-testdata/p4e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p4e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e3

<img src="seg/2006-testdata/p4e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p4e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e5

<img src="seg/2006-testdata/p4e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p4e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e6

<img src="seg/2006-testdata/p4e6/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p4e6/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e7

<img src="seg/2006-testdata/p4e7/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p4e7/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e1

<img src="seg/2006-testdata/p5e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p5e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e3
<img src="seg/2006-testdata/p5e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p5e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">



### p5e5
<img src="seg/2006-testdata/p5e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p5e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e6
<img src="seg/2006-testdata/p5e6/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p5e6/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e7

<img src="seg/2006-testdata/p5e7/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"><img src="seg/2006-testdata/p5e7/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e7/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


# Test Data 2009

### p1e1
<img src="seg/2009-testdata/p1e1/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p1e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p1e3
<img src="seg/2009-testdata/p1e3/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p1e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p1e5
<img src="seg/2009-testdata/p1e5/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p1e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### default  (p2e1)
<img src="seg/2009-testdata/default/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/default/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p2e3
<img src="seg/2009-testdata/p2e3/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p2e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p2e5
<img src="seg/2009-testdata/p2e5/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p2e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e1
<img src="seg/2009-testdata/p3e1/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p3e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e3
<img src="seg/2009-testdata/p3e3/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p3e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e5
<img src="seg/2009-testdata/p3e5/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p3e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e1
<img src="seg/2009-testdata/p4e1/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p4e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e3
<img src="seg/2009-testdata/p4e3/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p4e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e5
<img src="seg/2009-testdata/p4e5/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p4e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e6
<img src="seg/2009-testdata/p4e6/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p4e6/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e7
<img src="seg/2009-testdata/p4e7/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p4e7/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e1
<img src="seg/2009-testdata/p5e1/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p5e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e3
<img src="seg/2009-testdata/p5e3/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p5e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e5
<img src="seg/2009-testdata/p5e5/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p5e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e6
<img src="seg/2009-testdata/p5e6/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p5e6/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e7
<img src="seg/2009-testdata/p5e7/overlay_results/05180056_05180245_05180604.jpg" width="190"><img src="seg/2009-testdata/p5e7/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">
