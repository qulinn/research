# cospa: dataset7

## 💡 Dataset Info

- cospaのデータセットの一つ
- パッチ画像の生成元となる雲画像 : 2008年と2010年の雲画像データと，南極域に雲がかかっていない画像
- 南極域に雲がかかっていない画像
<img src="data/01210816_01210935_01210956_01211258_01211438_ch4.jpg" width="300">

- パッチ画像の作成方法
    - positive patches : 雲領域のパッチを手動で切り取る
    - negative patches : 雲以外の領域を手動で切り取る
    - unlabeled patches : プログラムを用いて自動生成
- patch size : 255 x 255 px
- 人間の目で見て（私の目で見て）わかりやすいところを切り取る
- 手動で作ったパッチ：positive: 506枚・negative: 836枚
- 訓練で使ったパッチ：positive: 500, negative: 500, unlabeled: 1000
- データセット：`/data/Users/izumi/cospa/dataset-all/train/extreme-patches-255/train-dataset`


## 目次
- [cospa: dataset7](#cospa-dataset7)
  - [💡 Dataset Info](#-dataset-info)
  - [目次](#目次)
  - [💡 訓練方法](#-訓練方法)
- [実験結果](#実験結果)
  - [定量評価：IoU](#定量評価iou)
    - [全テストデータ（2008,, 2006, 2009）に対しての各モデルのスコア](#全テストデータ2008-2006-2009に対しての各モデルのスコア)
    - [2008年のテストデータでテストした時のmIoU, mDice](#2008年のテストデータでテストした時のmiou-mdice)
    - [2006年のテストデータでテストした時のmIoU, mDice](#2006年のテストデータでテストした時のmiou-mdice)
    - [2009年のテストデータでテストした時のmIoU, mDice](#2009年のテストデータでテストした時のmiou-mdice)
  - [定性評価：画像](#定性評価画像)
    - [💡前置き](#前置き)
- [Test Data : 2008](#test-data--2008)
    - [p1e1](#p1e1)
    - [p1e3](#p1e3)
    - [p1e5](#p1e5)
    - [default  (p2e1)](#default--p2e1)
    - [p2e3](#p2e3)
    - [p2e5](#p2e5)
    - [p3e1](#p3e1)
    - [p3e3](#p3e3)
    - [p3e5](#p3e5)
    - [p4e1](#p4e1)
    - [p4e3](#p4e3)
    - [p4e5](#p4e5)
    - [p4e6](#p4e6)
    - [p4e7](#p4e7)
    - [p5e1](#p5e1)
    - [p5e3](#p5e3)
    - [p5e5](#p5e5)
    - [p5e6](#p5e6)
    - [p5e7](#p5e7)
- [Test Data 2006](#test-data-2006)
    - [p1e1](#p1e1-1)
    - [p1e3](#p1e3-1)
    - [p1e5](#p1e5-1)
    - [default  (p2e1)](#default--p2e1-1)
    - [p2e3](#p2e3-1)
    - [p2e5](#p2e5-1)
    - [p3e1](#p3e1-1)
    - [p3e3](#p3e3-1)
    - [p3e5](#p3e5-1)
    - [p4e1](#p4e1-1)
    - [p4e3](#p4e3-1)
    - [p4e5](#p4e5-1)
    - [p4e6](#p4e6-1)
    - [p4e7](#p4e7-1)
    - [p5e1](#p5e1-1)
    - [p5e3](#p5e3-1)
    - [p5e5](#p5e5-1)
    - [p5e6](#p5e6-1)
    - [p5e7](#p5e7-1)
- [Test Data 2009](#test-data-2009)
    - [p1e1](#p1e1-2)
    - [p1e3](#p1e3-2)
    - [p1e5](#p1e5-2)
    - [default  (p2e1)](#default--p2e1-2)
    - [p2e3](#p2e3-2)
    - [p2e5](#p2e5-2)
    - [p3e1](#p3e1-2)
    - [p3e3](#p3e3-2)
    - [p3e5](#p3e5-2)
    - [p4e1](#p4e1-2)
    - [p4e3](#p4e3-2)
    - [p4e5](#p4e5-2)
    - [p4e6](#p4e6-2)
    - [p4e7](#p4e7-2)
    - [p5e1](#p5e1-2)
    - [p5e3](#p5e3-2)
    - [p5e5](#p5e5-2)
    - [p5e6](#p5e6-2)
    - [p5e7](#p5e7-2)



## 💡 訓練方法

- prior=0.2, eta=0.1でepochsを上げる
- lossが横ばいになってきたepochsでprior/etaの値を変える
- stride: メモリ1枚を使うときにOOMにならない値





# 実験結果

## 定量評価：IoU
### 全テストデータ（2008,, 2006, 2009）に対しての各モデルのスコア
| **モデル**               |  | **average per model** |             |
|-----------------------|------|-----------------------|-----------------|
| **prior**                 | **eta**  | **mIoU (general)**        | **mDice (general)** |
| 0.2                   | 0.1  | 0.5250                | 0.6812          |
| 0.1                   | 0.1  | 0.5525366825          | 0.7051664657    |
| 0.1                   | 0.3  | 0.0000                | 0.0000          |
| 0.1                   | 0.5  | 0.3424                | 0.4792          |
| 0.2                   | 0.3  | 0.4979                | 0.6533          |
| 0.2                   | 0.5  | 0.5191                | 0.6712          |
| 0.3                   | 0.1  | 0.5256                | 0.6799          |
| 0.3                   | 0.3  | 0.5146                | 0.6724          |
| 0.3                   | 0.5  | 0.5204                | 0.6754          |
| 0.4                   | 0.1  | 0.5441                | 0.6951          |
| 0.4                   | 0.3  | 0.5222                | 0.6764          |
| 0.4                   | 0.5  | 0.5272                | 0.6794          |
| 0.4                   | 0.6  | 0.5159                | 0.6701          |
| 0.4                   | 0.7  | 0.5257                | 0.6783          |
| 0.5                   | 0.1  | 0.4799                | 0.6353          |
| 0.5                   | 0.3  | 0.5135                | 0.6672          |
| 0.5                   | 0.5  | 0.5109                | 0.6652          |
| 0.5                   | 0.6  | 0.5149                | 0.6692          |
| 0.5                   | 0.7  | 0.5139                | 0.6689          |
| **average per test data** |      | **0.4824**                | **0.6275**          |



### 2008年のテストデータでテストした時のmIoU, mDice
| **モデル**               |  | **テストデータ：2008** |             |
|-----------------------|------|-----------------|-----------------|
| **prior**                 | **eta**  | **mIoU (general)**  | **mDice (general)** |
| 0.2                   | 0.1  | 0.5345816156    | 0.6918286214    |
| 0.1                   | 0.1  | 0.5772269708    | 0.7254918816    |
| 0.1                   | 0.3  | 0               | 0               |
| 0.1                   | 0.5  | 0.5071878782    | 0.6699234556    |
| 0.2                   | 0.3  | 0.5706471732    | 0.7201905938    |
| 0.2                   | 0.5  | 0.546919117     | 0.6984802284    |
| 0.3                   | 0.1  | 0.5487010876    | 0.7045813032    |
| 0.3                   | 0.3  | 0.5477637796    | 0.7043950132    |
| 0.3                   | 0.5  | 0.5208581326    | 0.682112664     |
| 0.4                   | 0.1  | 0.555500309     | 0.7109270182    |
| 0.4                   | 0.3  | 0.5100413222    | 0.6722325346    |
| 0.4                   | 0.5  | 0.5251066124    | 0.6829800958    |
| 0.4                   | 0.6  | 0.5104611406    | 0.6679364824    |
| 0.4                   | 0.7  | 0.5286350408    | 0.6849299678    |
| 0.5                   | 0.1  | 0.4818245286    | 0.642252686     |
| 0.5                   | 0.3  | 0.5250181616    | 0.6821005868    |
| 0.5                   | 0.5  | 0.5113510486    | 0.6699001608    |
| 0.5                   | 0.6  | 0.5144847348    | 0.67333172      |
| 0.5                   | 0.7  | 0.5126954442    | 0.6711822596    |
| **average per test data** |      | 0.5015265314    | 0.6502514354    |

### 2006年のテストデータでテストした時のmIoU, mDice
| **モデル**               |  | **テストデータ：2006** |             |
|-----------------------|------|-----------------|-----------------|
| **prior**                 | **eta**  | **mIoU (general)**  | **mDice (general)** |
| 0.2                   | 0.1  | 0.545646544     | 0.697924924     |
| 0.1                   | 0.1  | 0.6213681       | 0.76647382      |
| 0.1                   | 0.3  | 0               | 0               |
| 0.1                   | 0.5  | 0.318337059     | 0.442002276     |
| 0.2                   | 0.3  | 0.505311317     | 0.654060778     |
| 0.2                   | 0.5  | 0.516004354     | 0.663231167     |
| 0.3                   | 0.1  | 0.540017597     | 0.689096814     |
| 0.3                   | 0.3  | 0.528257683     | 0.682839725     |
| 0.3                   | 0.5  | 0.506401193     | 0.661963998     |
| 0.4                   | 0.1  | 0.531920699     | 0.681015599     |
| 0.4                   | 0.3  | 0.524337857     | 0.677592238     |
| 0.4                   | 0.5  | 0.530866517     | 0.681559904     |
| 0.4                   | 0.6  | 0.526489004     | 0.677718108     |
| 0.4                   | 0.7  | 0.530979623     | 0.682679239     |
| 0.5                   | 0.1  | 0.475891953     | 0.632634417     |
| 0.5                   | 0.3  | 0.513625894     | 0.667934768     |
| 0.5                   | 0.5  | 0.521634668     | 0.674266437     |
| 0.5                   | 0.6  | 0.521744003     | 0.674972918     |
| 0.5                   | 0.7  | 0.505919118     | 0.66183654      |
| **average per test data** |      | 0.4876185886    | 0.6299896668    |

### 2009年のテストデータでテストした時のmIoU, mDice
| **モデル**               |  | **テストデータ：2009** |             |
|-----------------------|------|-----------------|-----------------|
| **prior**                 | **eta**  | **mIoU (general)**  | **mDice (general)** |
| 0.2                   | 0.1  | 0.4948706867    | 0.6539014733    |
| 0.1                   | 0.1  | 0.4590149767    | 0.6235336956    |
| 0.1                   | 0.3  | 0               | 0               |
| 0.1                   | 0.5  | 0.2016735767    | 0.3256562833    |
| 0.2                   | 0.3  | 0.4178679089    | 0.5856387167    |
| 0.2                   | 0.5  | 0.4942282478    | 0.6520331044    |
| 0.3                   | 0.1  | 0.4880900156    | 0.64598759      |
| 0.3                   | 0.3  | 0.4677113933    | 0.6300567544    |
| 0.3                   | 0.5  | 0.5339745133    | 0.6821476767    |
| 0.4                   | 0.1  | 0.5448855033    | 0.6934093233    |
| 0.4                   | 0.3  | 0.5323353211    | 0.6792862311    |
| 0.4                   | 0.5  | 0.5256436178    | 0.6737131378    |
| 0.4                   | 0.6  | 0.5108528667    | 0.6646568367    |
| 0.4                   | 0.7  | 0.51743813      | 0.6674398411    |
| 0.5                   | 0.1  | 0.4821021056    | 0.63086743      |
| 0.5                   | 0.3  | 0.5019853144    | 0.6515946333    |
| 0.5                   | 0.5  | 0.4998480367    | 0.65133966      |
| 0.5                   | 0.6  | 0.5084588133    | 0.6593824456    |
| 0.5                   | 0.7  | 0.5231451722    | 0.6735972767    |
| **average per test data** |      | 0.4581119053    | 0.6023285321    |



## 定性評価：画像

### 💡前置き
2008年のテストデータ：訓練画像に似ている
2006年と2009年のテストデータ：訓練画像に似ていない

# Test Data : 2008

訓練画像と近い雲画像

### p1e1

<img src="seg/2008-testdata/p1e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p1e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


### p1e3

予測がうまくいかなかった

<img src="seg/2008-testdata/p1e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p1e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


### p1e5

<img src="seg/2008-testdata/p1e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p1e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p1e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### default  (p2e1)

<img src="seg/2008-testdata/default/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/default/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/default/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p2e3

<img src="seg/2008-testdata/p2e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p2e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p2e5

<img src="seg/2008-testdata/p2e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p2e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p2e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e1

<img src="seg/2008-testdata/p3e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p3e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e3

<img src="seg/2008-testdata/p3e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p3e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p3e5

<img src="seg/2008-testdata/p3e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p3e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p3e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e1

<img src="seg/2008-testdata/p4e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p4e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e3

<img src="seg/2008-testdata/p4e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p4e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e5

<img src="seg/2008-testdata/p4e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p4e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e6

<img src="seg/2008-testdata/p4e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p4e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p4e7

<img src="seg/2008-testdata/p4e7/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p4e7/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p4e7/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e1

<img src="seg/2008-testdata/p5e1/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p5e1/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e1/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e3

<img src="seg/2008-testdata/p5e3/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p5e3/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e3/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e5

<img src="seg/2008-testdata/p5e5/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p5e5/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e5/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e6

<img src="seg/2008-testdata/p5e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p5e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">

### p5e7
<img src="seg/2008-testdata/p5e6/overlay_results/11270923_11271246_11271427_ch4.jpg" width="190"> <img src="seg/2008-testdata/p5e6/overlay_results/11271246_11271427_11271746_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12042203_12050122_12050255_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12160744_12160842_12160925_12161247_12161359_ch4.jpg" width="190">
<img src="seg/2008-testdata/p5e6/overlay_results/12170733_12170818_12170915_12171056_12171335_ch4.jpg" width="190">


# Test Data 2006

### p1e1

<img src="seg/2006-testdata/p1e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p1e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p1e3

<img src="seg/2006-testdata/p1e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p1e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p1e5

<img src="seg/2006-testdata/p1e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p1e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p1e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### default  (p2e1)

<img src="seg/2006-testdata/default/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/default/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/default/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p2e3

<img src="seg/2006-testdata/p2e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p2e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p2e5

<img src="seg/2006-testdata/p2e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p2e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p2e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">

### p3e1

<img src="seg/2006-testdata/p3e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p3e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p3e3

<img src="seg/2006-testdata/p3e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p3e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p3e5

<img src="seg/2006-testdata/p3e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p3e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p3e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e1

<img src="seg/2006-testdata/p4e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p4e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e3

<img src="seg/2006-testdata/p4e3/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p4e3/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e3/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e5

<img src="seg/2006-testdata/p4e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p4e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e6

<img src="seg/2006-testdata/p4e6/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p4e6/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e6/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p4e7

<img src="seg/2006-testdata/p4e7/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p4e7/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p4e7/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e1

<img src="seg/2006-testdata/p5e1/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p5e1/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">i
<img src="seg/2006-testdata/p5e1/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e1/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e3
<img src="seg/2006-testdata/p5e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p5e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">



### p5e5
<img src="seg/2006-testdata/p5e5/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p5e5/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e5/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e6
<img src="seg/2006-testdata/p5e6/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p5e6/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/04092234_04100152_04100253_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/04230255_04230613_04230856_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/05080211_05080528_05080803_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08170150_08170507_08170734_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08190243_08190601_08190855_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08211016_08211338_08211519_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/08220814_08221005_08221328_ch4.jpg" width="190">
<img src="seg/2006-testdata/p5e6/overlay_results/12300143_12300500_12300756_ch4.jpg" width="190">


### p5e7

<img src="seg/2006-testdata/p5e7/overlay_results/03022237_03030155_03030226_ch4.jpg" width="190"> <img src="seg/2006-testdata/p5e7/overlay_results/04052229_04060147_04060245_ch4.jpg" width="190">
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
<img src="seg/2009-testdata/p1e1/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p1e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p1e3
<img src="seg/2009-testdata/p1e3/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p1e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p1e5
<img src="seg/2009-testdata/p1e5/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p1e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p1e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### default  (p2e1)
<img src="seg/2009-testdata/default/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/default/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/default/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p2e3
<img src="seg/2009-testdata/p2e3/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p2e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p2e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p2e5
<img src="seg/2009-testdata/p2e5/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p2e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p2e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e1
<img src="seg/2009-testdata/p3e1/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p3e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e3
<img src="seg/2009-testdata/p3e3/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p3e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p3e5
<img src="seg/2009-testdata/p3e5/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p3e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p3e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e1
<img src="seg/2009-testdata/p4e1/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p4e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e3
<img src="seg/2009-testdata/p4e3/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p4e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e5
<img src="seg/2009-testdata/p4e5/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p4e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e6
<img src="seg/2009-testdata/p4e6/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p4e6/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e6/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p4e7
<img src="seg/2009-testdata/p4e7/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p4e7/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p4e7/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e1
<img src="seg/2009-testdata/p5e1/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p5e1/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e1/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e3
<img src="seg/2009-testdata/p5e3/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p5e3/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e3/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e5
<img src="seg/2009-testdata/p5e5/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p5e5/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e5/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e6
<img src="seg/2009-testdata/p5e6/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p5e6/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e6/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">

### p5e7
<img src="seg/2009-testdata/p5e7/overlay_results/05180056_05180245_05180604.jpg" width="190"> <img src="seg/2009-testdata/p5e7/overlay_results/05180245_05180604_05180745.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/05181248_05181749.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/06090855_06090917_06091036_06091358.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/06160010_06160101_06160131_06160155_06160449.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07030739_07030803_07030817_07030920_07030944_07031102_07031306.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07032112_07040030_07040251.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/07060912_07061031_07061054_07061212_07061322_07061415_07061556.jpg" width="190">
<img src="seg/2009-testdata/p5e7/overlay_results/09090835_09090928_09091250_09091431.jpg" width="190">
