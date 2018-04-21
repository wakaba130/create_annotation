# labelImg用のアノテーションデータ作成

※英語できないので、説明は日本語のみです。

本プログラムで出力した結果のデイレクトリを[labelImg](https://github.com/tzutalin/labelImg)というソフトで読み込むと認識結果が反映されて、アノテーションの効率化ができます。

ChainerCVに実装されているSSDを使用して、物体認識を行います。
認識結果をPascal VOCのフォーマット（一部labelImg仕様の箇所があります）で出力します。

## 1 準備

必要なインストール

```Shell
pip install cupy
pip install chainer
pip install chainercv
```

## 動作確認環境

+ CPU : Intel® Core™ i7-6700K CPU @ 4.00GHz × 8
+ Memory : 16GB
+ GPU : GeForce GTX 980 Ti(6GB)
+ OS : Ubuntu 16.04 LTS
+ CUDA : 8.0
+ Python version : 3.5.2
+ chaienr version : 3.5.0
+ cupy version : 2.5.0
+ chainercv version : 0.8.0

## 2 使い方

```
$ python3 create_ano.py [--model {ssd300,ssd512}] [--gpu GPU]
                        [--pretrained_model PRETRAINED_MODEL]
                        [--output_dir OUTPUT_DIR] [--no_copy]
                        image_dir

positional arguments:
  image_dir ・・・ データセットにしたい画像が入ったディレクトリ

optional arguments:
  -h, --help            show this help message and exit
  --model {ssd300,ssd512} ・・・　使用するモデルの選択
  --gpu GPU               ・・・　使用する GPU ID の指定
  --pretrained_model PRETRAINED_MODEL ・・・ 使用するモデルパラーメータ
  --output_dir OUTPUT_DIR ・・・ 出力するディレクトリ（指定しなければresultというディレクトリができる）
  --no_copy ・・・ 出力ディレクトリに画像を出力しない
```

実行オプションで、’--no_copy’を指定した場合、
labelImgでアノテーションデータの保存先に本プログラムの出力ディレクトリを指定する必要があります。

### オリジナル学習済みパラメータでの使用

通常は、ChainerCVに入っているPascal VOCの学習済みモデルが適用されますが、
ラベルを追加したり、変更したオリジナルのパラメータを使用する場合は、ラベル名を変更する必要があります。

ラベル名を変更する際は、`labels.txt`の中に１ラベル１行で追記または変更を行ってください。
以下は、default設定の'labels.txt'の内容です。

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```

## 3 出力について

出力フォーマットは以下のとおりです。
`path`の部分が本家のPascal VOCにはなく、labelImgの仕様だと思われます。

```xml
<annotation>
	<folder>hogehoge</folder>
	<filename>image.jpg</filename>
	<path>/home/hogehoge/image.jpg</path>
  <source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1125</width>
		<height>1600</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>4</xmin>
			<ymin>143</ymin>
			<xmax>947</xmax>
			<ymax>1600</ymax>
		</bndbox>
	</object>
</annotation>
```
