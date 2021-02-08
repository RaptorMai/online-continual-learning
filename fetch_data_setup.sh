DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir -p  $DIR/datasets

echo "Downloading Core50 (128x128 version)..."
echo $DIR'/datasets/core50/'
wget --directory-prefix=$DIR'/datasets/core50/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Unzipping data..."
unzip $DIR/datasets/core50/core50_128x128.zip -d $DIR/datasets/core50/

mv $DIR/datasets/core50/core50_128x128/* $DIR/datasets/core50/

wget --directory-prefix=$DIR'/datasets/core50/' https://vlomonaco.github.io/core50/data/paths.pkl
wget --directory-prefix=$DIR'/datasets/core50/' https://vlomonaco.github.io/core50/data/LUP.pkl
wget --directory-prefix=$DIR'/datasets/core50/' https://vlomonaco.github.io/core50/data/labels.pkl