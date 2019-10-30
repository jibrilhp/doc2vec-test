# doc2vec-test
It's just my research for Doc2Vec implementation on SPMI (https://spmi.ristekdikti.go.id/uploads/publications/Buku%20Pedoman%20SPMI%202018.pdf) 

# Installation
Python 3.6
pip install flask
pip install numpy
pip install pandas
pip install gensim
pip install sklearn

# Train the data
You can train the model from gabungan.csv and give the label

python train.py

Make sure before you run the api_processing, d2v has been created on your project folder

# Run 
python api_processing.py

# How to access that?
For a testing purposes use the PostMan or other 
An example:
## localhost:1311/spmi/ami/1
## Postdata ==> teks_analysis = "Tidak terdapat kebijakan tertulis maupun tidak tertulis tentang penyusunan dan pengembangan kurikulum"
## output ==> Nilai ke-1

