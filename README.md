# HumanvsArtificialIntelligence
 Bu proje, insan ve yapay zeka tarafından üretilen metinleri ayırt etmek için bir makine öğrenimi modeli geliştirmeyi amaçlamaktadır. Projede, BERT (Bidirectional Encoder Representations from Transformers) modeli kullanılarak metin sınıflandırma işlemi gerçekleştirilmiştir.

1. Projenin Amacı
İnsan ve yapay zeka tarafından yazılan metinleri ayırt etmek.

BERT gibi bir transformer modeli kullanarak yüksek doğrulukta sınıflandırma yapmak.

Modelin performansını ölçmek ve görselleştirmek.

2. Kullanılan Veri Seti
İnsan verisi: Başlıksızform.csv dosyasından alınan metinler.

Yapay zeka verisi: Proje içinde elle tanımlanmış örnek yanıtlar.

Etiketleme:

Kaynak = "İnsan" → 0

Kaynak = "Yapay Zeka" → 1

3. Veri Ön İşleme
a) Temel Temizlik İşlemleri
Küçük harfe çevirme: Tüm metinler küçük harfe dönüştürüldü.

Noktalama işaretlerini kaldırma: string.punctuation kullanılarak temizlendi.

Stop-words çıkarma: İngilizce stop-words (nltk.corpus.stopwords) kaldırıldı.

Köklerine indirgeme (Stemming): PorterStemmer ile kelimeler köklerine indirgendi.

b) Veri Birleştirme ve Bölme
İnsan ve yapay zeka verileri birleştirildi.

train_test_split ile %80 eğitim, %20 test verisi olarak ayrıldı.

NaN değerler boş string ("") ile dolduruldu.

4. Modelin Eğitimi
a) Kullanılan Model: BERT
Tokenizasyon: BertTokenizer.from_pretrained('bert-base-uncased')

Model: BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

Optimizasyon: AdamW (learning rate = 5e-5)

b) Eğitim Süreci
Batch boyutu: 8

Epoch sayısı: 3

Loss hesaplaması: Cross-entropy loss (outputs.loss)

Geri yayılım (Backpropagation): loss.backward()

Optimizasyon adımı: optimizer.step()

c) Loss Değerlerinin İzlenmesi
python
Epoch 1 Loss: 0.45
Epoch 2 Loss: 0.32
Epoch 3 Loss: 0.25
Loss grafiği çizdirilerek modelin öğrenme süreci görselleştirildi.

5. Modelin Değerlendirilmesi
a) Test Doğruluğu (Accuracy)
python
Test Accuracy: 95.00%
Model, test verisinde %95 doğruluk sağladı.

b) Tahmin Fonksiyonu
python
def tahmin_et():
    metin = input("Bir cümle girin: ")
    tahmin = model.predict(metin)  # 0 (İnsan) veya 1 (Yapay Zeka)
    print(f"Tahmin: {'İnsan' if tahmin == 0 else 'Yapay Zeka'}")
Kullanıcıdan alınan metin, modele gönderilerek tahmin yapılır.

6. Modelin Kaydedilmesi
python
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
Eğitilen model ve tokenizer, trained_model klasörüne kaydedilir.



**************************
This project aims to develop a machine learning model to distinguish between human-written and AI-generated text. The model uses BERT (Bidirectional Encoder Representations from Transformers) for text classification.

1. Project Objective
Differentiate between human and AI-generated text.

Achieve high classification accuracy using BERT.

Evaluate and visualize model performance.

2. Dataset
Human data: Text collected from Başlıksızform.csv.

AI-generated data: Manually defined sample responses.

Labeling:

Source = "Human" → 0

Source = "AI" → 1

3. Data Preprocessing
a) Basic Cleaning
Lowercasing: All text converted to lowercase.

Removing punctuation: Cleared using string.punctuation.

Stop-word removal: English stop-words (nltk.corpus.stopwords) removed.

Stemming: Words reduced to root form using PorterStemmer.

b) Data Merging & Splitting
Human and AI data combined.

Split into 80% training, 20% testing using train_test_split.

NaN values filled with empty strings ("").

4. Model Training
a) Model: BERT
Tokenization: BertTokenizer.from_pretrained('bert-base-uncased')

Model: BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

Optimizer: AdamW (learning rate = 5e-5)

b) Training Process
Batch size: 8

Epochs: 3

Loss calculation: Cross-entropy loss (outputs.loss)

Backpropagation: loss.backward()

Optimization step: optimizer.step()

c) Tracking Loss Values
python
Epoch 1 Loss: 0.45
Epoch 2 Loss: 0.32
Epoch 3 Loss: 0.25
Loss plot visualized to monitor training progress.

5. Model Evaluation
a) Test Accuracy
python
Test Accuracy: 95.00%
The model achieved 95% accuracy on test data.

b) Prediction Function
python
def predict_text():
    text = input("Enter a sentence: ")
    prediction = model.predict(text)  # 0 (Human) or 1 (AI)
    print(f"Prediction: {'Human' if prediction == 0 else 'AI'}")
Takes user input and returns a prediction.

6. Saving the Model
python
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
The trained model and tokenizer are saved in the trained_model folder.