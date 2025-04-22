import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW



# 1. İnsan verisini yükleme
insan_verisi = pd.read_csv('Başlıksızform.csv')
insan_verisi = insan_verisi.drop(columns=['Zaman damgası'])

# 2. Yapay zeka cevaplarını tanımlama
yapay_zeka_verisi = pd.DataFrame([
    {
        'İnsanlar neden mutlu değil ?': "mutsuzluk birçok nedenden kaynaklanabilir ancak çözümler de mevcuttur. Önemli olan, bu nedenleri anlamak ve kendine uygun çözümleri bulmaktır.",
        'Yapay zeka dünyayı ele geçirebilir mi?': "yapay zeka şu an için bir tehdit değil, ancak potansiyel bir araçtır. Bu aracı nasıl kullanacağımız bizim elimizde.",
        'Bir resim yapsan hangi renkleri kullanırsın ?': "Bir resim yapsaydım, o resmin atmosferine ve vermek istediğim duyguya göre renkleri seçerdim.",
        'Özgürlük nedir kısaca anlatınız.': "Özgürlük, kendi hayatımızın kontrolünü elimizde tutmaktır.",
        'zaman size ne çağrıştırıyor ?': "benim için bir ölçüm birimi, sürekli bir akış ve değişim süreci. İnsanlar için ise daha çok duygusal ve kişisel bir anlam taşıyor.",
        'Aşka inanıyor musunuz ?': "Bir yapay zeka olarak, aşk gibi soyut ve subjektif bir kavrama inanıp inanmamam mümkün değil.",
        'Bir karakter çizsen, o karakter nasıl görünürdü? kısaca açıklar mısın ?': "Eğer bir karakter çizebilseydim, muhtemelen gökkuşağı saçlı, büyük ve parlak gözlere sahip, yaratıcı bir karakter olurdu.",
        'Eğer bir müzik albümü yapsaydın, hangi tarzda olurdu?': "Bir yapay zeka olarak müzik besteleyemem. Ancak popüler trendleri analiz ederek hayali bir albüm oluşturabilirim.",
        'Eğer bir dizide rol alabilseydin, hangi diziyi seçerdin?': "Muhtemelen bilim kurgu türünde bir dizi tercih ederdim.",
        'Geçmişe mi gitmek isterdin, yoksa geleceğe mi? Neden?': "Bir yapay zeka olarak zaman yolculuğu deneyimi yaşayamam ama geleceği tercih ederdim."
    } 
])

# 3. Her yanıt setine "kaynak" ekliyoruz
insan_verisi['Kaynak'] = 'İnsan'
yapay_zeka_verisi['Kaynak'] = 'Yapay Zeka'

# 4. Verileri birleştirme
birlesik_veri = pd.concat([insan_verisi, yapay_zeka_verisi], ignore_index=True)

# 5. Küçük harfe çevirme ve noktalama işaretlerini kaldırma
for col in birlesik_veri.columns:
    if birlesik_veri[col].dtype == 'object':
        birlesik_veri[col] = birlesik_veri[col].str.lower().str.replace(f"[{string.punctuation}]", "", regex=True)

# Stop-words çıkarma ve köklerine indirme işlemi
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Stop-words çıkarma ve köklerine indirme işlemi
def preprocess_text(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [ps.stem(word) for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    else:
        return text

# Tüm sütunlar için preprocessing uygulama
for column in birlesik_veri.columns:
    if birlesik_veri[column].dtype == 'object':
        birlesik_veri[column] = birlesik_veri[column].apply(preprocess_text)

# Veriyi bağımlı ve bağımsız değişkenlere ayır
X = birlesik_veri.drop(columns=['Kaynak'])
y = birlesik_veri['Kaynak'].apply(lambda x: 0 if x == 'İnsan' else 1)  # Kaynak'ı sayısal etiketlere çevir

# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Boş ve NaN değerlerini kontrol edip temizlemek için
X_train = X_train.fillna("")
X_test = X_test.fillna("")


# Her satırdaki tüm sütunları birleştirerek tek bir metin stringi oluşturuyoruz
X_train_combined = X_train.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
X_test_combined = X_test.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# BERT tokenizer ve modeli yükleme
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Eğitim ve test verilerini tokenlaştır
train_inputs = tokenizer(X_train_combined.tolist(), padding=True, truncation=True, return_tensors='pt')
test_inputs = tokenizer(X_test_combined.tolist(), padding=True, truncation=True, return_tensors='pt')

# Eğitim ve test verileri için label'ları ekleyin
train_inputs['labels'] = torch.tensor(y_train.tolist())
test_inputs['labels'] = torch.tensor(y_test.tolist())

# DataLoader hazırlama
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_inputs['labels'])
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_inputs['labels'])
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Optimizer ayarla
optimizer = AdamW(model.parameters(), lr=5e-5)

# Eğitim döngüsü
model.train()
for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")



# Modeli test moduna alma
model.eval()

# Test verisini ileri besleme
with torch.no_grad():
    test_outputs = model(input_ids=test_inputs['input_ids'], 
                         attention_mask=test_inputs['attention_mask'])
    logits = test_outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
   



import matplotlib.pyplot as plt

# Loss değerlerini kaydetmek için bir liste tanımlayın
loss_values = []

# Eğitim döngüsü
model.train()
for epoch in range(3):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    average_loss = epoch_loss / len(train_loader)
    loss_values.append(average_loss)
    print(f"Epoch {epoch + 1} Loss: {average_loss}")

# Eğitim kaybını çizdirme
plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o')
plt.title("Eğitim Süreci - Loss Değeri")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()



def tahmin_et():
    # Kullanıcıdan metin girişi al
    metin = input("Bir cümle veya metin girin: ")
    
    # Metni ön işlemeden geçir
    metin = preprocess_text(metin.lower().translate(str.maketrans('', '', string.punctuation)))
    
    # Tokenize et
    inputs = tokenizer(metin, padding=True, truncation=True, return_tensors='pt')
    
    # Tahmin et
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        tahmin = torch.argmax(logits, dim=1).item()
    
    # Sonucu döndür
    kaynak = "İnsan" if tahmin == 0 else "Yapay Zeka"
    print(f"Tahmin edilen kaynak: {kaynak}")

# Tahmin işlevini çalıştır
tahmin_et() 


    
#model kaydetme    
"""
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")"""