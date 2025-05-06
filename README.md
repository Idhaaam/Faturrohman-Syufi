# Faturrohman-Syufi
Proyek ini mengimplementasikan pemrosesan data dari berbagai sumber serangan jaringan, yang kemudian digabungkan untuk pelatihan model klasifikasi. Dataset gabungan digunakan untuk melatih model menggunakan algoritma Decision Tree. Proyek ini mencakup preprocessing data, pemisahan fitur dan label, pelatihan model, evaluasi akurasi, serta visualisasi model pohon keputusan dan confusion matrix.

Import Library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as lol
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as lol
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Ekstraksi dan Pembacaan Data CSV
```python
import zipfile

with zipfile.ZipFile('drive-download-20250506T025215Z-1-001.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

dataset1 = pd.read_csv('MQTT Malformed.csv')
dataset2 = pd.read_csv('Recon OS Scan.csv')
dataset3 = pd.read_csv('Recon Ping Sweep.csv')
```
Tiga file CSV yang berisi data serangan:
MQTT Malformed.csv
Recon OS Scan.csv
Recon Ping Sweep.csv
File berada dalam format ZIP dan diekstrak sebelum dibaca.

Penggabungan DataFrame
```python
hasilgabung = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
```
Data dari ketiga sumber digabung menjadi satu DataFrame untuk pemrosesan selanjutnya.

Pemisahan Fitur dan Label
```python
x = hasilgabung.iloc[:, 7:76]
y = hasilgabung.iloc[:, 83:84]
```
Fitur (X): Kolom ke-7 hingga ke-75
Label (Y): Kolom ke-83, diasumsikan berisi target klasifikasi (jenis serangan)

Pembagian Data Latih dan Uji
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
Data dibagi menjadi:
80% data latih
20% data uji

Pelatihan Model Decision Tree
```python
alya = DecisionTreeClassifier(criterion='entropy', splitter='random')
alya.fit(x_train, y_train)
y_pred = alya.predict(x_test)
```
Model Decision Tree dibuat dan dilatih dengan parameter:
criterion='entropy': mengukur impurity
splitter='random': pembagian node secara acak

Evaluasi Akurasi
```python
accuracy = accuracy_score(y_test, y_pred)
```
Menghitung akurasi model dengan membandingkan hasil prediksi dan data aktual.

Visualisasi Pohon Keputusan
```python
fig = plt.figure(figsize=(10, 7))
tree.plot_tree(alya, feature_names=x.columns.values, class_names=np.array(['MQTT Malformed.csv', 'Recon OS Scan.csv', 'Recon Ping Sweep.csv']), filled=True)
plt.show()
```
Struktur pohon keputusan divisualisasikan, menunjukkan aturan klasifikasi berdasarkan fitur.

Confusion Matrix
```python
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
label = np.array(['MQTT Malformed', 'Recon OS Scan', 'Recon Ping Sweep'])
lol.heatmap(conf_matrix, annot=True, xticklabels=label, yticklabels=label)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.show()
```
Menampilkan matriks kebingungan dalam bentuk heatmap, yang menggambarkan kinerja klasifikasi model terhadap masing-masing kelas.

Pembagian DataFrame Lebih Lanjut
```python
hasilgabung = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
jumlah_kolom = hasilgabung.shape[1]
tengah = jumlah_kolom // 2
kiri = hasilgabung.iloc[:, :tengah]
kanan = hasilgabung.iloc[:, tengah:]

print("DataFrame Gabungan:")
print(hasilgabung)
print("\n")
print("Bagian Kiri:")
print(kiri)
print("\n")
print("Bagian Kanan:")
print(kanan)
```
Kode tambahan untuk membagi DataFrame gabungan menjadi dua bagian (kiri dan kanan) berdasarkan jumlah kolom. Ini mungkin untuk tujuan analisis atau pemrosesan data lebih lanjut.

Script Python ini menyusun pipeline analitik lengkap untuk klasifikasi serangan jaringan berbasis data dari berbagai sumber. Tahapan dimulai dari pembacaan dan penggabungan data, preprocessing fitur, pelatihan model klasifikasi Decision Tree, hingga visualisasi hasil evaluasi. Model mampu melakukan klasifikasi otomatis terhadap jenis serangan jaringan berdasarkan fitur-fitur yang ada. Visualisasi pohon keputusan dan confusion matrix membantu dalam mengevaluasi serta memahami pola pengambilan keputusan model. Kode tambahan untuk membagi DataFrame memberikan fleksibilitas untuk analisis data lebih lanjut.
