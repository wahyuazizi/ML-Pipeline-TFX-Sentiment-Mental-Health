# Submission 1: ML Pipeline of Sentiment Mental Health
Nama: Wahyu Azizi

Username dicoding: wahyuazizi

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Twitter Racism](https://huggingface.co/datasets/sakren/twitter_racism_dataset?library=pandas) |
| Masalah | Rasisme di media sosial adalah masalah kompleks yang mencakup ujaran kebencian, stereotip negatif, dan penghinaan berbasis ras. Anonimitas di platform digital membuat penyebaran konten rasis menjadi lebih mudah, dan algoritma media sosial dapat memperburuk situasi dengan memperluas jangkauan konten kontroversial. Pengguna yang baru dalam menggunakan sosial media sangat rentan untuk terpapar akan kebiasaan buruk ini. Oleh karena itu deteksi otomatis rasisme bisa sangat bermanfaat bagi pengguna supaya menciptakan komunikasi yang positif. |
| Solusi machine learning | model akan dibangun dengan pendekatan Deep Learning menggunakan Tensorflow yang dapat mendeteksi dan melakukan klasifikasi apakah teks atau kalimat terindikasi konten rasisme |
| Metode pengolahan | Pada data yang tersedia, pengembangan pipeline ini hanya menggunakan kolom ```text``` sebagai fitur dan ```oh_label``` sebagai label atau kelas. Dataset dibagi menjadi data train dan testing dengan persentase perbandingan 80:20. Kemudian melakukan preprocessing terhadap input yakni fitur dengan mentransformnya ke lower case, kemudian mengubah kelas kedalam bentuk integer |
| Arsitektur model | Model ini adalah model klasifikasi teks berbasis Neural Network yang mengambil input berupa teks, mengubahnya menjadi representasi numerik melalui TextVectorization, lalu melewatkannya ke Embedding Layer untuk mendapatkan vektor kata. Selanjutnya, vektor ini diringkas menggunakan Global Average Pooling untuk menghasilkan representasi tetap, sebelum diproses oleh beberapa Fully Connected Layers (ReLU) untuk ekstraksi fitur. Dropout digunakan untuk mencegah overfitting, dan akhirnya, Output Layer dengan aktivasi sigmoid menghasilkan probabilitas untuk klasifikasi biner. |
| Metrik evaluasi | Metrik evaluasi yang digunakan dalam proyek ini meliputi ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy. Evaluasi ini membantu dalam mengukur efektivitas model dalam mengklasifikasikan tweet yang rasis |
| Performa model | Dari hasil evaluasi didapatkan nilai AUC sebesar 88.4% dan Binary Accuracy mencapai 91.8% dengan False Positives, False Negatives, True Positives, dan True Negatives masing-masing 63, 164, 222, 2315. dari 2764 example count. |
