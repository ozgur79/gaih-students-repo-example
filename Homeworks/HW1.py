
"""
Homework 1

1) How would you define Machine Learning?
2) What are the differences between Supervised and Unsupervised Learning? Specify example 3 algorithms for each of these.
3) What are the test and validation set, and why would you want to use them?
4) What are the main preprocessing steps? Explain them in detail. Why we need to prepare our data?
5) How you can explore (countionus)continuous and discrete variables?
6) Analyse the plot given below. (What is the plot and variable type, check the distribution and make comment about how you can preproccess it.)

Cevaplar:
1)İnsan beyninin işleyemeyeceği yapıdaki bir veri yığınını, belirli filtreler ve belirli fonksiyonlar yardımıyla 
anlamlı bir yapıya dönüştürme ve bu veri yığınını yeniden inşa etme sürecine makine öğrenmesi denir. 
Örneğin; bir oyunun, yükleme başına maliyeti, sürekli değişen bir değere sahiptir. Bu değerin kabul edilebilir bir sınırı vardır.
Bu sınırın üzerindeki değere sahip bir oyun ölü bir oyundur yani elimizde, yatırım yapılmaması gereken bir ürün var demektir. 
Ayrıca indirilmesinden sonraki gün, oyuna tekrar girilip girilmediği yada oyunun ne kadar süre oynanıldığı gibi başka
değerler de vardır. Ek olarak, maliyetin yüksek olduğu durumlarda, oynama süresinin hiçbir değerinin olmadığı durumlar da vardır. 
Değişen şartlar, dataya yeni veri setlerinin girmesine neden olmaktadır.
Bu şekilde birçok parametre üzerinden, data ile duygusal bağ kurmadan, sürekli değişen bu dataya göre hızlı karar verilmesi gereken 
benzeri birçok durumda insanın alamayacağı kararlar, makine öğrenmesi algoritmaları ile alınabilir.

2)
Supervised Learning: Elimizde ML modelini eğitecek anlamlı bir veri vardır ve ne şekilde dağıldığı bilgisi ML algoritmasına verilir. 
Bu anlamlı verinin yapısı ve çıktısı bellidir. Bu yapısı belirli anlamlı veri ile ML modeli eğitilir.
Ardından ML modeline daha önce karşılaşmadığı, analiz edilecek data verilir. 
Algoritmanın, bu yeni veri ile olası durumları "predict" etmesi beklenir. Bu öğrenme modelinde ne tür bir data ile uğraşıldığı bilinmektedir 
ve gelecek tahmini de bu sayede yapılabilmektedir.

Unsupervised Learning: 
Eldeki data "unlabeled"dır, bu tanımlanamayan, yapısı belli olmayan data; biçimine/desenine göre kümelenir ve 
anlamlı bir bilgi elde edilmeye çalışılır.
Elimizde daha az test ve daha az veri dağılım bilgisi bulunduğu için, 
"Unsupervised Learning" algoritmaları, "Supervised Learning" algoritmalarına göre daha karmaşıktır. 



3)Makine Öğrenimi algoritmaları, gerçek dünya verileriyle başa çıkmadan önce veri dağılımları konusunda eğitilmeli, 
ardından doğrulanmalı ve test edilmelidir. Üzerinde çalışılan datanın, hangi şartlarda nasıl dağıldığının öğretilmesi
"validation set"ler üzerinden gerçekleşir.
Bir ML düşünme modeli tasarlarken, dağılımı bilinen bir dataset ML algoritmasına verilir. Dataset'teki verinin
dağılımının belirli ve bilinen bir verikümesi olması önemlidir.



4) Topladığımız data gerçekten ihtiyacımız olan datayı temsil ediyor mu ve bu data ile problemi çözebilecek miyiz? Ham datanın
elimizdeki problem ile tam uyumlu olması üreteceğimiz çözümün daha etkili olmasını sağlayacaktır.

head() : veri hakkında kabaca bilgi sahibi olmak için ilk beş elemanına bakılır
info() : veri içerisinde bulunan her bir sütun içerisinde kaç adet örnek var, 
bu verilerin veri tipleri nedir ve veri tiplerinin tüm veri içerisinde dağılımı nedir gibi 
veriyi tanımak için gerekli temel soruları cevaplamak için. Sütunlarında ki kayıp değer sayısı görülür
rename() : sütun isimlerinin düzenlersek veriyi daha kolay temizleriz
drop() : veri hakkında herhangi bir bilgi vermeyen veri içeren sütunlar yada satırlar çıkarılacaktır.
unique() : tekrar eden verinin tespit edilmesinde 
isnull() : boş data tespit edilir (veri setinden çıkartmak için)
isnan() : geçerli bir sayı olup olmadığının tespiti için
fillna() : boş veriyi, istatik hesaplamalarını etkilemeyecek şekide doldurmak istendiğinde
describe() : 
corr() :
loc[] :

Elimizdeki datayı ML'ye hazır hale getirmeliyiz. Böylece değişkenler arasındaki ilişkiyi daha net görebiliriz. Verideki anomaliyi
tespit etmek daha iyi görselleştirme yapmamızı sağlar.
Verinin Temizlenmesi
    Sütun İsimlerinin Düzenlenmesi
    Yararsız Verinin Çıkarılması ve Düzenlenmesi : veri hakkında herhangi bir bilgi vermediği için bazı sütunlar çıkarılacaktır. 
Tekrarlanan veriler çıkarılır. tutarsız 
    Kayıp Veri Sorunu : Kayıp veri kavramı veri içerisinde bulunan örneğin herhangi bir sütununda bulunan değerin olmaması anlamına gelir. 
NaN (Not a Number) yazan. Bu NaN yazısı örneğe ait ilgili verinin olmadığı ya da kayıp olduğu anlamına gelir.
Kayıp veri ile mücadele etmek için; Kayıp veriye ait örnekler veriden çıkarılabilir.Kayıp veriler asıl verinin dağılımını 
bozmayacak şekilde asıl verinin ortalama ya da medyan değerlerine göre doldurulabilir.




5)
Random variables: 
a)Katagorik a1)nominal:kadın/erkek a2)ordinal: s/m/l
b)Numerik b1)Discrete:Kesikli, çubuk b2)continuous: Sürekli, histogram

continuous variables : Sürekli, histogram : 
Daha küçük parçalara ayrılabiliyorsa
sıcaklık, ağırlık, uzunluk
Hayvanat bahçesinden rasgele seçilen bir hayvanın net ağırlığı : 0 ile 20 ton arasında olabilir : 
Tüm olasılıkları listesini tek tek çıkaramayız. 
100 metre yarışını kazanan sporcunun, 100 metreyi kesin koşma süresi: 9.57 saniyede olabilir, 9.571569842'de olabilir..

discrete variables : Kesikli, çubuk : yazı/tura
Daha küçük parçalara ayrılamıyorsa,
satın alınan bilet sayısı, sınıftaki öğrencilerin sayısı, gönderilen mesaj sayısı
Sınıftan rasgele seçilen bir öğrencinin doğum yılı : 92, 95, 2001 olabilir
Burada değerler sayılabilir. Listesi çıkarılabilir.
Bugün dünyada doğan çocuk sayısı
100 metre yarışını kazanan sporcunun, 100 metreyi koşma en yakın yüzlüğe yuvarlanmış süresi: 9.57


6)Histogram, gruplandırılmış bir veri dağılımının sütun grafiğiyle gösterimidir. 
Diğer bir ifadeyle, tekrarlı sayılardan oluşan verilerin, uygulanan işlemlerden sonra önce tabloya, 
tablodan yararlanarak grafiğe aktarılması, yani veri gruplarının grafiğinin dikdörtgen sütunlar halinde gösterilmesidir. 
İki farklı yaprak türüne ait dataseti görülmekte. Genişliği fazla türe ait datasetini ayrı, genişliği az olan yaprak türüne ait
datasetini ayrı değerlendirmek gerekir. Y ekseni frekansı verdiğine göre, geniş yapraklı bitkinin örnek uzaydaki sayısı daha az demektir.


"""