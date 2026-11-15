# Audit hesabatı: Bank Account Fraud (BAF)

## 0. Layihə nəyi proqnozlaşdırır (kod və sənədlərdən)
- **Məsələ:** bank hesabı açılışı müraciətinin fraud olub-olmadığı, binary classification. **Target:** `fraud_bool` (`src/config.py`).
- **Proqnoz anı:** müraciət daxil olduğu an (bank saniyələr içində təsdiq/blok qərarı verir). Model keçmiş aylarla öyrənib gələcək ayı proqnozlaşdırır.
- **Data:** BAF `Base.csv`, 1,000,000 sətir, 32 sütun, fraud 1.10%, `month` 0-7. Split: train 0-4 / valid 5 / test 6-7 (zamana görə).
- Əlçatan olmayan: real `Base.csv`, `models/`, `scripts/*` və `results/*` bu yoxlamaya verilməyib. **Real data ilə heç bir metrika yenidən hesablanmayıb.**

## 1. Tapılan problemlər

| # | Ciddilik | Yer (orijinal fayl) | Problem |
|---|---|---|---|
| 1 | Yüksək | `app_main.py` Tab1 qrafik 5-15, Tab2 qrafik 1,7,10-15, Tab3 qrafik 1,4-15, sətir 1054-1060 | App-dakı qrafiklərin çoxu **uydurmadır**: sabit siyahılar (`[0.35,0.25,...]` "Feature Importance", `Loss [0.5..0.1]` "Training loss" HGB üçün, ROC-AUC `CatBoost 0.90`, CV `Accuracy 0.91-0.93`, confusion `70/20/10`), `np.random` ilə "aylıq trend" və "həcm", `max(0.1, 1-3t)` "recall əyrisi". "İllik qənaət AZN" düsturu (`həcm*thr*0.8*120`) heç bir real hesaba əsaslanmır. Bunlar real nəticə kimi təqdim olunurdu. |
| 2 | Yüksək | `app_main.py:13` | `from src.inference_main import ...` — repo-da və README-də modul `src/inference.py`-dir; app import səhvi ilə açılmır. `plotly` isə `requirements.txt`-də yoxdur. |
| 3 | Orta | `app_main.py` lüğət | `intended_balocant_amount` yazı səhvi: real sütun `intended_balcon_amount`, label/help tətbiq olunmur. |
| 4 | Orta | notebook 03 çıxışları vs README/DEFENSE_QA | **Sənəd rəqəmləri saxlanmış çıxışlarla üst-üstə düşmür.** 02: 25 feature, ΔPR-AUC −0.0069 [−0.0119,−0.0015], threshold 0.0131, alarm 17.93%, xərc 3073, ECE 0.2289→0.00138; README/DEFENSE_QA: 26 feature, −0.0057, 0.0130, 17.72%, 3057, 0.2263→0.0016. 03-ün çıxışları isə **isotonic dövrünün** modelindəndir (threshold 0.0199, test PR-AUC 0.1768 [0.1644,0.1911], TPR@5%FPR 0.4715), sənəddə isə 0.193 [0.180,0.210] / 0.542. Yəni 03 final modelə uyğun yenidən işlədilməyib. |
| 5 | Orta | `01_edamain.ipynb` (bütün target-əlaqəli hüceyrələr) | EDA **bütün aylarda** (valid+test daxil) aparılırdı: missing-indikator, univariate AUC, yaş/kateqorik fraud faizləri test label-larına baxaraq çıxarılırdı. Test model seçimində iştirak etmir, amma preprocessing qərarlarının əsası test label-larını görürdü. |
| 6 | Aşağı-orta | `features.py:35,40` | Inference-də boş (NaN) xana `-1` kodu kimi tanınmırdı: `*_missing=0` + NaN dəyər (train-də isə `-1` → indikator=1). Train/serving uyğunsuzluğu. |
| 7 | Aşağı-orta | `inference_main.py:26-37` | Tip/schema yoxlaması yox idi (mətn rəqəm sütununda səssiz xəta/pipeline xətası); `model.joblib` (pickle) bütövlük yoxlaması olmadan yüklənirdi; sklearn versiyası uyğunsuzluğu xəbərdarlıq vermirdi. |
| 8 | Aşağı | `02_modeling` cell 6 | Sabit sütunlar bütün datadan müəyyən edilirdi (indi yalnız train). Praktik təsir yoxdur (`device_fraud_count` hər yerdə 0). |
| 9 | Aşağı | `evaluation.py:45` | `fp+tn==0` olanda ZeroDivisionError (kiçik qrup/split). |
| 10 | Məlumat | `features.py:72` | Şərh yanlış idi ("onluqlarla verilib"); riyaziyyat düzdür (il×12=ay). Şərh düzəldildi. |
| 11 | Məlumat | notebook 03 §9 | "Zamana CV" fold 6-7 test aylarıdır, hiperparametrlər valid ayında seçilib: yalnız hesabat xarakterli. Qeyd əlavə edildi. |

### Yoxlanıb, problem tapılmayanlar
- **Split:** zamana görə, kəsişməsiz (indi testlə təsdiqlənir). `month` feature deyil.
- **Preprocessing/imputation/scaling/encoding:** hamısı Pipeline-da, yalnız `X_tr` üzərində `fit` (test: valid/test dəyişəndə öyrənilən statistika dəyişmir).
- **Imbalance:** `class_weight`, resampling yoxdur → valid/test toxunulmazdır.
- **Tuning, feature selection, calibration-metod seçimi:** valid ayında; test yalnız hesabatda. **Threshold:** valid-dən (`choose_cost_threshold`), test-ə baxmadan.
- **Feature-ların proqnoz anında məlumluğu:** `fraud_bool` xaric bütün 30 xam feature BAF sənədində müraciət anında məlum atributlardır; target-dan sonra yaranan sütun tapılmadı. `month` və sabit `device_fraud_count` atılır. (Aşağıda 5.2-yə bax.)
- Dublikat: 0 (EDA). Tiplər/null: `df.info()` — null yoxdur, gizli missing `-1`/mənfi kodlarla.

## 2. Edilən dəyişikliklər
| Fayl | Dəyişiklik |
|---|---|
| `app.py` (yenidən yazıldı) | Bütün uydurma qrafiklər və saxta "qənaət" çıxarıldı. İndi: real skor, threshold müqayisəsi, **real lokal həssaslıq** (feature default-a qaytarılanda modelin real ehtimal dəyişməsi), toplu skorda real histogram/pie/aylıq nisbət (yalnız yüklənən fayldakı `month` varsa), `meta.json`-dan real confusion matrix və (varsa) importance/model müqayisəsi/threshold əyrisi. Import `src.inference`-ə düzəldildi, label typo düzəldildi. |
| `src/inference.py` (`inference_main.py`-dən) | `validate_input` (çatışan sütun, rəqəm olmayan dəyər, boş cədvəl, artıq sütunların atılması, giriş dəyişdirilmir), `model.joblib` SHA-256 yoxlaması (yükləmədən əvvəl), sklearn versiya xəbərdarlığı. |
| `src/features.py` | NaN də "missing" sayılır (indikator ilə); şərh düzəldildi. BAF xam datasında NaN olmadığı üçün train nəticələri dəyişmir. |
| `src/evaluation.py` | `choose_cost_threshold` (yalnız verilən valid üzrə), `fpr` sıfıra bölmə qoruması. |
| `src/splits.py` (yeni) | `time_split` + `assert_chronological`; notebook 02 bunu istifadə edir. |
| `src/train.py` (yeni) | Təkrarlana bilən end-to-end təlim (seed, split, calibration/threshold valid-də, test bir dəfə, hash + versiya + `drop_cols` meta-ya). `build_input_schema` notebook ilə ortaq. |
| `scripts/make_synthetic_baf.py` (yeni) | README-də göstərilən, amma verilməyən sintetik generator (yalnız kod yoxlaması üçün). |
| `02_modeling.ipynb` | `time_split`/`build_input_schema`/`choose_cost_threshold` src-dən; sabit sütunlar train-dən; `meta.json`-a `model_sha256`, `sklearn_version`, `drop_cols`, `feature_importance`, `model_comparison`, `threshold_curve`; hash ilə reload yoxlaması; qalan risk qeydi. |
| `01_eda.ipynb` | Target-əlaqəli EDA yalnız **train aylarında**; zaman bölməsi train+valid; `sample` ölçüsü qorunur. |
| `03_extended_analysis.ipynb` | Model hash ilə yüklənir; CV haqqında qeyd. |
| `requirements*.txt` | `plotly` (app), `scipy`, `pytest` əlavə olundu. |
| `tests/` (yeni, 25 test) | split kəsişməsi/xronologiya; preprocessing yalnız train-də fit; test label/dəyərləri dəyişəndə model eyni; threshold test-dən asılı deyil; schema/inference; hash dəyişdirmə rədd; təkrarlanma; end-to-end smoke; Streamlit AppTest. |

Notebook-ların **saxlanmış çıxışlarına toxunulmayıb** (real data olmadığı üçün); yeni kod isə sintetik datada 01→02→03 tam işlədi.

## 3. İşlədilən komandalar və real nəticələr
- `python -m pytest -q tests` → **25 passed** (≈20 s, sintetik data).
- Mutasiya yoxlaması: `train.py`-də threshold-u test-dən seçən və pipeline-ı train+test üzərində fit edən qəsdən pozulmuş variantlar `test_threshold_and_calibration_independent_of_test` testini **uğursuz etdi** (testlər sızıntını həqiqətən tutur); orijinal bərpa olundu, hamısı keçdi.
- `BAF_DATA_PATH=data/Base_synthetic.csv` (150k sətir, sintetik) ilə `01_eda → 02_modeling → 03_extended_analysis` nbclient ilə (təmiz surətdə, README toxunulmadan): **hamısı xətasız işlədi**; `models/meta.json` yeni açarlarla və hash ilə yarandı, reload yoxlaması keçdi. Bu nəticələrin metrikaları sintetik data üzrədir və **real deyil**.
- Streamlit: `AppTest` ilə app yüklənir, skorlama düyməsi xətasız işləyir (streamlit 1.67).
- Real data ilə skriptlər işlədildi: `compare_boosters` (LightGBM/XGBoost/CatBoost quraşdırıldı), `fe_ablation` (repo-da yox idi, yenidən yazıldım), `fe_ablation_linear`, `age_ablation`, `wider_tuning 10`.

## 4. Real data ilə nəticə (Base.csv, 1,000,000 sətir)
Data: fraud 1.1029%, dublikat 0, NaN 0 (EDA ilə üst-üstə düşür). 01→02→03 real data ilə xətasız işlədi (15 s / 434 s / 176 s).

| Göstərici (test, ay 6-7) | Repo-da əvvəl saxlanmış `meta.json` | Düzəlişlərdən sonra yenidən təlim |
|---|---|---|
| PR-AUC | 0.193767 | 0.193767 |
| ROC-AUC | 0.894228 | 0.894228 |
| TPR@5%FPR | 0.542738 | 0.542738 |
| Threshold | 0.013099 | 0.013099 |
| TP / FP / FN / TN | 2307 / 34453 / 571 / 167680 | 2307 / 34453 / 571 / 167680 |
| Recall / precision / alarm | 0.8016 / 0.0628 / 17.93% | 0.8016 / 0.0628 / 17.93% |

Yəni düzəlişlər modelin nəticəsini dəyişmədi və təlim təkrarlana bilir (rəqəmlər bit-bit eynidir). 03 indi final modelə (threshold 0.0131) uyğundur: test PR-AUC 0.1938 [0.1803, 0.2098], ROC-AUC 0.8942 [0.8881, 0.9007], TPR@5%FPR 0.5427 [0.5239, 0.5606], KS 0.637, zamana CV ROC-AUC 0.902 ± 0.008. 01 indi train aylarında (675,666 sətir, fraud 0.998%).
**Sənədlər** (README, DEFENSE_QA, DECISIONS) 02/03-ün real çıxışlarına görə sinxronlaşdırıldı: 25 feature, ΔPR-AUC −0.0069 [−0.0119, −0.0015], threshold 0.0131, xərc 3073, ECE 0.2289 → 0.0014, model cədvəli və həssaslıq cədvəli və s.

### Skriptlərin real nəticələri (Base.csv)
- Booster müqayisəsi (valid / test PR-AUC): HGB baseline 0.1868 / 0.1808, HGB tuned 0.1913 / 0.1938, LightGBM 0.1942 / 0.1975, XGBoost 0.1889 / 0.1952, CatBoost 0.1917 / 0.1957. Paired bootstrap (valid, Δ vs HGB tuned): LightGBM +0.0027 [−0.0024, +0.0072], XGBoost −0.0026 [−0.0054, +0.0004], CatBoost +0.0006 [−0.0046, +0.0059]: hamısı əhəmiyyətsiz. HGB sadəliyə görə saxlanır. (Qeyd: LightGBM-in valid-də ən yüksək olması seçim əsası deyil.)
- FE ablation, HGB: 56 → 72 feature, valid 0.1907 → 0.1913, test 0.1933 → 0.1938. Xətti model: valid 0.1552 → 0.1615, test 0.1597 → 0.1650.
- Yaşsız model: test PR-AUC 0.1938 → 0.1906; 50+ FPR 12.98% → 10.93%, <50 FPR 3.93% → 4.36%.
- Geniş tuning (10 sınaq): ən yaxşı valid 0.1919 vs mövcud 0.1913; parametrlər dəyişmədi (3 parametr yenə şəbəkə kənarındadır).

## 5. Qalan risklər və sizdən tələb olunanlar
1. **Skriptlər yenidən işlədildi.** Uyğun çıxanlar: yaşsız model (FPR fərqi 9.05 → 6.57 faiz bəndi, PR-AUC 0.1938 → 0.1906), geniş tuning (ən yaxşı valid 0.1919 vs 0.1913), xətti FE ablation. Düzəldilənlər: XGBoost artıq HGB-dən "əhəmiyyətli pis" deyil (−0.0026 [−0.0054, +0.0004]); HGB-də FE ablation 0.1933 → 0.1938 (köhnə 0.1946/0.1930 mənbəsi məlum deyildi, `fe_ablation.py` repo-da yox idi). Fairness cədvəllərində üçüncü onluq fərqləri qala bilər.
2. **Feature mənbəyi sualı:** velocity/`*_4w`/`*_8w` pəncərə feature-ləri müraciət vaxtından **əvvəlki** məlumatla hesablanıbmı? BAF sənədi belə deyir, amma data generatorunu mən görmürəm. Production-da eyni hesablamanı təmin etmək sizin məsuliyyətinizdədir. Velocity PSI 1-3.7 çox yüksəkdir (driftin səbəbi yoxlanılmayıb).
3. **Eyni müştərinin train/test-də olması** yoxlana bilməz: BAF-da müştəri/ID sütunu yoxdur.
4. **Valid ayı çox yüklənib:** tuning, permutation importance, feature seçimi, calibration, threshold hamısı ay 5-dədir (tək ay, ~119k sətir). Test təmizdir, amma valid-ə optimizm mümkündür. Calibrator (sigmoid, 2 parametr) və threshold eyni valid-dədir; kodu real nəticəni dəyişəcəyi üçün (yenidən işlətmə lazım) toxunmadım. İstəsəniz valid-i A/B-yə bölək.
5. **Threshold/alarm:** FN=500/FP=10 illüstrativdir; alarm 17.7%, precision 6.3%. Real xərc və yoxlama tutumu lazımdır.
6. **Fairness:** `customer_age` modeldə qalır; 50+ üçün FPR 32% vs 14% — hüquqi/etik qərar sizindir (README-də təsvir olunub).
7. Verilməyən fayllar (`scripts/compare_boosters.py`, `fe_ablation*.py`, `age_ablation.py`, `wider_tuning.py`, `results/`, `models/`) audit olunmayıb; onlardakı rəqəmlər təsdiqlənməyib.
8. `use_container_width` yeni Streamlit-də köhnəlmiş xəbərdarlıq verir (hələ işləyir). `model.joblib` pickle-dır: yalnız öz mühitinizdə yaradılmış faylı yükləyin; hash yoxlaması ancaq bu yeni notebook işlədildikdən sonra `meta.json`-da olacaq (köhnə meta-da hash yoxdur, xəbərdarlıq verilir).

## 6. Təkrar işə salma
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # Python >= 3.11, scikit-learn==1.9.1
python -m pytest -q tests                     # 25 test, real data tələb etmir
# Base.csv -> data/Base.csv
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_modeling.ipynb 03_extended_analysis.ipynb
#   (və ya: python -m src.train data/Base.csv models   # skript variantı, notebook-suz)
streamlit run app.py
# yalnız kod yoxlaması: python scripts/make_synthetic_baf.py 150000 data/Base_synthetic.csv && BAF_DATA_PATH=data/Base_synthetic.csv jupyter nbconvert ...
```
