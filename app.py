"""Streamlit app: Bank Account Fraud risk paneli.

Bütün qrafiklər real məlumatdan gəlir: (1) istifadəçinin daxil etdiyi/yüklədiyi sətirlər üzərində modelin real
çıxışı, (2) models/meta.json-dakı notebook nəticələri. Meta-da olmayan hissələr (feature importance, model
müqayisəsi, threshold əyrisi) göstərilmir, 02_modeling.ipynb-ni işlətmək kifayətdir.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    flag_by_capacity,
    load_artifacts,
    score_frame,
    values_to_frame,
)

st.set_page_config(page_title="Fraud Risk Paneli", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts(ROOT)


try:
    model, meta = get_artifacts()
except FileNotFoundError:
    st.error("models/model.joblib tapılmadı. Əvvəlcə 02_modeling.ipynb-ni (və ya python -m src.train) işə salın.")
    st.stop()
except ValueError as e:  # hash uyğunsuzluğu
    st.error(f"Model yüklənmədi: {e}")
    st.stop()

schema, thr = meta["input_schema"], float(meta["threshold"])

with st.sidebar:
    st.header("🎛️ Parametrlər")
    dynamic_thr = st.slider("Qərar həddi (threshold)", 0.001, 0.200, float(min(max(thr, 0.001), 0.2)), 0.001, format="%.3f")
    st.caption(f"Notebook-da validation-dan seçilmiş threshold: {thr:.4f}")
    st.markdown("---")
    st.markdown("👨‍🎓 **Lead Data Scientist:** Amin Rzazada")
    st.markdown("🏛️ **Universitet:** AzTU - SABAH")

st.title("🛡️ Bank Account Fraud: Risk Paneli")
st.caption(f"Model: {meta['model']} + {meta['calibration']} calibration | Aktiv threshold: {dynamic_thr:.4f}")

# Biznes terminləri lüğəti
FEATURE_LABELS = {
    "income": {
        "label": "Müştərinin Aylıq Gəliri (Skalalanmış)",
        "help": "Müştərinin rəsmi təsdiqlənmiş aylıq gəliri.",
    },
    "name_email_similarity": {
        "label": "Ad və E-poçt Uyğunluğu",
        "help": "Ad ilə e-poçt ünvanının oxşarlığı (0-1).",
    },
    "prev_address_months_count": {
        "label": "Əvvəlki Ünvanda Yaşama Müddəti (Ay)",
        "help": "Əvvəlki ünvanda qalma müddəti (-1: məlumat yoxdur).",
    },
    "current_address_months_count": {
        "label": "Cari Ünvanda Yaşama Müddəti (Ay)",
        "help": "İndiki ünvanda qalma müddəti.",
    },
    "customer_age": {
        "label": "Müştərinin Yaşı",
        "help": "Müştərinin yaşı.",
    },
    "days_since_request": {
        "label": "Müraciətdən Keçən Vaxt",
        "help": "Qeydiyyat tələbindən sonra keçən gün sayı.",
    },
    "intended_balcon_amount": {
        "label": "Nəzərdə Tutulan İlkin Balans",
        "help": "Hesaba yatırılacaq ilkin məbləğ.",
    },
    "payment_type": {
        "label": "Ödəmə Tipi / Kanalı",
        "help": "İstifadə olunan ödəniş növü.",
    },
    "zip_count_4w": {
        "label": "Poçt İndeksi Sorğu Sayı (4 həftə)",
        "help": "Eyni poçt indeksindən müraciət intensivliyi.",
    },
    "velocity_6h": {
        "label": "Son 6 Saatda Əməliyyat Tezliyi",
        "help": "Son 6 saatdakı tranzaksiya sayı.",
    },
    "velocity_24h": {
        "label": "Son 24 Saatda Əməliyyat Tezliyi",
        "help": "Bir gün ərzindəki əməliyyat sayı.",
    },
    "velocity_4w": {
        "label": "Son 4 Həftədə Əməliyyat Tezliyi",
        "help": "Son 4 həftədəki aktivlik.",
    },
    "bank_branch_count_8w": {
        "label": "Filial Yaxınlığındakı Müraciətlər (8 həftə)",
        "help": "Filial ətrafındakı müraciət sayı.",
    },
    "date_of_birth_distinct_emails_4w": {
        "label": "Doğum Tarixi Üzrə Fərqli E-poçtlar",
        "help": "Eyni doğum tarixi ilə fərqli e-poçtlar.",
    },
    "employment_status": {
        "label": "Məşğulluq Statusu",
        "help": "Müştərinin iş statusu.",
    },
    "credit_risk_score": {
        "label": "Kredit Risk Hesabı (Skoru)",
        "help": "Kredit tarixçəsinə əsaslanan risk balı.",
    },
    "email_is_free": {
        "label": "Pulsuz E-poçt Provideri?",
        "help": "Gmail, Hotmail və s. istifadəsi (0/1).",
    },
    "housing_status": {
        "label": "Mənzil / Yaşayış Statusu",
        "help": "Ev sahibi və ya kirayə.",
    },
    "phone_home_valid": {
        "label": "Ev Telefonunun Doğruluğu",
        "help": "Ev nömrəsinin etibarlılığı (0/1).",
    },
    "phone_mobile_valid": {
        "label": "Mobil Telefonun Doğruluğu",
        "help": "Mobil nömrənin aktivliyi.",
    },
    "bank_months_count": {
        "label": "Bankda Hesabın Müddəti (Ay)",
        "help": "Bankda hesabın ümumi ay sayı.",
    },
    "has_other_cards": {
        "label": "Digər Bank Kartlarının Mövcudluğu",
        "help": "Başqa banklarda kartın olması (0/1).",
    },
    "proposed_credit_limit": {
        "label": "Təklif Olunan Kredit Limiti",
        "help": "İlkin kredit limiti məbləği.",
    },
    "foreign_request": {
        "label": "Xarici Ölkədən Müraciət?",
        "help": "Xaricdən edilib-edilməməsi (0/1).",
    },
    "source": {
        "label": "Müraciət Mənbəyi / Kanalı",
        "help": "İnternet, Mobil və s.",
    },
    "session_length_in_minutes": {
        "label": "Sessiya Müddəti (Dəqiqə)",
        "help": "Saytda keçirilən ümumi vaxt.",
    },
    "device_os": {
        "label": "Cihazın Əməliyyat Sistemi",
        "help": "Windows, iOS, Android və s.",
    },
    "keep_alive_session": {
        "label": "Sessiyanın Saxlanılması",
        "help": "Bağlantının aktivlik statusu.",
    },
    "device_distinct_emails_8w": {
        "label": "Cihaz Üzrə Fərqli E-poçtlar (8 həftə)",
        "help": "Eyni cihazdan qeydiyyatlar.",
    },
}

tab1, tab2, tab3 = st.tabs(["👤 Fərdi müraciət", "📁 Toplu skorinq", "📊 Model göstəriciləri"])

# ============================ TAB 1 ============================
with tab1:
    st.markdown("### Tək müraciətin skorlanması")
    preset = st.selectbox("Hazır nümunə", ["Median dəyərlər (standart)"] + list(meta["examples"]))
    base = {c: s["default"] for c, s in schema.items()}
    if preset in meta["examples"]:
        base.update(meta["examples"][preset])

    values = {}
    cols = st.columns(3)
    for i, (name, spec) in enumerate(schema.items()):
        box = cols[i % 3]
        key = f"{preset}|{name}"
        info = FEATURE_LABELS.get(name, {"label": name, "help": name})
        if spec["type"] == "cat":
            idx = spec["choices"].index(base[name]) if base[name] in spec["choices"] else 0
            values[name] = box.selectbox(info["label"], spec["choices"], index=idx, key=key, help=info["help"])
        elif spec["type"] == "int" and spec["min"] == 0 and spec["max"] == 1:
            values[name] = box.selectbox(info["label"], [0, 1], index=int(base[name]), key=key, help=info["help"])
        elif spec["type"] == "int":
            val = int(min(max(base[name], spec["min"]), spec["max"]))
            values[name] = box.number_input(info["label"], int(spec["min"]), int(spec["max"]), val, step=1, key=key, help=info["help"])
        else:
            val = float(min(max(base[name], spec["min"]), spec["max"]))
            step = (spec["max"] - spec["min"]) / 100 or 0.01
            values[name] = box.number_input(info["label"], spec["min"], spec["max"], val, step=step, format="%.4f", key=key, help=info["help"])

    st.markdown("---")
    if st.button("🚀 Risk ehtimalını hesabla", type="primary"):
        row = values_to_frame(values, meta)
        p = float(model.predict_proba(row)[:, 1][0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud ehtimalı (calibrated)", f"{p:.2%}")
        c2.metric("Aktiv threshold", f"{dynamic_thr:.2%}")
        c3.metric("Ehtimal / threshold", f"{p / dynamic_thr:.1f}x")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= dynamic_thr:
            st.error(f"🚨 Müraciət işarələndi (ehtimal {p:.2%} ≥ threshold {dynamic_thr:.2%}).")
        else:
            st.success(f"✅ Müraciət işarələnmədi (ehtimal {p:.2%} < threshold {dynamic_thr:.2%}).")

        # Real lokal həssaslıq: hər feature tək-tək default (median/mod) dəyərə qaytarılanda modelin ehtimalı necə dəyişir
        deltas = {}
        for name, spec in schema.items():
            alt = dict(values)
            alt[name] = spec["default"] if spec["type"] == "cat" else (int(spec["default"]) if spec["type"] == "int" else float(spec["default"]))
            deltas[name] = p - float(model.predict_proba(values_to_frame(alt, meta))[:, 1][0])
        d = pd.Series(deltas)
        d = d[d.abs() > 1e-9].sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        if len(d):
            fig = px.bar(d[::-1].rename("Δ ehtimal").reset_index(), x="Δ ehtimal", y="index", orientation="h",
                         title="Bu müraciətdə ehtimala ən çox təsir edən feature-lər (default dəyərə qaytarılanda fərq)")
            fig.update_layout(yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Bu, modelin real çıxışı üzərində bir-bir feature dəyişdirmə hesabıdır (lokal izah). Səbəb sübutu deyil.")

# ============================ TAB 2 ============================
with tab2:
    st.markdown("### Toplu CSV skorinqi")
    rule = st.radio("Qərar strategiyası", ["Threshold", "Tutum limiti (ən riskli X%)"])
    capacity = st.slider("Yoxlama tutumu (%)", 1, 30, 6) if rule.startswith("Tutum") else None
    up = st.file_uploader("Müraciətlər CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        try:
            res = score_frame(model, meta, data)
        except ValueError as e:
            st.error(f"Xəta: {e}")
        else:
            res["flagged"] = flag_by_capacity(res["fraud_probability"], capacity) if capacity is not None else res["fraud_probability"] >= dynamic_thr
            n_fl = int(res["flagged"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Ümumi müraciət", f"{len(res):,}")
            m2.metric("🚨 İşarələnən", f"{n_fl:,}", f"{res['flagged'].mean():.2%}")
            m3.metric("✅ İşarələnməyən", f"{len(res) - n_fl:,}")

            a, b = st.columns(2)
            a.plotly_chart(px.histogram(res, x="fraud_probability", nbins=40, log_y=True, title="Ehtimal paylanması (log şkala)"), use_container_width=True)
            b.plotly_chart(px.pie(pd.DataFrame({"Status": ["İşarələnən", "İşarələnməyən"], "Say": [n_fl, len(res) - n_fl]}),
                                  names="Status", values="Say", hole=0.4, title="Qərar nisbəti"), use_container_width=True)
            if "month" in res.columns:
                mm = res.groupby("month").agg(flag_rate=("flagged", "mean"), mean_prob=("fraud_probability", "mean")).reset_index()
                st.plotly_chart(px.line(mm, x="month", y=["flag_rate", "mean_prob"], markers=True,
                                        title="Aylar üzrə işarələnmə nisbəti və orta ehtimal (yüklənən datadan)"), use_container_width=True)
            if "fraud_bool" in res.columns:
                st.info(f"Yüklənən faylda real etiket var: real fraud nisbəti {res['fraud_bool'].mean():.2%}. "
                        f"Qeyd: bu fayl təlim aylarını ehtiva edirsə, nəticələr obyektiv qiymətləndirmə deyil.")
            top = res.sort_values("fraud_probability", ascending=False).head(100)
            st.markdown("#### Ən riskli 100 müraciət")
            st.dataframe(top)
            st.download_button("📥 Nəticəni CSV yüklə", res.to_csv(index=False).encode("utf-8"), "scored.csv", "text/csv")

# ============================ TAB 3 ============================
with tab3:
    st.markdown("### Notebook-da hesablanmış göstəricilər (test = ay 6-7)")
    st.dataframe(pd.DataFrame(meta["test_metrics"]).T.round(4))
    op = meta["operating_point_test"]
    st.info(f"Seçilən threshold {op['threshold']:.4f}: recall {op['recall']:.1%}, precision {op['precision']:.1%}, "
            f"alarm nisbəti {op['flag_rate']:.1%}, xərc/1000 müraciət {op['cost_per_1000']:.0f} (xərc fərziyyələri illüstrativdir: "
            f"FN={meta['cost_fn']:.0f}, FP={meta['cost_fp']:.0f}).")
    cm = pd.DataFrame([[op["tn"], op["fp"]], [op["fn"], op["tp"]]], index=["Real legit", "Real fraud"], columns=["Proqnoz legit", "Proqnoz fraud"])
    st.plotly_chart(px.imshow(cm, text_auto=",d", color_continuous_scale="Blues", title="Test confusion matrix (real say)"), use_container_width=True)

    if meta.get("feature_importance"):
        fi = pd.Series(meta["feature_importance"]).sort_values()
        st.plotly_chart(px.bar(fi.rename("ΔPR-AUC").reset_index(), x="ΔPR-AUC", y="index", orientation="h",
                               title="Permutation importance (valid, top)").update_layout(yaxis_title=""), use_container_width=True)
    if meta.get("model_comparison"):
        mc = pd.DataFrame(meta["model_comparison"]).T
        st.markdown("#### Model müqayisəsi (notebook 02)")
        st.dataframe(mc.round(4))
    if meta.get("threshold_curve"):
        tc = pd.DataFrame(meta["threshold_curve"])
        st.plotly_chart(px.line(tc, x="threshold", y="cost_per_1000", log_x=True, title="Xərc vs threshold (valid)")
                        .add_vline(x=dynamic_thr, line_dash="dash", line_color="red"), use_container_width=True)
    if not any(meta.get(k) for k in ("feature_importance", "model_comparison", "threshold_curve")):
        st.caption("Feature importance, model müqayisəsi və threshold əyrisi üçün 02_modeling.ipynb-ni yenidən işlədin (meta.json bu bölmələri yazır).")
