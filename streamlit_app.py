import streamlit as st

st.set_page_config(page_title="Tłumacz EN → DE", layout="centered")


@st.cache_resource(show_spinner=False)
def load_translator():
    from transformers import MarianMTModel, MarianTokenizer
    model_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(text: str) -> str:
    tokenizer, model = load_translator()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


st.title("Tłumacz angielsko-niemiecki")
st.write("Tłumaczenie tekstu z języka angielskiego na niemiecki. Model: Helsinki-NLP/opus-mt-en-de.")
st.markdown("---")

source_text = st.text_area("Tekst (angielski):", height=150)

if st.button("Przetłumacz"):
    if not source_text.strip():
        st.warning("Wprowadź tekst do przetłumaczenia.")
    else:
        with st.spinner("Tłumaczenie..."):
            try:
                translation = translate(source_text)
                st.success("Gotowe.")
                st.subheader("Tłumaczenie (niemiecki):")
                st.write(translation)
            except Exception as exc:
                st.error(f"Błąd: {exc}")

st.markdown("---")
st.caption("s26634")
