import streamlit as st
import spacy
from spacy import displacy

st.title("SpaCy explorer")
st.header("Dependency visualizer")

nlp = spacy.load('en_core_web_md')

#import streamlit.components.v1 as components
#components.html("https://explosion-demos.netlify.app/js/displacy.js")

#import os
#st.text((os.path.dirname(st.__file__)))

input_text = st.text_input('Text string to analyze:', 'Jennifer drove to Seattle.')

doc= nlp(input_text)

dep_svg = displacy.render(doc, style="dep", jupyter=False)

st.image(dep_svg, width=400, use_column_width='never')


st.header("Entity visualizer")

ent_html = displacy.render(doc, style="ent", jupyter=False)

st.markdown(ent_html, unsafe_allow_html=True)
