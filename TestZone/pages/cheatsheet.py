import streamlit as st

# Add text to the app
st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.text("Text")
st.markdown("Markdown")

# Add widgets to the app
st.button("Button")
st.checkbox("Checkbox")
st.radio("Radio", ["Option 1", "Option 2", "Option 3"])
st.selectbox("Selectbox", ["Option 1", "Option 2", "Option 3"])
st.multiselect("Multiselect", ["Option 1", "Option 2", "Option 3"])
st.slider("Slider", 0, 10, 5)
st.select_slider("Select slider", options=["Option 1", "Option 2", "Option 3"])
st.text_input("Text input")
st.text_area("Text area")
st.date_input("Date input")
st.time_input("Time input")
st.file_uploader("File uploader")

# Display data in the app
st.write("This is a simple text")
st.dataframe(data)
st.table(data)
st.plotly_chart(fig)
st.bokeh_chart(fig)

# Add interactivity to the app
if st.button("Button"):
    st.write("Button clicked")
    
if st.checkbox("Checkbox"):
    st.write("Checkbox checked")
    
option = st.radio("Radio", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", option)

options = st.multiselect("Multiselect", ["Option 1", "Option 2", "Option 3"])
st.write("You selected:", options)

value = st.slider("Slider", 0, 10, 5)
st.write("You selected:", value)

text = st.text_input("Text input")
st.write("You entered:", text)

date = st.date_input("Date input")
st.write("You selected:", date)

# Add columns to the app
col1, col2 = st.columns(2)
with col1:
    st.write("This is column 1")
    
with col2:
    st.write("This is column 2")
