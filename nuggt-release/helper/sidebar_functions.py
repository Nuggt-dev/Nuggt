
import base64
import streamlit as st

@st.cache_data()
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position="50% 0%",
    margin_top="10%",
    image_width="30%",
    image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return f"""
            <style>
                [data-testid="stSidebarNav"] {{
                    background-image: url("data:image/png;base64,{binary_string}");
                    background-repeat: no-repeat;
                    background-position: {background_position};
                    margin-top: {margin_top};
                    background-size: {image_width} {image_height};
                }}
            </style>
            """


def sidebar_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )
