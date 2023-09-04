import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

import model as model


def load_data():
    """
    Load the preprocessed data from a CSV file.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    df = pd.read_csv("../data/processed/data.csv")
    return df


@st.cache_data()
def load_model():
    """
    Load the machine learning model.

    Returns:
        my_module.model.Model: The machine learning model.
    """
    model_factory = model.Model(
        data_path="../data/processed/data.csv",
        embed_path="../data/embed.npy",
        model_path="../model/gnn.pt",
        mapping_path="../model/model_mapping.json",
        downloads_path="../model/downloads_mapping.json",
    )
    return model_factory


st.set_page_config(
    page_title="Hugging Face Model Recommender System (HFMRS)",
    page_icon=":hugging_face:",
    layout="wide",
)


df = load_data()
# sort dropdown by desc downloads
model_ids = df.sort_values("downloads", ascending=False).iloc[:, 0].values

COS_SIM = "Cosine Similarity"
JAC_SIM = "Jaccard Similarity"
GNN = "Graph Neural Network"
RECOMMENDERS = [COS_SIM, JAC_SIM, GNN]
DISPLAY_COLS = ["modelId", "downloads", "score"]
SORT_BY_METHOD = {
    "None": lambda x: x,
    "Downloads (Ascending)": lambda x: x.sort_values("downloads", ascending=True),
    "Downloads (Descending)": lambda x: x.sort_values("downloads", ascending=False),
}
model_fac = load_model()

# # Create a Streamlit app
st.title("🤗 Hugging Face Model Recommender System (HFMRS)")
st.write("Select a model to get recommendations.")

model_id = st.selectbox("Model Id", model_ids)
recommender = st.selectbox("Recommender Algorithm", RECOMMENDERS)
recommend_no = st.slider("Number of Recommendations", 1, 30)
sort_by = st.selectbox("Sort By", SORT_BY_METHOD)


def make_clickable(modelId):
    """
    Create a clickable link to a Hugging Face model.

    Parameters:
    -----------
    modelId : str
        The ID of the Hugging Face model.

    Returns:
    --------
    str
        A string containing an HTML link to the model.

    Example:
    --------
    >>> make_clickable("bert-base-uncased")
    '<a href="https://huggingface.co/bert-base-uncased" target="_blank">bert-base-uncased</a>'
    """
    return f'<a href="https://huggingface.co/{modelId}" target="_blank">{modelId}</a>'


def bar_plot(df, metric_name):
    """
    Generate a bar plot showing the cosine similarity score for each model in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be plotted. Must have columns "modelId" and "score".
    metric_name: str
        Metric name selected

    Returns:
    --------
    None

    Example:
    --------
    >>> df = pd.DataFrame({"modelId": ["bert-base-uncased", "distilbert-base-uncased"], "score": [0.8, 0.7]})
    >>> bar_plot(df)
    """
    # Sort the DataFrame by cosine similarity score in descending order
    df = df.sort_values(by="score", ascending=False)
    fig = px.bar(
        df,
        x="score",
        y="modelId",
        orientation="h",
        title=f"{metric_name} Score vs. Product Name",
    )
    fig.update_layout(xaxis_title="Cosine Similarity Score", yaxis_title="Model ID")
    st.plotly_chart(fig, use_container_width=True)


if st.button("Recommend"):
    if recommender == COS_SIM:
        result = (
            model_fac.get_recommendation(model_id, metric="cosine")
            .head(recommend_no)
            .reset_index()
        )
    elif recommender == JAC_SIM:
        result = (
            model_fac.get_recommendation(model_id, metric="jaccard")
            .head(recommend_no)
            .reset_index()
        )
    elif recommender == GNN:
        result = (
            model_fac.get_recommendation(model_id, metric="gnn")
            .head(recommend_no)
            .reset_index()
        )
        result["score"] = result["score"] / np.linalg.norm(result["score"])
    else:
        st.error(f"Recommender feature '{recommender}' coming soon")

    # reset index to start from 1 for display
    result.index = result.index + 1
    result["modelId"] = result.apply(lambda row: make_clickable(row["modelId"]), axis=1)

    # Sort by option
    result = SORT_BY_METHOD[sort_by](result)

    # center result
    st.write(
        "<style>table {margin: 0 auto; text-align: center;}</style>",
        unsafe_allow_html=True,
    )
    st.write(f"Top {recommend_no} recommendations:")

    bar_plot(result, recommender)
    # use st.dataframe for sortable columns
    # st.dataframe(
    #     result[DISPLAY_COLS], use_container_width=True, unsafe_allow_html=True
    # )

    # use st.write for clickable links
    st.write(result[DISPLAY_COLS].to_html(escape=False), unsafe_allow_html=True)
