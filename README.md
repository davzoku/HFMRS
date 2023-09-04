# HFMRS

## Usage

### Streamlit

**Default Port**: `8051`

You can run HFMRS as a Streamlit webapp locally.

From root directory, run the following command

```
streamlit run src/streamlit.py
```

#### Running Streamlit in Docker

Run the following command

```
docker compose -f docker-compose.streamlit.yml up
```

### FastAPI

**Default Port**: `8000`

You can also run HFMRS as an backend API using FastAPI.

From root directory, run the following command

```
uvicorn src.fastapiapp:app --reload
```

#### Running FastAPI in Docker

Run the following command

```
docker compose -f docker-compose.fastapi.yml up
```

## API Documentation

The Swagger API documentation can be found at this link after running uvicorn

```
http://localhost:8000/docs
```

Alternatively, you can try out the APIs by importing the Postman collection in the `assets` folder into Postman.

## Chrome Extension

We can use HFMRS as a browser extension for chromium-based browsers eg. Chrome. For example, while browsing for the [gpt2](https://huggingface.co/gpt2) model, HFMRS will suggest similar models on the sidebar.

To try this locally, the FastAPI has to be running in the background.

Subsequently, you can follow the steps below:

- Go to Chrome settings, click on `Extensions` on the left sidebar
- Enable `Developer mode`
- Click on `Load Unpacked`
- Import the `chrome-ext` folder in this repository
- Go to any HuggingFace model page such as [gpt2](https://huggingface.co/gpt2) model and observe the last section of the right sidebard

## Future Work

- Add more configurations such as
  - toggle Recommender algorithms
  - configure number of suggestions
  - and more
