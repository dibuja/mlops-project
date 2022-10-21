FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR /app

COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt --no-cache-dir

RUN cd /app && mkdir -p src/data && mkdir -p data/raw
COPY src/data /app/src/data
COPY data/ /app/data/
RUN python src/data/make_dataset.py

COPY src/ src/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]