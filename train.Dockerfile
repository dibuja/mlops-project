FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR /app

COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt --no-cache-dir

# DVC-adapted
RUN cd /app && mkdir -p src/data && mkdir -p data/raw
COPY .dvc /app/.dvc
COPY data.dvc /app/data.dvc
RUN dvc config core.no_scm true
RUN dvc pull
RUN python src/data/make_dataset.py

COPY src/ src/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]