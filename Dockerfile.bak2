FROM python:slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn pymysql cryptography

COPY app app
COPY migrations migrations
COPY InvestigationApp.py config.py boot.sh ./
RUN chmod a+x boot.sh

ENV FLASK_APP InvestigationApp.py
RUN flask translate compile

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
